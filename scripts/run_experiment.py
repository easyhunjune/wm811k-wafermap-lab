from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from wm811k.constants import LABEL_TO_INDEX, LABELS
from wm811k.data import prepare_labeled_frame
from wm811k.dataset import WaferDataset
from wm811k.metrics import (
    class_recall_cluster_interval,
    classification_summary,
    exact_binomial_interval,
    macro_f1_cluster_interval,
)
from wm811k.model import WaferCNN
from wm811k.provenance import verify_dataset_matches_splits, verify_split_file_matches_metadata
from wm811k.splitting import SPLIT_NAMES, split_audit
from wm811k.training import evaluate, fit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a configured experiment.")
    parser.add_argument(
        "--experiment",
        choices=("E0", "E1", "E2", "E3", "E3b", "E3c"),
        required=True,
    )
    parser.add_argument("--stage", choices=("validation", "test"), default="validation")
    parser.add_argument("--confirm-test", action="store_true")
    parser.add_argument("--data", type=Path, default=PROJECT_ROOT / "data" / "LSWMD.pkl")
    parser.add_argument("--splits", type=Path, default=PROJECT_ROOT / "artifacts" / "splits.csv")
    parser.add_argument(
        "--config", type=Path, default=PROJECT_ROOT / "configs" / "experiments.json"
    )
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "artifacts")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Training seed. When set, save under artifacts/repeat/{EXP}/seed{N}.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override the maximum epochs (intended for smoke tests only).",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def select_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_run_dir(output_dir: Path, experiment: str, seed_override: int | None) -> Path:
    if seed_override is None:
        return output_dir / experiment
    return output_dir / "repeat" / experiment / f"seed{seed_override}"


def result_path_for(run_dir: Path, stage: str) -> Path:
    return run_dir / f"{stage}_metrics.json"


def guard_existing_results(run_dir: Path, *, stage: str, seed_override: int | None) -> None:
    """Refuse runs that would overwrite a completed result or a confirmed checkpoint."""
    result_path = result_path_for(run_dir, stage)
    if seed_override is not None and result_path.exists():
        raise SystemExit(
            f"Completed repeat result already exists: {result_path}. "
            "Remove that seed directory explicitly before rerunning it."
        )
    if stage == "test" and result_path.exists():
        raise SystemExit(
            f"Test result already exists: {result_path}. "
            "Do not overwrite it; archive the run before an intentional new evaluation."
        )
    confirmed_test = result_path_for(run_dir, "test")
    if stage == "validation" and confirmed_test.exists():
        raise SystemExit(
            f"Test evaluation is already confirmed here: {confirmed_test}. "
            f"Retraining would replace {run_dir / 'best.pt'}, the checkpoint those numbers "
            "came from. Archive the run directory before rerunning validation."
        )


def guard_selected_experiment(experiment_name: str, selected: str) -> None:
    """Keep the single test evaluation on the experiment frozen after validation."""
    if experiment_name != selected:
        raise SystemExit(
            f"Only the selected experiment may be tested: {selected}, not {experiment_name}. "
            "Testing a second experiment would turn the held-out set into a selection "
            "criterion. Change final_selected_experiment in the config if the choice "
            "really did change, and record why."
        )


def guard_checkpoint_matches(
    checkpoint: dict,
    *,
    experiment_name: str,
    experiment: dict,
    split_hash: str,
    seed: int,
) -> None:
    """Refuse a checkpoint that was not produced by the run now being evaluated."""
    expected = {
        "experiment": experiment_name,
        "experiment_config": experiment,
        "label_to_index": LABEL_TO_INDEX,
        "split_sha256": split_hash,
        "seed": seed,
    }
    mismatched = sorted(key for key, value in expected.items() if checkpoint.get(key) != value)
    if mismatched:
        raise ValueError(
            f"Checkpoint does not match this run; mismatched: {mismatched}. "
            "Retrain the experiment or restore the checkpoint these settings produced."
        )


def load_confirmed_checkpoint(
    checkpoint_path: Path,
    *,
    device: torch.device,
    experiment_name: str,
    experiment: dict,
    split_hash: str,
    seed: int,
) -> dict:
    """Load the checkpoint for a non-training stage, refusing one from a different run."""
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Best checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    guard_checkpoint_matches(
        checkpoint,
        experiment_name=experiment_name,
        experiment=experiment,
        split_hash=split_hash,
        seed=seed,
    )
    return checkpoint


def git_commit() -> str | None:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return completed.stdout.strip() or None if completed.returncode == 0 else None


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_frame(data_path: Path, split_path: Path) -> pd.DataFrame:
    if not data_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {data_path}")
    if not split_path.is_file():
        raise FileNotFoundError(f"Split file not found: {split_path}")
    verify_dataset_matches_splits(data_path, split_path)
    verify_split_file_matches_metadata(split_path)
    raw = pd.read_pickle(data_path)
    frame = prepare_labeled_frame(raw, labels=LABELS)
    splits = pd.read_csv(split_path)
    expected = {"source_row_id", "split"}
    if not expected.issubset(splits.columns):
        raise ValueError(f"Split file must contain {sorted(expected)}.")
    if not splits["split"].isin(SPLIT_NAMES).all():
        raise ValueError(f"Split values must be valid split names: {SPLIT_NAMES}.")
    if splits["source_row_id"].duplicated().any():
        raise ValueError("Duplicate source_row_id values; each row must have exactly one split.")
    merged = frame.merge(
        splits[["source_row_id", "split"]],
        on="source_row_id",
        how="left",
        validate="one_to_one",
    )
    if merged["split"].isna().any():
        raise ValueError("Some labeled rows have no split assignment.")
    split_audit(merged, labels=LABELS)
    return merged


def class_weights(
    train_frame: pd.DataFrame,
    device: torch.device,
    *,
    formula: str,
    clip: dict | None = None,
    normalization: str = "mean_one",
) -> tuple[torch.Tensor, dict]:
    indices = train_frame["failure_label"].map(LABEL_TO_INDEX).to_numpy()
    counts = np.bincount(indices, minlength=len(LABELS)).astype(np.float64)
    if np.any(counts == 0):
        raise ValueError(f"Cannot calculate class weights with zero train count: {counts.tolist()}")
    if formula == "inverse_frequency":
        weights = 1.0 / counts
    elif formula == "sqrt_inverse_frequency":
        weights = 1.0 / np.sqrt(counts)
    else:
        raise ValueError(f"Unsupported class-weight formula: {formula}")
    unclipped_ratio = float(weights.max() / weights.min())
    clipped_count = 0
    if clip is not None:
        max_min_ratio = float(clip["max_min_ratio"])
        if max_min_ratio <= 1.0:
            raise ValueError("class-weight max_min_ratio must be greater than 1.")
        upper_bound = weights.min() * max_min_ratio
        clipped_count = int(np.count_nonzero(weights > upper_bound))
        weights = np.minimum(weights, upper_bound)
    if normalization == "mean_one":
        # The arithmetic mean over the nine classes is 1. Because training averages the
        # per-sample losses over a batch, the scale that actually reaches the optimizer is
        # the sample-weighted mean below, which falls well under 1 on an imbalanced split.
        weights /= weights.mean()
    elif normalization == "sample_mean_one":
        # The mean over training samples is 1, so a weighted run keeps the same loss scale
        # as an unweighted one and shares its learning rate on equal terms.
        weights /= np.average(weights, weights=counts)
    else:
        raise ValueError(f"Unsupported class-weight normalization: {normalization}")
    metadata = {
        "formula": formula,
        "normalization": normalization,
        "clip": clip,
        "counts": {label: int(counts[index]) for index, label in enumerate(LABELS)},
        "weights": {label: float(weights[index]) for index, label in enumerate(LABELS)},
        "unclipped_max_min_ratio": unclipped_ratio,
        "max_min_ratio": float(weights.max() / weights.min()),
        "clipped_class_count": clipped_count,
        # What the batch-averaged loss is actually multiplied by, relative to an
        # unweighted run. 1.0 means the weighted and unweighted losses share a scale.
        "effective_sample_scale": float(np.average(weights, weights=counts)),
    }
    return torch.as_tensor(weights, dtype=torch.float32, device=device), metadata


def predictions_frame(
    source: pd.DataFrame,
    truth: np.ndarray,
    prediction: np.ndarray,
) -> pd.DataFrame:
    inverse_labels = dict(enumerate(LABELS))
    return pd.DataFrame(
        {
            "source_row_id": source["source_row_id"].to_numpy(),
            "lotName": source["lotName"].astype(str).to_numpy(),
            "y_true": truth,
            "y_pred": prediction,
            "true_label": [inverse_labels[int(value)] for value in truth],
            "predicted_label": [inverse_labels[int(value)] for value in prediction],
        }
    )


def bootstrap_report(
    truth: np.ndarray,
    prediction: np.ndarray,
    groups: np.ndarray,
    *,
    n_bootstrap: int,
    seed: int,
) -> dict[str, object]:
    label_indices = tuple(range(len(LABELS)))
    macro = macro_f1_cluster_interval(
        truth,
        prediction,
        groups,
        labels=label_indices,
        n_bootstrap=n_bootstrap,
        seed=seed,
    )
    recall = {}
    for index, label in enumerate(LABELS):
        class_mask = truth == index
        cluster_interval = class_recall_cluster_interval(
            truth,
            prediction,
            groups,
            target_label=index,
            n_bootstrap=n_bootstrap,
            seed=seed + index + 1,
        ).to_dict()
        cluster_interval["iid_exact_binomial_reference"] = exact_binomial_interval(
            int(np.sum(prediction[class_mask] == index)),
            int(class_mask.sum()),
        )
        recall[label] = cluster_interval
    return {"macro_f1": macro.to_dict(), "per_class_recall": recall}


def run_majority(
    frame: pd.DataFrame,
    *,
    stage: str,
    n_bootstrap: int,
    seed: int,
) -> tuple[dict[str, object], pd.DataFrame]:
    train = frame[frame["split"] == "train"]
    target = frame[frame["split"] == stage]
    majority_label = train["failure_label"].value_counts().idxmax()
    majority_index = LABEL_TO_INDEX[majority_label]
    truth = target["failure_label"].map(LABEL_TO_INDEX).to_numpy(dtype=np.int64)
    prediction = np.full_like(truth, majority_index)
    result = {
        "majority_label": majority_label,
        "metrics": classification_summary(truth, prediction, labels=range(len(LABELS))),
    }
    if stage == "test":
        result["cluster_bootstrap_ci"] = bootstrap_report(
            truth,
            prediction,
            target["lotName"].astype(str).to_numpy(),
            n_bootstrap=n_bootstrap,
            seed=seed,
        )
    return result, predictions_frame(target, truth, prediction)


def run_cnn(
    frame: pd.DataFrame,
    *,
    experiment_name: str,
    experiment: dict,
    training: dict,
    stage: str,
    run_dir: Path,
    split_hash: str,
    seed: int,
    n_bootstrap: int,
) -> tuple[dict[str, object], pd.DataFrame]:
    device = select_device()
    train_frame = frame[frame["split"] == "train"].reset_index(drop=True)
    validation_frame = frame[frame["split"] == "validation"].reset_index(drop=True)
    target_frame = frame[frame["split"] == stage].reset_index(drop=True)
    dataset_args = {
        "mode": experiment["input_mode"],
        "output_size": experiment["input_size"],
    }
    loader_args = {
        "batch_size": training["batch_size"],
        "num_workers": training["num_workers"],
        "pin_memory": device.type == "cuda",
    }
    train_loader = DataLoader(
        WaferDataset(train_frame, **dataset_args),
        shuffle=True,
        **loader_args,
    )
    validation_loader = DataLoader(
        WaferDataset(validation_frame, **dataset_args),
        shuffle=False,
        **loader_args,
    )
    target_loader = DataLoader(
        WaferDataset(target_frame, **dataset_args),
        shuffle=False,
        **loader_args,
    )
    model = WaferCNN(
        input_channels=experiment["input_channels"],
        num_classes=len(LABELS),
    ).to(device)
    weight_metadata = None
    weights = None
    if experiment["loss"] == "weighted_cross_entropy":
        weights, weight_metadata = class_weights(
            train_frame,
            device,
            formula=experiment["class_weight"]["formula"],
            clip=experiment["class_weight"].get("clip"),
            normalization=experiment["class_weight"].get("normalization", "mean_one"),
        )
    criterion = nn.CrossEntropyLoss(weight=weights, reduction="none")
    checkpoint_path = run_dir / "best.pt"

    if stage == "validation":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=training["learning_rate"],
            weight_decay=training["weight_decay"],
        )
        history, checkpoint = fit(
            model,
            train_loader,
            validation_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epochs=training["epochs"],
            patience=training["early_stopping_patience"],
            min_delta=training["early_stopping_min_delta"],
            checkpoint_path=checkpoint_path,
            checkpoint_metadata={
                "experiment": experiment_name,
                "experiment_config": experiment,
                "label_to_index": LABEL_TO_INDEX,
                "split_sha256": split_hash,
                "class_weight": weight_metadata,
                "seed": seed,
            },
        )
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        checkpoint = load_confirmed_checkpoint(
            checkpoint_path,
            device=device,
            experiment_name=experiment_name,
            experiment=experiment,
            split_hash=split_hash,
            seed=seed,
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        history = None

    evaluated = evaluate(model, target_loader, criterion=criterion, device=device)
    result = {
        "device": str(device),
        "best_epoch": checkpoint["epoch"],
        "best_validation_macro_f1": checkpoint["best_validation_macro_f1"],
        "checkpoint_sha256": sha256(checkpoint_path),
        "metrics": classification_summary(
            evaluated.y_true,
            evaluated.y_pred,
            labels=range(len(LABELS)),
        ),
        "class_weight": weight_metadata or checkpoint.get("class_weight"),
        "history": history,
    }
    if stage == "test":
        result["cluster_bootstrap_ci"] = bootstrap_report(
            evaluated.y_true,
            evaluated.y_pred,
            target_frame["lotName"].astype(str).to_numpy(),
            n_bootstrap=n_bootstrap,
            seed=seed,
        )
    return result, predictions_frame(target_frame, evaluated.y_true, evaluated.y_pred)


def main() -> int:
    args = parse_args()
    if args.epochs is not None and args.epochs < 1:
        raise SystemExit("--epochs must be at least 1.")
    if args.seed is not None and args.stage == "test":
        raise SystemExit("Repeat-seed runs are validation-only and must not touch the test set.")
    if args.stage == "test" and not args.confirm_test:
        raise SystemExit(
            "Test evaluation is locked. Freeze the configuration, then add --confirm-test."
        )
    with args.config.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    project = config["project"]
    if tuple(project["labels"]) != LABELS:
        raise ValueError("Configured label order does not match wm811k.constants.LABELS.")
    training = dict(project["training"])
    if args.epochs is not None:
        training["epochs"] = args.epochs
    if args.stage == "test":
        guard_selected_experiment(args.experiment, project["final_selected_experiment"])
    experiment = config["experiments"][args.experiment]
    seed = int(project["seed"]) if args.seed is None else int(args.seed)
    set_seed(seed)
    split_hash = sha256(args.splits)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_dir = resolve_run_dir(args.output_dir, args.experiment, args.seed)
    result_path = result_path_for(run_dir, args.stage)
    guard_existing_results(run_dir, stage=args.stage, seed_override=args.seed)

    run_dir.mkdir(parents=True, exist_ok=True)
    started_at = utc_now()
    started = time.perf_counter()
    base_meta = {
        "experiment": args.experiment,
        "seed": seed,
        "stage": args.stage,
        "split_sha256": split_hash,
        "started_at": started_at,
        "device": str(select_device()),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "git_commit": git_commit(),
    }
    try:
        frame = load_frame(args.data, args.splits)
        if args.experiment == "E0":
            result, predictions = run_majority(
                frame,
                stage=args.stage,
                n_bootstrap=training["bootstrap_replicates"],
                seed=seed,
            )
        else:
            result, predictions = run_cnn(
                frame,
                experiment_name=args.experiment,
                experiment=experiment,
                training=training,
                stage=args.stage,
                run_dir=run_dir,
                split_hash=split_hash,
                seed=seed,
                n_bootstrap=training["bootstrap_replicates"],
            )
        result.update(
            {
                "experiment": args.experiment,
                "stage": args.stage,
                "seed": seed,
                "split_sha256": split_hash,
                "test_confirmed": bool(args.stage == "test" and args.confirm_test),
            }
        )
        prediction_path = run_dir / f"{args.stage}_predictions.csv"
        result_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        predictions.to_csv(prediction_path, index=False)
        history = result.get("history") or []
        run_meta = {
            **base_meta,
            "finished_at": utc_now(),
            "duration_seconds": time.perf_counter() - started,
            "epochs_run": len(history),
            "best_epoch": result.get("best_epoch"),
            "early_stopped": bool(history and len(history) < training["epochs"]),
            "status": "completed",
        }
    except BaseException as error:
        run_meta = {
            **base_meta,
            "finished_at": utc_now(),
            "duration_seconds": time.perf_counter() - started,
            "epochs_run": None,
            "best_epoch": None,
            "early_stopped": False,
            "status": "failed",
            "error": f"{type(error).__name__}: {error}",
        }
        (run_dir / "run_meta.json").write_text(
            json.dumps(run_meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        raise
    (run_dir / "run_meta.json").write_text(
        json.dumps(run_meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"saved: {result_path}")
    print(f"saved: {prediction_path}")
    print(f"saved: {run_dir / 'run_meta.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
