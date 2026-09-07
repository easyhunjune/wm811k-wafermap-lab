from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from wm811k.constants import LABEL_TO_INDEX, LABELS
from wm811k.data import prepare_labeled_frame
from wm811k.dataset import WaferDataset
from wm811k.interpretability import cam_overlap_metrics, grad_cam
from wm811k.model import WaferCNN
from wm811k.preprocessing import encode_wafer_map

EXPECTED_HASHES = {
    "data/LSWMD.pkl": "1d04fccb3dd3176b276878b926b20fead7e077c5751e4d353ea9741a5e7b5c65",
    "artifacts/splits.csv": "11fa5f986749f905e6b498b0a40bd4464c0575d6f73fa41215f6269b79f36090",
    "artifacts/E2/best.pt": "8743739f2402765904d9bc6e192def9f28f66bc1b8f192dfedf667fdd51a9903",
    "artifacts/E2/validation_predictions.csv": (
        "a9118669304c6e771104380d4ece2bfaa177cbf44308b00361ea38b11770da67"
    ),
    "artifacts/E2/test_predictions.csv": (
        "bf53a84800968adedb633cd6c689bcb16ace504d55f25b62616c8381d7341941"
    ),
    # Pinned over the LF bytes git checks out (.gitattributes sets eol=lf for every text
    # file). The previous pin was taken from a CRLF working copy on Windows, so every
    # fresh clone failed the integrity check before doing anything. The reported metrics
    # are unchanged; tests/test_publication.py keeps this pin aligned with the tracked file.
    "artifacts/E2/test_metrics.json": (
        "3cf96b04d0e6747c8090a387e030059df19128cd93194c82e0a808c761863358"
    ),
}
GROUP_SPECS = (
    ("scratch_correct", "Scratch", "Scratch"),
    ("scratch_to_none", "Scratch", "none"),
    ("loc_correct", "Loc", "Loc"),
    ("loc_to_none", "Loc", "none"),
)
EXPECTED_TEST_GROUP_COUNTS = {
    "scratch_correct": 109,
    "scratch_to_none": 89,
    "loc_correct": 410,
    "loc_to_none": 223,
}
PROCEDURE = {
    "version": "p2-gradcam-v1",
    "model": "E2",
    "target_layer": "model.features[3][0]",
    "input_mode": "two_channel_masks",
    "output_size": [64, 64],
    "groups": [list(item) for item in GROUP_SPECS],
    "correct_cam": "actual_class_logit",
    "miss_cam": ["actual_class_logit", "none_logit"],
    "normalization": "per_map_min_max_after_relu",
    "metrics": ["defect_cam_mass_ratio", "dilated_defect_cam_mass_ratio"],
    "dilation": "3x3_one_pixel",
    "sample_seed": 42,
    "samples_per_group": 10,
    "cluster_bootstrap_replicates": 2000,
}
PROCEDURE_SIGNATURE = hashlib.sha256(
    json.dumps(PROCEDURE, sort_keys=True, separators=(",", ":")).encode()
).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the preregistered P2 Grad-CAM analysis.")
    parser.add_argument("--stage", choices=("validation", "test"), required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--data", type=Path, default=PROJECT_ROOT / "data" / "LSWMD.pkl")
    parser.add_argument(
        "--splits",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "splits.csv",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "E2" / "best.pt",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "p2",
    )
    parser.add_argument(
        "--figure-dir",
        type=Path,
        default=PROJECT_ROOT / "reports" / "figures",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def verify_fixed_hashes(
    *, data_path: Path, splits_path: Path, checkpoint_path: Path
) -> dict[str, str]:
    """Confirm the P2 inputs are the exact artefacts the reported results came from.

    These pins are tied to one confirmed E2 test run. Re-running that test — after
    archiving artifacts/E2 as guard_existing_results() requires — changes best.pt and
    test_metrics.json, so every pin above must be regenerated together and the reported
    P2 figures rebuilt. run_experiment.py also records checkpoint_sha256 inside
    *_metrics.json now, which is one more reason a re-run shifts the JSON hash.

    These are byte-level pins over the files this run actually reads: --data, --splits
    and --checkpoint as given on the command line, plus the E2 prediction and metrics
    files under the repository. Editing a pinned JSON for any reason — even an addition
    that leaves every reported number alone — invalidates them and must be paired with
    a pin update.
    """
    load_paths = {
        "data/LSWMD.pkl": data_path,
        "artifacts/splits.csv": splits_path,
        "artifacts/E2/best.pt": checkpoint_path,
    }
    observed = {}
    for relative_path, expected in EXPECTED_HASHES.items():
        path = load_paths.get(relative_path, PROJECT_ROOT / relative_path)
        if not path.is_file():
            raise FileNotFoundError(f"Required fixed file not found: {path}")
        actual = sha256(path)
        if actual != expected:
            raise ValueError(
                f"Integrity failure for {relative_path} at {path}: "
                f"expected {expected}, observed {actual}"
            )
        observed[relative_path] = actual
    return observed


def select_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(requested)


def load_model(checkpoint_path: Path, split_hash: str, device: torch.device) -> WaferCNN:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    expected_config = {
        "kind": "cnn",
        "input_mode": "two_channel_masks",
        "input_channels": 2,
        "input_size": 64,
        "loss": "cross_entropy",
    }
    if checkpoint.get("experiment") != "E2":
        raise ValueError("Checkpoint is not the fixed E2 experiment.")
    if checkpoint.get("experiment_config") != expected_config:
        raise ValueError("Checkpoint experiment configuration is not the registered E2 setup.")
    if checkpoint.get("label_to_index") != LABEL_TO_INDEX:
        raise ValueError("Checkpoint label mapping does not match the project label mapping.")
    if checkpoint.get("split_sha256") != split_hash:
        raise ValueError("Checkpoint and current fixed split hashes do not match.")
    if checkpoint.get("seed") != 42 or checkpoint.get("epoch") != 18:
        raise ValueError("Checkpoint seed or best epoch differs from the registered values.")
    if checkpoint.get("class_weight") is not None:
        raise ValueError("The registered E2 checkpoint must not use class weights.")
    if "features.3.0.weight" not in checkpoint["model_state_dict"]:
        raise ValueError("Registered Grad-CAM target layer is absent from the checkpoint.")

    model = WaferCNN(input_channels=2, num_classes=len(LABELS)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return model


def load_aligned_stage(
    data_path: Path,
    split_path: Path,
    predictions_path: Path,
    stage: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = pd.read_pickle(data_path)
    frame = prepare_labeled_frame(raw, labels=LABELS)
    splits = pd.read_csv(split_path)
    if not {"source_row_id", "split"}.issubset(splits.columns):
        raise ValueError("Fixed split file is missing required columns.")
    frame = frame.merge(
        splits[["source_row_id", "split"]],
        on="source_row_id",
        how="left",
        validate="one_to_one",
    )
    if frame["split"].isna().any():
        raise ValueError("Some labeled source rows do not have a split assignment.")
    stage_frame = frame[frame["split"] == stage].reset_index(drop=True)
    predictions = pd.read_csv(predictions_path)
    required_prediction_columns = {
        "source_row_id",
        "lotName",
        "y_true",
        "y_pred",
        "true_label",
        "predicted_label",
    }
    if not required_prediction_columns.issubset(predictions.columns):
        raise ValueError("Stored prediction file is missing required columns.")
    if predictions["source_row_id"].duplicated().any():
        raise ValueError("Stored predictions contain duplicate source_row_id values.")
    if len(stage_frame) != len(predictions):
        raise ValueError("Stored prediction row count differs from the fixed split.")
    if not np.array_equal(
        stage_frame["source_row_id"].to_numpy(),
        predictions["source_row_id"].to_numpy(),
    ):
        raise ValueError("Stored prediction rows are not aligned to the fixed split order.")
    expected_truth = stage_frame["failure_label"].map(LABEL_TO_INDEX).to_numpy()
    if not np.array_equal(expected_truth, predictions["y_true"].to_numpy()):
        raise ValueError("Stored prediction truth labels do not align with source data.")
    if not np.array_equal(
        stage_frame["failure_label"].astype(str).to_numpy(),
        predictions["true_label"].astype(str).to_numpy(),
    ):
        raise ValueError("Stored prediction label names do not align with source data.")
    if not np.array_equal(
        stage_frame["lotName"].astype(str).to_numpy(),
        predictions["lotName"].astype(str).to_numpy(),
    ):
        raise ValueError("Stored prediction lot names do not align with source data.")
    return stage_frame, predictions


def reinfer_predictions(
    model: WaferCNN,
    frame: pd.DataFrame,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    loader = DataLoader(
        WaferDataset(frame, mode="two_channel_masks", output_size=64),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    predictions = []
    with torch.inference_mode():
        for inputs, _targets in loader:
            logits = model(inputs.to(device, non_blocking=True))
            predictions.append(logits.argmax(dim=1).cpu().numpy())
    return np.concatenate(predictions)


def build_groups(predictions: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for group_name, true_label, predicted_label in GROUP_SPECS:
        subset = predictions[
            (predictions["true_label"] == true_label)
            & (predictions["predicted_label"] == predicted_label)
        ].copy()
        subset["case_group"] = group_name
        frames.append(subset)
    groups = pd.concat(frames, ignore_index=True)
    if groups["source_row_id"].duplicated().any():
        raise ValueError("A source row was assigned to more than one P2 case group.")
    return groups


def select_manifest(groups: pd.DataFrame, stage: str) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    selected_frames = []
    for group_name, _true_label, _predicted_label in GROUP_SPECS:
        candidates = groups[groups["case_group"] == group_name]
        count = min(10, len(candidates))
        positions = rng.choice(len(candidates), size=count, replace=False)
        selected = candidates.iloc[np.sort(positions)].copy()
        selected.insert(0, "stage", stage)
        selected_frames.append(selected)
    return pd.concat(selected_frames, ignore_index=True)


def build_cam_tasks(groups: pd.DataFrame) -> pd.DataFrame:
    tasks = []
    for row in groups.itertuples(index=False):
        tasks.append(
            {
                "source_row_id": int(row.source_row_id),
                "lotName": str(row.lotName),
                "true_label": row.true_label,
                "predicted_label": row.predicted_label,
                "case_group": row.case_group,
                "cam_target_role": "actual",
                "cam_target_label": row.true_label,
                "cam_target_index": LABEL_TO_INDEX[row.true_label],
            }
        )
        if row.predicted_label == "none" and row.true_label != "none":
            tasks.append(
                {
                    "source_row_id": int(row.source_row_id),
                    "lotName": str(row.lotName),
                    "true_label": row.true_label,
                    "predicted_label": row.predicted_label,
                    "case_group": row.case_group,
                    "cam_target_role": "none",
                    "cam_target_label": "none",
                    "cam_target_index": LABEL_TO_INDEX["none"],
                }
            )
    return pd.DataFrame(tasks)


def calculate_cams(
    model: WaferCNN,
    target_layer: torch.nn.Module,
    stage_frame: pd.DataFrame,
    tasks: pd.DataFrame,
    selected_ids: set[int],
    device: torch.device,
    batch_size: int,
) -> tuple[pd.DataFrame, dict[tuple[int, str], np.ndarray]]:
    source = stage_frame.set_index("source_row_id", verify_integrity=True)
    metric_rows = []
    selected_cams = {}
    for start in range(0, len(tasks), batch_size):
        batch_tasks = tasks.iloc[start : start + batch_size]
        encoded = torch.stack(
            [
                encode_wafer_map(
                    source.at[int(row.source_row_id), "waferMap"],
                    output_size=64,
                    mode="two_channel_masks",
                    validate_values=False,
                )
                for row in batch_tasks.itertuples(index=False)
            ]
        )
        targets = torch.as_tensor(
            batch_tasks["cam_target_index"].to_numpy(),
            dtype=torch.long,
            device=device,
        )
        result = grad_cam(
            model,
            target_layer,
            encoded.to(device, non_blocking=True),
            targets,
            output_size=(64, 64),
        )
        inferred = result.logits.argmax(dim=1).cpu().numpy()
        expected = batch_tasks["predicted_label"].map(LABEL_TO_INDEX).to_numpy()
        if not np.array_equal(inferred, expected):
            raise ValueError("Grad-CAM forward predictions differ from stored E2 predictions.")
        probabilities = result.logits.softmax(dim=1)
        target_probabilities = probabilities.gather(1, targets.view(-1, 1)).squeeze(1)

        cams = result.cams.cpu()
        zero_maps = result.zero_maps.cpu().numpy()
        target_probabilities = target_probabilities.detach().cpu().numpy()
        for offset, row in enumerate(batch_tasks.itertuples(index=False)):
            cam = cams[offset]
            die_mask = encoded[offset, 0]
            defect_mask = encoded[offset, 1]
            metrics = cam_overlap_metrics(cam, die_mask, defect_mask)
            values = (
                metrics.defect_mass_ratio,
                metrics.dilated_defect_mass_ratio,
                metrics.padding_cam_mass_ratio,
            )
            if any(np.isfinite(value) and not 0.0 <= value <= 1.0 for value in values):
                raise ValueError("A CAM overlap metric is outside [0, 1].")
            metric_rows.append(
                {
                    "source_row_id": int(row.source_row_id),
                    "lotName": row.lotName,
                    "true_label": row.true_label,
                    "predicted_label": row.predicted_label,
                    "case_group": row.case_group,
                    "cam_target_role": row.cam_target_role,
                    "cam_target_label": row.cam_target_label,
                    "target_probability": float(target_probabilities[offset]),
                    "zero_cam": bool(zero_maps[offset]),
                    "die_cam_mass": metrics.die_cam_mass,
                    "padding_cam_mass_ratio": metrics.padding_cam_mass_ratio,
                    "defect_cam_mass_ratio": metrics.defect_mass_ratio,
                    "dilated_defect_cam_mass_ratio": (
                        metrics.dilated_defect_mass_ratio
                    ),
                }
            )
            key = (int(row.source_row_id), row.cam_target_role)
            if int(row.source_row_id) in selected_ids:
                selected_cams[key] = cam.numpy()
    return pd.DataFrame(metric_rows), selected_cams


def cluster_bootstrap_difference(
    correct: pd.DataFrame,
    missed: pd.DataFrame,
    metric: str,
    *,
    seed: int,
    replicates: int = 2000,
) -> tuple[float, float, float, int]:
    point = float(correct[metric].median() - missed[metric].median())
    lots = np.array(sorted(set(correct["lotName"]).union(missed["lotName"])))
    correct_by_lot = {
        lot: correct.loc[correct["lotName"] == lot, metric].dropna().to_numpy()
        for lot in lots
    }
    missed_by_lot = {
        lot: missed.loc[missed["lotName"] == lot, metric].dropna().to_numpy()
        for lot in lots
    }
    rng = np.random.default_rng(seed)
    differences = []
    for _ in range(replicates):
        sampled_lots = rng.choice(lots, size=len(lots), replace=True)
        correct_values = [
            correct_by_lot[lot] for lot in sampled_lots if correct_by_lot[lot].size
        ]
        missed_values = [
            missed_by_lot[lot] for lot in sampled_lots if missed_by_lot[lot].size
        ]
        if not correct_values or not missed_values:
            continue
        differences.append(
            float(
                np.median(np.concatenate(correct_values))
                - np.median(np.concatenate(missed_values))
            )
        )
    if not differences:
        return point, float("nan"), float("nan"), 0
    low, high = np.quantile(differences, [0.025, 0.975])
    return point, float(low), float(high), len(differences)


def summarize_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (case_group, role), subset in metrics.groupby(
        ["case_group", "cam_target_role"],
        sort=False,
    ):
        exact = subset["defect_cam_mass_ratio"]
        dilated = subset["dilated_defect_cam_mass_ratio"]
        rows.append(
            {
                "case_group": case_group,
                "true_label": subset["true_label"].iloc[0],
                "outcome": "missed" if case_group.endswith("to_none") else "correct",
                "cam_target_role": role,
                "n_cases": len(subset),
                "n_valid": int(exact.notna().sum()),
                "n_missing": int(exact.isna().sum()),
                "n_zero_cam": int(subset["zero_cam"].sum()),
                "exact_q1": exact.quantile(0.25),
                "exact_median": exact.median(),
                "exact_q3": exact.quantile(0.75),
                "dilated_q1": dilated.quantile(0.25),
                "dilated_median": dilated.median(),
                "dilated_q3": dilated.quantile(0.75),
                "padding_mass_ratio_median": subset["padding_cam_mass_ratio"].median(),
            }
        )
    summary = pd.DataFrame(rows)
    for label_index, label in enumerate(("Scratch", "Loc")):
        correct = metrics[
            (metrics["true_label"] == label)
            & (metrics["predicted_label"] == label)
            & (metrics["cam_target_role"] == "actual")
        ]
        missed = metrics[
            (metrics["true_label"] == label)
            & (metrics["predicted_label"] == "none")
            & (metrics["cam_target_role"] == "actual")
        ]
        comparison_values = {}
        for metric_index, metric in enumerate(
            ("defect_cam_mass_ratio", "dilated_defect_cam_mass_ratio")
        ):
            point, low, high, valid = cluster_bootstrap_difference(
                correct,
                missed,
                metric,
                seed=42 + label_index * 10 + metric_index,
            )
            prefix = "exact" if metric_index == 0 else "dilated"
            comparison_values.update(
                {
                    f"{prefix}_correct_minus_missed": point,
                    f"{prefix}_difference_ci_low": low,
                    f"{prefix}_difference_ci_high": high,
                    f"{prefix}_bootstrap_valid": valid,
                }
            )
        mask = (summary["true_label"] == label) & (
            summary["cam_target_role"] == "actual"
        )
        for column, value in comparison_values.items():
            summary.loc[mask, column] = value
    return summary


def wafer_background(encoded: torch.Tensor) -> np.ndarray:
    die = encoded[0].numpy()
    defect = encoded[1].numpy()
    background = np.ones((64, 64, 3), dtype=np.float32)
    background[die > 0] = (0.82, 0.84, 0.87)
    background[defect > 0] = (0.05, 0.05, 0.05)
    return background


def draw_cam_axis(
    axis: plt.Axes,
    encoded: torch.Tensor,
    cam: np.ndarray,
    title: str,
) -> None:
    axis.imshow(wafer_background(encoded), interpolation="nearest")
    axis.imshow(cam, cmap="magma", vmin=0, vmax=1, alpha=0.68, interpolation="bilinear")
    defect = encoded[1].numpy()
    if defect.any():
        axis.contour(defect, levels=[0.5], colors=["#00e5ff"], linewidths=0.45)
    axis.set_title(title, fontsize=8)
    axis.axis("off")


def plot_group(
    selected: pd.DataFrame,
    stage_frame: pd.DataFrame,
    cams: dict[tuple[int, str], np.ndarray],
    group_name: str,
    destination: Path,
) -> None:
    rows = selected[selected["case_group"] == group_name].reset_index(drop=True)
    source = stage_frame.set_index("source_row_id", verify_integrity=True)
    is_missed = group_name.endswith("to_none")
    if is_missed:
        figure, axes = plt.subplots(5, 4, figsize=(13, 16))
        flat_axes = axes.ravel()
        for axis in flat_axes:
            axis.axis("off")
        for index, row in rows.iterrows():
            encoded = encode_wafer_map(
                source.at[int(row["source_row_id"]), "waferMap"],
                output_size=64,
                mode="two_channel_masks",
                validate_values=False,
            )
            base = (index // 2) * 4 + (index % 2) * 2
            row_id = int(row["source_row_id"])
            draw_cam_axis(
                flat_axes[base],
                encoded,
                cams[(row_id, "actual")],
                f"row {row_id} · {row['true_label']} CAM",
            )
            draw_cam_axis(
                flat_axes[base + 1],
                encoded,
                cams[(row_id, "none")],
                f"row {row_id} · none CAM",
            )
    else:
        figure, axes = plt.subplots(2, 5, figsize=(14, 7))
        flat_axes = axes.ravel()
        for axis in flat_axes:
            axis.axis("off")
        for axis, row in zip(flat_axes, rows.itertuples(index=False), strict=False):
            encoded = encode_wafer_map(
                source.at[int(row.source_row_id), "waferMap"],
                output_size=64,
                mode="two_channel_masks",
                validate_values=False,
            )
            draw_cam_axis(
                axis,
                encoded,
                cams[(int(row.source_row_id), "actual")],
                f"row {int(row.source_row_id)} · {row.true_label} CAM",
            )
    pretty_name = group_name.replace("_", " ")
    figure.suptitle(
        f"E2 test Grad-CAM · {pretty_name}\ncyan outline: encoded defect mask",
        fontsize=14,
    )
    figure.subplots_adjust(
        top=0.90 if is_missed else 0.82,
        bottom=0.03,
        hspace=0.26,
        wspace=0.12,
    )
    figure.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_overlap_comparison(metrics: pd.DataFrame, destination: Path) -> None:
    actual = metrics[metrics["cam_target_role"] == "actual"].copy()
    order = [item[0] for item in GROUP_SPECS]
    labels = [name.replace("_", "\n") for name in order]
    figure, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)
    for axis, metric, title in zip(
        axes,
        ("defect_cam_mass_ratio", "dilated_defect_cam_mass_ratio"),
        ("Exact defect mask", "3×3 dilated defect mask"),
        strict=True,
    ):
        values = [
            actual.loc[actual["case_group"] == group, metric].dropna().to_numpy()
            for group in order
        ]
        axis.boxplot(values, tick_labels=labels, showfliers=False)
        axis.set_title(title)
        axis.set_ylim(0, 1)
        axis.grid(axis="y", alpha=0.25)
        axis.tick_params(axis="x", labelsize=8)
    axes[0].set_ylabel("CAM mass ratio within defect region")
    figure.suptitle("E2 test · actual-class Grad-CAM overlap")
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    figure.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(figure)


def write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def validate_test_gate(output_dir: Path, hashes: dict[str, str]) -> dict:
    gate_path = output_dir / "validation_gate.json"
    if not gate_path.is_file():
        raise FileNotFoundError("Validation gate is absent; run --stage validation first.")
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    if not gate.get("passed"):
        raise ValueError("Validation Grad-CAM gate did not pass.")
    if gate.get("procedure_signature") != PROCEDURE_SIGNATURE:
        raise ValueError("Validation gate used a different Grad-CAM procedure.")
    if gate.get("fixed_hashes") != hashes:
        raise ValueError("Fixed files changed after validation.")
    return gate


def main() -> int:
    args = parse_args()
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be positive.")
    started_at = datetime.now(timezone.utc)
    hashes = verify_fixed_hashes(
        data_path=args.data, splits_path=args.splits, checkpoint_path=args.checkpoint
    )
    device = select_device(args.device)
    torch.manual_seed(42)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    model = load_model(
        args.checkpoint,
        hashes["artifacts/splits.csv"],
        device,
    )
    target_layer = model.features[3][0]
    predictions_path = (
        PROJECT_ROOT / "artifacts" / "E2" / f"{args.stage}_predictions.csv"
    )
    stage_frame, predictions = load_aligned_stage(
        args.data,
        args.splits,
        predictions_path,
        args.stage,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.stage == "test":
        validation_gate = validate_test_gate(args.output_dir, hashes)
    else:
        validation_gate = None

    reinferred = reinfer_predictions(
        model,
        stage_frame,
        device,
        args.batch_size,
    )
    mismatch_count = int(np.count_nonzero(reinferred != predictions["y_pred"].to_numpy()))
    if mismatch_count:
        raise ValueError(
            f"Stored and reinferred {args.stage} predictions differ in "
            f"{mismatch_count} rows."
        )

    groups = build_groups(predictions)
    group_counts = groups["case_group"].value_counts().to_dict()
    if any(group_counts.get(name, 0) == 0 for name, _true, _predicted in GROUP_SPECS):
        raise ValueError(f"One or more registered case groups are empty: {group_counts}")
    if args.stage == "test" and group_counts != EXPECTED_TEST_GROUP_COUNTS:
        raise ValueError(
            "Fixed test case counts differ from the preregistered values: "
            f"{group_counts}"
        )

    manifest = select_manifest(groups, args.stage)
    tasks = build_cam_tasks(groups)
    metrics, selected_cams = calculate_cams(
        model,
        target_layer,
        stage_frame,
        tasks,
        set(manifest["source_row_id"].astype(int)),
        device,
        args.batch_size,
    )
    if metrics["zero_cam"].all():
        raise ValueError("All registered cases produced zero CAMs.")
    if not (metrics["die_cam_mass"] > 0).any():
        raise ValueError("All CAM mass is confined outside the encoded wafer.")
    summary = summarize_metrics(metrics)

    prefix = "validation_" if args.stage == "validation" else ""
    metrics.to_csv(args.output_dir / f"{prefix}overlap_metrics.csv", index=False)
    summary.to_csv(args.output_dir / f"{prefix}group_summary.csv", index=False)
    if args.stage == "test":
        manifest.to_csv(args.output_dir / "case_manifest.csv", index=False)
        args.figure_dir.mkdir(parents=True, exist_ok=True)
        figure_names = {
            "scratch_correct": "p2_scratch_correct_gradcam.png",
            "scratch_to_none": "p2_scratch_to_none_gradcam.png",
            "loc_correct": "p2_loc_correct_gradcam.png",
            "loc_to_none": "p2_loc_to_none_gradcam.png",
        }
        for group_name, figure_name in figure_names.items():
            plot_group(
                manifest,
                stage_frame,
                selected_cams,
                group_name,
                args.figure_dir / figure_name,
            )
        plot_overlap_comparison(
            metrics,
            args.figure_dir / "p2_overlap_comparison.png",
        )

    completed_at = datetime.now(timezone.utc)
    run_meta = {
        "stage": args.stage,
        "started_at_utc": started_at.isoformat(),
        "completed_at_utc": completed_at.isoformat(),
        "duration_seconds": (completed_at - started_at).total_seconds(),
        "device": str(device),
        "torch_version": torch.__version__,
        "procedure": PROCEDURE,
        "procedure_signature": PROCEDURE_SIGNATURE,
        "fixed_hashes": hashes,
        "checkpoint_metadata": {
            "experiment": "E2",
            "seed": 42,
            "best_epoch": 18,
            "target_layer": "model.features[3][0]",
            "split_sha256": hashes["artifacts/splits.csv"],
        },
        "stage_rows": len(stage_frame),
        "prediction_mismatch_count": mismatch_count,
        "group_counts": group_counts,
        "cam_task_count": len(tasks),
        "zero_cam_count": int(metrics["zero_cam"].sum()),
        "missing_overlap_count": int(metrics["defect_cam_mass_ratio"].isna().sum()),
        "validation_gate": validation_gate,
    }
    if args.stage == "validation":
        gate = {
            "passed": True,
            "stage": "validation",
            "procedure_signature": PROCEDURE_SIGNATURE,
            "fixed_hashes": hashes,
            "prediction_mismatch_count": mismatch_count,
            "group_counts": group_counts,
            "cam_task_count": len(tasks),
            "zero_cam_count": int(metrics["zero_cam"].sum()),
            "missing_overlap_count": int(
                metrics["defect_cam_mass_ratio"].isna().sum()
            ),
            "all_cam_shapes": [64, 64],
            "all_cam_ranges_valid": True,
            "all_overlap_ranges_valid": True,
            "some_cam_mass_inside_die": True,
        }
        write_json(args.output_dir / "validation_gate.json", gate)
        write_json(args.output_dir / "validation_run_meta.json", run_meta)
    else:
        write_json(args.output_dir / "run_meta.json", run_meta)

    print(
        f"P2 {args.stage} passed: rows={len(stage_frame)}, groups={group_counts}, "
        f"tasks={len(tasks)}, zero_cams={int(metrics['zero_cam'].sum())}, "
        f"prediction_mismatches={mismatch_count}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
