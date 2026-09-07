from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from scripts.analyze_cases_and_spatial import CASE_CMAP, CASE_LEGEND
from scripts.run_experiment import load_confirmed_checkpoint, select_device, sha256
from scripts.run_p2_interpretability import draw_cam_axis, wafer_background
from wm811k.constants import LABEL_TO_INDEX, LABELS
from wm811k.interpretability import grad_cam
from wm811k.model import WaferCNN
from wm811k.preprocessing import encode_wafer_map
from wm811k.provenance import verify_dataset_matches_splits, verify_split_file_matches_metadata


def select_wafer_row(
    predictions: pd.DataFrame,
    *,
    seed: int,
    source_row_id: int | None,
    true_label: str | None,
    only_errors: bool,
) -> pd.Series:
    """Select one prediction row deterministically, without performing I/O."""
    if source_row_id is not None:
        selected = predictions.loc[predictions["source_row_id"] == source_row_id]
        if selected.empty:
            raise ValueError(f"source_row_id={source_row_id}인 행이 예측 CSV에 없습니다.")
        return selected.iloc[0]

    candidates = predictions
    if true_label is not None:
        candidates = candidates.loc[candidates["true_label"] == true_label]
    if only_errors:
        candidates = candidates.loc[candidates["y_true"] != candidates["y_pred"]]

    if candidates.empty:
        conditions: list[str] = []
        if true_label is not None:
            conditions.append(f"true_label={true_label}")
        if only_errors:
            conditions.append("오분류만")
        description = ", ".join(conditions) if conditions else "전체 예측"
        raise ValueError(f"선택 조건({description})에 맞는 웨이퍼 후보가 없습니다.")

    candidates = candidates.sort_values("source_row_id", kind="stable")
    rng = np.random.default_rng(seed)
    return candidates.iloc[int(rng.integers(0, len(candidates)))]


def predictions_path_for(
    project_root: Path,
    experiment: str,
    split: str,
    *,
    final_selected_experiment: str,
) -> Path:
    """Return the saved prediction CSV path, with a reproducible build hint."""
    path = project_root / "artifacts" / experiment / f"{split}_predictions.csv"
    if path.exists():
        return path

    if split == "test" and experiment != final_selected_experiment:
        raise FileNotFoundError(
            f"예측 CSV가 없습니다: {path}\n"
            "테스트 평가는 config의 final_selected_experiment"
            f"({final_selected_experiment})에만 허용됩니다. "
            f"{experiment}로는 테스트 예측 CSV를 만들 수 없습니다."
        )

    command = (
        rf".\.venv\Scripts\python.exe .\scripts\run_experiment.py "
        f"--experiment {experiment}"
    )
    if split == "test":
        command += " --stage test --confirm-test"
    raise FileNotFoundError(
        f"예측 CSV가 없습니다: {path}\n다음 명령으로 생성하세요: {command}"
    )


def default_output_path(
    project_root: Path,
    experiment: str,
    split: str,
    source_row_id: int,
) -> Path:
    """Derive the default private, per-wafer figure path."""
    return (
        project_root
        / "reports"
        / "figures"
        / f"sample_wafer_{experiment}_{split}_row{source_row_id}.png"
    )


def guard_output_path(output: Path, project_root: Path) -> None:
    """Keep per-wafer figures inside the explicitly ignored private namespace."""
    figure_directory = (project_root / "reports" / "figures").resolve()
    resolved_output = output.resolve()
    if resolved_output.parent != figure_directory:
        raise SystemExit(
            "--output은 reports/figures 바로 아래여야 합니다. "
            "개별 웨이퍼·로트 식별자는 공개 대상이 아닙니다."
        )
    if not resolved_output.name.startswith("sample_wafer_"):
        raise SystemExit(
            "--output 파일명은 sample_wafer_로 시작해야 합니다. "
            "개별 웨이퍼·로트 식별자는 공개 대상이 아닙니다."
        )


def _read_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def _parse_args() -> argparse.Namespace:
    initial = argparse.ArgumentParser(add_help=False)
    initial.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "experiments.json",
    )
    initial.add_argument("--experiment")
    known, _ = initial.parse_known_args()

    config = _read_config(known.config)
    experiments = config["experiments"]
    # P1의 E3b/E3c는 repeat-seed 경로에서만 실행했으므로 새 베이스 런을 만들지 않는다.
    base_experiment_names = ("E1", "E2", "E3")
    cnn_experiments = [
        name
        for name in base_experiment_names
        if name in experiments and experiments[name]["kind"] == "cnn"
    ]
    default_experiment = config["project"]["final_selected_experiment"]

    parser = argparse.ArgumentParser(
        description="저장된 CNN 체크포인트로 웨이퍼 한 장을 추론하고 시각화합니다."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=PROJECT_ROOT / "data" / "LSWMD.pkl",
    )
    parser.add_argument(
        "--splits",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "splits.csv",
    )
    parser.add_argument("--config", type=Path, default=known.config)
    parser.add_argument(
        "--experiment",
        choices=cnn_experiments,
        default=default_experiment,
    )
    parser.add_argument(
        "--split",
        choices=("validation", "test"),
        default="validation",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--source-row-id", type=int)
    parser.add_argument("--true-label", choices=LABELS)
    parser.add_argument("--only-errors", action="store_true")
    parser.add_argument("--show-lot", action="store_true")
    parser.add_argument("--output", type=Path)

    if (
        known.experiment in experiments
        and experiments[known.experiment]["kind"] != "cnn"
    ):
        parser.error(
            f"실험 {known.experiment}은 CNN 실험이 아니므로 "
            "체크포인트도 conv layer도 없다."
        )
    return parser.parse_args()


def _input_label_grid(encoded: torch.Tensor) -> np.ndarray:
    encoded_array = encoded.detach().cpu().numpy()
    if encoded_array.shape[0] == 1:
        return encoded_array[0].astype(np.uint8)

    die_mask = encoded_array[0]
    defect_mask = encoded_array[1]
    grid = np.zeros(die_mask.shape, dtype=np.uint8)
    grid[(die_mask == 1) & (defect_mask == 0)] = 1
    grid[defect_mask == 1] = 2
    return grid


def _cam_encoded(input_grid: np.ndarray) -> torch.Tensor:
    return torch.stack(
        (
            torch.as_tensor(input_grid > 0, dtype=torch.float32),
            torch.as_tensor(input_grid == 2, dtype=torch.float32),
        )
    )


def _build_title(
    selected: pd.Series,
    *,
    experiment: str,
    split: str,
    probability: float,
    correct: bool,
    show_lot: bool,
    cam_degenerate: bool,
) -> str:
    title = (
        f"Experiment: {experiment} | Split: {split} | "
        f"source_row_id: {int(selected['source_row_id'])} | "
        f"True: {selected['true_label']} | "
        f"Pred: {selected['predicted_label']} ({probability:.1%}) | "
        f"Correct: {'Yes' if correct else 'No'}"
    )
    if show_lot:
        title += f" | Lot: {selected['lotName']}"
    if cam_degenerate:
        title += " | CAM degenerate: zero"
    return title


def _checkpoint_load_kwargs(
    config: dict,
    args: argparse.Namespace,
    experiment_config: dict,
    split_hash: str,
    device: torch.device,
) -> dict:
    return {
        "device": device,
        "experiment_name": args.experiment,
        "experiment": experiment_config,
        "split_hash": split_hash,
        "seed": int(config["project"]["seed"]),
    }


def _validate_true_label(selected: pd.Series) -> None:
    stored_index = int(selected["y_true"])
    stored_label = str(selected["true_label"])
    if LABEL_TO_INDEX.get(stored_label) != stored_index:
        raise ValueError(
            "실제 라벨 정합성 오류: "
            f"y_true={stored_index}, true_label={stored_label}"
        )


def _save_figure(
    *,
    wafer_map: np.ndarray,
    input_grid: np.ndarray,
    cam: np.ndarray,
    selected: pd.Series,
    experiment: str,
    split: str,
    probability: float,
    correct: bool,
    show_lot: bool,
    cam_degenerate: bool,
    output_path: Path,
) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)

    axes[0].imshow(
        wafer_map,
        cmap=CASE_CMAP,
        vmin=0,
        vmax=2,
        interpolation="nearest",
    )
    axes[0].set_title("Original wafer map")
    axes[0].legend(
        handles=CASE_LEGEND,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.17),
        ncol=3,
        frameon=False,
    )

    axes[1].imshow(
        input_grid,
        cmap=CASE_CMAP,
        vmin=0,
        vmax=2,
        interpolation="nearest",
    )
    axes[1].set_title(f"Model input ({input_grid.shape[1]}x{input_grid.shape[0]})")

    cam_encoded = _cam_encoded(input_grid)
    axes[2].imshow(wafer_background(cam_encoded), interpolation="nearest")
    cam_title = "Grad-CAM (predicted class)"
    if cam_degenerate:
        cam_title += " [CAM degenerate: zero]"
    draw_cam_axis(axes[2], cam_encoded, cam, cam_title)

    for axis in axes:
        axis.set_axis_off()

    figure.suptitle(
        _build_title(
            selected,
            experiment=experiment,
            split=split,
            probability=probability,
            correct=correct,
            show_lot=show_lot,
            cam_degenerate=cam_degenerate,
        )
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> int:
    args = _parse_args()
    config = _read_config(args.config)
    experiment_config = config["experiments"][args.experiment]

    if args.output is not None:
        guard_output_path(args.output, PROJECT_ROOT)

    if args.source_row_id is not None and (
        args.true_label is not None or args.only_errors
    ):
        print("안내: --source-row-id가 지정되어 --true-label과 --only-errors를 무시합니다.")

    prediction_path = predictions_path_for(
        PROJECT_ROOT,
        args.experiment,
        args.split,
        final_selected_experiment=config["project"]["final_selected_experiment"],
    )
    predictions = pd.read_csv(prediction_path)
    selected = select_wafer_row(
        predictions,
        seed=args.seed,
        source_row_id=args.source_row_id,
        true_label=args.true_label,
        only_errors=args.only_errors,
    )
    _validate_true_label(selected)
    source_row_id = int(selected["source_row_id"])

    verify_dataset_matches_splits(args.data, args.splits)
    verify_split_file_matches_metadata(args.splits)
    raw = pd.read_pickle(args.data)
    wafer_map = np.asarray(raw.iloc[source_row_id]["waferMap"])
    encoded = encode_wafer_map(
        wafer_map,
        output_size=experiment_config["input_size"],
        mode=experiment_config["input_mode"],
        validate_values=False,
    )
    inputs = encoded.to(dtype=torch.float32)

    device = select_device()
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    checkpoint_path = PROJECT_ROOT / "artifacts" / args.experiment / "best.pt"
    checkpoint = load_confirmed_checkpoint(
        checkpoint_path,
        **_checkpoint_load_kwargs(
            config,
            args,
            experiment_config,
            sha256(args.splits),
            device,
        ),
    )
    model = WaferCNN(
        input_channels=experiment_config["input_channels"],
        num_classes=len(LABELS),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    batch = inputs.unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(batch)
        probabilities = torch.softmax(logits, dim=1)
        predicted_index = int(probabilities.argmax(dim=1).item())
        probability = float(probabilities[0, predicted_index].item())
    predicted_label = LABELS[predicted_index]

    stored_index = int(selected["y_pred"])
    stored_label = str(selected["predicted_label"])
    stored_label_index = LABEL_TO_INDEX.get(stored_label)
    if (
        predicted_index != stored_index
        or predicted_label != stored_label
        or stored_label_index != stored_index
    ):
        top_values = logits.topk(2, dim=1).values
        top2_margin = float((top_values[0, 0] - top_values[0, 1]).item())
        if top2_margin <= 1e-3:
            margin_hint = "마진이 좁아 수치 잡음일 수 있습니다."
        else:
            margin_hint = "마진이 넓어 다른 체크포인트일 가능성이 있습니다."
        raise ValueError(
            "재추론 결과가 예측 CSV와 일치하지 않습니다: "
            f"재추론={predicted_index}/{predicted_label}, "
            f"저장값={stored_index}/{stored_label}, "
            f"top-2 로짓 마진={top2_margin:.6g}. {margin_hint}"
        )

    grad_cam_output = grad_cam(
        model,
        model.features[-1][0],
        batch,
        torch.tensor([predicted_index]),
        output_size=(
            experiment_config["input_size"],
            experiment_config["input_size"],
        ),
    )
    cam = grad_cam_output.cams[0].detach().cpu().numpy()
    cam_degenerate = bool(grad_cam_output.zero_maps[0].item())
    input_grid = _input_label_grid(encoded)
    correct = int(selected["y_true"]) == predicted_index

    output_path = args.output or default_output_path(
        PROJECT_ROOT,
        args.experiment,
        args.split,
        source_row_id,
    )
    _save_figure(
        wafer_map=wafer_map,
        input_grid=input_grid,
        cam=cam,
        selected=selected,
        experiment=args.experiment,
        split=args.split,
        probability=probability,
        correct=correct,
        show_lot=args.show_lot,
        cam_degenerate=cam_degenerate,
        output_path=output_path,
    )

    if cam_degenerate:
        print(
            "경고: Grad-CAM이 전부 0으로 나왔습니다. 모델이 결함을 안 본 것이 아니라 "
            "CAM 계산 자체가 퇴화했습니다."
        )
    print(
        f"선택한 웨이퍼: source_row_id={source_row_id}, "
        f"실제={selected['true_label']}, 예측={predicted_label}, "
        f"확률={probability:.4f}, 정답={'예' if correct else '아니오'}"
    )
    print(
        "재현 명령: "
        rf".\.venv\Scripts\python.exe .\scripts\show_one_wafer.py "
        f"--experiment {args.experiment} --split {args.split} "
        f"--source-row-id {source_row_id}"
    )
    print(f"저장 파일: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
