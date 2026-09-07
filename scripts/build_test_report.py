from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

from wm811k.constants import LABELS

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def describe_run(predictions_path: Path) -> tuple[str, str]:
    """Name a predictions file by its run directory and stage: ("E2 Test", "e2_test")."""
    stem = predictions_path.stem
    suffix = "_predictions"
    stage = stem.removesuffix(suffix)
    run_dir = predictions_path.parent
    # Repeat runs live in artifacts/repeat/{experiment}/seed{N}, so the seed directory
    # alone would not say which experiment produced the file.
    names = [run_dir.name]
    if run_dir.name.startswith("seed"):
        names.insert(0, run_dir.parent.name)
    label = " ".join([*names, stage.capitalize()])
    slug = "_".join(name.lower() for name in [*names, stage])
    return label, slug


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build class metrics and confusion matrix.")
    parser.add_argument(
        "--predictions",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "E2" / "test_predictions.csv",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Defaults to reports/{experiment}_{stage}_classification_report.csv.",
    )
    parser.add_argument(
        "--figure",
        type=Path,
        default=None,
        help="Defaults to reports/figures/{experiment}_{stage}_confusion_matrix.png.",
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Figure title. Defaults to the experiment and stage of --predictions.",
    )
    args = parser.parse_args()
    label, slug = describe_run(args.predictions)
    if args.report is None:
        args.report = PROJECT_ROOT / "reports" / f"{slug}_classification_report.csv"
    if args.figure is None:
        args.figure = PROJECT_ROOT / "reports" / "figures" / f"{slug}_confusion_matrix.png"
    if args.title is None:
        args.title = f"{label} Confusion Matrix — Row Normalized"
    return args


def main() -> int:
    args = parse_args()
    predictions = pd.read_csv(args.predictions)
    labels = list(range(len(LABELS)))
    report = classification_report(
        predictions["y_true"],
        predictions["y_pred"],
        labels=labels,
        target_names=LABELS,
        output_dict=True,
        zero_division=0,
    )
    report_frame = pd.DataFrame(report).transpose()
    args.report.parent.mkdir(parents=True, exist_ok=True)
    report_frame.to_csv(args.report, index_label="class")

    matrix = confusion_matrix(
        predictions["y_true"],
        predictions["y_pred"],
        labels=labels,
        normalize="true",
    )
    args.figure.parent.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="white")
    figure, axis = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        matrix,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        vmin=0,
        vmax=1,
        xticklabels=LABELS,
        yticklabels=LABELS,
        square=True,
        cbar_kws={"label": "Row-normalized rate"},
        ax=axis,
    )
    axis.set_title(args.title)
    axis.set_xlabel("Predicted label")
    axis.set_ylabel("True label")
    figure.tight_layout()
    figure.savefig(args.figure, dpi=180, bbox_inches="tight")
    plt.close(figure)
    print(f"saved: {args.report}")
    print(f"saved: {args.figure}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
