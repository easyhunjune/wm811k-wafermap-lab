from __future__ import annotations

import argparse
from itertools import pairwise
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from wm811k.constants import LABELS
from wm811k.provenance import verify_dataset_matches_splits, verify_split_file_matches_metadata
from wm811k.spatial import edge_counts, radial_counts

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CASE_CMAP = ListedColormap(["#d1d5db", "#f8fafc", "#dc2626"])
CASE_LEGEND = [
    Patch(facecolor="#d1d5db", label="No die"),
    Patch(facecolor="#f8fafc", edgecolor="#9ca3af", label="Normal die"),
    Patch(facecolor="#dc2626", label="Defect die"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build test error and spatial analysis.")
    parser.add_argument("--data", type=Path, default=PROJECT_ROOT / "data" / "LSWMD.pkl")
    parser.add_argument(
        "--predictions",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "E2" / "test_predictions.csv",
    )
    parser.add_argument(
        "--reports",
        type=Path,
        default=PROJECT_ROOT / "reports",
    )
    parser.add_argument(
        "--splits",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "splits.csv",
        help="Hashed against its metadata, which also ties source_row_id to --data.",
    )
    parser.add_argument("--examples-per-case", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def safe_slug(label: str) -> str:
    return label.lower().replace("-", "_")


def sample_rows(frame: pd.DataFrame, count: int, rng: np.random.Generator) -> pd.DataFrame:
    if len(frame) <= count:
        return frame.copy()
    positions = rng.choice(len(frame), size=count, replace=False)
    return frame.iloc[np.sort(positions)].copy()


def plot_case_grid(
    selected: pd.DataFrame,
    raw: pd.DataFrame,
    *,
    true_label: str,
    case_type: str,
    destination: Path,
) -> None:
    figure, axes = plt.subplots(2, 5, figsize=(13, 5.6))
    flat_axes = axes.ravel()
    for axis in flat_axes:
        axis.axis("off")

    for axis, (_, row) in zip(flat_axes, selected.iterrows(), strict=False):
        wafer_map = np.asarray(raw.iloc[int(row["source_row_id"])]["waferMap"])
        axis.imshow(wafer_map, cmap=CASE_CMAP, vmin=0, vmax=2, interpolation="nearest")
        prediction = row["predicted_label"]
        title = (
            f"pred: {prediction}\nrow: {int(row['source_row_id'])}"
            if case_type == "error"
            else f"correct\nrow: {int(row['source_row_id'])}"
        )
        axis.set_title(title, fontsize=8)
        axis.axis("off")

    if selected.empty:
        flat_axes[0].text(
            0.5,
            0.5,
            "No observed cases",
            ha="center",
            va="center",
            transform=flat_axes[0].transAxes,
        )
    figure.suptitle(f"E2 test — {true_label} — {case_type} cases", fontsize=14)
    figure.legend(
        handles=CASE_LEGEND,
        loc="lower center",
        ncol=3,
        frameon=False,
    )
    figure.tight_layout(rect=(0, 0.08, 1, 0.94))
    figure.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(figure)


def build_case_figures(
    predictions: pd.DataFrame,
    raw: pd.DataFrame,
    *,
    reports: Path,
    examples_per_case: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    figure_directory = reports / "figures"
    figure_directory.mkdir(parents=True, exist_ok=True)
    selections: list[pd.DataFrame] = []

    for label in LABELS:
        true_rows = predictions[predictions["true_label"] == label]
        cases = {
            "correct": true_rows[true_rows["y_true"] == true_rows["y_pred"]],
            "error": true_rows[true_rows["y_true"] != true_rows["y_pred"]],
        }
        for case_type, candidates in cases.items():
            selected = sample_rows(candidates, examples_per_case, rng)
            selected["case_type"] = case_type
            selections.append(selected)
            destination = figure_directory / f"e2_{safe_slug(label)}_{case_type}_cases.png"
            plot_case_grid(
                selected,
                raw,
                true_label=label,
                case_type=case_type,
                destination=destination,
            )

    manifest = pd.concat(selections, ignore_index=True)
    manifest.to_csv(reports / "e2_case_manifest.csv", index=False)
    return manifest


def build_spatial_tables(
    predictions: pd.DataFrame,
    raw: pd.DataFrame,
    *,
    reports: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    bin_edges = np.linspace(0.0, 1.0, 11)
    radial_totals = {
        label: {
            "die": np.zeros(10, dtype=np.int64),
            "defect": np.zeros(10, dtype=np.int64),
        }
        for label in LABELS
    }
    edge_totals = {
        label: {threshold: {"die": 0, "defect": 0} for threshold in (0.90, 0.95)}
        for label in LABELS
    }

    for row in predictions.itertuples(index=False):
        wafer_map = np.asarray(raw.iloc[int(row.source_row_id)]["waferMap"])
        die_count, defect_count = radial_counts(wafer_map, bin_edges)
        radial_totals[row.true_label]["die"] += die_count
        radial_totals[row.true_label]["defect"] += defect_count
        for threshold in (0.90, 0.95):
            edge_die, edge_defect = edge_counts(wafer_map, threshold)
            edge_totals[row.true_label][threshold]["die"] += edge_die
            edge_totals[row.true_label][threshold]["defect"] += edge_defect

    radial_rows = []
    for label in LABELS:
        for index, (start, end) in enumerate(pairwise(bin_edges)):
            die_count = int(radial_totals[label]["die"][index])
            defect_count = int(radial_totals[label]["defect"][index])
            radial_rows.append(
                {
                    "class": label,
                    "radius_start": start,
                    "radius_end": end,
                    "die_count": die_count,
                    "defect_count": defect_count,
                    "defect_density": defect_count / die_count if die_count else np.nan,
                }
            )
    radial_frame = pd.DataFrame(radial_rows)
    radial_frame.to_csv(reports / "e2_test_radial_profile.csv", index=False)

    edge_rows = []
    for label in LABELS:
        for threshold in (0.90, 0.95):
            counts = edge_totals[label][threshold]
            edge_rows.append(
                {
                    "class": label,
                    "zone": f"outer_{round((1 - threshold) * 100)}pct",
                    "radius_threshold": threshold,
                    "die_count": counts["die"],
                    "defect_count": counts["defect"],
                    "defect_density": (
                        counts["defect"] / counts["die"] if counts["die"] else np.nan
                    ),
                }
            )
    edge_frame = pd.DataFrame(edge_rows)
    edge_frame.to_csv(reports / "e2_test_edge_zones.csv", index=False)
    return radial_frame, edge_frame


def plot_radial_profiles(frame: pd.DataFrame, destination: Path) -> None:
    figure, axes = plt.subplots(3, 3, figsize=(12, 10), sharex=True, sharey=True)
    for axis, label in zip(axes.ravel(), LABELS, strict=True):
        subset = frame[frame["class"] == label]
        centers = (subset["radius_start"] + subset["radius_end"]) / 2
        axis.plot(centers, subset["defect_density"], marker="o", linewidth=1.8)
        axis.set_title(label)
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1)
        axis.grid(alpha=0.25)
    figure.supxlabel("Normalized radius r/R")
    figure.supylabel("Die-weighted defect density")
    figure.suptitle("E2 test — radial defect density by true class", fontsize=14)
    figure.tight_layout(rect=(0.03, 0.03, 1, 0.96))
    figure.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_edge_zones(frame: pd.DataFrame, destination: Path) -> None:
    pivot = frame.pivot(index="class", columns="zone", values="defect_density").reindex(LABELS)
    positions = np.arange(len(LABELS))
    width = 0.38
    figure, axis = plt.subplots(figsize=(12, 5.8))
    axis.bar(
        positions - width / 2,
        pivot["outer_10pct"],
        width,
        label="outer 10% (r/R ≥ 0.90)",
    )
    axis.bar(
        positions + width / 2,
        pivot["outer_5pct"],
        width,
        label="outer 5% (r/R ≥ 0.95)",
    )
    axis.set_xticks(positions, LABELS, rotation=35, ha="right")
    axis.set_ylim(0, 1)
    axis.set_ylabel("Die-weighted defect density")
    axis.set_title("E2 test — normalized edge-zone defect density")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> int:
    args = parse_args()
    args.reports.mkdir(parents=True, exist_ok=True)
    # source_row_id below indexes raw positionally, so the dataset has to be the
    # one the splits were built from.
    verify_dataset_matches_splits(args.data, args.splits)
    verify_split_file_matches_metadata(args.splits)
    raw = pd.read_pickle(args.data)
    predictions = pd.read_csv(args.predictions)
    build_case_figures(
        predictions,
        raw,
        reports=args.reports,
        examples_per_case=args.examples_per_case,
        seed=args.seed,
    )
    radial_frame, edge_frame = build_spatial_tables(
        predictions,
        raw,
        reports=args.reports,
    )
    plot_radial_profiles(
        radial_frame,
        args.reports / "figures" / "e2_test_radial_profiles.png",
    )
    plot_edge_zones(
        edge_frame,
        args.reports / "figures" / "e2_test_edge_zones.png",
    )
    print(f"saved reports under: {args.reports}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
