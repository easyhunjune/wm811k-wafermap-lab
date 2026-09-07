from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from wm811k.constants import LABELS
from wm811k.data import normalize_failure_label
from wm811k.paths import portable_dataset_path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit the local WM-811K pickle.")
    parser.add_argument("--data", type=Path, default=PROJECT_ROOT / "data" / "LSWMD.pkl")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "data_audit.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    frame = pd.read_pickle(args.data)
    labels = frame["failureType"].map(normalize_failure_label)
    recognized = labels.isin(LABELS)
    unexpected = labels[labels.notna() & ~recognized].value_counts().to_dict()

    shape_counts: Counter[tuple[int, int]] = Counter()
    values: set[int | float] = set()
    invalid_dimensions = 0
    empty_maps = 0
    for wafer_map in frame["waferMap"]:
        array = np.asarray(wafer_map)
        if array.ndim != 2:
            invalid_dimensions += 1
            continue
        if 0 in array.shape:
            empty_maps += 1
            continue
        shape_counts[tuple(int(value) for value in array.shape)] += 1
        values.update(np.unique(array).tolist())

    split_column = next(
        (name for name in ("trainTestLabel", "trianTestLabel") if name in frame.columns),
        None,
    )
    report = {
        "dataset_path": portable_dataset_path(args.data),
        "file_size_bytes": args.data.stat().st_size,
        "sha256": sha256(args.data),
        "rows": len(frame),
        "columns": frame.columns.tolist(),
        "required_columns_present": {
            column: column in frame.columns
            for column in ("waferMap", "dieSize", "lotName", "waferIndex", "failureType")
        },
        "provided_split_column": split_column,
        "label_audit": {
            "unlabeled_rows": int(labels.isna().sum()),
            "recognized_labeled_rows": int(recognized.sum()),
            "counts": {label: int((labels == label).sum()) for label in LABELS},
            "unexpected_labels": unexpected,
        },
        "lot_audit": {
            "missing_lot_name": int(frame["lotName"].isna().sum()),
            "distinct_lots_all_rows": int(frame["lotName"].nunique(dropna=True)),
            "distinct_lots_labeled_rows": int(
                frame.loc[recognized, "lotName"].nunique(dropna=True)
            ),
        },
        "wafer_map_audit": {
            "observed_values": sorted(values),
            "values_valid_0_1_2": values.issubset({0, 1, 2}),
            "invalid_dimension_count": invalid_dimensions,
            "empty_map_count": empty_maps,
            "distinct_shapes": len(shape_counts),
            "height_range": [
                min(shape[0] for shape in shape_counts),
                max(shape[0] for shape in shape_counts),
            ],
            "width_range": [
                min(shape[1] for shape in shape_counts),
                max(shape[1] for shape in shape_counts),
            ],
            "most_common_shapes": [
                {"shape": list(shape), "count": count}
                for shape, count in shape_counts.most_common(20)
            ],
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
