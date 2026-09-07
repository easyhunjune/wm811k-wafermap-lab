from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from wm811k.data import prepare_labeled_frame
from wm811k.paths import portable_dataset_path
from wm811k.splitting import find_group_split, split_audit

__all__ = ["file_sha256", "main", "portable_dataset_path"]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create leakage-aware WM-811K splits.")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--config", type=Path, default=PROJECT_ROOT / "configs" / "experiments.json"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.data.is_file():
        raise FileNotFoundError(f"Dataset not found: {args.data}")

    with args.config.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    split_config = config["project"]["split"]
    labels = tuple(config["project"]["labels"])

    raw = pd.read_pickle(args.data)
    frame = prepare_labeled_frame(raw, labels=labels)
    split_frame, metadata = find_group_split(
        frame,
        labels=labels,
        candidate_seeds=split_config["candidate_seeds"],
        label_column="failure_label",
        group_column=split_config["group_column"],
        id_column="source_row_id",
    )
    audit = split_audit(
        frame.merge(split_frame, on="source_row_id", validate="one_to_one"),
        labels=labels,
        label_column="failure_label",
        group_column=split_config["group_column"],
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    split_frame.to_csv(args.output, index=False)
    metadata_path = args.output.with_suffix(".metadata.json")
    metadata.update(
        {
            "dataset_path": portable_dataset_path(args.data),
            "dataset_sha256": file_sha256(args.data),
            "split_file_sha256": file_sha256(args.output),
            "audit": audit,
        }
    )
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"saved: {args.output}")
    print(f"saved: {metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
