"""Check that a consumer is reading the dataset the splits were built from, and
the split file that was written from it.

`source_row_id` is a positional index into the frame as it stood when
`prepare_splits.py` ran. Anything that re-opens the pickle and indexes by that
column — case galleries, spatial statistics, the training merge — is silently
wrong if the row order differs, and a different distribution or a re-extraction
is enough to change it. Nothing about the failure looks like an error: the
numbers still compute, they just describe other wafers.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

CHUNK_BYTES = 1024 * 1024


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def split_metadata_path(splits_path: Path) -> Path:
    """prepare_splits.py writes metadata beside the split file it produced."""
    return splits_path.with_suffix(".metadata.json")


def verify_dataset_matches_splits(data_path: Path, splits_path: Path) -> str:
    """Refuse a dataset whose hash differs from the one the splits were built on.

    Returns the verified hash. Raises FileNotFoundError when the metadata written
    alongside the splits is missing, and ValueError when it records a different
    dataset or omits the hash entirely.
    """
    metadata_path = split_metadata_path(splits_path)
    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"Split metadata not found: {metadata_path}. "
            "Regenerate the splits so the dataset hash is recorded alongside them."
        )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    expected = metadata.get("dataset_sha256")
    if not expected:
        raise ValueError(
            f"{metadata_path} records no dataset_sha256, so source_row_id cannot be "
            "tied to a dataset. Regenerate the splits."
        )
    actual = file_sha256(data_path)
    if actual != expected:
        raise ValueError(
            f"Dataset does not match the splits: {data_path} hashes to {actual}, "
            f"but {metadata_path} was built from {expected}. source_row_id would "
            "point at different wafers."
        )
    return actual


def verify_split_file_matches_metadata(splits_path: Path) -> str:
    """Refuse a split file whose hash differs from its recorded metadata.

    Returns the verified hash. Raises FileNotFoundError when the metadata written
    alongside the splits is missing, and ValueError when it records a different
    split file or omits the hash entirely.
    """
    metadata_path = split_metadata_path(splits_path)
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Split metadata not found: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    expected = metadata.get("split_file_sha256")
    if not expected:
        raise ValueError(f"{metadata_path} records no split_file_sha256.")
    actual = file_sha256(splits_path)
    if actual != expected:
        raise ValueError(
            f"Split file does not match metadata: {splits_path} hashes to {actual}, "
            f"but {metadata_path} records {expected}."
        )
    return actual
