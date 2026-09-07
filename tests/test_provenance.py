from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from scripts.run_experiment import load_frame
from wm811k.constants import LABELS
from wm811k.provenance import (
    file_sha256,
    split_metadata_path,
    verify_dataset_matches_splits,
    verify_split_file_matches_metadata,
)
from wm811k.splitting import SPLIT_NAMES

EMPTY_SHA = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"


def _dataset_and_splits(tmp_path: Path, *, recorded_sha: str | None) -> tuple[Path, Path]:
    data = tmp_path / "LSWMD.pkl"
    data.write_bytes(b"wafer rows in a fixed order")
    splits = tmp_path / "splits.csv"
    splits.write_text("source_row_id,split\n0,train\n", encoding="utf-8")
    metadata = {"dataset_path": "data/LSWMD.pkl"}
    if recorded_sha is not None:
        metadata["dataset_sha256"] = recorded_sha
    split_metadata_path(splits).write_text(json.dumps(metadata), encoding="utf-8")
    return data, splits


def test_metadata_sits_beside_the_split_file():
    assert split_metadata_path(Path("artifacts/splits.csv")) == Path(
        "artifacts/splits.metadata.json"
    )


def test_file_sha256_matches_hashlib_on_an_empty_file(tmp_path):
    empty = tmp_path / "empty"
    empty.write_bytes(b"")

    assert file_sha256(empty) == EMPTY_SHA


def test_the_dataset_the_splits_were_built_from_is_accepted(tmp_path):
    data = tmp_path / "LSWMD.pkl"
    data.write_bytes(b"wafer rows in a fixed order")
    splits = tmp_path / "splits.csv"
    splits.write_text("source_row_id,split\n0,train\n", encoding="utf-8")
    split_metadata_path(splits).write_text(
        json.dumps({"dataset_sha256": file_sha256(data)}), encoding="utf-8"
    )

    assert verify_dataset_matches_splits(data, splits) == file_sha256(data)


def test_a_different_dataset_is_refused_before_any_row_is_read(tmp_path):
    data, splits = _dataset_and_splits(tmp_path, recorded_sha="0" * 64)

    with pytest.raises(ValueError, match="Dataset does not match the splits"):
        verify_dataset_matches_splits(data, splits)


def test_missing_split_metadata_is_refused(tmp_path):
    data, splits = _dataset_and_splits(tmp_path, recorded_sha="0" * 64)
    split_metadata_path(splits).unlink()

    with pytest.raises(FileNotFoundError, match="Split metadata not found"):
        verify_dataset_matches_splits(data, splits)


def test_metadata_without_a_recorded_hash_is_refused(tmp_path):
    data, splits = _dataset_and_splits(tmp_path, recorded_sha=None)

    with pytest.raises(ValueError, match="records no dataset_sha256"):
        verify_dataset_matches_splits(data, splits)


def test_load_frame_refuses_a_mismatched_dataset_before_reading_it(tmp_path, monkeypatch):
    data, splits = _dataset_and_splits(tmp_path, recorded_sha="0" * 64)
    monkeypatch.setattr(
        "scripts.run_experiment.pd.read_pickle",
        lambda *args, **kwargs: pytest.fail("dataset was read despite the hash mismatch"),
    )

    with pytest.raises(ValueError, match="Dataset does not match the splits"):
        load_frame(data, splits)


def _write_split_metadata(data: Path, splits: Path) -> None:
    split_metadata_path(splits).write_text(
        json.dumps(
            {"dataset_sha256": file_sha256(data), "split_file_sha256": file_sha256(splits)}
        ),
        encoding="utf-8",
    )


def _synthetic_dataset_and_splits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, Path, pd.DataFrame]:
    frame = pd.DataFrame(
        [
            {
                "waferMap": [[1, 2]],
                "lotName": f"{split}-{lot}",
                "failureType": label,
                "split": split,
            }
            for split in SPLIT_NAMES
            for lot in range(2)
            for label in LABELS
        ]
    )
    frame.insert(0, "source_row_id", range(len(frame)))
    data = tmp_path / "LSWMD.pkl"
    data.write_bytes(b"synthetic wafer dataset")
    splits = tmp_path / "splits.csv"
    frame[["source_row_id", "split"]].to_csv(splits, index=False)
    _write_split_metadata(data, splits)

    def read_pickle(path: Path) -> pd.DataFrame:
        assert path == data
        # The real preparation step adds positional IDs to the raw pickle columns.
        return frame.drop(columns=["source_row_id", "split"]).copy()

    monkeypatch.setattr("scripts.run_experiment.pd.read_pickle", read_pickle)
    return data, splits, frame


def test_a_mismatched_split_file_is_refused(tmp_path: Path) -> None:
    data, splits = _dataset_and_splits(tmp_path, recorded_sha="0" * 64)
    _write_split_metadata(data, splits)
    expected = file_sha256(splits)
    splits.write_text("source_row_id,split\n0,test\n", encoding="utf-8")

    with pytest.raises(ValueError) as excinfo:
        verify_split_file_matches_metadata(splits)

    message = str(excinfo.value)
    assert str(splits) in message
    assert str(split_metadata_path(splits)) in message
    assert expected in message
    assert file_sha256(splits) in message


def test_metadata_without_a_split_hash_is_refused(tmp_path: Path) -> None:
    _, splits = _dataset_and_splits(tmp_path, recorded_sha="0" * 64)

    with pytest.raises(ValueError, match="records no split_file_sha256"):
        verify_split_file_matches_metadata(splits)


def test_split_verification_requires_metadata(tmp_path: Path) -> None:
    _, splits = _dataset_and_splits(tmp_path, recorded_sha="0" * 64)
    split_metadata_path(splits).unlink()

    with pytest.raises(FileNotFoundError, match="Split metadata not found"):
        verify_split_file_matches_metadata(splits)


def test_load_frame_refuses_lot_leakage_with_matching_hashes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data, splits, _ = _synthetic_dataset_and_splits(tmp_path, monkeypatch)
    assignments = pd.read_csv(splits)
    assignments.loc[0, "split"] = "validation"
    assignments.to_csv(splits, index=False)
    _write_split_metadata(data, splits)

    with pytest.raises(AssertionError, match="leakage"):
        load_frame(data, splits)


def test_load_frame_accepts_disjoint_lots_with_all_classes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data, splits, frame = _synthetic_dataset_and_splits(tmp_path, monkeypatch)

    assert verify_split_file_matches_metadata(splits) == file_sha256(splits)
    merged = load_frame(data, splits)

    assert isinstance(merged, pd.DataFrame)
    pd.testing.assert_frame_equal(merged[frame.columns], frame)
    assert merged["failure_label"].tolist() == frame["failureType"].tolist()


@pytest.mark.parametrize("invalid_split", ["training", None])
def test_load_frame_refuses_invalid_split_names(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, invalid_split: str | None
) -> None:
    data, splits, _ = _synthetic_dataset_and_splits(tmp_path, monkeypatch)
    assignments = pd.read_csv(splits)
    assignments.loc[0, "split"] = invalid_split
    assignments.to_csv(splits, index=False)
    _write_split_metadata(data, splits)

    with pytest.raises(ValueError, match="valid split names"):
        load_frame(data, splits)


def test_load_frame_refuses_duplicate_row_ids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data, splits, _ = _synthetic_dataset_and_splits(tmp_path, monkeypatch)
    assignments = pd.read_csv(splits)
    pd.concat([assignments, assignments.iloc[[0]]]).to_csv(splits, index=False)
    _write_split_metadata(data, splits)

    with pytest.raises(ValueError, match="Duplicate source_row_id"):
        load_frame(data, splits)


def test_load_frame_refuses_missing_class_coverage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data, splits, frame = _synthetic_dataset_and_splits(tmp_path, monkeypatch)
    frame.loc[
        (frame["split"] == "test") & (frame["failureType"] == LABELS[0]), "failureType"
    ] = LABELS[1]

    with pytest.raises(AssertionError, match="coverage"):
        load_frame(data, splits)


def test_load_frame_refuses_a_mismatched_split_before_reading_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data, splits, _ = _synthetic_dataset_and_splits(tmp_path, monkeypatch)
    splits.write_text("source_row_id,split\n0,test\n", encoding="utf-8")

    def read_pickle(path: Path) -> pd.DataFrame:
        pytest.fail("dataset was read despite the split hash mismatch")

    monkeypatch.setattr("scripts.run_experiment.pd.read_pickle", read_pickle)
    with pytest.raises(ValueError, match="Split file does not match metadata"):
        load_frame(data, splits)
