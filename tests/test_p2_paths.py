from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from scripts import run_p2_interpretability as module


@pytest.fixture
def pinned_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[dict[str, Path], dict[str, str]]:
    paths = {
        "data/LSWMD.pkl": tmp_path / "external" / "LSWMD.pkl",
        "artifacts/splits.csv": tmp_path / "external" / "splits.csv",
        "artifacts/E2/best.pt": tmp_path / "external" / "best.pt",
        "artifacts/E2/validation_predictions.csv": (
            tmp_path / "artifacts" / "E2" / "validation_predictions.csv"
        ),
        "artifacts/E2/test_predictions.csv": (
            tmp_path / "artifacts" / "E2" / "test_predictions.csv"
        ),
        "artifacts/E2/test_metrics.json": tmp_path / "artifacts" / "E2" / "test_metrics.json",
    }
    expected = {}
    for relative_path, path in paths.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"pinned contents for {relative_path}".encode())
        expected[relative_path] = hashlib.sha256(path.read_bytes()).hexdigest()
    monkeypatch.setattr(module, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(module, "EXPECTED_HASHES", expected)
    arguments = {
        "data_path": paths["data/LSWMD.pkl"],
        "splits_path": paths["artifacts/splits.csv"],
        "checkpoint_path": paths["artifacts/E2/best.pt"],
    }
    return arguments, expected


def test_fixed_hashes_verify_actual_load_paths(
    pinned_paths: tuple[dict[str, Path], dict[str, str]],
) -> None:
    arguments, expected = pinned_paths

    observed = module.verify_fixed_hashes(**arguments)

    assert len(observed) == 6
    assert observed == expected


@pytest.mark.parametrize(
    ("argument", "relative_path"),
    [
        ("checkpoint_path", "artifacts/E2/best.pt"),
        ("splits_path", "artifacts/splits.csv"),
        ("data_path", "data/LSWMD.pkl"),
    ],
)
def test_fixed_hashes_refuse_modified_load_paths(
    pinned_paths: tuple[dict[str, Path], dict[str, str]],
    argument: str,
    relative_path: str,
) -> None:
    arguments, expected = pinned_paths
    path = arguments[argument]
    path.write_bytes(b"modified contents")

    with pytest.raises(ValueError) as excinfo:
        module.verify_fixed_hashes(**arguments)

    message = str(excinfo.value)
    assert "Integrity failure" in message
    assert str(path) in message
    assert relative_path in message
    assert expected[relative_path] in message
    assert hashlib.sha256(path.read_bytes()).hexdigest() in message


def test_fixed_hashes_refuse_missing_checkpoint(
    pinned_paths: tuple[dict[str, Path], dict[str, str]],
) -> None:
    arguments, _ = pinned_paths
    path = arguments["checkpoint_path"]
    path.unlink()

    with pytest.raises(FileNotFoundError) as excinfo:
        module.verify_fixed_hashes(**arguments)

    assert str(path) in str(excinfo.value)
