from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from scripts.show_one_wafer import (
    _build_title,
    _checkpoint_load_kwargs,
    _input_label_grid,
    _validate_true_label,
    default_output_path,
    guard_output_path,
    predictions_path_for,
    select_wafer_row,
)


@pytest.fixture
def predictions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "source_row_id": [80, 10, 60, 30, 70, 20, 50, 40],
            "lotName": [f"Lot{i}" for i in range(8)],
            "y_true": [0, 7, 2, 7, 2, 0, 7, 2],
            "y_pred": [0, 2, 2, 7, 0, 0, 0, 2],
            "true_label": [
                "Center",
                "Scratch",
                "Edge-Loc",
                "Scratch",
                "Edge-Loc",
                "Center",
                "Scratch",
                "Edge-Loc",
            ],
            "predicted_label": [
                "Center",
                "Edge-Loc",
                "Edge-Loc",
                "Scratch",
                "Center",
                "Center",
                "Center",
                "Edge-Loc",
            ],
        }
    )


def test_select_wafer_row_same_seed_is_reproducible(
    predictions: pd.DataFrame,
) -> None:
    first = select_wafer_row(
        predictions,
        seed=7,
        source_row_id=None,
        true_label=None,
        only_errors=False,
    )
    second = select_wafer_row(
        predictions.sample(frac=1, random_state=123),
        seed=7,
        source_row_id=None,
        true_label=None,
        only_errors=False,
    )

    assert first["source_row_id"] == second["source_row_id"]


def test_select_wafer_row_filters_true_label(predictions: pd.DataFrame) -> None:
    selected = select_wafer_row(
        predictions,
        seed=3,
        source_row_id=None,
        true_label="Scratch",
        only_errors=False,
    )

    assert selected["true_label"] == "Scratch"


def test_select_wafer_row_filters_only_errors(predictions: pd.DataFrame) -> None:
    selected = select_wafer_row(
        predictions,
        seed=4,
        source_row_id=None,
        true_label=None,
        only_errors=True,
    )

    assert selected["y_true"] != selected["y_pred"]


def test_select_wafer_row_source_id_ignores_filters(
    predictions: pd.DataFrame,
) -> None:
    selected = select_wafer_row(
        predictions,
        seed=1,
        source_row_id=80,
        true_label="Scratch",
        only_errors=True,
    )

    assert selected["source_row_id"] == 80
    assert selected["true_label"] == "Center"
    assert selected["y_true"] == selected["y_pred"]


def test_select_wafer_row_raises_for_empty_candidates(
    predictions: pd.DataFrame,
) -> None:
    with pytest.raises(ValueError, match="후보가 없습니다"):
        select_wafer_row(
            predictions,
            seed=1,
            source_row_id=None,
            true_label="Donut",
            only_errors=False,
        )

    with pytest.raises(ValueError, match="source_row_id=999"):
        select_wafer_row(
            predictions,
            seed=1,
            source_row_id=999,
            true_label=None,
            only_errors=False,
        )


def test_default_output_path() -> None:
    path = default_output_path(Path("project"), "E2", "validation", 123)

    assert path == Path(
        "project/reports/figures/sample_wafer_E2_validation_row123.png"
    )


@pytest.mark.parametrize(
    ("split", "command_suffix"),
    [
        ("validation", "--experiment E2"),
        ("test", "--experiment E2 --stage test --confirm-test"),
    ],
)
def test_predictions_path_for_missing_file_has_command(
    tmp_path: Path,
    split: str,
    command_suffix: str,
) -> None:
    with pytest.raises(FileNotFoundError) as error:
        predictions_path_for(
            tmp_path,
            "E2",
            split,
            final_selected_experiment="E2",
        )

    assert command_suffix in str(error.value)
    assert (
        r".\.venv\Scripts\python.exe .\scripts\run_experiment.py"
        in str(error.value)
    )


def test_predictions_path_for_rejects_non_selected_test_hint(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="허용됩니다") as error:
        predictions_path_for(
            tmp_path,
            "E1",
            "test",
            final_selected_experiment="E2",
        )

    assert "E1로는 테스트 예측 CSV를 만들 수 없습니다" in str(error.value)
    assert "run_experiment.py" not in str(error.value)


def test_input_label_grid_from_two_channel_masks() -> None:
    encoded = torch.tensor(
        [
            [[0, 1], [1, 1]],
            [[0, 0], [1, 0]],
        ],
        dtype=torch.float32,
    )

    result = _input_label_grid(encoded)

    np.testing.assert_array_equal(result, np.array([[0, 1], [2, 1]]))


def test_input_label_grid_from_categorical_single_channel() -> None:
    encoded = torch.tensor([[[0, 1], [2, 1]]], dtype=torch.float32)

    result = _input_label_grid(encoded)

    np.testing.assert_array_equal(result, np.array([[0, 1], [2, 1]]))


def test_build_title_hides_and_shows_lot_and_marks_degenerate_cam(
    predictions: pd.DataFrame,
) -> None:
    selected = predictions.iloc[0]
    hidden = _build_title(
        selected,
        experiment="E2",
        split="validation",
        probability=0.75,
        correct=True,
        show_lot=False,
        cam_degenerate=False,
    )
    shown = _build_title(
        selected,
        experiment="E2",
        split="test",
        probability=0.75,
        correct=True,
        show_lot=True,
        cam_degenerate=True,
    )

    assert "Experiment: E2" in hidden
    assert "Split: validation" in hidden
    assert "Lot:" not in hidden
    assert selected["lotName"] not in hidden
    assert "Split: test" in shown
    assert f"Lot: {selected['lotName']}" in shown
    assert "CAM degenerate: zero" in shown


def test_checkpoint_load_kwargs_uses_training_seed_not_selection_seed() -> None:
    config = {"project": {"seed": 42}}
    args = Namespace(experiment="E2", seed=7)
    experiment_config = {"kind": "cnn", "input_channels": 2}
    device = torch.device("cpu")

    kwargs = _checkpoint_load_kwargs(
        config,
        args,
        experiment_config,
        "split-hash",
        device,
    )

    assert kwargs["seed"] == 42
    assert kwargs["seed"] != args.seed
    assert kwargs["experiment_name"] == "E2"
    assert kwargs["experiment"] is experiment_config
    assert kwargs["split_hash"] == "split-hash"
    assert kwargs["device"] == device


def test_guard_output_path_accepts_private_sample_name(tmp_path: Path) -> None:
    output = tmp_path / "reports" / "figures" / "sample_wafer_manual.png"

    guard_output_path(output, tmp_path)


@pytest.mark.parametrize(
    "output",
    [
        Path("reports/figures/my_wafer.png"),
        Path("artifacts/E2/look.png"),
    ],
)
def test_guard_output_path_rejects_publication_boundary_bypasses(
    tmp_path: Path,
    output: Path,
) -> None:
    with pytest.raises(SystemExit, match="공개 대상이 아닙니다"):
        guard_output_path(tmp_path / output, tmp_path)


def test_validate_true_label_rejects_mismatched_saved_values(
    predictions: pd.DataFrame,
) -> None:
    selected = predictions.iloc[0].copy()
    selected["true_label"] = "Scratch"

    with pytest.raises(ValueError, match="실제 라벨 정합성 오류"):
        _validate_true_label(selected)
