import json
import sys
from pathlib import Path

import pandas as pd
import pytest
import torch

from scripts.run_experiment import (
    class_weights,
    guard_checkpoint_matches,
    guard_existing_results,
    guard_selected_experiment,
    load_confirmed_checkpoint,
    main,
    resolve_run_dir,
)
from wm811k.constants import LABEL_TO_INDEX, LABELS

EXPERIMENT_CONFIG = {"kind": "cnn", "input_channels": 2, "loss": "cross_entropy"}


def _write_result(run_dir: Path, stage: str) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / f"{stage}_metrics.json"
    path.write_text("{}", encoding="utf-8")
    return path


def _checkpoint() -> dict:
    return {
        "experiment": "E2",
        "experiment_config": dict(EXPERIMENT_CONFIG),
        "label_to_index": dict(LABEL_TO_INDEX),
        "split_sha256": "a" * 64,
        "seed": 42,
    }


def _guard_checkpoint(checkpoint: dict) -> None:
    guard_checkpoint_matches(
        checkpoint,
        experiment_name="E2",
        experiment=EXPERIMENT_CONFIG,
        split_hash="a" * 64,
        seed=42,
    )


def test_resolve_run_dir_preserves_legacy_path_without_seed():
    assert resolve_run_dir(Path("artifacts"), "E2", None) == Path("artifacts/E2")


def test_validation_rerun_is_refused_after_the_test_is_confirmed(tmp_path):
    run_dir = tmp_path / "E2"
    _write_result(run_dir, "test")
    (run_dir / "best.pt").write_bytes(b"weights")

    with pytest.raises(SystemExit, match="Test evaluation is already confirmed"):
        guard_existing_results(run_dir, stage="validation", seed_override=None)


def test_validation_rerun_is_allowed_when_only_validation_finished(tmp_path):
    run_dir = tmp_path / "E2"
    _write_result(run_dir, "validation")

    guard_existing_results(run_dir, stage="validation", seed_override=None)


def test_repeat_seed_run_is_not_blocked_by_a_confirmed_test_elsewhere(tmp_path):
    _write_result(tmp_path / "E2", "test")
    repeat_dir = resolve_run_dir(tmp_path, "E2", 52)

    guard_existing_results(repeat_dir, stage="validation", seed_override=52)


def test_completed_test_and_repeat_results_are_still_protected(tmp_path):
    run_dir = tmp_path / "E2"
    _write_result(run_dir, "test")
    with pytest.raises(SystemExit, match="Test result already exists"):
        guard_existing_results(run_dir, stage="test", seed_override=None)

    repeat_dir = resolve_run_dir(tmp_path, "E2", 52)
    _write_result(repeat_dir, "validation")
    with pytest.raises(SystemExit, match="Completed repeat result already exists"):
        guard_existing_results(repeat_dir, stage="validation", seed_override=52)


def test_resolve_run_dir_isolates_repeat_seed_output():
    assert resolve_run_dir(Path("artifacts"), "E2", 52) == Path(
        "artifacts/repeat/E2/seed52"
    )


def test_only_the_selected_experiment_may_reach_the_test_set():
    guard_selected_experiment("E2", "E2")

    with pytest.raises(SystemExit, match="Only the selected experiment may be tested"):
        guard_selected_experiment("E3", "E2")


def test_matching_checkpoint_is_accepted():
    _guard_checkpoint(_checkpoint())


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("experiment", "E3"),
        ("experiment_config", {"kind": "cnn", "input_channels": 1, "loss": "cross_entropy"}),
        ("label_to_index", {"none": 0}),
        ("split_sha256", "b" * 64),
        ("seed", 52),
    ],
)
def test_checkpoint_from_a_different_run_is_refused(field, value):
    checkpoint = _checkpoint()
    checkpoint[field] = value

    with pytest.raises(ValueError, match=f"mismatched: \\['{field}'\\]"):
        _guard_checkpoint(checkpoint)


def test_checkpoint_missing_provenance_fields_is_refused():
    with pytest.raises(ValueError, match="mismatched"):
        _guard_checkpoint({"model_state_dict": {}})


def test_sqrt_inverse_frequency_reduces_weight_ratio():
    frame = pd.DataFrame(
        {
            "failure_label": [
                label
                for count, label in enumerate(LABELS, start=1)
                for _ in range(count)
            ]
        }
    )
    inverse, inverse_meta = class_weights(
        frame, torch.device("cpu"), formula="inverse_frequency"
    )
    sqrt_inverse, sqrt_meta = class_weights(
        frame, torch.device("cpu"), formula="sqrt_inverse_frequency"
    )

    assert inverse_meta["max_min_ratio"] == pytest.approx(9.0)
    assert sqrt_meta["max_min_ratio"] == pytest.approx(3.0)
    assert inverse.mean().item() == pytest.approx(1.0)
    assert sqrt_inverse.mean().item() == pytest.approx(1.0)


def _imbalanced_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "failure_label": [
                label
                for count, label in zip((5, 5, 5, 5, 5, 5, 5, 5, 500), LABELS, strict=True)
                for _ in range(count)
            ]
        }
    )


def test_class_mean_normalisation_shrinks_the_loss_the_optimizer_sees():
    _, metadata = class_weights(
        _imbalanced_frame(),
        torch.device("cpu"),
        formula="inverse_frequency",
        normalization="mean_one",
    )

    assert metadata["normalization"] == "mean_one"
    # The nine class weights average 1, but almost every sample belongs to the majority
    # class, so the batch-averaged loss is scaled far below an unweighted run.
    assert metadata["effective_sample_scale"] < 0.2


def test_sample_mean_normalisation_keeps_the_unweighted_loss_scale():
    weights, metadata = class_weights(
        _imbalanced_frame(),
        torch.device("cpu"),
        formula="inverse_frequency",
        normalization="sample_mean_one",
    )

    assert metadata["effective_sample_scale"] == pytest.approx(1.0)
    assert metadata["max_min_ratio"] == pytest.approx(100.0)
    assert weights.min().item() > 0


def test_normalisation_choices_differ_only_by_a_constant_factor():
    frame = _imbalanced_frame()
    class_mean, _ = class_weights(
        frame, torch.device("cpu"), formula="inverse_frequency", normalization="mean_one"
    )
    sample_mean, _ = class_weights(
        frame,
        torch.device("cpu"),
        formula="inverse_frequency",
        normalization="sample_mean_one",
    )
    ratios = (sample_mean / class_mean).tolist()

    assert ratios == pytest.approx([ratios[0]] * len(ratios))


def test_an_unknown_normalisation_is_refused_rather_than_ignored():
    with pytest.raises(ValueError, match="Unsupported class-weight normalization"):
        class_weights(
            _imbalanced_frame(),
            torch.device("cpu"),
            formula="inverse_frequency",
            normalization="whatever",
        )


def test_inverse_frequency_clip_limits_dynamic_range():
    frame = pd.DataFrame(
        {
            "failure_label": [
                label
                for count, label in enumerate(LABELS, start=1)
                for _ in range(count)
            ]
        }
    )
    weights, metadata = class_weights(
        frame,
        torch.device("cpu"),
        formula="inverse_frequency",
        clip={"max_min_ratio": 3.0},
    )

    assert metadata["unclipped_max_min_ratio"] == pytest.approx(9.0)
    assert metadata["max_min_ratio"] == pytest.approx(3.0)
    assert metadata["clipped_class_count"] == 2
    assert weights.mean().item() == pytest.approx(1.0)


def _minimal_config(tmp_path: Path, selected: str = "E2") -> Path:
    config = {
        "project": {
            "seed": 42,
            "final_selected_experiment": selected,
            "labels": list(LABELS),
            "training": {"epochs": 1, "bootstrap_replicates": 10},
        },
        "experiments": {
            "E2": {"kind": "cnn", "loss": "cross_entropy"},
            "E3": {"kind": "cnn", "loss": "weighted_cross_entropy"},
        },
    }
    path = tmp_path / "experiments.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    return path


def _run_main(monkeypatch, tmp_path: Path, argv: list[str]) -> None:
    splits = tmp_path / "splits.csv"
    splits.write_text("source_row_id,split\n0,train\n", encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_experiment.py",
            "--config",
            str(_minimal_config(tmp_path)),
            "--splits",
            str(splits),
            "--output-dir",
            str(tmp_path / "artifacts"),
            *argv,
        ],
    )
    # Guards must fire before any data is touched; loading would need the 2GB pickle.
    monkeypatch.setattr(
        "scripts.run_experiment.load_frame",
        lambda *args, **kwargs: pytest.fail("guard did not stop the run before load_frame"),
    )
    main()


def test_main_refuses_to_test_an_experiment_that_was_not_selected(monkeypatch, tmp_path):
    with pytest.raises(SystemExit, match="Only the selected experiment may be tested"):
        _run_main(
            monkeypatch,
            tmp_path,
            ["--experiment", "E3", "--stage", "test", "--confirm-test"],
        )


def test_main_refuses_to_revalidate_a_directory_with_a_confirmed_test(monkeypatch, tmp_path):
    _write_result(tmp_path / "artifacts" / "E2", "test")

    with pytest.raises(SystemExit, match="Test evaluation is already confirmed"):
        _run_main(monkeypatch, tmp_path, ["--experiment", "E2"])


def test_a_checkpoint_from_another_experiment_is_refused_on_load(tmp_path):
    path = tmp_path / "best.pt"
    torch.save({**_checkpoint(), "model_state_dict": {}}, path)

    with pytest.raises(ValueError, match=r"mismatched: \['experiment'\]"):
        load_confirmed_checkpoint(
            path,
            device=torch.device("cpu"),
            experiment_name="E3",
            experiment=EXPERIMENT_CONFIG,
            split_hash="a" * 64,
            seed=42,
        )


def test_a_missing_checkpoint_is_reported_by_path(tmp_path):
    with pytest.raises(FileNotFoundError, match="Best checkpoint not found"):
        load_confirmed_checkpoint(
            tmp_path / "absent.pt",
            device=torch.device("cpu"),
            experiment_name="E2",
            experiment=EXPERIMENT_CONFIG,
            split_hash="a" * 64,
            seed=42,
        )
