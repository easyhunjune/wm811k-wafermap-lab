import json
from pathlib import Path

import pytest

from wm811k.dashboard_data import (
    load_class_report,
    load_repeat_summary,
    load_test_metrics,
    load_validation_summary,
    read_json,
    required_dashboard_paths,
    selected_experiment,
)


def test_read_json_round_trip(tmp_path):
    path = tmp_path / "sample.json"
    path.write_text(json.dumps({"value": 3}), encoding="utf-8")
    assert read_json(path) == {"value": 3}


def test_validation_summary_preserves_experiment_order_and_selection(tmp_path):
    for index, experiment in enumerate(("E0", "E1", "E2", "E3")):
        run = tmp_path / experiment
        run.mkdir()
        payload = {
            "best_epoch": None if experiment == "E0" else index,
            "metrics": {
                "macro_f1": index / 10,
                "balanced_accuracy": index / 20,
            },
        }
        (run / "validation_metrics.json").write_text(
            json.dumps(payload),
            encoding="utf-8",
        )
    summary = load_validation_summary(tmp_path)
    assert summary["experiment"].tolist() == ["E0", "E1", "E2", "E3"]
    assert summary.loc[summary["selected"], "experiment"].tolist() == ["E2"]


def test_read_json_raises_for_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_json(tmp_path / "missing.json")


def test_repeat_summary_keeps_latest_duplicate_seed(tmp_path):
    repeat = tmp_path / "repeat"
    repeat.mkdir()
    (repeat / "seed_summary.csv").write_text(
        "experiment,seed,macro_f1,balanced_accuracy\n"
        "E2,42,0.80,0.81\n"
        "E2,52,0.82,0.83\n",
        encoding="utf-8",
    )
    (repeat / "p1_e3c_summary.csv").write_text(
        "experiment,seed,macro_f1,balanced_accuracy\n"
        "E2,42,0.90,0.91\n"
        "E3c,42,0.88,0.89\n",
        encoding="utf-8",
    )
    summary = load_repeat_summary(tmp_path).set_index("experiment")
    assert summary.loc["E2", "seeds"] == 2
    assert summary.loc["E2", "macro_f1_mean"] == pytest.approx(0.86)
    assert summary.loc["E3c", "seeds"] == 1


def test_case_manifest_is_not_a_global_dashboard_requirement(tmp_path):
    required = required_dashboard_paths(tmp_path)
    assert tmp_path / "reports" / "e2_case_manifest.csv" not in required


def _write_config(tmp_path: Path, selected: str) -> None:
    configs = tmp_path / "configs"
    configs.mkdir(parents=True, exist_ok=True)
    (configs / "experiments.json").write_text(
        json.dumps({"project": {"final_selected_experiment": selected}}), encoding="utf-8"
    )


def test_required_paths_come_from_the_configured_experiment(tmp_path):
    _write_config(tmp_path, "E2")

    paths = [p.as_posix() for p in required_dashboard_paths(tmp_path)]

    assert any(p.endswith("artifacts/E2/test_metrics.json") for p in paths)
    assert any(p.endswith("reports/e2_test_classification_report.csv") for p in paths)


def test_a_selection_the_dashboard_cannot_render_is_refused(tmp_path):
    # The case, radial and edge outputs only exist with an e2_ prefix, and app.py names
    # E2 in its headings, so silently accepting E3b would demand files nothing produces.
    _write_config(tmp_path, "E3b")

    with pytest.raises(ValueError, match="dashboard is built around E2"):
        selected_experiment(tmp_path)

    with pytest.raises(ValueError, match="dashboard is built around E2"):
        required_dashboard_paths(tmp_path)


def test_a_missing_config_falls_back_to_the_supported_experiment(tmp_path):
    assert selected_experiment(tmp_path) == "E2"


def test_dashboard_follows_the_configured_selected_experiment(tmp_path):
    reports = tmp_path / "reports"
    reports.mkdir()
    (reports / "e3b_test_classification_report.csv").write_text(
        "class,precision,recall,f1-score,support\nScratch,0.5,0.5,0.5,10\n",
        encoding="utf-8",
    )
    artifacts = tmp_path / "artifacts" / "E3b"
    artifacts.mkdir(parents=True)
    (artifacts / "test_metrics.json").write_text('{"experiment": "E3b"}', encoding="utf-8")

    metrics = load_test_metrics(tmp_path / "artifacts", "E3b")
    report = load_class_report(reports, "E3b")

    assert metrics["experiment"] == "E3b"
    assert list(report["class"]) == ["Scratch"]
