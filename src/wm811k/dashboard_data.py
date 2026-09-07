from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

EXPERIMENTS = ("E0", "E1", "E2", "E3")
REPEAT_SUMMARIES = (
    "p1_e3c_summary.csv",
    "p1_e3b_summary.csv",
    "seed_summary.csv",
)


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_validation_summary(
    artifacts: Path,
    *,
    selected_experiment: str = "E2",
) -> pd.DataFrame:
    rows = []
    for experiment in EXPERIMENTS:
        metrics = read_json(artifacts / experiment / "validation_metrics.json")
        rows.append(
            {
                "experiment": experiment,
                "best_epoch": metrics.get("best_epoch"),
                "macro_f1": metrics["metrics"]["macro_f1"],
                "balanced_accuracy": metrics["metrics"]["balanced_accuracy"],
                "selected": experiment == selected_experiment,
            }
        )
    return pd.DataFrame(rows)


def load_repeat_summary(artifacts: Path) -> pd.DataFrame:
    """Load the most complete available repeat-seed results."""
    repeat_root = artifacts / "repeat"
    sources = [
        repeat_root / name
        for name in REPEAT_SUMMARIES
        if (repeat_root / name).is_file()
    ]
    if not sources:
        return pd.DataFrame(
            columns=[
                "experiment",
                "seeds",
                "macro_f1_mean",
                "macro_f1_std",
                "balanced_accuracy_mean",
                "balanced_accuracy_std",
            ]
        )
    frames = [pd.read_csv(path) for path in reversed(sources)]
    repeats = (
        pd.concat(frames, ignore_index=True)
        .drop_duplicates(["experiment", "seed"], keep="last")
    )
    return (
        repeats.groupby("experiment", sort=False)
        .agg(
            seeds=("seed", "nunique"),
            macro_f1_mean=("macro_f1", "mean"),
            macro_f1_std=("macro_f1", "std"),
            balanced_accuracy_mean=("balanced_accuracy", "mean"),
            balanced_accuracy_std=("balanced_accuracy", "std"),
        )
        .reset_index()
    )


def load_test_metrics(artifacts: Path, selected_experiment: str = "E2") -> dict:
    return read_json(artifacts / selected_experiment / "test_metrics.json")


def load_class_report(reports: Path, selected_experiment: str = "E2") -> pd.DataFrame:
    # build_test_report.py names its output after the run that produced it, so the
    # dashboard has to follow the same rule rather than assume E2.
    name = f"{selected_experiment.lower()}_test_classification_report.csv"
    frame = pd.read_csv(reports / name)
    return frame[
        frame["class"].isin(
            [
                "Center",
                "Donut",
                "Edge-Loc",
                "Edge-Ring",
                "Loc",
                "Near-full",
                "Random",
                "Scratch",
                "none",
            ]
        )
    ].reset_index(drop=True)


def load_dashboard_tables(project_root: Path) -> dict[str, object]:
    artifacts = project_root / "artifacts"
    reports = project_root / "reports"
    selected = selected_experiment(project_root)
    case_manifest = reports / f"{selected.lower()}_case_manifest.csv"
    return {
        "validation": load_validation_summary(
            artifacts,
            selected_experiment=selected,
        ),
        "repeat_validation": load_repeat_summary(artifacts),
        "test": load_test_metrics(artifacts, selected),
        "class_report": load_class_report(reports, selected),
        "radial": pd.read_csv(reports / f"{selected.lower()}_test_radial_profile.csv"),
        "edge": pd.read_csv(reports / f"{selected.lower()}_test_edge_zones.csv"),
        "cases": pd.read_csv(case_manifest) if case_manifest.is_file() else None,
        "split": read_json(artifacts / "splits.metadata.json"),
        "audit": read_json(artifacts / "data_audit.json"),
    }


SUPPORTED_DASHBOARD_EXPERIMENT = "E2"


def selected_experiment(project_root: Path) -> str:
    """The experiment frozen for the single test evaluation, per the experiment config.

    The dashboard reads files that only the E2 analysis run produces:
    analyze_cases_and_spatial.py writes e2_-prefixed radial, edge and case outputs, and
    app.py names E2 in its headings and in the paragraph recording why it was chosen —
    that paragraph is a record, not a template. So a different selection is refused here
    rather than left to fail later on a filename nothing in the repository can produce.
    """
    config_path = project_root / "configs" / "experiments.json"
    if not config_path.is_file():
        return SUPPORTED_DASHBOARD_EXPERIMENT
    selected = read_json(config_path)["project"].get(
        "final_selected_experiment", SUPPORTED_DASHBOARD_EXPERIMENT
    )
    if selected != SUPPORTED_DASHBOARD_EXPERIMENT:
        raise ValueError(
            f"The dashboard is built around {SUPPORTED_DASHBOARD_EXPERIMENT}, but the "
            f"config selects {selected}. Regenerate the case, radial and edge outputs "
            f"under a {selected.lower()}_ prefix and update app.py's headings before "
            "pointing the dashboard at another experiment."
        )
    return selected


def required_dashboard_paths(project_root: Path) -> tuple[Path, ...]:
    selected = selected_experiment(project_root)
    prefix = selected.lower()
    return (
        # E0~E3 are the validation comparison itself, so they stay fixed. Everything
        # downstream of the test evaluation follows whichever experiment was frozen.
        project_root / "artifacts" / "E0" / "validation_metrics.json",
        project_root / "artifacts" / "E1" / "validation_metrics.json",
        project_root / "artifacts" / "E2" / "validation_metrics.json",
        project_root / "artifacts" / "E3" / "validation_metrics.json",
        project_root / "artifacts" / selected / "test_metrics.json",
        project_root / "reports" / f"{prefix}_test_classification_report.csv",
        project_root / "reports" / f"{prefix}_test_radial_profile.csv",
        project_root / "reports" / f"{prefix}_test_edge_zones.csv",
        project_root / "artifacts" / "splits.metadata.json",
        project_root / "artifacts" / "data_audit.json",
        project_root / "configs" / "experiments.json",
    )
