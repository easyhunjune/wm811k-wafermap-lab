from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _read(relative_path: str) -> str:
    return (PROJECT_ROOT / relative_path).read_text(encoding="utf-8")


def _local_path_markers() -> tuple[str, str]:
    """Windows home-directory prefixes, plain and JSON-escaped.

    Assembled at runtime so this file does not match the pattern it searches for.
    """
    separator = chr(92)
    return f"C:{separator}Users{separator}", f"C:{separator * 2}Users{separator * 2}"


def test_github_publication_boundary_is_explicit() -> None:
    gitignore = _read(".gitignore")
    required_rules = {
        "site/",
        "*.egg-info/",
        "artifacts/splits.csv",
        "artifacts/**/*_predictions.csv",
        "artifacts/p2/case_manifest.csv",
        "artifacts/p2/overlap_metrics.csv",
        "artifacts/p2/validation_overlap_metrics.csv",
        "reports/e2_case_manifest.csv",
        "reports/figures/sample_wafer_*.png",
        "reports/figures/e2_*_correct_cases.png",
        "reports/figures/e2_*_error_cases.png",
        "reports/figures/p2_*_correct_gradcam.png",
        "reports/figures/p2_*_to_none_gradcam.png",
        "*.pt",
        "*.pkl",
        "/EVALUATION.md",
        "/EXECUTION_CHECKLIST.md",
        "/GITHUB_PUBLISH_CHECKLIST.md",
        "/NEXT_STEPS_HANDOFF.md",
        "/NEXT_STEPS_P0_GUIDE.md",
        "/P1_BEGINNER_CAUSAL_GUIDE.md",
        "/P2_SESSION_HANDOFF.md",
        "/REVIEW.md",
        "/reports/P3_GITHUB_PUBLICATION.md",
        "/reports/P4_CAREER_PACKAGE*.md",
    }

    assert required_rules.issubset(set(gitignore.splitlines()))


def test_public_documents_do_not_advertise_a_site_or_local_user_path() -> None:
    public_documents = [
        "README.md",
        "data/DATA_CARD.md",
        "reports/TEST_RESULTS.md",
        "reports/P2_INTERPRETABILITY.md",
    ]

    plain, _ = _local_path_markers()
    for relative_path in public_documents:
        content = _read(relative_path)
        assert "chatgpt" + ".site" not in content
        assert plain not in content


def test_no_tracked_text_file_carries_a_developer_absolute_path() -> None:
    # Naming the documents to check missed artifacts/data_audit.json, which recorded the
    # dataset's absolute path and so the developer's Windows account name. Scan whatever
    # git actually tracks instead of a hand-kept list.
    tracked = subprocess.run(
        ["git", "ls-files", "-z", "*.md", "*.json", "*.csv", "*.py", "*.toml", "*.txt"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split("\0")

    plain, escaped = _local_path_markers()
    offenders = []
    for relative_path in filter(None, tracked):
        path = PROJECT_ROOT / relative_path
        if not path.is_file():
            continue
        content = path.read_text(encoding="utf-8", errors="replace")
        if plain in content or escaped in content:
            offenders.append(relative_path)

    assert offenders == []


def test_readme_claims_are_traceable_to_public_reports() -> None:
    readme = _read("README.md")
    test_report = _read("reports/TEST_RESULTS.md")
    p2_report = _read("reports/P2_INTERPRETABILITY.md")

    for value in ("0.8349", "[0.8158, 0.8494]"):
        assert value in readme
        assert value in test_report

    for value in ("+0.0290", "+0.0421"):
        assert value in readme
        assert value in p2_report


def test_p2_pin_matches_the_tracked_test_metrics_file() -> None:
    # run_p2_interpretability.py pins artifacts/E2/test_metrics.json byte for byte. An
    # earlier pin was taken from a CRLF working copy, so a fresh clone, which checks the
    # file out with LF, failed the integrity check before doing anything. Keep the pin on
    # the bytes git actually delivers.
    from scripts.run_p2_interpretability import EXPECTED_HASHES

    path = PROJECT_ROOT / "artifacts" / "E2" / "test_metrics.json"
    observed = hashlib.sha256(path.read_bytes()).hexdigest()
    assert observed == EXPECTED_HASHES["artifacts/E2/test_metrics.json"]
