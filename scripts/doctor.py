from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REQUIRED_PACKAGES = {
    "numpy": "numpy",
    "pandas": "pandas",
    "scikit-learn": "sklearn",
    "torch": "torch",
    "matplotlib": "matplotlib",
    "seaborn": "seaborn",
}
OPTIONAL_PACKAGES = {
    "pytest": "pytest",
    "streamlit": "streamlit",
}


def sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def find_data_path() -> Path:
    configured = os.getenv("WM811K_DATA")
    if configured:
        return Path(configured).expanduser().resolve()
    return PROJECT_ROOT / "data" / "LSWMD.pkl"


def package_status() -> tuple[dict[str, bool], dict[str, bool]]:
    required = {
        display: importlib.util.find_spec(module) is not None
        for display, module in REQUIRED_PACKAGES.items()
    }
    optional = {
        display: importlib.util.find_spec(module) is not None
        for display, module in OPTIONAL_PACKAGES.items()
    }
    return required, optional


def inspect() -> dict[str, object]:
    data_path = find_data_path()
    required, optional = package_status()
    config_path = PROJECT_ROOT / "configs" / "experiments.json"
    python_ok = sys.version_info >= (3, 10)
    data_exists = data_path.is_file()
    config_ok = False
    config_error = None

    try:
        with config_path.open("r", encoding="utf-8") as handle:
            config = json.load(handle)
        config_ok = all(key in config for key in ("project", "experiments"))
    except (OSError, json.JSONDecodeError) as exc:
        config_error = str(exc)

    ready = python_ok and data_exists and config_ok and all(required.values())
    return {
        "ready": ready,
        "python": {
            "version": platform.python_version(),
            "executable": sys.executable,
            "supported": python_ok,
        },
        "data": {
            "path": str(data_path),
            "exists": data_exists,
            "size_bytes": data_path.stat().st_size if data_exists else None,
            "sha256": sha256(data_path) if data_exists else None,
        },
        "config": {
            "path": str(config_path),
            "valid": config_ok,
            "error": config_error,
        },
        "required_packages": required,
        "optional_packages": optional,
    }


def print_report(report: dict[str, object]) -> None:
    python = report["python"]
    data = report["data"]
    config = report["config"]
    print("WM-811K Project 1 readiness")
    print(f"[{'OK' if python['supported'] else 'NO'}] Python {python['version']}")
    print(f"[{'OK' if config['valid'] else 'NO'}] experiment config")
    print(f"[{'OK' if data['exists'] else 'NO'}] dataset: {data['path']}")
    for name, installed in report["required_packages"].items():
        print(f"[{'OK' if installed else 'NO'}] required package: {name}")
    for name, installed in report["optional_packages"].items():
        print(f"[{'OK' if installed else '--'}] optional package: {name}")
    if data["exists"]:
        print(f"dataset sha256: {data['sha256']}")
    print("READY" if report["ready"] else "NOT READY")


def main() -> int:
    parser = argparse.ArgumentParser(description="Check Project 1 execution prerequisites.")
    parser.add_argument("--json", action="store_true", help="Print JSON instead of text.")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Return a non-zero exit code when required items are missing.",
    )
    args = parser.parse_args()
    report = inspect()
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print_report(report)
    return 1 if args.strict and not report["ready"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
