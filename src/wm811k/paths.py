"""Path handling for artefacts that are committed and eventually published."""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def portable_dataset_path(path: Path) -> str:
    """Store a portable path instead of leaking a developer's absolute path."""
    try:
        relative = os.path.relpath(path.resolve(), PROJECT_ROOT.resolve())
    except ValueError:
        return path.name
    return Path(relative).as_posix()
