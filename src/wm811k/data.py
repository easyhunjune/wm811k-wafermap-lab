from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from .constants import LABELS


def unwrap_scalar(value: object) -> object:
    """Unwrap the nested singleton arrays used by common WM-811K pickle mirrors."""
    current = value
    for _ in range(8):
        if isinstance(current, np.ndarray):
            if current.size == 0:
                return None
            if current.size != 1:
                return current
            current = current.reshape(-1)[0]
            continue
        if isinstance(current, (list, tuple)):
            if len(current) == 0:
                return None
            if len(current) != 1:
                return current
            current = current[0]
            continue
        break
    return current


def normalize_failure_label(value: object) -> str | None:
    value = unwrap_scalar(value)
    if value is None:
        return None
    try:
        if bool(pd.isna(value)):
            return None
    except (TypeError, ValueError):
        return None
    text = str(value).strip()
    return text if text else None


def prepare_labeled_frame(
    frame: pd.DataFrame,
    *,
    labels: Iterable[str] = LABELS,
) -> pd.DataFrame:
    required = {"waferMap", "lotName", "failureType"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    allowed = set(labels)
    prepared = frame.copy()
    prepared.insert(0, "source_row_id", np.arange(len(prepared), dtype=np.int64))
    prepared["failure_label"] = prepared["failureType"].map(normalize_failure_label)
    prepared = prepared[prepared["failure_label"].isin(allowed)].copy()
    if prepared.empty:
        raise ValueError("No labeled rows matched the configured nine classes.")
    if prepared["lotName"].isna().any():
        raise ValueError("Labeled rows contain missing lotName values.")
    return prepared.reset_index(drop=True)
