from __future__ import annotations

from collections.abc import Iterable, Sequence
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

SPLIT_NAMES = ("train", "validation", "test")


def _distribution_deviation(
    y: np.ndarray,
    split_indices: Sequence[np.ndarray],
    labels: Sequence[str],
) -> float:
    global_counts = pd.Series(y).value_counts(normalize=True).reindex(labels, fill_value=0.0)
    deviations = []
    for indices in split_indices:
        split_counts = (
            pd.Series(y[indices]).value_counts(normalize=True).reindex(labels, fill_value=0.0)
        )
        deviations.append(float(np.abs(split_counts - global_counts).mean()))
    return float(np.mean(deviations))


def _has_all_labels(y: np.ndarray, indices: np.ndarray, labels: set[str]) -> bool:
    return set(y[indices].tolist()) == labels


def _groups_are_disjoint(groups: np.ndarray, split_indices: Sequence[np.ndarray]) -> bool:
    group_sets = [set(groups[indices].tolist()) for indices in split_indices]
    return all(left.isdisjoint(right) for left, right in combinations(group_sets, 2))


def find_group_split(
    frame: pd.DataFrame,
    *,
    labels: Iterable[str],
    candidate_seeds: Iterable[int] = (42,),
    label_column: str = "failure_label",
    group_column: str = "lotName",
    id_column: str = "source_row_id",
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Find a 60/20/20-like split without consulting model or test performance."""
    required = {label_column, group_column, id_column}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing split columns: {sorted(missing)}")

    label_order = tuple(labels)
    label_set = set(label_order)
    y = frame[label_column].astype(str).to_numpy()
    groups = frame[group_column].astype(str).to_numpy()
    row_ids = frame[id_column].to_numpy()

    if set(y.tolist()) != label_set:
        missing_labels = sorted(label_set.difference(y.tolist()))
        unexpected = sorted(set(y.tolist()).difference(label_set))
        raise ValueError(
            f"Label mismatch before splitting; missing={missing_labels}, unexpected={unexpected}"
        )
    if len(set(groups.tolist())) < 5:
        raise ValueError("At least five distinct lots are required for the outer group split.")

    best: tuple[float, int, int, int, tuple[np.ndarray, np.ndarray, np.ndarray]] | None = None
    rejected = 0
    for seed in candidate_seeds:
        outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
        for outer_fold, (train_validation, test) in enumerate(outer.split(row_ids, y, groups)):
            inner_y = y[train_validation]
            inner_groups = groups[train_validation]
            inner = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=seed + 1)
            for inner_fold, (train_relative, validation_relative) in enumerate(
                inner.split(train_validation, inner_y, inner_groups)
            ):
                train = train_validation[train_relative]
                validation = train_validation[validation_relative]
                candidate = (train, validation, test)
                if not all(_has_all_labels(y, indices, label_set) for indices in candidate):
                    rejected += 1
                    continue
                if not _groups_are_disjoint(groups, candidate):
                    rejected += 1
                    continue
                score = _distribution_deviation(y, candidate, label_order)
                item = (score, seed, outer_fold, inner_fold, candidate)
                if best is None or item[:4] < best[:4]:
                    best = item

    if best is None:
        raise RuntimeError(
            "No candidate split contained every class with disjoint lots. "
            "Inspect class-by-lot support or expand candidate seeds."
        )

    score, seed, outer_fold, inner_fold, indices = best
    assignments = np.empty(len(frame), dtype=object)
    for split_name, split_indices in zip(SPLIT_NAMES, indices, strict=True):
        assignments[split_indices] = split_name
    split_frame = pd.DataFrame({id_column: row_ids, "split": assignments})
    metadata = {
        "seed": seed,
        "outer_fold": outer_fold,
        "inner_fold": inner_fold,
        "distribution_deviation": score,
        "rejected_candidates": rejected,
        "selection_uses_model_metrics": False,
    }
    return split_frame, metadata


def split_audit(
    frame: pd.DataFrame,
    *,
    labels: Iterable[str],
    label_column: str = "failure_label",
    group_column: str = "lotName",
    split_column: str = "split",
) -> dict[str, object]:
    label_order = tuple(labels)
    group_sets = {
        split: set(frame.loc[frame[split_column] == split, group_column].astype(str))
        for split in SPLIT_NAMES
    }
    overlaps = {
        f"{left}__{right}": len(group_sets[left].intersection(group_sets[right]))
        for left, right in combinations(SPLIT_NAMES, 2)
    }
    counts = {
        split: {
            label: int(
                ((frame[split_column] == split) & (frame[label_column].astype(str) == label)).sum()
            )
            for label in label_order
        }
        for split in SPLIT_NAMES
    }
    missing = {
        split: [label for label, count in split_counts.items() if count == 0]
        for split, split_counts in counts.items()
    }
    if any(overlaps.values()):
        raise AssertionError(f"Group leakage detected: {overlaps}")
    if any(missing.values()):
        raise AssertionError(f"Class coverage failure: {missing}")
    return {
        "rows": {split: int((frame[split_column] == split).sum()) for split in SPLIT_NAMES},
        "lots": {split: len(groups) for split, groups in group_sets.items()},
        "group_overlap": overlaps,
        "class_counts": counts,
        "missing_classes": missing,
    }
