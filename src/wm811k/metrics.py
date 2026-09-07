from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass

import numpy as np
from scipy.stats import beta
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    recall_score,
)


@dataclass(frozen=True)
class Interval:
    point: float
    lower: float
    upper: float
    valid_replicates: int
    attempted_replicates: int

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def exact_binomial_interval(
    successes: int,
    trials: int,
    *,
    confidence: float = 0.95,
) -> dict[str, float | int | str]:
    """Return a Clopper-Pearson interval as an IID sensitivity reference."""
    if trials <= 0 or not 0 <= successes <= trials:
        raise ValueError("successes and trials must satisfy 0 <= successes <= trials.")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie between zero and one.")
    alpha = 1.0 - confidence
    lower = (
        0.0
        if successes == 0
        else float(beta.ppf(alpha / 2.0, successes, trials - successes + 1))
    )
    upper = (
        1.0
        if successes == trials
        else float(beta.ppf(1.0 - alpha / 2.0, successes + 1, trials - successes))
    )
    return {
        "point": successes / trials,
        "lower": lower,
        "upper": upper,
        "successes": successes,
        "trials": trials,
        "method": "clopper_pearson_iid_reference",
    }


def classification_summary(
    y_true: Iterable[int],
    y_pred: Iterable[int],
    *,
    labels: Iterable[int],
) -> dict[str, object]:
    truth = np.asarray(tuple(y_true), dtype=np.int64)
    prediction = np.asarray(tuple(y_pred), dtype=np.int64)
    label_order = np.asarray(tuple(labels), dtype=np.int64)
    if truth.shape != prediction.shape or truth.ndim != 1:
        raise ValueError("y_true and y_pred must be one-dimensional arrays of equal length.")
    matrix = confusion_matrix(truth, prediction, labels=label_order)
    row_totals = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(
        matrix,
        row_totals,
        out=np.zeros_like(matrix, dtype=np.float64),
        where=row_totals != 0,
    )
    return {
        "macro_f1": float(
            f1_score(truth, prediction, labels=label_order, average="macro", zero_division=0)
        ),
        "balanced_accuracy": float(balanced_accuracy_score(truth, prediction)),
        "per_class_recall": recall_score(
            truth,
            prediction,
            labels=label_order,
            average=None,
            zero_division=0,
        ).tolist(),
        "confusion_matrix": matrix.tolist(),
        "row_normalized_confusion_matrix": normalized.tolist(),
    }


def _cluster_indices(groups: np.ndarray) -> dict[object, np.ndarray]:
    return {group: np.flatnonzero(groups == group) for group in np.unique(groups)}


def cluster_bootstrap_interval(
    y_true: Iterable[int],
    y_pred: Iterable[int],
    groups: Iterable[object],
    *,
    metric: Callable[[np.ndarray, np.ndarray], float],
    required_labels: Iterable[int],
    n_bootstrap: int = 2000,
    confidence: float = 0.95,
    seed: int = 42,
    max_attempt_multiplier: int = 20,
) -> Interval:
    truth = np.asarray(tuple(y_true), dtype=np.int64)
    prediction = np.asarray(tuple(y_pred), dtype=np.int64)
    group_array = np.asarray(tuple(groups), dtype=object)
    if not (truth.shape == prediction.shape == group_array.shape) or truth.ndim != 1:
        raise ValueError("Predictions, labels and groups must be aligned one-dimensional arrays.")
    if not 0 < confidence < 1:
        raise ValueError("confidence must lie between zero and one.")
    if n_bootstrap <= 0:
        raise ValueError("n_bootstrap must be positive.")

    required = set(required_labels)
    lookup = _cluster_indices(group_array)
    unique_groups = np.asarray(tuple(lookup), dtype=object)
    if len(unique_groups) < 2:
        raise ValueError("Cluster bootstrap requires at least two lots.")

    rng = np.random.default_rng(seed)
    values: list[float] = []
    attempts = 0
    max_attempts = n_bootstrap * max_attempt_multiplier
    while len(values) < n_bootstrap and attempts < max_attempts:
        attempts += 1
        sampled_groups = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        indices = np.concatenate([lookup[group] for group in sampled_groups])
        sampled_truth = truth[indices]
        if not required.issubset(set(sampled_truth.tolist())):
            continue
        values.append(float(metric(sampled_truth, prediction[indices])))

    if not values:
        raise RuntimeError("No valid bootstrap replicate contained the required classes.")
    alpha = (1.0 - confidence) / 2.0
    return Interval(
        point=float(metric(truth, prediction)),
        lower=float(np.quantile(values, alpha)),
        upper=float(np.quantile(values, 1.0 - alpha)),
        valid_replicates=len(values),
        attempted_replicates=attempts,
    )


def macro_f1_cluster_interval(
    y_true: Iterable[int],
    y_pred: Iterable[int],
    groups: Iterable[object],
    *,
    labels: Iterable[int],
    n_bootstrap: int = 2000,
    seed: int = 42,
) -> Interval:
    label_order = tuple(labels)

    def metric(truth: np.ndarray, prediction: np.ndarray) -> float:
        return float(
            f1_score(
                truth,
                prediction,
                labels=label_order,
                average="macro",
                zero_division=0,
            )
        )

    return cluster_bootstrap_interval(
        y_true,
        y_pred,
        groups,
        metric=metric,
        required_labels=label_order,
        n_bootstrap=n_bootstrap,
        seed=seed,
    )


def class_recall_cluster_interval(
    y_true: Iterable[int],
    y_pred: Iterable[int],
    groups: Iterable[object],
    *,
    target_label: int,
    n_bootstrap: int = 2000,
    seed: int = 42,
) -> Interval:
    def metric(truth: np.ndarray, prediction: np.ndarray) -> float:
        return float(
            recall_score(
                truth,
                prediction,
                labels=[target_label],
                average=None,
                zero_division=0,
            )[0]
        )

    return cluster_bootstrap_interval(
        y_true,
        y_pred,
        groups,
        metric=metric,
        required_labels=(target_label,),
        n_bootstrap=n_bootstrap,
        seed=seed,
    )
