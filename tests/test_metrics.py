import numpy as np
import pytest

from wm811k.metrics import (
    classification_summary,
    exact_binomial_interval,
    macro_f1_cluster_interval,
)


def test_classification_summary_uses_sample_level_confusion_matrix():
    summary = classification_summary([0, 0, 1, 1], [0, 1, 1, 1], labels=[0, 1])
    assert summary["confusion_matrix"] == [[1, 1], [0, 2]]
    assert summary["row_normalized_confusion_matrix"] == [[0.5, 0.5], [0.0, 1.0]]


def test_cluster_bootstrap_is_deterministic_for_fixed_seed():
    truth = np.array([0, 1, 0, 1, 0, 1])
    prediction = np.array([0, 1, 1, 1, 0, 0])
    groups = np.array(["a", "a", "b", "b", "c", "c"])
    first = macro_f1_cluster_interval(
        truth,
        prediction,
        groups,
        labels=[0, 1],
        n_bootstrap=50,
        seed=7,
    )
    second = macro_f1_cluster_interval(
        truth,
        prediction,
        groups,
        labels=[0, 1],
        n_bootstrap=50,
        seed=7,
    )
    assert first == second
    assert 0.0 <= first.lower <= first.upper <= 1.0


def test_cluster_bootstrap_requires_multiple_lots():
    with pytest.raises(ValueError, match="at least two lots"):
        macro_f1_cluster_interval(
            [0, 1],
            [0, 1],
            ["same", "same"],
            labels=[0, 1],
            n_bootstrap=10,
        )


def test_exact_binomial_interval_is_not_degenerate_for_perfect_recall():
    interval = exact_binomial_interval(32, 32)
    assert interval["point"] == 1.0
    assert interval["lower"] == pytest.approx(0.8911188393)
    assert interval["upper"] == 1.0
    assert interval["method"] == "clopper_pearson_iid_reference"
