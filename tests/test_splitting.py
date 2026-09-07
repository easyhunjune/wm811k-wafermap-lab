import numpy as np
import pandas as pd
import pytest

from wm811k.constants import LABELS
from wm811k.splitting import (
    _distribution_deviation,
    _groups_are_disjoint,
    _has_all_labels,
    find_group_split,
    split_audit,
)

SPLIT_NAMES = ("train", "validation", "test")
RARE_LABEL = "Scratch"
assert RARE_LABEL in LABELS


def _make_balanced_frame(*, num_lots: int = 25) -> pd.DataFrame:
    rows = []
    for lot_index in range(num_lots):
        for label in LABELS:
            rows.append(
                {
                    "source_row_id": 1000 + 7 * len(rows),
                    "lotName": f"lot-{lot_index:02d}",
                    "failure_label": label,
                }
            )
    return pd.DataFrame(rows)


def _make_scoring_frame() -> pd.DataFrame:
    lot_specs = (
        ("lot-test", 2, None, 0),
        ("lot-skew-center", 1, "Center", 20),
        ("lot-skew-donut", 1, "Donut", 20),
        ("lot-balanced-validation", 3, None, 0),
        ("lot-balanced-train-a", 2, None, 0),
        ("lot-balanced-train-b", 2, None, 0),
    )
    rows = []
    for lot_name, base_count, dominant_label, dominant_count in lot_specs:
        for label in LABELS:
            count = dominant_count if label == dominant_label else base_count
            for _ in range(count):
                rows.append(
                    {
                        "source_row_id": 5000 + 11 * len(rows),
                        "lotName": lot_name,
                        "failure_label": label,
                    }
                )
    return pd.DataFrame(rows)


def _make_audit_frame(
    *,
    group_names: dict[str, str] | None = None,
    missing: tuple[str, str] | None = None,
) -> pd.DataFrame:
    groups = group_names or {split: f"{split}-lot" for split in SPLIT_NAMES}
    rows = []
    for split in SPLIT_NAMES:
        for label in LABELS:
            if missing == (split, label):
                continue
            rows.append(
                {
                    "lotName": groups[split],
                    "failure_label": label,
                    "split": split,
                }
            )
    return pd.DataFrame(rows)


def _fake_splitter_class(
    *,
    include_disjoint_candidate: bool,
    overlap_with_test: bool = False,
):
    class FakeSplitter:
        def __init__(self, n_splits, shuffle, random_state):
            self.n_splits = n_splits

        def split(self, X, y, groups):
            lot_size = len(LABELS)
            if self.n_splits == 5:
                if overlap_with_test:
                    test = np.append(0, np.arange(lot_size, lot_size * 2))
                    train_validation = np.setdiff1d(np.arange(len(X)), test)
                    assert "lot-00" in set(groups[test])
                else:
                    test = np.arange(lot_size)
                    train_validation = np.arange(lot_size, len(X))
                assert set(y[test]) == set(LABELS)
                yield train_validation, test
                return

            assert self.n_splits == 4
            if overlap_with_test:
                validation = np.arange(lot_size - 1, lot_size * 2 - 1)
                train = np.setdiff1d(np.arange(len(X)), validation)
                assert "lot-00" in set(groups[train])
                assert set(y[train]) == set(LABELS)
                assert set(y[validation]) == set(LABELS)
                yield train, validation
                return

            validation = np.arange(lot_size)
            train = np.arange(lot_size, len(X))
            # Relative row 18 belongs to lot-03 after outer test rows are removed.
            shared_row_relative_index = lot_size * 2
            overlapping_validation = np.append(validation, shared_row_relative_index)
            overlapping_train = train[train != shared_row_relative_index]

            assert set(y[overlapping_train]) == set(LABELS)
            assert set(y[overlapping_validation]) == set(LABELS)
            yield overlapping_train, overlapping_validation

            if include_disjoint_candidate:
                assert set(y[train]) == set(LABELS)
                assert set(y[validation]) == set(LABELS)
                yield train, validation

    return FakeSplitter


def _scoring_splitter_class():
    class FakeSplitter:
        def __init__(self, n_splits, shuffle, random_state):
            self.n_splits = n_splits

        def split(self, X, y, groups):
            groups = np.asarray(groups)
            if self.n_splits == 5:
                test = np.flatnonzero(groups == "lot-test")
                train_validation = np.flatnonzero(groups != "lot-test")
                assert set(y[test]) == set(LABELS)
                yield train_validation, test
                return

            assert self.n_splits == 4
            worse_validation = np.flatnonzero(groups == "lot-skew-center")
            worse_train = np.flatnonzero(groups != "lot-skew-center")
            better_validation = np.flatnonzero(groups == "lot-balanced-validation")
            better_train = np.flatnonzero(groups != "lot-balanced-validation")

            for train, validation in (
                (worse_train, worse_validation),
                (better_train, better_validation),
            ):
                assert set(y[train]) == set(LABELS)
                assert set(y[validation]) == set(LABELS)
                yield train, validation

    return FakeSplitter


def test_distribution_deviation_uses_absolute_class_proportion_errors():
    y = np.array(["A", "A", "B", "B"])
    labels = ("A", "B")
    matching_distribution = (np.array([0, 2]), np.array([1, 3]))
    opposite_skews = (np.array([0, 1]), np.array([2, 3]))

    assert _distribution_deviation(y, matching_distribution, labels) == pytest.approx(0.0)
    assert _distribution_deviation(y, opposite_skews, labels) == pytest.approx(0.5)
    assert _distribution_deviation(y, opposite_skews, labels) > 0.0


def test_find_group_split_covers_rows_classes_and_has_pairwise_disjoint_lots():
    frame = _make_balanced_frame()

    split_frame, metadata = find_group_split(frame, labels=LABELS, candidate_seeds=(17,))
    merged = frame.merge(split_frame, on="source_row_id", validate="one_to_one")
    audit = split_audit(merged, labels=LABELS)

    assert len(split_frame) == len(frame)
    assert len(merged) == len(frame)
    assert not np.array_equal(frame["source_row_id"].to_numpy(), frame.index.to_numpy())
    assert set(split_frame["split"]) == set(SPLIT_NAMES)
    # Contract pin: candidate selection must never consult model metrics.
    assert metadata["selection_uses_model_metrics"] is False
    assert all(overlap == 0 for overlap in audit["group_overlap"].values())
    assert all(not missing for missing in audit["missing_classes"].values())

    row_ids = {
        split: set(merged.loc[merged["split"] == split, "source_row_id"])
        for split in SPLIT_NAMES
    }
    assert row_ids["train"] | row_ids["validation"] | row_ids["test"] == set(
        frame["source_row_id"]
    )
    assert row_ids["train"].isdisjoint(row_ids["validation"])
    assert row_ids["train"].isdisjoint(row_ids["test"])
    assert row_ids["validation"].isdisjoint(row_ids["test"])
    assert len(row_ids["train"]) > len(row_ids["validation"])
    assert len(row_ids["train"]) > len(row_ids["test"])
    assert len(row_ids["train"]) / len(frame) == pytest.approx(0.6, abs=0.05)

    lots = {
        split: set(merged.loc[merged["split"] == split, "lotName"])
        for split in SPLIT_NAMES
    }
    assert lots["train"].intersection(lots["validation"]) == set()
    assert lots["train"].intersection(lots["test"]) == set()
    assert lots["validation"].intersection(lots["test"]) == set()
    assert len(lots["train"]) > len(lots["validation"])
    assert len(lots["train"]) > len(lots["test"])
    assert len(lots["train"]) / frame["lotName"].nunique() == pytest.approx(0.6, abs=0.05)

    for split in SPLIT_NAMES:
        split_labels = set(merged.loc[merged["split"] == split, "failure_label"])
        assert split_labels == set(LABELS)


def test_find_group_split_selects_candidate_with_lowest_distribution_deviation(monkeypatch):
    frame = _make_scoring_frame()
    y = frame["failure_label"].to_numpy()
    groups = frame["lotName"].to_numpy()
    test = np.flatnonzero(groups == "lot-test")
    worse_validation = np.flatnonzero(groups == "lot-skew-center")
    worse_train = np.flatnonzero(
        (groups != "lot-test") & (groups != "lot-skew-center")
    )
    better_validation = np.flatnonzero(groups == "lot-balanced-validation")
    better_train = np.flatnonzero(
        (groups != "lot-test") & (groups != "lot-balanced-validation")
    )
    worse_score = _distribution_deviation(
        y,
        (worse_train, worse_validation, test),
        LABELS,
    )
    better_score = _distribution_deviation(
        y,
        (better_train, better_validation, test),
        LABELS,
    )
    assert better_score < worse_score

    fake_splitter = _scoring_splitter_class()
    monkeypatch.setattr("wm811k.splitting.StratifiedGroupKFold", fake_splitter)
    split_frame, metadata = find_group_split(frame, labels=LABELS, candidate_seeds=(17,))
    merged = frame.merge(split_frame, on="source_row_id", validate="one_to_one")

    assert metadata["distribution_deviation"] == pytest.approx(better_score)
    assert set(merged.loc[merged["split"] == "validation", "lotName"]) == {
        "lot-balanced-validation"
    }
    assert set(merged.loc[merged["split"] == "test", "lotName"]) == {"lot-test"}


def test_find_group_split_rejects_every_label_complete_candidate_with_group_overlap(
    monkeypatch,
):
    frame = _make_balanced_frame()
    fake_splitter = _fake_splitter_class(include_disjoint_candidate=False)
    monkeypatch.setattr("wm811k.splitting.StratifiedGroupKFold", fake_splitter)

    with pytest.raises(
        RuntimeError,
        match="No candidate split contained every class with disjoint lots",
    ):
        find_group_split(frame, labels=LABELS, candidate_seeds=(17,))


def test_find_group_split_counts_overlap_and_selects_disjoint_candidate(monkeypatch):
    frame = _make_balanced_frame()
    fake_splitter = _fake_splitter_class(include_disjoint_candidate=True)
    monkeypatch.setattr("wm811k.splitting.StratifiedGroupKFold", fake_splitter)

    split_frame, metadata = find_group_split(frame, labels=LABELS, candidate_seeds=(17,))
    merged = frame.merge(split_frame, on="source_row_id", validate="one_to_one")

    assert metadata["rejected_candidates"] == 1
    lots = {
        split: set(merged.loc[merged["split"] == split, "lotName"])
        for split in SPLIT_NAMES
    }
    assert lots["train"].isdisjoint(lots["validation"])
    assert lots["train"].isdisjoint(lots["test"])
    assert lots["validation"].isdisjoint(lots["test"])


def test_find_group_split_rejects_label_complete_train_test_group_overlap(monkeypatch):
    frame = _make_balanced_frame()
    fake_splitter = _fake_splitter_class(
        include_disjoint_candidate=False,
        overlap_with_test=True,
    )
    monkeypatch.setattr("wm811k.splitting.StratifiedGroupKFold", fake_splitter)

    with pytest.raises(
        RuntimeError,
        match="No candidate split contained every class with disjoint lots",
    ):
        find_group_split(frame, labels=LABELS, candidate_seeds=(17,))


def test_split_audit_rejects_group_leakage():
    frame = _make_audit_frame(
        group_names={
            "train": "shared-lot",
            "validation": "shared-lot",
            "test": "test-lot",
        }
    )

    with pytest.raises(AssertionError, match="Group leakage"):
        split_audit(frame, labels=LABELS)


@pytest.mark.parametrize("split_sharing_test_lot", ["train", "validation"])
def test_split_audit_rejects_group_leakage_involving_test(split_sharing_test_lot):
    group_names = {split: f"{split}-lot" for split in SPLIT_NAMES}
    group_names[split_sharing_test_lot] = "shared-test-lot"
    group_names["test"] = "shared-test-lot"
    frame = _make_audit_frame(group_names=group_names)

    with pytest.raises(AssertionError, match="Group leakage"):
        split_audit(frame, labels=LABELS)


def test_groups_are_disjoint_detects_overlap_and_accepts_disjoint_indices():
    groups = np.array(["lot-a", "lot-b", "lot-a", "lot-c"])
    overlapping = (np.array([0]), np.array([2]), np.array([3]))
    disjoint = (np.array([0]), np.array([1]), np.array([3]))

    assert not _groups_are_disjoint(groups, overlapping)
    assert _groups_are_disjoint(groups, disjoint)


def test_split_audit_rejects_missing_class():
    frame = _make_audit_frame(missing=("validation", RARE_LABEL))

    with pytest.raises(AssertionError, match="Class coverage failure"):
        split_audit(frame, labels=LABELS)


def test_find_group_split_rejects_missing_required_columns():
    frame = _make_balanced_frame().drop(columns=["lotName"])

    with pytest.raises(ValueError, match="Missing split columns"):
        find_group_split(frame, labels=LABELS, candidate_seeds=(17,))


@pytest.mark.parametrize("mismatch", ["missing", "unexpected"])
def test_find_group_split_rejects_label_mismatch(mismatch):
    frame = _make_balanced_frame()
    if mismatch == "missing":
        frame = frame.loc[frame["failure_label"] != RARE_LABEL].reset_index(drop=True)
    else:
        frame.loc[0, "failure_label"] = "unexpected-label"

    with pytest.raises(ValueError, match="Label mismatch"):
        find_group_split(frame, labels=LABELS, candidate_seeds=(17,))


def test_find_group_split_requires_at_least_five_lots():
    frame = _make_balanced_frame(num_lots=4)

    with pytest.raises(ValueError, match="At least five distinct lots"):
        find_group_split(frame, labels=LABELS, candidate_seeds=(17,))


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_find_group_split_rejects_candidates_when_rare_label_is_in_one_lot():
    frame = _make_balanced_frame()
    rare_label_outside_first_lot = (frame["failure_label"] == RARE_LABEL) & (
        frame["lotName"] != "lot-00"
    )
    frame = frame.loc[~rare_label_outside_first_lot].reset_index(drop=True)

    assert set(frame["failure_label"]) == set(LABELS)
    assert frame.loc[frame["failure_label"] == RARE_LABEL, "lotName"].nunique() == 1
    with pytest.raises(
        RuntimeError,
        match="No candidate split contained every class with disjoint lots",
    ):
        find_group_split(frame, labels=LABELS, candidate_seeds=(17,))


def test_has_all_labels_requires_every_expected_label():
    labels = np.array(LABELS)
    all_indices = np.arange(len(labels))
    missing_one_index = np.arange(len(labels) - 1)

    assert _has_all_labels(labels, all_indices, set(LABELS))
    assert not _has_all_labels(labels, missing_one_index, set(LABELS))
