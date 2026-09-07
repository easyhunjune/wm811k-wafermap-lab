import numpy as np
import pandas as pd

from wm811k.data import normalize_failure_label, prepare_labeled_frame


def test_normalize_nested_labels_and_empty_values():
    assert normalize_failure_label(np.array([["Center"]], dtype=object)) == "Center"
    assert normalize_failure_label(np.array([], dtype=object)) is None
    assert normalize_failure_label("") is None


def test_prepare_labeled_frame_does_not_turn_empty_into_none():
    frame = pd.DataFrame(
        {
            "waferMap": [np.ones((2, 2)), np.ones((2, 2))],
            "lotName": ["lot-a", "lot-b"],
            "failureType": [np.array([["none"]], dtype=object), np.array([], dtype=object)],
        }
    )
    result = prepare_labeled_frame(frame)
    assert result["failure_label"].tolist() == ["none"]
    assert result["source_row_id"].tolist() == [0]
