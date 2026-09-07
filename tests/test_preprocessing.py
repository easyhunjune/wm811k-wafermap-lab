import numpy as np
import pytest
import torch

from wm811k.preprocessing import encode_wafer_map


def test_two_channel_encoding_preserves_binary_masks_and_shape():
    wafer = np.array([[0, 1, 2], [0, 1, 1]], dtype=np.uint8)
    encoded = encode_wafer_map(wafer, output_size=8, mode="two_channel_masks")
    assert encoded.shape == (2, 8, 8)
    assert set(torch.unique(encoded).tolist()).issubset({0.0, 1.0})
    assert torch.all(encoded[1] <= encoded[0])


def test_rejects_unknown_wafer_values():
    with pytest.raises(ValueError, match="unsupported values"):
        encode_wafer_map(np.array([[0, 3]]), output_size=8)
