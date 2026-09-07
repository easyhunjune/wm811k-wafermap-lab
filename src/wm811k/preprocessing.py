from __future__ import annotations

import numpy as np
import torch
from torch.nn import functional


def _validate_wafer_map(
    wafer_map: np.ndarray,
    *,
    validate_values: bool,
) -> np.ndarray:
    array = np.asarray(wafer_map)
    if array.ndim != 2:
        raise ValueError(f"waferMap must be 2-D, got shape {array.shape}.")
    if 0 in array.shape:
        raise ValueError("waferMap cannot have an empty dimension.")
    if validate_values:
        values = set(np.unique(array).tolist())
        if not values.issubset({0, 1, 2}):
            raise ValueError(f"waferMap contains unsupported values: {sorted(values)}")
    return array


def resize_with_padding(tensor: torch.Tensor, output_size: int) -> torch.Tensor:
    if tensor.ndim != 3:
        raise ValueError("Expected a CxHxW tensor.")
    if output_size <= 0:
        raise ValueError("output_size must be positive.")

    _, height, width = tensor.shape
    scale = min(output_size / height, output_size / width)
    resized_height = max(1, round(height * scale))
    resized_width = max(1, round(width * scale))
    resized = functional.interpolate(
        tensor.unsqueeze(0),
        size=(resized_height, resized_width),
        mode="nearest",
    ).squeeze(0)

    pad_height = output_size - resized_height
    pad_width = output_size - resized_width
    left = pad_width // 2
    right = pad_width - left
    top = pad_height // 2
    bottom = pad_height - top
    return functional.pad(resized, (left, right, top, bottom), value=0.0)


def encode_wafer_map(
    wafer_map: np.ndarray,
    *,
    output_size: int = 64,
    mode: str = "two_channel_masks",
    validate_values: bool = True,
) -> torch.Tensor:
    array = _validate_wafer_map(wafer_map, validate_values=validate_values)
    raw = torch.as_tensor(array, dtype=torch.float32)

    if mode == "categorical_single":
        channels = raw.unsqueeze(0)
    elif mode == "two_channel_masks":
        die_mask = (raw > 0).to(torch.float32)
        defect_mask = (raw == 2).to(torch.float32)
        channels = torch.stack((die_mask, defect_mask), dim=0)
    else:
        raise ValueError(f"Unsupported input mode: {mode}")
    return resize_with_padding(channels, output_size)
