from __future__ import annotations

import numpy as np


def normalized_radius_grid(wafer_map: np.ndarray) -> np.ndarray:
    """Return die-centroid radius normalized by the outermost observed die."""
    array = np.asarray(wafer_map)
    if array.ndim != 2:
        raise ValueError(f"waferMap must be 2-D, got shape {array.shape}.")
    die_mask = array > 0
    coordinates = np.argwhere(die_mask)
    if len(coordinates) == 0:
        raise ValueError("waferMap contains no die positions.")

    center = coordinates.mean(axis=0)
    row_grid, column_grid = np.indices(array.shape)
    distance = np.sqrt((row_grid - center[0]) ** 2 + (column_grid - center[1]) ** 2)
    radius_scale = float(distance[die_mask].max())
    normalized = np.full(array.shape, np.nan, dtype=np.float64)
    if radius_scale == 0:
        normalized[die_mask] = 0.0
    else:
        normalized[die_mask] = distance[die_mask] / radius_scale
    return normalized


def radial_counts(
    wafer_map: np.ndarray,
    bin_edges: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    array = np.asarray(wafer_map)
    edges = np.asarray(bin_edges, dtype=np.float64)
    if edges.ndim != 1 or len(edges) < 2:
        raise ValueError("bin_edges must be a one-dimensional array with at least two values.")
    if not np.all(np.diff(edges) > 0):
        raise ValueError("bin_edges must be strictly increasing.")

    die_mask = array > 0
    defect_mask = array == 2
    radii = normalized_radius_grid(array)[die_mask]
    bin_indices = np.searchsorted(edges, radii, side="right") - 1
    bin_indices = np.clip(bin_indices, 0, len(edges) - 2)
    die_counts = np.bincount(bin_indices, minlength=len(edges) - 1)
    defect_counts = np.bincount(
        bin_indices,
        weights=defect_mask[die_mask].astype(np.int64),
        minlength=len(edges) - 1,
    ).astype(np.int64)
    return die_counts.astype(np.int64), defect_counts


def edge_counts(
    wafer_map: np.ndarray,
    radius_threshold: float,
) -> tuple[int, int]:
    if not 0 <= radius_threshold <= 1:
        raise ValueError("radius_threshold must be between zero and one.")
    array = np.asarray(wafer_map)
    die_mask = array > 0
    radii = normalized_radius_grid(array)
    edge_mask = die_mask & (radii >= radius_threshold)
    return int(edge_mask.sum()), int(((array == 2) & edge_mask).sum())
