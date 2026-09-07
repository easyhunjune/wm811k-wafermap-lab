import numpy as np
import pytest

from wm811k.spatial import edge_counts, normalized_radius_grid, radial_counts


def test_normalized_radius_uses_die_centroid_and_outermost_die():
    wafer = np.array(
        [
            [0, 1, 0],
            [1, 2, 1],
            [0, 1, 0],
        ]
    )
    radii = normalized_radius_grid(wafer)
    assert radii[1, 1] == pytest.approx(0.0)
    assert radii[0, 1] == pytest.approx(1.0)
    assert np.isnan(radii[0, 0])


def test_radial_and_edge_counts_preserve_die_and_defect_totals():
    wafer = np.array(
        [
            [0, 1, 0],
            [1, 2, 1],
            [0, 1, 0],
        ]
    )
    die_counts, defect_counts = radial_counts(wafer, np.linspace(0, 1, 3))
    assert die_counts.sum() == 5
    assert defect_counts.sum() == 1
    edge_die, edge_defect = edge_counts(wafer, 0.9)
    assert edge_die == 4
    assert edge_defect == 0


def test_rejects_map_without_die_positions():
    with pytest.raises(ValueError, match="no die"):
        normalized_radius_grid(np.zeros((3, 3), dtype=np.uint8))
