from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from torch import nn

from wm811k.interpretability import cam_overlap_metrics, grad_cam
from wm811k.model import WaferCNN


def test_grad_cam_shape_range_reproducibility_and_hook_cleanup() -> None:
    torch.manual_seed(42)
    model = WaferCNN(input_channels=2, num_classes=9)
    target_layer = model.features[3][0]
    inputs = torch.rand(2, 2, 64, 64)
    targets = torch.tensor([4, 7])
    forward_hooks_before = len(target_layer._forward_hooks)

    first = grad_cam(model, target_layer, inputs, targets)
    second = grad_cam(model, target_layer, inputs, targets)

    assert first.cams.shape == (2, 64, 64)
    assert torch.isfinite(first.cams).all()
    assert torch.all((first.cams >= 0.0) & (first.cams <= 1.0))
    assert torch.equal(first.cams, second.cams)
    assert torch.equal(first.zero_maps, second.zero_maps)
    assert len(target_layer._forward_hooks) == forward_hooks_before


def test_grad_cam_rejects_invalid_targets_and_removes_hook() -> None:
    model = WaferCNN(input_channels=2, num_classes=9)
    target_layer = model.features[3][0]
    forward_hooks_before = len(target_layer._forward_hooks)

    with pytest.raises(ValueError, match="one class index"):
        grad_cam(model, target_layer, torch.rand(2, 2, 64, 64), torch.tensor([4]))
    with pytest.raises(ValueError, match="out-of-range"):
        grad_cam(model, target_layer, torch.rand(2, 2, 64, 64), torch.tensor([4, 9]))

    assert len(target_layer._forward_hooks) == forward_hooks_before


def test_grad_cam_constant_map_is_returned_as_zero() -> None:
    model = WaferCNN(input_channels=2, num_classes=9)
    for parameter in model.parameters():
        torch.nn.init.zeros_(parameter)
    result = grad_cam(
        model,
        model.features[3][0],
        torch.ones(1, 2, 64, 64),
        torch.tensor([4]),
    )
    assert result.zero_maps.tolist() == [True]
    assert torch.count_nonzero(result.cams) == 0


def test_overlap_metrics_exact_dilated_and_zero_denominators() -> None:
    cam = np.ones((4, 4), dtype=np.float32)
    die = np.zeros((4, 4), dtype=np.float32)
    die[1:3, 1:3] = 1
    defect = np.zeros((4, 4), dtype=np.float32)
    defect[1, 1] = 1

    metrics = cam_overlap_metrics(cam, die, defect)
    assert metrics.defect_mass_ratio == pytest.approx(0.25)
    assert metrics.dilated_defect_mass_ratio == pytest.approx(1.0)
    assert metrics.padding_cam_mass_ratio == pytest.approx(0.75)

    empty_cam = cam_overlap_metrics(np.zeros((4, 4)), die, defect)
    assert math.isnan(empty_cam.defect_mass_ratio)
    assert math.isnan(empty_cam.dilated_defect_mass_ratio)
    assert math.isnan(empty_cam.padding_cam_mass_ratio)

    empty_die = cam_overlap_metrics(cam, np.zeros((4, 4)), defect)
    assert math.isnan(empty_die.defect_mass_ratio)
    assert math.isnan(empty_die.dilated_defect_mass_ratio)


def test_grad_cam_restores_the_callers_training_mode():
    model = nn.Sequential(nn.Conv2d(1, 2, 3, padding=1), nn.Flatten(), nn.Linear(2 * 16, 3))
    inputs = torch.rand(2, 1, 4, 4)
    targets = torch.tensor([0, 1])

    model.train()
    grad_cam(model, model[0], inputs, targets)
    assert model.training

    model.eval()
    grad_cam(model, model[0], inputs, targets)
    assert not model.training
