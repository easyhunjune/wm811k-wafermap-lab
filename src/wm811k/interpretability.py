from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional


@dataclass(frozen=True)
class GradCamOutput:
    cams: torch.Tensor
    logits: torch.Tensor
    zero_maps: torch.Tensor


@dataclass(frozen=True)
class OverlapMetrics:
    defect_mass_ratio: float
    dilated_defect_mass_ratio: float
    die_cam_mass: float
    padding_cam_mass_ratio: float


def grad_cam(
    model: nn.Module,
    target_layer: nn.Module,
    inputs: torch.Tensor,
    target_classes: torch.Tensor,
    *,
    output_size: tuple[int, int] | None = None,
) -> GradCamOutput:
    """Calculate batched Grad-CAM maps and remove all temporary hooks."""
    if inputs.ndim != 4:
        raise ValueError("inputs must be a BxCxHxW tensor.")
    if target_classes.ndim != 1 or target_classes.shape[0] != inputs.shape[0]:
        raise ValueError("target_classes must contain one class index per input.")
    if inputs.shape[0] == 0:
        raise ValueError("inputs cannot be empty.")

    # Grad-CAM needs deterministic BatchNorm and no dropout, but the caller's mode is
    # theirs to keep: a diagnostic call from inside a training loop must not leave the
    # model in eval afterwards.
    was_training = model.training
    model.eval()
    activations: list[torch.Tensor] = []
    gradients: list[torch.Tensor] = []
    tensor_hook: torch.utils.hooks.RemovableHandle | None = None

    def capture_gradient(gradient: torch.Tensor) -> None:
        gradients.append(gradient.detach())

    def capture_activation(
        _module: nn.Module,
        _arguments: tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> None:
        nonlocal tensor_hook
        activations.append(output.detach())
        tensor_hook = output.register_hook(capture_gradient)

    module_hook = target_layer.register_forward_hook(capture_activation)
    try:
        model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            logits = model(inputs)
            if logits.ndim != 2:
                raise ValueError("model output must be a BxK tensor.")
            if torch.any(target_classes < 0) or torch.any(target_classes >= logits.shape[1]):
                raise ValueError("target_classes contains an out-of-range class index.")
            selected = logits.gather(1, target_classes.to(logits.device).view(-1, 1))
            selected.sum().backward()

        if len(activations) != 1 or len(gradients) != 1:
            raise RuntimeError("Grad-CAM hooks did not capture exactly one forward/backward pass.")
        activation = activations[0]
        gradient = gradients[0]
        if activation.shape != gradient.shape or activation.ndim != 4:
            raise RuntimeError("Activation and gradient shapes do not match.")

        weights = gradient.mean(dim=(2, 3), keepdim=True)
        cams = torch.relu((weights * activation).sum(dim=1, keepdim=True))
        requested_size = output_size or tuple(int(value) for value in inputs.shape[-2:])
        cams = functional.interpolate(
            cams,
            size=requested_size,
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)

        flat = cams.flatten(start_dim=1)
        minima = flat.min(dim=1).values
        maxima = flat.max(dim=1).values
        ranges = maxima - minima
        zero_maps = ranges <= torch.finfo(cams.dtype).eps
        safe_ranges = torch.where(zero_maps, torch.ones_like(ranges), ranges)
        cams = (cams - minima[:, None, None]) / safe_ranges[:, None, None]
        cams = torch.where(zero_maps[:, None, None], torch.zeros_like(cams), cams)
        cams = cams.clamp(0.0, 1.0)
        if not torch.isfinite(cams).all():
            raise RuntimeError("Grad-CAM produced non-finite values.")
        return GradCamOutput(
            cams=cams.detach(),
            logits=logits.detach(),
            zero_maps=zero_maps.detach(),
        )
    finally:
        if tensor_hook is not None:
            tensor_hook.remove()
        module_hook.remove()
        model.zero_grad(set_to_none=True)
        model.train(was_training)


def _as_float_mask(value: torch.Tensor | np.ndarray, *, name: str) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=torch.float64)
    if tensor.ndim != 2:
        raise ValueError(f"{name} must be a 2-D array.")
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{name} contains non-finite values.")
    return (tensor > 0).to(torch.float64)


def cam_overlap_metrics(
    cam: torch.Tensor | np.ndarray,
    die_mask: torch.Tensor | np.ndarray,
    defect_mask: torch.Tensor | np.ndarray,
) -> OverlapMetrics:
    """Measure CAM mass on exact and one-pixel-dilated defect masks."""
    cam_tensor = torch.as_tensor(cam, dtype=torch.float64)
    if cam_tensor.ndim != 2:
        raise ValueError("cam must be a 2-D array.")
    if not torch.isfinite(cam_tensor).all():
        raise ValueError("cam contains non-finite values.")
    if torch.any(cam_tensor < 0):
        raise ValueError("cam cannot contain negative values.")

    die = _as_float_mask(die_mask, name="die_mask")
    defect = _as_float_mask(defect_mask, name="defect_mask")
    if cam_tensor.shape != die.shape or cam_tensor.shape != defect.shape:
        raise ValueError("cam, die_mask and defect_mask must have identical shapes.")
    defect = defect * die
    dilated = functional.max_pool2d(
        defect[None, None],
        kernel_size=3,
        stride=1,
        padding=1,
    ).squeeze(0).squeeze(0)
    dilated = dilated * die

    die_mass = float((cam_tensor * die).sum().item())
    total_mass = float(cam_tensor.sum().item())
    padding_mass = float((cam_tensor * (1.0 - die)).sum().item())
    padding_ratio = padding_mass / total_mass if total_mass > 0.0 else float("nan")
    if die_mass <= 0.0:
        exact_ratio = float("nan")
        dilated_ratio = float("nan")
    else:
        exact_ratio = float((cam_tensor * defect).sum().item()) / die_mass
        dilated_ratio = float((cam_tensor * dilated).sum().item()) / die_mass
    return OverlapMetrics(
        defect_mass_ratio=exact_ratio,
        dilated_defect_mass_ratio=dilated_ratio,
        die_cam_mass=die_mass,
        padding_cam_mass_ratio=padding_ratio,
    )
