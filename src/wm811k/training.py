from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader

from .constants import LABELS


@dataclass(frozen=True)
class EpochOutput:
    loss: float
    accuracy: float
    macro_f1: float
    y_true: np.ndarray
    y_pred: np.ndarray


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    *,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> EpochOutput:
    is_training = optimizer is not None
    model.train(is_training)
    loss_sum = 0.0
    correct = 0
    sample_count = 0
    truth_batches: list[np.ndarray] = []
    prediction_batches: list[np.ndarray] = []

    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True).long().view(-1)
        if is_training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(is_training):
            logits = model(inputs)
            per_sample_loss = criterion(logits, targets)
            if per_sample_loss.ndim != 1:
                raise ValueError("criterion must return one loss value per sample.")
            loss = per_sample_loss.mean()
            if is_training:
                loss.backward()
                optimizer.step()

        predictions = logits.argmax(dim=1)
        batch_size = targets.size(0)
        loss_sum += float(per_sample_loss.detach().sum().item())
        correct += int((predictions == targets).sum().item())
        sample_count += batch_size
        truth_batches.append(targets.detach().cpu().numpy())
        prediction_batches.append(predictions.detach().cpu().numpy())

    if sample_count == 0:
        raise ValueError("DataLoader produced no samples.")
    y_true = np.concatenate(truth_batches)
    y_pred = np.concatenate(prediction_batches)
    return EpochOutput(
        loss=loss_sum / sample_count,
        accuracy=correct / sample_count,
        macro_f1=float(
            f1_score(
                y_true,
                y_pred,
                labels=range(len(LABELS)),
                average="macro",
                zero_division=0,
            )
        ),
        y_true=y_true,
        y_pred=y_pred,
    )


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    *,
    criterion: nn.Module,
    device: torch.device,
) -> EpochOutput:
    return run_epoch(model, loader, criterion=criterion, device=device)


def fit(
    model: nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    *,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epochs: int,
    patience: int,
    min_delta: float,
    checkpoint_path: Path,
    checkpoint_metadata: dict,
) -> tuple[list[dict[str, float | int]], dict]:
    history: list[dict[str, float | int]] = []
    best_metric = float("-inf")
    patience_reference = float("-inf")
    best_checkpoint: dict | None = None
    epochs_without_improvement = 0
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        train = run_epoch(
            model,
            train_loader,
            criterion=criterion,
            device=device,
            optimizer=optimizer,
        )
        validation = evaluate(
            model,
            validation_loader,
            criterion=criterion,
            device=device,
        )
        row = {
            "epoch": epoch,
            "train_loss": train.loss,
            "train_accuracy": train.accuracy,
            "train_macro_f1": train.macro_f1,
            "validation_loss": validation.loss,
            "validation_accuracy": validation.accuracy,
            "validation_macro_f1": validation.macro_f1,
        }
        history.append(row)
        print(
            f"epoch={epoch:03d} train_loss={train.loss:.5f} val_macro_f1={validation.macro_f1:.5f}"
        )
        # Saving and early stopping ask different questions. The checkpoint must hold the
        # highest validation macro F1 seen, so any improvement saves. Patience asks whether
        # the run is still making progress worth waiting for, so it keeps min_delta and
        # measures against the last epoch that cleared it.
        if validation.macro_f1 > best_metric:
            best_metric = validation.macro_f1
            best_checkpoint = {
                **checkpoint_metadata,
                "epoch": epoch,
                "best_validation_macro_f1": best_metric,
                "model_state_dict": copy.deepcopy(model.state_dict()),
                "optimizer_state_dict": copy.deepcopy(optimizer.state_dict()),
            }
            torch.save(best_checkpoint, checkpoint_path)
        if validation.macro_f1 > patience_reference + min_delta:
            patience_reference = validation.macro_f1
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"early_stopping epoch={epoch:03d} patience={patience}")
                break

    if best_checkpoint is None:
        raise RuntimeError("Training ended without a checkpoint.")
    return history, best_checkpoint
