import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from wm811k.training import EpochOutput, fit, run_epoch

MIN_DELTA = 0.0001


def _validation_output(macro_f1: float) -> EpochOutput:
    return EpochOutput(
        loss=1.0,
        accuracy=0.5,
        macro_f1=macro_f1,
        y_true=np.array([0]),
        y_pred=np.array([0]),
    )


def _fit_with_validation_scores(
    monkeypatch,
    tmp_path,
    scores,
    *,
    patience: int,
    epochs: int | None = None,
):
    """Drive fit() through a scripted sequence of validation scores."""
    scripted = iter([_validation_output(score) for score in scores])
    monkeypatch.setattr(
        "wm811k.training.run_epoch",
        lambda *args, **kwargs: _validation_output(0.0),
    )
    monkeypatch.setattr(
        "wm811k.training.evaluate",
        lambda *args, **kwargs: next(scripted),
    )
    model = nn.Linear(1, 2, bias=False)
    loader = DataLoader(
        TensorDataset(torch.zeros(1, 1), torch.zeros(1, dtype=torch.long)),
    )
    return fit(
        model,
        loader,
        loader,
        criterion=nn.CrossEntropyLoss(reduction="none"),
        optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
        device=torch.device("cpu"),
        epochs=len(scores) if epochs is None else epochs,
        patience=patience,
        min_delta=MIN_DELTA,
        checkpoint_path=tmp_path / "best.pt",
        checkpoint_metadata={"experiment": "T"},
    )


def test_epoch_loss_is_sample_weighted_when_last_batch_is_smaller():
    inputs = torch.tensor([[0.0], [1.0], [2.0]])
    targets = torch.tensor([0, 1, 1])
    loader = DataLoader(TensorDataset(inputs, targets), batch_size=2, shuffle=False)
    model = nn.Linear(1, 2, bias=False)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[0.0], [1.0]]))
    criterion = nn.CrossEntropyLoss(reduction="none")

    output = run_epoch(
        model,
        loader,
        criterion=criterion,
        device=torch.device("cpu"),
    )
    logits = model(inputs)
    expected = criterion(logits, targets).mean().item()
    assert output.loss == pytest.approx(expected)


def test_checkpoint_keeps_the_best_epoch_when_the_gain_is_below_min_delta(
    monkeypatch, tmp_path
):
    scores = [0.80000, 0.80005]

    history, checkpoint = _fit_with_validation_scores(
        monkeypatch, tmp_path, scores, patience=5
    )

    assert scores[1] - scores[0] < MIN_DELTA
    assert checkpoint["epoch"] == 2
    assert checkpoint["best_validation_macro_f1"] == pytest.approx(max(scores))
    assert len(history) == 2

    saved = torch.load(tmp_path / "best.pt", weights_only=False)
    assert saved["epoch"] == checkpoint["epoch"]
    assert saved["best_validation_macro_f1"] == pytest.approx(max(scores))


def test_early_stopping_still_requires_gains_above_min_delta(monkeypatch, tmp_path):
    scores = [0.80000, 0.80005, 0.80007, 0.90000, 0.95000]

    history, checkpoint = _fit_with_validation_scores(
        monkeypatch, tmp_path, scores, patience=2
    )

    assert len(history) == 3
    assert checkpoint["best_validation_macro_f1"] == pytest.approx(0.80007)


def test_patience_measures_from_the_last_epoch_that_cleared_min_delta(
    monkeypatch, tmp_path
):
    scores = [0.80000, 0.80006, 0.80012, 0.80013]

    history, _ = _fit_with_validation_scores(monkeypatch, tmp_path, scores, patience=2)

    assert scores[2] - scores[1] < MIN_DELTA
    assert scores[2] - scores[0] > MIN_DELTA
    assert len(history) == len(scores)


def test_training_stops_after_patience_epochs_without_improvement(monkeypatch, tmp_path):
    scores = [0.90000, 0.50000, 0.50000, 0.99000]

    history, checkpoint = _fit_with_validation_scores(
        monkeypatch, tmp_path, scores, patience=2
    )

    assert len(history) == 3
    assert checkpoint["epoch"] == 1
    assert checkpoint["best_validation_macro_f1"] == pytest.approx(0.90000)
