"""Tests for `ct_iqa.training.trainer.train`'s checkpoint-selection logic.

Regression coverage for the repository audit's "Model Selection Mismatch in
Training Loop" finding: `train()` previously selected the "best" checkpoint
purely by minimum validation loss in normalized [0, 1] space, with no
PLCC/SROCC tracked anywhere in the loop. No existing test exercised
`train()`'s multi-epoch loop at all before this file.

Uses a tiny linear stand-in model (not `OhashiResNet50`) so the loop runs in
milliseconds -- `train()` doesn't care about model architecture, only that
it's an `nn.Module` mapping a batch of inputs to one prediction per sample.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ct_iqa.config import ExperimentConfig
from ct_iqa.training.trainer import train


class _TinyRegressor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.fc(x)).squeeze(-1)


def _make_loader(images: torch.Tensor, scores: torch.Tensor, batch_size: int = 4) -> DataLoader:
    return DataLoader(TensorDataset(images, scores), batch_size=batch_size)


def test_train_tracks_plcc_and_srocc_every_epoch(tmp_path):
    torch.manual_seed(0)
    n = 16
    images = torch.randn(n, 4)
    scores = torch.linspace(0.0, 4.0, n)  # real variance -> well-defined plcc/srocc

    config = ExperimentConfig(
        epochs=3,
        batch_size=4,
        experiment_dir=str(tmp_path / "exp"),
        selection_metric="plcc",
    )
    history = train(_TinyRegressor(), _make_loader(images, scores), _make_loader(images, scores), config)

    assert len(history.val_plcc) == config.epochs
    assert len(history.val_srocc) == config.epochs
    assert history.selection_metric == "plcc"
    assert history.best_epoch >= 0


def test_train_selects_checkpoint_on_plcc_not_only_val_loss(tmp_path):
    torch.manual_seed(0)
    n = 16
    images = torch.randn(n, 4)
    scores = torch.linspace(0.0, 4.0, n)

    config = ExperimentConfig(
        epochs=3,
        batch_size=4,
        experiment_dir=str(tmp_path / "exp"),
        selection_metric="plcc",
    )
    history = train(_TinyRegressor(), _make_loader(images, scores), _make_loader(images, scores), config)

    checkpoint_path = Path(config.checkpoint_dir) / "best.pt"
    assert checkpoint_path.exists()
    saved = torch.load(checkpoint_path, map_location="cpu")

    # The checkpoint on disk must be the one `history` says is best, and must
    # be labeled with the metric actually used to select it -- not silently
    # labeled/selected by val_loss the way the pre-fix trainer always did.
    assert saved["checkpoint_type"] == "best_plcc"
    assert saved["epoch"] == history.best_epoch
    assert saved["val_loss"] == history.val_loss[history.best_epoch]


def test_train_rejects_unknown_selection_metric(tmp_path):
    torch.manual_seed(0)
    images = torch.randn(8, 4)
    scores = torch.linspace(0.0, 4.0, 8)

    config = ExperimentConfig(
        epochs=1,
        batch_size=4,
        experiment_dir=str(tmp_path / "exp"),
        selection_metric="not_a_real_metric",
    )

    with pytest.raises(ValueError, match="not_a_real_metric"):
        train(_TinyRegressor(), _make_loader(images, scores), _make_loader(images, scores), config)


def test_train_can_still_select_on_val_loss_for_backward_compatibility(tmp_path):
    torch.manual_seed(0)
    images = torch.randn(16, 4)
    scores = torch.linspace(0.0, 4.0, 16)

    config = ExperimentConfig(
        epochs=3,
        batch_size=4,
        experiment_dir=str(tmp_path / "exp"),
        selection_metric="val_loss",
    )
    history = train(_TinyRegressor(), _make_loader(images, scores), _make_loader(images, scores), config)

    saved = torch.load(Path(config.checkpoint_dir) / "best.pt", map_location="cpu")
    assert saved["checkpoint_type"] == "best_val_loss"
    assert history.val_loss[history.best_epoch] == history.best_val_loss
