"""Tests for the training components extracted from `ct_iqa.training.trainer`
into `ct_iqa.training.optimizers`, `ct_iqa.training.losses`, and
`ct_iqa.training.checkpointing` (see docs/repository_architecture_audit.md,
section 3).
"""

from __future__ import annotations

import torch
from torch import nn

from ct_iqa.config import ExperimentConfig
from ct_iqa.models.ohashi_resnet50 import OhashiResNet50
from ct_iqa.training.checkpointing import best_checkpoint_path, load_checkpoint, save_checkpoint
from ct_iqa.training.losses import build_loss
from ct_iqa.training.optimizers import build_optimizer


def test_build_optimizer_returns_adam():
    model = nn.Linear(4, 1)
    config = ExperimentConfig(optimizer="adam", learning_rate=0.01)
    optimizer = build_optimizer(model, config)
    assert isinstance(optimizer, torch.optim.Adam)
    assert optimizer.param_groups[0]["lr"] == 0.01


def test_build_optimizer_rejects_unsupported():
    model = nn.Linear(4, 1)
    config = ExperimentConfig(optimizer="sgd")
    try:
        build_optimizer(model, config)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_build_loss_returns_mse():
    config = ExperimentConfig(loss="mse")
    criterion = build_loss(config)
    assert isinstance(criterion, nn.MSELoss)


def test_build_loss_rejects_unsupported():
    config = ExperimentConfig(loss="l1")
    try:
        build_loss(config)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_save_and_load_checkpoint_roundtrip(tmp_path):
    model = OhashiResNet50(dropout_p=0.5, in_channels=1)
    checkpoint_dir = tmp_path / "checkpoint"

    saved_path = save_checkpoint(model, checkpoint_dir)
    assert saved_path == best_checkpoint_path(checkpoint_dir)
    assert saved_path.exists()

    other_model = OhashiResNet50(dropout_p=0.5, in_channels=1)
    load_checkpoint(other_model, checkpoint_dir)

    for p1, p2 in zip(model.parameters(), other_model.parameters()):
        assert torch.equal(p1, p2)
