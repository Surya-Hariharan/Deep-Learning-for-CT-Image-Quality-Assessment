"""Training loop for the Ohashi-ResNet50 / LDCT-IQAC baseline.

Implements only the Ohashi replication configuration (Part 9 of the task):
Adam optimizer, MSE loss, batch size 64, 30 epochs, learning rate 1e-3,
224x224 input. No scheduler, warmup, weight decay, early stopping,
augmentation, or mixed precision is added -- all are deliberately absent
so the first experiment stays close to the paper's stated configuration.

DataLoader construction (`ct_iqa.data.loader`), the optimizer/loss builders
(`ct_iqa.training.optimizers`/`ct_iqa.training.losses`), and checkpoint
saving (`ct_iqa.training.checkpointing`) have been factored out into their
own modules -- this file keeps only the training-loop orchestration itself
(`train_one_step`, `evaluate_loader`, `train`, `TrainingHistory`).

Validation split: created ONLY from the training set (Part 10), via
`ct_iqa.data.loader.build_dataloaders`. The official LDCT-IQAC test set
(`data/raw/ldct_iqac/test/`) is never touched by that function and must only
be used for final held-out evaluation, never for model selection.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import torch
from torch import nn
from torch.utils.data import DataLoader

from ct_iqa.data.ldct_iqac import denormalize_score, normalize_score
from ct_iqa.training.checkpointing import save_checkpoint
from ct_iqa.training.losses import build_loss
from ct_iqa.training.optimizers import build_optimizer

logger = logging.getLogger(__name__)


def train_one_step(
    model: nn.Module,
    batch: tuple[torch.Tensor, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: str = "cpu",
) -> float:
    """Run a single optimization step on one batch. Returns the scalar loss.

    Loss is computed in NORMALIZED [0, 1] target space (matching the
    Sigmoid output head), via `normalize_score` -- never in raw [0, 4]
    score space, per Part 8 of the task.
    """
    model.train()
    images, raw_scores = batch
    images = images.to(device)
    targets = normalize_score(raw_scores.to(device))

    optimizer.zero_grad()
    predictions = model(images)
    loss = criterion(predictions, targets)
    loss.backward()
    optimizer.step()
    return float(loss.item())


@torch.no_grad()
def evaluate_loader(
    model: nn.Module, loader: DataLoader, criterion: nn.Module, device: str = "cpu"
) -> tuple[float, torch.Tensor, torch.Tensor]:
    """Compute average normalized-space loss over a loader.

    Returns (avg_loss, raw_predictions, raw_targets) where both tensors
    are denormalized back to the [SCORE_MIN, SCORE_MAX] scale for
    downstream reporting/metrics.
    """
    model.eval()
    total_loss = 0.0
    n_samples = 0
    all_preds, all_targets = [], []

    for images, raw_scores in loader:
        images = images.to(device)
        raw_scores = raw_scores.to(device)
        targets = normalize_score(raw_scores)

        predictions = model(images)
        loss = criterion(predictions, targets)

        batch_size = images.shape[0]
        total_loss += float(loss.item()) * batch_size
        n_samples += batch_size

        all_preds.append(denormalize_score(predictions.cpu()))
        all_targets.append(raw_scores.cpu())

    avg_loss = total_loss / n_samples
    return avg_loss, torch.cat(all_preds), torch.cat(all_targets)


@dataclass
class TrainingHistory:
    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    best_epoch: int = -1
    best_val_loss: float = float("inf")


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config,
) -> TrainingHistory:
    """Full Ohashi-configuration training loop with best-checkpoint saving on val loss.

    `config` is a `ct_iqa.config.ExperimentConfig`. Best-checkpoint saving
    uses `config.checkpoint_dir` (`<experiment_dir>/checkpoint/best.pt`),
    via `ct_iqa.training.checkpointing.save_checkpoint` -- each save
    includes the optimizer state, epoch, val_loss, resolved config, and
    seed alongside the model weights, so the checkpoint alone documents
    what produced it.
    """
    device = config.device
    model.to(device)

    optimizer = build_optimizer(model, config)
    criterion = build_loss(config)
    history = TrainingHistory()

    for epoch in range(config.epochs):
        epoch_loss, n_samples = 0.0, 0
        for batch in train_loader:
            batch_size = batch[0].shape[0]
            loss = train_one_step(model, batch, optimizer, criterion, device=device)
            epoch_loss += loss * batch_size
            n_samples += batch_size
        train_loss = epoch_loss / n_samples

        val_loss, _, _ = evaluate_loader(model, val_loader, criterion, device=device)

        history.train_loss.append(train_loss)
        history.val_loss.append(val_loss)
        logger.info(
            f"epoch {epoch + 1}/{config.epochs}  train_loss={train_loss:.6f}  val_loss={val_loss:.6f}"
        )

        if val_loss < history.best_val_loss:
            history.best_val_loss = val_loss
            history.best_epoch = epoch
            save_checkpoint(
                model,
                config.checkpoint_dir,
                optimizer=optimizer,
                epoch=epoch,
                val_loss=val_loss,
                config=config,
                seed=config.seed,
            )

    return history
