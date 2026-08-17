"""Training loop for the Ohashi-ResNet50 / LDCT-IQAC baseline.

Implements only the Ohashi replication configuration (Part 9 of the task):
Adam optimizer, MSE loss, batch size 64, 30 epochs, learning rate 1e-3,
224x224 input. No scheduler, warmup, weight decay, early stopping,
augmentation, or mixed precision is added -- all are deliberately absent
so the first experiment stays close to the paper's stated configuration.

Validation split: created ONLY from the training set (Part 10). The
official LDCT-IQAC test set (`data/testing/`) is never touched by
`build_dataloaders` and must only be used for final held-out evaluation,
never for model selection.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split

from ct_iqa.config import ExperimentConfig
from ct_iqa.data.ldct_iqac import LDCTIQACDataset, denormalize_score, normalize_score

logger = logging.getLogger(__name__)


def build_dataloaders(config: ExperimentConfig) -> tuple[DataLoader, DataLoader]:
    """Build (train_loader, val_loader) from the training split only.

    The validation subset is carved out of `LDCTIQACDataset(train_image_dir,
    train_json_path, ...)` using `torch.utils.data.random_split` with a
    generator seeded from `config.seed`, so the split is exactly
    reproducible given the same seed and `val_fraction`. The test set is
    not referenced here at all.
    """
    full_train_dataset = LDCTIQACDataset(
        image_dir=config.train_image_dir,
        json_path=config.train_json_path,
        image_size=config.image_size,
    )

    n_val = int(round(len(full_train_dataset) * config.val_fraction))
    n_train = len(full_train_dataset) - n_val
    generator = torch.Generator().manual_seed(config.seed)
    train_subset, val_subset = random_split(
        full_train_dataset, [n_train, n_val], generator=generator
    )

    train_loader = DataLoader(
        train_subset, batch_size=config.batch_size, shuffle=True, drop_last=False
    )
    val_loader = DataLoader(
        val_subset, batch_size=config.batch_size, shuffle=False, drop_last=False
    )
    return train_loader, val_loader


def build_test_dataloader(config: ExperimentConfig) -> DataLoader:
    """Build the held-out test DataLoader from the official LDCT-IQAC test split.

    Use only for final evaluation, never during training or model selection.
    """
    test_dataset = LDCTIQACDataset(
        image_dir=config.test_image_dir,
        json_path=config.test_json_path,
        image_size=config.image_size,
    )
    return DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)


def build_optimizer(model: nn.Module, config: ExperimentConfig) -> torch.optim.Optimizer:
    if config.optimizer.lower() != "adam":
        raise ValueError(
            f"Unsupported optimizer {config.optimizer!r}. "
            f"Ohashi replication uses Adam; use a separate config for anything else."
        )
    return torch.optim.Adam(model.parameters(), lr=config.learning_rate)


def build_loss(config: ExperimentConfig) -> nn.Module:
    if config.loss.lower() != "mse":
        raise ValueError(
            f"Unsupported loss {config.loss!r}. Ohashi replication uses MSE."
        )
    return nn.MSELoss()


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
    config: ExperimentConfig,
) -> TrainingHistory:
    """Full Ohashi-configuration training loop with best-checkpoint saving on val loss."""
    device = config.device
    model.to(device)

    optimizer = build_optimizer(model, config)
    criterion = build_loss(config)
    history = TrainingHistory()

    checkpoint_dir = Path(config.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_checkpoint_path = checkpoint_dir / "best.pt"

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
        logger.info(f"epoch {epoch + 1}/{config.epochs}  train_loss={train_loss:.6f}  val_loss={val_loss:.6f}")

        if val_loss < history.best_val_loss:
            history.best_val_loss = val_loss
            history.best_epoch = epoch
            torch.save(model.state_dict(), best_checkpoint_path)

    return history
