"""Training loop for the Ohashi-ResNet50 / LDCT-IQAC baseline.

Implements only the Ohashi replication configuration (Part 9 of the task):
Adam optimizer, MSE loss, batch size 64, 30 epochs, learning rate 1e-3,
224x224 input. No scheduler, warmup, weight decay, or mixed precision is
added -- these are deliberately absent so the first experiment stays close
to the paper's stated configuration. Best-checkpoint saving IS a form of
validation-based model selection (see `config.selection_metric` below) and
is not claimed to be paper-specified -- it's an explicit, documented
IMPLEMENTATION DECISION (see docs/replication/deviations.md), not something
this project pretends the paper calls for.

DataLoader construction (`ct_iqa.data.loader`), the optimizer/loss builders
(`ct_iqa.training.optimizers`/`ct_iqa.training.losses`), and checkpoint
saving (`ct_iqa.training.checkpointing`) have been factored out into their
own modules -- this file keeps only the training-loop orchestration itself
(`train_one_step`, `evaluate_loader`, `train`, `TrainingHistory`).

Validation split: created ONLY from the training set (Part 10), via
`ct_iqa.data.loader.build_dataloaders`. The official LDCT-IQAC test set
(`data/raw/ldct_iqac/test/`) is never touched by that function and must only
be used for final held-out evaluation, never for model selection.

Checkpoint selection metric (updated 2026-09-14, following the repository
audit's "Model Selection Mismatch in Training Loop" finding): the previous
version of this module selected the "best" checkpoint purely by minimum
validation loss in normalized [0, 1] target space. Minimum MSE does not
guarantee maximum PLCC/SROCC -- the metrics this project actually reports
and compares experiments on (see README.md, docs/replication/final_results.md)
-- so `train()` now computes PLCC/SROCC each epoch (on raw [0, 4] scores,
via `ct_iqa.evaluation.correlation`) and selects on `config.selection_metric`
(default `"plcc"`); `"val_loss"` remains available for backward
compatibility/ablation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import torch
from torch import nn
from torch.utils.data import DataLoader

from ct_iqa.data.ldct_iqac import denormalize_score, normalize_score
from ct_iqa.evaluation.correlation import plcc, srocc
from ct_iqa.training.checkpointing import save_checkpoint
from ct_iqa.training.losses import build_loss
from ct_iqa.training.optimizers import build_optimizer

# Metrics `train()` can select the "best" checkpoint on, and whether higher
# or lower values are better for each. IMPLEMENTATION DECISION: the paper
# does not specify a selection rule -- see docs/replication/deviations.md.
_HIGHER_IS_BETTER = {"plcc": True, "srocc": True, "val_loss": False}

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
    val_plcc: list[float] = field(default_factory=list)
    val_srocc: list[float] = field(default_factory=list)
    best_epoch: int = -1
    best_val_loss: float = float("inf")
    # Best value seen so far for whichever metric `config.selection_metric`
    # names -- initialized once `train()` knows the selection direction.
    best_selection_value: float = field(default=float("nan"))
    selection_metric: str = "val_loss"


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config,
) -> TrainingHistory:
    """Full Ohashi-configuration training loop with best-checkpoint saving.

    `config` is a `ct_iqa.config.ExperimentConfig`. Best-checkpoint saving
    uses `config.checkpoint_dir` (`<experiment_dir>/checkpoint/best.pt`),
    via `ct_iqa.training.checkpointing.save_checkpoint` -- each save
    includes the optimizer state, epoch, val_loss, resolved config, and
    seed alongside the model weights, so the checkpoint alone documents
    what produced it.

    The checkpoint is selected on `config.selection_metric` (`"plcc"` by
    default, or `"srocc"`/`"val_loss"`) computed each epoch on the
    validation split's raw [0, 4] predictions -- not on validation loss
    alone, which optimizes a different (and only loosely related) quantity
    from the PLCC/SROCC this project actually reports and compares
    experiments on. See the module docstring for background.
    """
    device = config.device
    model.to(device)

    selection_metric = getattr(config, "selection_metric", "val_loss")
    if selection_metric not in _HIGHER_IS_BETTER:
        raise ValueError(
            f"Unknown selection_metric {selection_metric!r}; expected one of "
            f"{sorted(_HIGHER_IS_BETTER)}."
        )
    higher_is_better = _HIGHER_IS_BETTER[selection_metric]

    optimizer = build_optimizer(model, config)
    criterion = build_loss(config)
    history = TrainingHistory(selection_metric=selection_metric)
    history.best_selection_value = float("-inf") if higher_is_better else float("inf")

    for epoch in range(config.epochs):
        epoch_loss, n_samples = 0.0, 0
        for batch in train_loader:
            batch_size = batch[0].shape[0]
            loss = train_one_step(model, batch, optimizer, criterion, device=device)
            epoch_loss += loss * batch_size
            n_samples += batch_size
        train_loss = epoch_loss / n_samples

        val_loss, val_preds, val_targets = evaluate_loader(model, val_loader, criterion, device=device)
        val_plcc = plcc(val_targets, val_preds)
        val_srocc = srocc(val_targets, val_preds)

        history.train_loss.append(train_loss)
        history.val_loss.append(val_loss)
        history.val_plcc.append(val_plcc)
        history.val_srocc.append(val_srocc)
        if val_loss < history.best_val_loss:
            history.best_val_loss = val_loss
        logger.info(
            f"epoch {epoch + 1}/{config.epochs}  train_loss={train_loss:.6f}  "
            f"val_loss={val_loss:.6f}  val_plcc={val_plcc:.4f}  val_srocc={val_srocc:.4f}"
        )

        selection_value = {"plcc": val_plcc, "srocc": val_srocc, "val_loss": val_loss}[selection_metric]
        improved = (
            selection_value > history.best_selection_value
            if higher_is_better
            else selection_value < history.best_selection_value
        )
        if improved:
            history.best_selection_value = selection_value
            history.best_epoch = epoch
            save_checkpoint(
                model,
                config.checkpoint_dir,
                optimizer=optimizer,
                epoch=epoch,
                val_loss=val_loss,
                config=config,
                seed=config.seed,
                checkpoint_type=f"best_{selection_metric}",
            )

    return history
