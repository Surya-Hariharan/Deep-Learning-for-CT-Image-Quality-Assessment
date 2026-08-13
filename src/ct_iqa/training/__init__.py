"""Training procedure specification. NOT EXECUTED at this phase."""

from ct_iqa.training.checkpointing import CheckpointingNotAuthorized
from ct_iqa.training.losses import LOSS_NAME, mse_loss
from ct_iqa.training.trainer import (
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE_GRID,
    OPTIMIZER,
    REPORTED_BEST_LR,
    SELECTION_METRIC,
    PlannedRun,
    Trainer,
    TrainingNotAuthorized,
    plan_lr_search,
    select_best,
)

__all__ = [
    "LEARNING_RATE_GRID", "BATCH_SIZE", "EPOCHS", "OPTIMIZER", "SELECTION_METRIC",
    "REPORTED_BEST_LR", "PlannedRun", "plan_lr_search", "select_best",
    "Trainer", "TrainingNotAuthorized", "LOSS_NAME", "mse_loss", "CheckpointingNotAuthorized",
]
