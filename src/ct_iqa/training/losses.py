"""Training loss — OHASHI-SPECIFIED as MSE, for both the synthetic-VIF stage
and reported as a Stage-1 evaluation metric."""

from __future__ import annotations

LOSS_NAME = "mse"  # OHASHI-SPECIFIED


def mse_loss(predicted, target):
    """Thin re-export so training code and evaluation code share one MSE
    implementation. See ``ct_iqa.evaluation.metrics.mean_squared_error``."""
    from ct_iqa.evaluation.metrics import mean_squared_error

    return mean_squared_error(predicted, target)
