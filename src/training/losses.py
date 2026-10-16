"""Loss construction for CT-IQA training runs.

Extracted from `ct_iqa.training.trainer` with no change in behavior. Only
the Ohashi replication's loss (MSE) is supported.
"""

from __future__ import annotations

from torch import nn

from ct_iqa.config import ExperimentConfig


def build_loss(config: ExperimentConfig) -> nn.Module:
    if config.loss.lower() != "mse":
        raise ValueError(f"Unsupported loss {config.loss!r}. Ohashi replication uses MSE.")
    return nn.MSELoss()
