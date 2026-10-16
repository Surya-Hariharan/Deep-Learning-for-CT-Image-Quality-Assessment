"""Optimizer construction for CT-IQA training runs.

Extracted from `ct_iqa.training.trainer` with no change in behavior. Only
the Ohashi replication's optimizer (Adam) is supported -- unsupported values
raise rather than silently falling back, since only the paper-specified
configuration is implemented.
"""

from __future__ import annotations

import torch
from torch import nn

from ct_iqa.config import ExperimentConfig


def build_optimizer(model: nn.Module, config: ExperimentConfig) -> torch.optim.Optimizer:
    if config.optimizer.lower() != "adam":
        raise ValueError(
            f"Unsupported optimizer {config.optimizer!r}. "
            f"Ohashi replication uses Adam; use a separate config for anything else."
        )
    return torch.optim.Adam(model.parameters(), lr=config.learning_rate)


def build_study_optimizer(model: nn.Module, cfg) -> torch.optim.Optimizer:
    """Optimizer from a `ct_iqa.config.schema.OptimConfig` (study v2). The function above is the historical one."""
    if cfg.name != "adam":
        raise ValueError(f"Unsupported optimizer {cfg.name!r}")
    return torch.optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
