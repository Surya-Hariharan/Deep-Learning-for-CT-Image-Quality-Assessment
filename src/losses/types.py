"""Shared return type for all losses."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class LossResult:
    total: torch.Tensor                                   # scalar tensor with grad, used for backward()
    parts: dict[str, float] = field(default_factory=dict)  # detached diagnostic scalars (e.g. mse, rank, n_pairs)
