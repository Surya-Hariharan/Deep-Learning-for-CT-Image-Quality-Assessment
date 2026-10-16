"""Mean-squared-error objective (Experiments A, and the pointwise term of B)."""

from __future__ import annotations

import torch

from ct_iqa.losses.types import LossResult
from ct_iqa.models.registry import ModelOutput


class MSELoss:
    """MSE between the predicted mean and the target, both in the normalized [0, 1] score space."""

    name = "mse"

    def __call__(self, output: ModelOutput, target_norm: torch.Tensor, raw_target: torch.Tensor) -> LossResult:
        value = torch.mean((output.mean - target_norm) ** 2)
        return LossResult(total=value, parts={"mse": float(value.detach())})
