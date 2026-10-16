"""Heteroscedastic Gaussian negative log-likelihood (Experiment C).

With predicted mean mu and predicted log-variance s (both in the normalized [0, 1] score space) and target y:

    L_C = 1/(2N) * sum_i [ exp(-s_i) * (y_i - mu_i)^2 + s_i ]

(the constant 0.5*log(2*pi) is omitted). Predicting s = log(sigma^2) instead of sigma keeps the variance
positive without a constraint; `ct_iqa.models.heads.clamp_log_var` bounds s so exp(+/-s) stays finite.
A single deterministic head captures aleatoric (data) uncertainty only, not epistemic uncertainty.
"""

from __future__ import annotations

import torch

from ct_iqa.losses.types import LossResult
from ct_iqa.models.registry import ModelOutput


class GaussianNLLLoss:
    name = "gaussian_nll"

    def __call__(self, output: ModelOutput, target_norm: torch.Tensor, raw_target: torch.Tensor) -> LossResult:
        if output.log_var is None:
            raise ValueError("gaussian_nll requires a model that outputs log_var (resnet50_gaussian)")
        s = output.log_var
        value = 0.5 * torch.mean(torch.exp(-s) * (target_norm - output.mean) ** 2 + s)
        mse = torch.mean((output.mean - target_norm) ** 2)
        return LossResult(total=value, parts={"nll": float(value.detach()), "mse": float(mse.detach()),
                                              "mean_log_var": float(s.detach().mean())})
