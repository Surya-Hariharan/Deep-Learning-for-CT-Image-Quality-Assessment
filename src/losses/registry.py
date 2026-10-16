"""Loss registry: configuration -> loss object."""

from __future__ import annotations

from ct_iqa.config.schema import ConfigError, LossConfig
from ct_iqa.losses.gaussian_nll import GaussianNLLLoss
from ct_iqa.losses.mse import MSELoss
from ct_iqa.losses.ranking import MSERankingLoss


def build_loss(cfg: LossConfig):
    if cfg.name == "mse":
        return MSELoss()
    if cfg.name == "mse_ranking":
        r = cfg.ranking
        return MSERankingLoss(r.lambda_rank, r.margin, r.min_score_diff)
    if cfg.name == "gaussian_nll":
        return GaussianNLLLoss()
    raise ConfigError(f"Unknown loss {cfg.name!r}")
