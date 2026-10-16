"""Ranking-aware objective (Experiment B):  L_B = L_MSE + lambda * L_rank.

For a mini-batch with predicted means f (normalized [0, 1]) and raw targets y ([0, 4]), let P be the set of
ordered pairs (i, j) with  y_i - y_j >= min_score_diff  (i has the higher quality score; see
`ct_iqa.data.sampling.ordered_pair_mask`). The pairwise margin ranking term is

    L_rank = mean_{(i, j) in P}  max(0, margin - (f_i - f_j))

i.e. a pair contributes when the model does not rank i above j by at least `margin` (margin is in the
normalized [0, 1] space). The hinge is exactly zero for satisfied pairs and has a constant gradient
otherwise, so it is numerically stable (no exponentials). If the batch has no valid pair, L_rank = 0 and
no gradient flows through it. With lambda = 0 the total equals the MSE term exactly.

This term supervises ORDER only. It does not claim to improve rank correlation; that is an empirical question.
"""

from __future__ import annotations

import torch

from ct_iqa.data.sampling import ordered_pair_mask
from ct_iqa.losses.types import LossResult
from ct_iqa.models.registry import ModelOutput


def pairwise_margin_ranking(pred: torch.Tensor, raw_target: torch.Tensor, margin: float,
                            min_score_diff: float) -> tuple[torch.Tensor, int]:
    """Return (mean hinge over valid pairs, number of valid pairs)."""
    mask = ordered_pair_mask(raw_target, min_score_diff)
    n_pairs = int(mask.sum())
    if n_pairs == 0:
        return pred.sum() * 0.0, 0                      # graph-connected zero: finite, zero gradient
    diff = pred.unsqueeze(1) - pred.unsqueeze(0)        # diff[i, j] = f_i - f_j
    hinge = torch.clamp(margin - diff, min=0.0)
    return hinge[mask].mean(), n_pairs


class MSERankingLoss:
    name = "mse_ranking"

    def __init__(self, lambda_rank: float, margin: float, min_score_diff: float) -> None:
        if lambda_rank < 0 or margin < 0 or min_score_diff < 0:
            raise ValueError("lambda_rank, margin and min_score_diff must be non-negative")
        self.lambda_rank, self.margin, self.min_score_diff = lambda_rank, margin, min_score_diff

    def __call__(self, output: ModelOutput, target_norm: torch.Tensor, raw_target: torch.Tensor) -> LossResult:
        mse = torch.mean((output.mean - target_norm) ** 2)
        rank, n_pairs = pairwise_margin_ranking(output.mean, raw_target, self.margin, self.min_score_diff)
        total = mse + self.lambda_rank * rank
        n = raw_target.shape[0]
        return LossResult(total=total, parts={
            "mse": float(mse.detach()), "rank": float(rank.detach()),
            "n_valid_pairs": float(n_pairs), "n_possible_pairs": float(n * (n - 1) // 2),
        })
