"""Pair construction for the ranking-aware objective (Experiment B).

Pairs are formed WITHIN a mini-batch (no sampling across the dataset), so the pair tensor is at most
B x B (4096 entries for B = 64) and no all-pairs tensor over the full dataset is ever materialized.
There is no random pair subsampling: every valid in-batch pair is used. The only stochasticity is the
batch composition, which is controlled by the seeded DataLoader shuffle generator (`train.seed`).
"""

from __future__ import annotations

import torch


def ordered_pair_mask(raw_scores: torch.Tensor, min_score_diff: float) -> torch.Tensor:
    """Boolean (B, B) mask M with M[i, j] = True iff  raw_scores[i] - raw_scores[j] >= min_score_diff.

    Direction: i is the item with the HIGHER quality score, j the lower. Each unordered pair appears at most
    once (only the higher-first orientation can satisfy the inequality), and the diagonal and all pairs
    whose scores are equal or closer than `min_score_diff` are excluded, so ties and near-ties create no
    ranking constraint. `min_score_diff` is in raw [0, 4] score units; a threshold of 0 is raised to a tiny
    positive value so exact ties are never paired.
    """
    if raw_scores.dim() != 1:
        raise ValueError(f"raw_scores must be 1-D, got shape {tuple(raw_scores.shape)}")
    diff = raw_scores.unsqueeze(1) - raw_scores.unsqueeze(0)
    threshold = max(float(min_score_diff), 1e-12)   # strictly positive: exact ties are never pairs
    return diff >= threshold
