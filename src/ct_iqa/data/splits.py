"""Train/validation split logic for the LDCT-IQAC training set.

Extracted from `ct_iqa.training.trainer.build_dataloaders` with no change in
behavior. The official LDCT-IQAC test set is never involved in this split --
see `ct_iqa.data.loader.build_test_dataloader` for the separate, independent
test-set loader.
"""

from __future__ import annotations

import torch
from torch.utils.data import Dataset, Subset, random_split


def train_val_split(
    dataset: Dataset, val_fraction: float, seed: int
) -> tuple[Subset, Subset]:
    """Split `dataset` into (train_subset, val_subset) via a seeded random_split.

    Reproducible given the same `seed` and `val_fraction`.
    """
    n_val = int(round(len(dataset) * val_fraction))
    n_train = len(dataset) - n_val
    generator = torch.Generator().manual_seed(seed)
    train_subset, val_subset = random_split(dataset, [n_train, n_val], generator=generator)
    return train_subset, val_subset
