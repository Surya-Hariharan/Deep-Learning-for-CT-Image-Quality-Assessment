"""Deterministic seeding helper."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch RNGs, and request deterministic cuDNN behavior.

    Best-effort determinism: some CUDA ops remain nondeterministic
    regardless of these flags. Sufficient for reproducible data splits and
    weight initialization on CPU, which is what this baseline relies on.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
