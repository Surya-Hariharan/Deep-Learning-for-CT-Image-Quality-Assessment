"""Deterministic seeding.

The author's `utils/config.py` seeded random/numpy/torch as an import-time
side effect using a fixed seed (35202). That seed is preserved as
`NGPNetConfig.seed`, but seeding is now an explicit call.
"""

import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    ''' Seed `random`, `numpy` and `torch` (including all CUDA devices). '''
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
