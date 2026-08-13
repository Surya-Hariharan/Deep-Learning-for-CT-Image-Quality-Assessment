"""Seed every RNG this project touches, from the single project seed.

PROJECT ADAPTATION -- reproducibility infrastructure, not part of Ohashi's
methodology. TensorFlow/PyTorch are seeded only if importable, so this stays
usable before either framework is installed.
"""

from __future__ import annotations

import random

import numpy as np


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)

    try:
        import tensorflow as tf

        tf.random.set_seed(seed)
    except ImportError:
        pass

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
