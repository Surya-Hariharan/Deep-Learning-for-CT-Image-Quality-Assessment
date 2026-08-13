"""Ohashi-method synthetic degradation engine (Gaussian noise + Gaussian blur).

OHASHI-SPECIFIED. Dataset-agnostic: the same grids and engine apply whichever
reference-image source is configured (see docs/dataset_adaptation.md for the
LDCT-IQAC vs DeepLesion/CQ500 adaptation).
"""

from ct_iqa.degradation.combined import Condition, DegradationType, Order, apply_condition, condition_rng
from ct_iqa.degradation.gaussian_blur import BLUR_SIGMAS, apply_gaussian_blur
from ct_iqa.degradation.gaussian_noise import NOISE_SIGMAS, apply_gaussian_noise
from ct_iqa.degradation.generator import build_grid, iter_grid

__all__ = [
    "Condition", "DegradationType", "Order", "apply_condition", "condition_rng",
    "BLUR_SIGMAS", "apply_gaussian_blur",
    "NOISE_SIGMAS", "apply_gaussian_noise",
    "build_grid", "iter_grid",
]
