"""Gaussian blur degradation -- OHASHI-SPECIFIED sigma grid."""

from __future__ import annotations

import numpy as np

# OHASHI-SPECIFIED grid (see docs/ohashi_methodology.md).
BLUR_SIGMAS: tuple[float, ...] = (0.1, 0.15, 0.2, 0.3, 0.45, 0.6, 0.75, 0.9, 1.4, 2.1, 3.0, 5.0)


def apply_gaussian_blur(
    image: np.ndarray,
    sigma: float,
    truncate: float = 4.0,
    mode: str = "nearest",
) -> np.ndarray:
    """Blur with an isotropic Gaussian kernel (sigma in pixels).

    ``truncate``/``mode`` are PROJECT ADAPTATION defaults (scipy convention);
    Ohashi does not specify kernel extent or boundary handling.
    """
    from scipy.ndimage import gaussian_filter

    return gaussian_filter(image, sigma=sigma, truncate=truncate, mode=mode)
