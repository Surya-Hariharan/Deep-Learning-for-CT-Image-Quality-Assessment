"""Gaussian noise degradation -- OHASHI-SPECIFIED sigma grid."""

from __future__ import annotations

import numpy as np

# OHASHI-SPECIFIED grid (see docs/ohashi_methodology.md).
NOISE_SIGMAS: tuple[float, ...] = (1, 1.5, 2, 3, 4.5, 6, 7.5, 9, 14, 21, 30, 50)


def apply_gaussian_noise(
    image: np.ndarray,
    sigma: float,
    sigma_scale: float,
    rng: np.random.Generator,
    clip_range: tuple[float, float] | None = None,
) -> np.ndarray:
    """Add zero-mean white Gaussian noise.

    Args:
        image: float array.
        sigma: sigma from the OHASHI-SPECIFIED grid.
        sigma_scale: multiplier converting grid sigma into the image's own
            intensity units. REQUIRED and never defaulted: NOT SPECIFIED BY
            OHASHI whether the grid is in HU or in 8-bit grey levels; pass 1.0
            to treat the grid as already being in image units (the case for
            the LDCT-IQAC adaptation, whose images are 8-bit PNG).
        rng: seeded generator -- degradation must be reproducible.
        clip_range: optional (lo, hi) clamp. ``None`` leaves values unclipped,
            since clipping behaviour is NOT SPECIFIED BY OHASHI.
    """
    noisy = image + rng.normal(loc=0.0, scale=sigma * sigma_scale, size=image.shape)
    if clip_range is not None:
        noisy = np.clip(noisy, *clip_range)
    return noisy
