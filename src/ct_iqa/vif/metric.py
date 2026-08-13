"""Pixel-domain VIF (VIFp) metric implementation.

The original Ohashi paper used MATLAB R2024a for VIF (Section 12 of the
project brief). This is a from-scratch Python implementation, following
Sheikh & Bovik, "Image Information and Visual Quality", IEEE TIP 15(2), 2006,
via the authors' released ``vifp_mscale`` reference: 4 scales, Gaussian
windows of side 2^(4-scale)+1 with sigma = side/5, and sigma_nsq = 2.0.

NOT yet numerically cross-validated against a MATLAB reference run or a
trusted Python package -- see docs/deviations_from_ohashi.md, DEV-03.
"""

from __future__ import annotations

import numpy as np

#: Noise variance of the HVS model. This is a constant of the reference
#: implementation, not a free parameter we chose.
SIGMA_NSQ = 2.0


def vif_p(reference: np.ndarray, distorted: np.ndarray, sigma_nsq: float = SIGMA_NSQ) -> float:
    """Pixel-domain VIF (VIFp) between a reference and a distorted image.

    Both inputs must be 2-D, the same shape, and on the same intensity scale.
    Returns a scalar; identical inputs give 1.0.
    """
    from scipy.ndimage import gaussian_filter

    ref = np.asarray(reference, dtype=np.float64)
    dist = np.asarray(distorted, dtype=np.float64)
    if ref.shape != dist.shape:
        raise ValueError(f"shape mismatch: {ref.shape} vs {dist.shape}")
    if ref.ndim != 2:
        raise ValueError(f"expected a 2-D image, got shape {ref.shape}")

    num = 0.0
    den = 0.0

    for scale in range(1, 5):
        side = 2 ** (4 - scale + 1) + 1
        sd = side / 5.0

        if scale > 1:
            # Downsample by 2 after low-pass filtering, as in the reference code.
            ref = gaussian_filter(ref, sd)[::2, ::2]
            dist = gaussian_filter(dist, sd)[::2, ::2]

        mu1 = gaussian_filter(ref, sd)
        mu2 = gaussian_filter(dist, sd)
        mu1_sq, mu2_sq, mu1_mu2 = mu1 * mu1, mu2 * mu2, mu1 * mu2

        sigma1_sq = gaussian_filter(ref * ref, sd) - mu1_sq
        sigma2_sq = gaussian_filter(dist * dist, sd) - mu2_sq
        sigma12 = gaussian_filter(ref * dist, sd) - mu1_mu2

        # Negative variances are numerical artefacts of the subtraction above.
        sigma1_sq = np.maximum(sigma1_sq, 0.0)
        sigma2_sq = np.maximum(sigma2_sq, 0.0)

        g = np.divide(sigma12, sigma1_sq, out=np.zeros_like(sigma12), where=sigma1_sq > 1e-10)
        sv_sq = sigma2_sq - g * sigma12

        # Degenerate-region handling, exactly as in the reference implementation.
        g = np.where(sigma1_sq < 1e-10, 0.0, g)
        sv_sq = np.where(sigma1_sq < 1e-10, sigma2_sq, sv_sq)
        sigma1_sq = np.where(sigma1_sq < 1e-10, 0.0, sigma1_sq)

        g = np.where(sigma2_sq < 1e-10, 0.0, g)
        sv_sq = np.where(sigma2_sq < 1e-10, 0.0, sv_sq)

        sv_sq = np.where(g < 0.0, sigma2_sq, sv_sq)
        g = np.maximum(g, 0.0)
        sv_sq = np.maximum(sv_sq, 1e-10)

        num += float(np.sum(np.log10(1.0 + (g**2) * sigma1_sq / (sv_sq + sigma_nsq))))
        den += float(np.sum(np.log10(1.0 + sigma1_sq / sigma_nsq)))

    if den == 0.0:
        # A completely flat reference carries no information to preserve.
        return float("nan")
    return num / den
