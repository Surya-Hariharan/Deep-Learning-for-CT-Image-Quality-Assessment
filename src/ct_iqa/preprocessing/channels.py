"""Grayscale-to-3-channel conversion.

NOT SPECIFIED BY OHASHI: the exact grayscale-to-3-channel implementation.
Channel replication is the PROJECT ADAPTATION used here -- it is also what the
LDCT-IQAC PNG files already do natively (verified: all three RGB channels are
identical in all 1,000 images, see docs/dataset_audit.md).
"""

from __future__ import annotations

import numpy as np


def grayscale_to_rgb(image: np.ndarray) -> np.ndarray:
    """Replicate a single-channel image across 3 channels.

    A no-op when the input already has 3 channels.
    """
    image = np.asarray(image)
    if image.ndim == 2:
        return np.stack([image, image, image], axis=-1)
    if image.ndim == 3 and image.shape[-1] == 3:
        return image
    if image.ndim == 3 and image.shape[-1] == 1:
        return np.repeat(image, 3, axis=-1)
    raise ValueError(f"unexpected image shape for grayscale_to_rgb: {image.shape}")
