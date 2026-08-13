"""Central crop -- OHASHI-SPECIFIED input preprocessing (Section 9/10)."""

from __future__ import annotations

import numpy as np


def central_crop(image: np.ndarray, size: int = 224) -> np.ndarray:
    """Crop a ``size x size`` window from the image centre.

    OHASHI-SPECIFIED that a central crop to 224x224 is used. NOT SPECIFIED BY
    OHASHI: whether the source image is resized before cropping, and what
    happens when the source is smaller than ``size`` in either dimension.
    This function does neither silently -- a source smaller than ``size``
    raises rather than padding or upsampling, because guessing which one
    Ohashi used would bake an unverified choice into every training image.

    LDCT-IQAC images are 512x512 (verified, see docs/dataset_audit.md), so a
    plain central crop to 224x224 is well-defined without any resize.
    """
    image = np.asarray(image)
    h, w = image.shape[:2]
    if h < size or w < size:
        raise ValueError(
            f"image is {h}x{w}, smaller than the {size}x{size} central crop; "
            "pre-crop resizing is NOT SPECIFIED BY OHASHI and must be decided "
            "explicitly, not assumed by this function"
        )
    top = (h - size) // 2
    left = (w - size) // 2
    return image[top : top + size, left : left + size, ...]
