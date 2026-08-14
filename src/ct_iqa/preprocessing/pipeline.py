"""The single authoritative baseline preprocessing pipeline: a raw
reference image in, the exact tensor the RadImageNet ResNet50 backbone
expects out.

Full specification recorded in ``docs/training_environment.md``,
"Preprocessing Specification" -- this module is that specification's
executable form. Composes three independently-provenanced steps, each
documented in its own module:

    1. ``central_crop``       (cropping.py)      -- OHASHI-SPECIFIED, no resize
    2. ``grayscale_to_rgb``   (channels.py)       -- PROJECT ADAPTATION
    3. ``resnet50_preprocess_input`` (normalization.py) -- PROJECT-ADAPTATION (A-23)

Deterministic and side-effect-free: no randomness, no augmentation, per
Ohashi's explicit "no augmentation beyond the central crop" (Section 10).
"""

from __future__ import annotations

import numpy as np

from ct_iqa.preprocessing.channels import grayscale_to_rgb
from ct_iqa.preprocessing.cropping import central_crop
from ct_iqa.preprocessing.normalization import resnet50_preprocess_input

OUTPUT_SIZE = 224
OUTPUT_SHAPE = (OUTPUT_SIZE, OUTPUT_SIZE, 3)
OUTPUT_DTYPE = np.float32


def preprocess_for_model(image: np.ndarray, *, crop_size: int = OUTPUT_SIZE) -> np.ndarray:
    """Central crop (no resize) -> channel replication -> normalization.

    Args:
        image: 2-D (grayscale) or 3-D (``(H, W, 3)``, RGB) array, raw
            ``[0, 255]``-scale pixel values (uint8 or float). Must be at
            least ``crop_size`` in both spatial dimensions -- LDCT-IQAC's
            512x512 images comfortably satisfy this for ``crop_size=224``.

    Returns:
        ``float32`` array shaped ``(crop_size, crop_size, 3)``, BGR
        channel order, ImageNet-mean-subtracted (decision A-23).
    """
    cropped = central_crop(image, size=crop_size)
    rgb = grayscale_to_rgb(cropped)
    normalized = resnet50_preprocess_input(rgb)
    expected_shape = (crop_size, crop_size, 3)
    if normalized.shape != expected_shape:
        raise ValueError(f"pipeline produced shape {normalized.shape}, expected {expected_shape}")
    return normalized.astype(OUTPUT_DTYPE)
