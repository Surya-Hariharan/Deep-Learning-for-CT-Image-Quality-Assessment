"""Image loading, saving and basic shape/dtype validation.

PROJECT ADAPTATION: thin wrappers around Pillow, kept separate from
``cropping.py``/``channels.py`` so image IO concerns don't mix with the
Ohashi-specified transforms.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def load_image(path: Path) -> np.ndarray:
    """Load an image file as a numpy array, preserving its native mode."""
    from PIL import Image

    with Image.open(path) as im:
        return np.array(im)


def save_image(array: np.ndarray, path: Path) -> None:
    """Save a numpy array as an image file. Creates parent directories."""
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path)


def validate_ct_image(image: np.ndarray, expected_size: tuple[int, int] | None = None) -> None:
    """Raise if an image does not meet the basic shape/dtype expectations.

    PROJECT ADAPTATION: a defensive check, not part of Ohashi's methodology.
    """
    image = np.asarray(image)
    if image.ndim not in (2, 3):
        raise ValueError(f"expected a 2-D or 3-D image array, got shape {image.shape}")
    if expected_size is not None and image.shape[:2] != expected_size:
        raise ValueError(f"expected size {expected_size}, got {image.shape[:2]}")
