"""Composed CT-image preprocessing transform.

Combines `ct_iqa.preprocessing.crop.center_crop` and
`ct_iqa.preprocessing.normalize.to_radimagenet_range` into the single
preprocessing pipeline used by `ct_iqa.data.ldct_iqac.LDCTIQACDataset`:
center crop (not resize), then [-1, 1] pixel normalization. Extracted with
no change in behavior or ordering from the pre-extraction inline
implementation.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ct_iqa.preprocessing.crop import center_crop
from ct_iqa.preprocessing.normalize import to_radimagenet_range


def preprocess_ct_image(
    array: np.ndarray, crop_size: int, source_path: Path | str | None = None
) -> np.ndarray:
    """Center-crop `array` to `crop_size` x `crop_size`, then normalize to [-1, 1]."""
    cropped = center_crop(array, crop_size, source_path=source_path)
    return to_radimagenet_range(cropped)
