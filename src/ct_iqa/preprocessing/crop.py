"""Center-crop preprocessing for CT images.

Extracted from `ct_iqa.data.ldct_iqac` (see
docs/repository_architecture_audit.md, section 6 "Data Pipeline Audit") with
no change in behavior: this is still a CROP, never a resize, per Ohashi et
al.: "each image was cropped to the central region according to the input
size required by the model" -- done to preserve the original CT images'
spatial resolution.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


class CropSizeError(ValueError):
    """Raised when an image is smaller than the requested crop size."""


def center_crop(
    array: np.ndarray, crop_size: int, source_path: Path | str | None = None
) -> np.ndarray:
    """Crop the central `crop_size` x `crop_size` region out of a 2-D array.

    Raises `CropSizeError` if either dimension of `array` is smaller than
    `crop_size`. `source_path` is optional and only used to make the error
    message identify which file failed.
    """
    h, w = array.shape
    if h < crop_size or w < crop_size:
        location = f" {source_path}" if source_path is not None else ""
        raise CropSizeError(
            f"Image{location} has shape {(h, w)}, smaller than the "
            f"requested center-crop size {crop_size}x{crop_size}"
        )
    top = (h - crop_size) // 2
    left = (w - crop_size) // 2
    return array[top : top + crop_size, left : left + crop_size]
