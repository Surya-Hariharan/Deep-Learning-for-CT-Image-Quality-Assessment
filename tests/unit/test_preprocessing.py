"""Tests for the preprocessing functions extracted from `ct_iqa.data.ldct_iqac`
into `ct_iqa.preprocessing` (see docs/internal/repository_architecture_audit.md,
section 6). These cover the same crop/normalize behavior previously only
exercised indirectly through `LDCTIQACDataset.__getitem__`.
"""

from __future__ import annotations

import numpy as np
import pytest

from ct_iqa.preprocessing.crop import CropSizeError, center_crop
from ct_iqa.preprocessing.normalize import to_radimagenet_range
from ct_iqa.preprocessing.transforms import preprocess_ct_image


def _row_gradient(size: tuple[int, int]) -> np.ndarray:
    h, w = size
    rows = np.arange(h, dtype=np.float32) / (h - 1)
    return np.repeat(rows[:, None], w, axis=1)


def test_center_crop_output_shape():
    array = np.zeros((64, 64), dtype=np.float32)
    cropped = center_crop(array, crop_size=32)
    assert cropped.shape == (32, 32)


def test_center_crop_is_exactly_centered():
    array = _row_gradient((64, 64))
    cropped = center_crop(array, crop_size=32)
    # a centered 32-crop out of 64 rows keeps rows [16:48)
    assert cropped[0, 0] == pytest.approx(16 / 63.0)
    assert cropped[-1, 0] == pytest.approx(47 / 63.0)


def test_center_crop_does_not_resize():
    """A resize would interpolate/blend rows; a crop keeps exact source values."""
    array = _row_gradient((64, 64))
    cropped = center_crop(array, crop_size=32)
    # row 0 of a resize would come from source row 0 (value 0.0); a crop's
    # row 0 comes from source row 16, not 0.
    assert cropped[0, 0] != pytest.approx(0.0, abs=1e-6)


def test_center_crop_raises_when_source_smaller_than_crop():
    array = np.zeros((16, 16), dtype=np.float32)
    with pytest.raises(CropSizeError, match="smaller than"):
        center_crop(array, crop_size=32)


def test_center_crop_matches_real_dataset_dimensions():
    """512x512 source -> 224x224 center crop, matching the real LDCT-IQAC/Ohashi setup."""
    array = np.zeros((512, 512), dtype=np.float32)
    cropped = center_crop(array, crop_size=224)
    assert cropped.shape == (224, 224)


def test_to_radimagenet_range_formula():
    array = np.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype=np.float32)
    result = to_radimagenet_range(array)
    expected = np.array([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=np.float32)
    assert np.allclose(result, expected)


def test_to_radimagenet_range_bounds():
    array = np.random.default_rng(0).uniform(0.0, 1.0, size=(10, 10)).astype(np.float32)
    result = to_radimagenet_range(array)
    assert result.min() >= -1.0
    assert result.max() <= 1.0


def test_preprocess_ct_image_crops_then_normalizes():
    array = _row_gradient((64, 64))
    result = preprocess_ct_image(array, crop_size=32)

    # equivalent to center_crop then to_radimagenet_range, in that order
    expected = to_radimagenet_range(center_crop(array, crop_size=32))
    assert np.allclose(result, expected)
    assert result.min() >= -1.0 and result.max() <= 1.0


def test_preprocess_ct_image_propagates_crop_error_with_source_path():
    array = np.zeros((16, 16), dtype=np.float32)
    with pytest.raises(CropSizeError, match="smaller than"):
        preprocess_ct_image(array, crop_size=32, source_path="fake/path.tif")
