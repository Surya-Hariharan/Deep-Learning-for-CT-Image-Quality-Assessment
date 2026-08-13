"""Tests for ct_iqa.preprocessing (cropping, channels, image_io)."""

from __future__ import annotations

import numpy as np
import pytest

from ct_iqa.preprocessing.channels import grayscale_to_rgb
from ct_iqa.preprocessing.cropping import central_crop
from ct_iqa.preprocessing.image_io import load_image, save_image, validate_ct_image


def test_central_crop_takes_the_middle():
    img = np.arange(16 * 16).reshape(16, 16)
    cropped = central_crop(img, size=4)
    assert cropped.shape == (4, 4)
    np.testing.assert_array_equal(cropped, img[6:10, 6:10])


def test_central_crop_matches_ohashi_input_size():
    img = np.zeros((512, 512))
    assert central_crop(img, size=224).shape == (224, 224)


def test_central_crop_refuses_to_upsample_smaller_images():
    """Padding/upsampling behaviour is NOT SPECIFIED BY OHASHI; must not be guessed."""
    with pytest.raises(ValueError, match="NOT SPECIFIED"):
        central_crop(np.zeros((100, 100)), size=224)


def test_grayscale_to_rgb_replicates_channel():
    img = np.arange(9).reshape(3, 3)
    rgb = grayscale_to_rgb(img)
    assert rgb.shape == (3, 3, 3)
    np.testing.assert_array_equal(rgb[..., 0], rgb[..., 1])
    np.testing.assert_array_equal(rgb[..., 1], rgb[..., 2])


def test_grayscale_to_rgb_is_noop_for_existing_rgb():
    img = np.random.default_rng(0).integers(0, 255, size=(4, 4, 3))
    np.testing.assert_array_equal(grayscale_to_rgb(img), img)


def test_grayscale_to_rgb_expands_single_channel_axis():
    img = np.arange(9).reshape(3, 3, 1)
    rgb = grayscale_to_rgb(img)
    assert rgb.shape == (3, 3, 3)


def test_grayscale_to_rgb_rejects_unexpected_shape():
    with pytest.raises(ValueError):
        grayscale_to_rgb(np.zeros((4, 4, 5)))


def test_validate_ct_image_accepts_expected_size():
    validate_ct_image(np.zeros((512, 512)), expected_size=(512, 512))


def test_validate_ct_image_rejects_wrong_size():
    with pytest.raises(ValueError):
        validate_ct_image(np.zeros((256, 256)), expected_size=(512, 512))


def test_validate_ct_image_rejects_bad_ndim():
    with pytest.raises(ValueError):
        validate_ct_image(np.zeros((2, 2, 2, 2)))


def test_save_then_load_image_roundtrips(tmp_path):
    array = np.random.default_rng(0).integers(0, 255, size=(32, 32), dtype=np.uint8)
    out_path = tmp_path / "nested" / "img.png"
    save_image(array, out_path)
    assert out_path.is_file()
    loaded = load_image(out_path)
    np.testing.assert_array_equal(loaded, array)
