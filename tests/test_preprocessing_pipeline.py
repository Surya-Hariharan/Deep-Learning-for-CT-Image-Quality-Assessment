"""Tests for the baseline model-input preprocessing contract
(``ct_iqa.preprocessing.normalization``, ``ct_iqa.preprocessing.pipeline``).

Resolves U-M05 (decision A-23, docs/research_decisions.md): input
normalization for the RadImageNet ResNet50 backbone. These tests verify
the preprocessing *contract* only (shape, dtype, value range,
determinism, crop/normalization behavior) -- never model accuracy, and
never anything requiring TensorFlow (this module is pure numpy, usable in
the main Python 3.13 environment where TensorFlow is not installed).
"""

from __future__ import annotations

import numpy as np
import pytest

from ct_iqa.preprocessing.normalization import BGR_MEAN, resnet50_preprocess_input
from ct_iqa.preprocessing.pipeline import (
    OUTPUT_DTYPE,
    OUTPUT_SHAPE,
    OUTPUT_SIZE,
    preprocess_for_model,
)


# ---------------------------------------------------------------- normalization


def test_resnet50_preprocess_input_reverses_channel_order():
    img = np.zeros((1, 1, 3), dtype=np.float64)
    img[0, 0] = [10.0, 20.0, 30.0]  # R, G, B
    out = resnet50_preprocess_input(img)
    # after BGR reorder: [30, 20, 10], then subtract BGR_MEAN
    expected = np.array([30.0, 20.0, 10.0]) - BGR_MEAN
    np.testing.assert_allclose(out[0, 0], expected)


def test_resnet50_preprocess_input_subtracts_imagenet_bgr_mean():
    img = np.zeros((2, 2, 3), dtype=np.float64)
    out = resnet50_preprocess_input(img)
    # zero input, reversed is still zero, so output is exactly -BGR_MEAN everywhere
    expected = -BGR_MEAN
    for i in range(2):
        for j in range(2):
            np.testing.assert_allclose(out[i, j], expected)


def test_resnet50_preprocess_input_matches_tensorflow_reference_values():
    """Cross-checked bit-exact against tensorflow.keras.applications.resnet50.
    preprocess_input in the isolated training environment
    (docs/training_environment.md) -- hardcoded expected values from that
    cross-check, so this test catches any future drift without requiring
    TensorFlow to be installed here."""
    img = np.array([[[0.0, 128.0, 255.0]]])  # single pixel, R=0 G=128 B=255
    out = resnet50_preprocess_input(img)
    # BGR reorder: [255, 128, 0], then subtract [103.939, 116.779, 123.68]
    expected = np.array([255.0 - 103.939, 128.0 - 116.779, 0.0 - 123.68])
    np.testing.assert_allclose(out[0, 0], expected, atol=1e-9)


def test_resnet50_preprocess_input_output_dtype_is_float64():
    img = np.zeros((2, 2, 3), dtype=np.uint8)
    out = resnet50_preprocess_input(img)
    assert out.dtype == np.float64


def test_resnet50_preprocess_input_accepts_uint8_input():
    img = np.full((2, 2, 3), 200, dtype=np.uint8)
    out = resnet50_preprocess_input(img)
    assert np.isfinite(out).all()


def test_resnet50_preprocess_input_rejects_non_3channel_input():
    with pytest.raises(ValueError):
        resnet50_preprocess_input(np.zeros((4, 4, 5)))
    with pytest.raises(ValueError):
        resnet50_preprocess_input(np.zeros((4, 4)))


def test_resnet50_preprocess_input_is_deterministic():
    img = np.random.default_rng(0).integers(0, 255, size=(8, 8, 3), dtype=np.uint8)
    a = resnet50_preprocess_input(img)
    b = resnet50_preprocess_input(img)
    np.testing.assert_array_equal(a, b)


# ---------------------------------------------------------------- full pipeline


def test_preprocess_for_model_output_shape_is_224_224_3():
    img = np.random.default_rng(0).integers(0, 255, size=(512, 512), dtype=np.uint8)
    out = preprocess_for_model(img)
    assert out.shape == (224, 224, 3) == OUTPUT_SHAPE
    assert OUTPUT_SIZE == 224


def test_preprocess_for_model_output_dtype_is_float32():
    img = np.random.default_rng(0).integers(0, 255, size=(512, 512), dtype=np.uint8)
    out = preprocess_for_model(img)
    assert out.dtype == np.float32 == OUTPUT_DTYPE


def test_preprocess_for_model_handles_grayscale_2d_input():
    img = np.random.default_rng(1).integers(0, 255, size=(512, 512), dtype=np.uint8)
    out = preprocess_for_model(img)
    assert out.shape == (224, 224, 3)


def test_preprocess_for_model_handles_rgb_3channel_input():
    img = np.random.default_rng(2).integers(0, 255, size=(512, 512, 3), dtype=np.uint8)
    out = preprocess_for_model(img)
    assert out.shape == (224, 224, 3)


def test_preprocess_for_model_is_central_crop_not_resize():
    """A 512x512 image with a distinct marker only in the exact center
    224x224 region must appear unchanged in the crop (after undoing
    normalization) -- proves no resize/interpolation occurred."""
    size = 512
    img = np.zeros((size, size), dtype=np.uint8)
    top = (size - 224) // 2
    left = (size - 224) // 2
    marker = np.random.default_rng(3).integers(0, 255, size=(224, 224), dtype=np.uint8)
    img[top : top + 224, left : left + 224] = marker

    out = preprocess_for_model(img)
    # undo: BGR->RGB, add mean back, take one channel (all 3 equal pre-normalization)
    recovered_bgr = out.astype(np.float64) + BGR_MEAN
    recovered_rgb = recovered_bgr[..., ::-1]
    np.testing.assert_allclose(recovered_rgb[..., 0], marker, atol=1e-3)


def test_preprocess_for_model_value_range_is_not_0_1_scaled():
    """Guards against accidentally reintroducing the redundant /255 rescale
    decision A-23 explicitly rejected -- output values should span a range
    consistent with raw [0,255] input minus ImageNet means, not a [0,1]-ish
    range."""
    img = np.full((512, 512), 255, dtype=np.uint8)
    out = preprocess_for_model(img)
    assert out.max() > 100  # would be ~1.14 if the redundant /255 were present


def test_preprocess_for_model_rejects_undersized_image():
    with pytest.raises(ValueError, match="NOT SPECIFIED"):
        preprocess_for_model(np.zeros((100, 100), dtype=np.uint8))


def test_preprocess_for_model_is_deterministic():
    img = np.random.default_rng(4).integers(0, 255, size=(512, 512), dtype=np.uint8)
    a = preprocess_for_model(img)
    b = preprocess_for_model(img)
    np.testing.assert_array_equal(a, b)


def test_preprocess_for_model_no_augmentation_randomness_across_calls():
    """OHASHI-SPECIFIED: no augmentation beyond the central crop -- two
    distinct images that share the same center region must produce the
    same crop result regardless of what's outside it."""
    size = 512
    top = (size - 224) // 2
    left = (size - 224) // 2
    marker = np.random.default_rng(5).integers(0, 255, size=(224, 224), dtype=np.uint8)

    img_a = np.zeros((size, size), dtype=np.uint8)
    img_a[top : top + 224, left : left + 224] = marker
    img_b = np.full((size, size), 200, dtype=np.uint8)
    img_b[top : top + 224, left : left + 224] = marker

    np.testing.assert_array_equal(preprocess_for_model(img_a), preprocess_for_model(img_b))
