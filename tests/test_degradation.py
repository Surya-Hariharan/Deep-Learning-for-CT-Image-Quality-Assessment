"""Tests for the Ohashi degradation grid and engine."""

from __future__ import annotations

import numpy as np
import pytest

from ct_iqa.degradation import (
    BLUR_SIGMAS,
    NOISE_SIGMAS,
    Condition,
    apply_condition,
    build_grid,
    condition_rng,
)


def test_grid_sizes_match_paper():
    """105 references x 169 conditions must equal the reported 17,745 images."""
    assert len(NOISE_SIGMAS) == 12
    assert len(BLUR_SIGMAS) == 12
    grid = build_grid()
    assert len(grid) == 169
    assert 105 * len(grid) == 17745


def test_grid_composition():
    grid = build_grid()
    kinds = [c.degradation_type for c in grid]
    assert kinds.count("clean") == 1
    assert kinds.count("noise") == 12
    assert kinds.count("blur") == 12
    assert kinds.count("noise_blur") == 144


def test_grid_without_clean_is_17640():
    """The variant excluding the clean reference, kept for comparison."""
    assert 105 * len(build_grid(include_clean=False)) == 17640


def test_ldctiqac_scale_grid_matches_169000():
    """The active adaptation: 1,000 LDCT-IQAC references x 169 = 169,000."""
    assert 1000 * len(build_grid()) == 169000


def test_condition_ids_are_unique_and_readable():
    suffixes = [c.suffix() for c in build_grid()]
    assert len(set(suffixes)) == len(suffixes)
    assert "noise_3_blur_0.45" in suffixes
    assert "clean" in suffixes


def test_clean_condition_is_identity():
    img = np.random.default_rng(0).normal(size=(32, 32))
    out = apply_condition(
        img, Condition("clean"), order="blur_then_noise", sigma_scale=1.0,
        rng=np.random.default_rng(0),
    )
    np.testing.assert_allclose(out, img)


def test_noise_increases_variance_monotonically():
    img = np.zeros((64, 64))
    stds = [
        apply_condition(
            img, Condition("noise", noise_sigma=s), order="blur_then_noise",
            sigma_scale=1.0, rng=condition_rng(1, "REF", Condition("noise", noise_sigma=s)),
        ).std()
        for s in (1, 6, 30)
    ]
    assert stds[0] < stds[1] < stds[2]


def test_blur_reduces_high_frequency_energy():
    rng = np.random.default_rng(0)
    img = rng.normal(size=(64, 64))
    prev = img.std()
    for sigma in (0.3, 1.4, 5.0):
        blurred = apply_condition(
            img, Condition("blur", blur_sigma=sigma), order="blur_then_noise",
            sigma_scale=1.0, rng=rng,
        )
        assert blurred.std() < prev
        prev = blurred.std()


def test_generation_is_reproducible_with_fixed_seed():
    """Same seed + same ids must reproduce the identical noise realisation."""
    img = np.zeros((16, 16))
    cond = Condition("noise", noise_sigma=6)
    a = apply_condition(img, cond, order="blur_then_noise", sigma_scale=1.0,
                        rng=condition_rng(42, "DL_001", cond))
    b = apply_condition(img, cond, order="blur_then_noise", sigma_scale=1.0,
                        rng=condition_rng(42, "DL_001", cond))
    np.testing.assert_array_equal(a, b)


def test_different_references_get_different_noise():
    img = np.zeros((16, 16))
    cond = Condition("noise", noise_sigma=6)
    a = apply_condition(img, cond, order="blur_then_noise", sigma_scale=1.0,
                        rng=condition_rng(42, "DL_001", cond))
    b = apply_condition(img, cond, order="blur_then_noise", sigma_scale=1.0,
                        rng=condition_rng(42, "DL_002", cond))
    assert not np.array_equal(a, b)


def test_combination_order_actually_matters():
    """Justifies why `order` is a required argument rather than a default."""
    cond = Condition("noise_blur", noise_sigma=30, blur_sigma=2.1)
    img = np.random.default_rng(3).normal(size=(48, 48))
    a = apply_condition(img, cond, order="blur_then_noise", sigma_scale=1.0,
                        rng=condition_rng(7, "R", cond))
    b = apply_condition(img, cond, order="noise_then_blur", sigma_scale=1.0,
                        rng=condition_rng(7, "R", cond))
    assert abs(a.std() - b.std()) > 1.0


def test_clip_range_is_respected():
    img = np.full((32, 32), 128.0)
    out = apply_condition(
        img, Condition("noise", noise_sigma=50), order="blur_then_noise",
        sigma_scale=1.0, rng=np.random.default_rng(0), clip_range=(0.0, 255.0),
    )
    assert out.min() >= 0.0 and out.max() <= 255.0


def test_sigma_scale_is_required():
    with pytest.raises(TypeError):
        apply_condition(  # type: ignore[call-arg]
            np.zeros((8, 8)), Condition("noise", noise_sigma=1), order="blur_then_noise"
        )


def test_order_is_required_for_combined_condition_call():
    with pytest.raises(TypeError):
        apply_condition(  # type: ignore[call-arg]
            np.zeros((8, 8)), Condition("noise_blur", noise_sigma=1, blur_sigma=1),
            sigma_scale=1.0, rng=np.random.default_rng(0),
        )
