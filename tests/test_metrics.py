"""Tests for MSE, PLCC, SROCC, five-parameter logistic calibration, and VIF."""

from __future__ import annotations

import numpy as np
import pytest

from ct_iqa.evaluation.logistic_calibration import (
    default_initial_guess,
    fit_calibration,
    five_param_logistic,
)
from ct_iqa.evaluation.metrics import evaluate_correlation, mean_squared_error
from ct_iqa.vif.labeling import compute_vif
from ct_iqa.vif.metric import vif_p


def _phantom(size: int = 128, seed: int = 0) -> np.ndarray:
    """A textured test image; VIF is undefined on a perfectly flat one."""
    rng = np.random.default_rng(seed)
    base = np.zeros((size, size))
    base[size // 4 : 3 * size // 4, size // 4 : 3 * size // 4] = 100.0
    return base + rng.normal(scale=5.0, size=(size, size))


# ------------------------------------------------------------------- MSE

def test_mse_of_identical_arrays_is_zero():
    x = np.array([0.1, 0.5, 0.9])
    assert mean_squared_error(x, x) == pytest.approx(0.0)


def test_mse_matches_hand_computation():
    predicted = np.array([1.0, 2.0, 3.0])
    target = np.array([1.5, 2.0, 2.0])
    assert mean_squared_error(predicted, target) == pytest.approx(((-0.5) ** 2 + 0 + 1.0**2) / 3)


def test_mse_rejects_shape_mismatch():
    with pytest.raises(ValueError):
        mean_squared_error(np.zeros(3), np.zeros(4))


# -------------------------------------------------------------------- VIF

def test_vif_of_identical_images_is_one():
    img = _phantom()
    assert vif_p(img, img) == pytest.approx(1.0, abs=1e-6)


def test_vif_decreases_with_noise_severity():
    img = _phantom()
    rng = np.random.default_rng(1)
    scores = [vif_p(img, img + rng.normal(scale=s, size=img.shape)) for s in (1, 6, 30)]
    assert scores[0] > scores[1] > scores[2]


def test_vif_decreases_with_blur_severity():
    from scipy.ndimage import gaussian_filter

    img = _phantom()
    scores = [vif_p(img, gaussian_filter(img, s)) for s in (0.3, 1.4, 5.0)]
    assert scores[0] > scores[1] > scores[2]


def test_vif_rejects_shape_mismatch():
    with pytest.raises(ValueError):
        vif_p(_phantom(64), _phantom(128))


def test_vif_rejects_non_2d():
    with pytest.raises(ValueError):
        vif_p(np.zeros((8, 8, 3)), np.zeros((8, 8, 3)))


def test_compute_vif_requires_explicit_variant():
    """No default variant: the choice is unverified and must be stated."""
    img = _phantom(64)
    with pytest.raises(TypeError):
        compute_vif(img, img)  # type: ignore[call-arg]


def test_wavelet_vif_refuses_rather_than_approximating():
    img = _phantom(64)
    with pytest.raises(NotImplementedError, match="NOT SPECIFIED BY OHASHI"):
        compute_vif(img, img, variant="vif_wavelet")


def test_unknown_variant_rejected():
    img = _phantom(64)
    with pytest.raises(ValueError):
        compute_vif(img, img, variant="ssim")  # type: ignore[arg-type]


# ------------------------------------------------------------- calibration

def test_logistic_is_finite_for_extreme_inputs():
    """The exponent clip must prevent overflow at the optimiser's extremes."""
    y = np.array([-1e6, 0.0, 1e6])
    out = five_param_logistic(y, 1.0, 1000.0, 0.5, 0.0, 0.0)
    assert np.all(np.isfinite(out))


def test_logistic_is_monotonic_for_positive_parameters():
    y = np.linspace(0, 1, 50)
    out = five_param_logistic(y, 1.0, 10.0, 0.5, 0.5, 0.0)
    assert np.all(np.diff(out) > 0)


def test_calibration_recovers_known_parameters():
    truth = (0.8, 8.0, 0.45, 0.3, 0.05)
    y = np.linspace(0.0, 1.0, 300)
    target = five_param_logistic(y, *truth)
    fit = fit_calibration(y, target)
    assert fit.converged
    np.testing.assert_allclose(fit.apply(y), target, atol=1e-3)


def test_calibration_improves_linearity_of_a_nonlinear_predictor():
    """The point of calibration: raise PLCC without changing SROCC.

    The raw score is related to VIF by an S-shaped compression -- the situation
    the five-parameter logistic exists to undo.
    """
    rng = np.random.default_rng(0)
    raw = rng.uniform(0.05, 0.95, size=400)
    vif = 0.9 / (1.0 + np.exp(-9.0 * (raw - 0.5))) + 0.05

    before = evaluate_correlation(raw, vif)
    fit = fit_calibration(raw, vif)
    after = evaluate_correlation(fit.apply(raw), vif)

    assert fit.converged
    assert after.plcc > before.plcc
    assert after.plcc > 0.999
    assert after.srocc == pytest.approx(before.srocc, abs=1e-9)


def test_calibration_handles_inverse_s_curvature():
    """Justifies the multi-start: a negative b1 is unreachable from one start."""
    truth = (-0.7, 9.0, 0.5, 0.9, 0.1)
    y = np.linspace(0.0, 1.0, 300)
    target = five_param_logistic(y, *truth)

    fit = fit_calibration(y, target)
    assert fit.converged
    np.testing.assert_allclose(fit.apply(y), target, atol=1e-3)


def test_calibration_records_the_winning_start():
    """The start is a project decision, so it must be recoverable from the fit."""
    y = np.linspace(0, 1, 50)
    fit = fit_calibration(y, y * 0.5)
    assert fit.initial_guess is not None
    assert len(fit.initial_guess) == 5
    assert fit.n_samples == 50


def test_explicit_initial_guess_is_honoured():
    """An explicit start disables the multi-start and is used verbatim."""
    y = np.linspace(0, 1, 50)
    guess = default_initial_guess(y, y * 0.5)
    fit = fit_calibration(y, y * 0.5, initial_guess=guess)
    assert fit.initial_guess == guess


def test_calibration_needs_enough_samples():
    with pytest.raises(ValueError):
        fit_calibration(np.array([0.1, 0.2]), np.array([0.1, 0.2]))


def test_calibration_rejects_shape_mismatch():
    with pytest.raises(ValueError):
        fit_calibration(np.zeros(10), np.zeros(9))


# -------------------------------------------------------------- correlation

def test_correlations_of_perfect_prediction():
    x = np.linspace(0, 1, 20)
    result = evaluate_correlation(x, x)
    assert result.plcc == pytest.approx(1.0)
    assert result.srocc == pytest.approx(1.0)
    assert result.n == 20


def test_srocc_is_invariant_to_monotonic_transform():
    x = np.linspace(0.1, 0.9, 40)
    y = np.exp(3 * x)
    assert evaluate_correlation(x, y).srocc == pytest.approx(1.0)


def test_non_finite_pairs_are_dropped():
    x = np.array([0.1, 0.2, np.nan, 0.4])
    y = np.array([0.1, 0.2, 0.3, 0.4])
    assert evaluate_correlation(x, y).n == 3
