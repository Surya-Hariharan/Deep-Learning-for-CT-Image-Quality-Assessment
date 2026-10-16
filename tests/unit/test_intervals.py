"""Unit tests for split-conformal calibration and interval metrics. All data are SYNTHETIC."""

from __future__ import annotations

import math

import numpy as np
import pytest

from ct_iqa.evaluation.intervals import (
    SCORE_RANGE, build_intervals, calibrate_and_evaluate, conformal_quantile, conformal_rank,
    coverage_by_sigma_tercile, gaussian_nll_score_scale, interval_metrics, nonconformity_scores,
    to_score_scale, uncertainty_error_correlation,
)


def test_rank_formula_and_float_noise():
    assert conformal_rank(100, 0.90) == 91                 # ceil(101 * 0.9) = ceil(90.9)
    assert conformal_rank(99, 0.90) == 90                  # 100 * 0.9 = 90.00000000000001 must NOT become 91
    assert conformal_rank(9, 0.90) == 9                    # ceil(10 * 0.9)
    assert conformal_rank(10, 0.95) == 11                  # ceil(11 * 0.95) = ceil(10.45) = 11 > n
    with pytest.raises(ValueError):
        conformal_rank(0, 0.9)
    with pytest.raises(ValueError):
        conformal_rank(10, 1.0)


def test_quantile_picks_kth_smallest_and_handles_ties():
    s = np.arange(1.0, 101.0)                              # 1..100, n = 100, k = 91
    q, k = conformal_quantile(s, 0.90)
    assert (q, k) == (91.0, 91)
    q, k = conformal_quantile(np.array([2.0] * 50 + [1.0] * 50), 0.90)   # heavy ties
    assert (q, k) == (2.0, 91)


def test_k_greater_than_n_is_unbounded_before_clipping():
    q, k = conformal_quantile(np.array([0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]), 0.95)
    assert k == 11 and math.isinf(q)
    iv = build_intervals(np.array([1.0, 2.0]), np.array([0.3, 0.4]), q)
    assert np.isinf(iv["lower"]).all() and np.isinf(iv["upper"]).all()
    assert iv["lower_clipped"].tolist() == [0.0, 0.0] and iv["upper_clipped"].tolist() == [4.0, 4.0]
    m = interval_metrics([1.0, 3.0], iv["lower"], iv["upper"])
    assert m["coverage"] == 1.0 and math.isinf(m["mean_width"])


def test_scale_conversion_of_mean_and_std():
    mu, sigma = to_score_scale(np.array([0.25, 1.0]), np.array([0.0, 2 * math.log(0.1)]))
    assert mu.tolist() == [1.0, 4.0]
    assert sigma == pytest.approx([SCORE_RANGE * 1.0, SCORE_RANGE * 0.1])     # std = 4 * exp(s/2)
    assert SCORE_RANGE == 4.0


def test_scores_and_coverage_invariant_to_working_scale_and_halfwidth_scales_4x():
    rng = np.random.default_rng(0)
    mean_n, logvar_n = rng.uniform(0.1, 0.9, 200), rng.uniform(-6, -2, 200)
    y_n = np.clip(mean_n + rng.normal(0, np.exp(0.5 * logvar_n) * 1.3), 0, 1)
    r_norm = np.abs(y_n - mean_n) / np.exp(0.5 * logvar_n)                      # normalized working scale
    mu, sigma = to_score_scale(mean_n, logvar_n)
    r_raw = nonconformity_scores(4.0 * y_n, mu, sigma)                          # [0, 4] working scale
    assert np.allclose(r_norm, r_raw)
    q, _ = conformal_quantile(r_raw, 0.9)
    iv = build_intervals(mu, sigma, q)
    half_raw = (iv["upper"] - iv["lower"]) / 2
    half_norm = q * np.exp(0.5 * logvar_n)
    assert np.allclose(half_raw, 4.0 * half_norm)                               # exactly 4x
    cov_raw = interval_metrics(4.0 * y_n, iv["lower"], iv["upper"])["coverage"]
    cov_norm = float(np.mean(np.abs(y_n - mean_n) <= half_norm))
    assert cov_raw == cov_norm


def test_empirical_coverage_close_to_target_on_exchangeable_data():
    rng = np.random.default_rng(1)

    def draw(n):
        mean_n = rng.uniform(0.2, 0.8, n)
        logvar_n = rng.uniform(-6, -3, n)
        y = np.clip(mean_n + rng.normal(0, 1.7 * np.exp(0.5 * logvar_n)), 0, 1) * 4  # sigma misspecified by 1.7x
        return y, mean_n, logvar_n

    res = calibrate_and_evaluate(*draw(2000), *draw(20000), coverage=0.90)
    assert res["unclipped"]["coverage"] == pytest.approx(0.90, abs=0.02)
    assert res["clipped_to_0_4"]["coverage"] == res["unclipped"]["coverage"]    # labels lie in [0, 4]
    assert res["clipped_to_0_4"]["mean_width"] <= res["unclipped"]["mean_width"]
    assert all(np.isfinite(v) for v in (res["nll_score_scale"], res["mean_sigma"]))
    assert [t["tercile"] for t in res["coverage_by_sigma_tercile"]] == [1, 2, 3]


def test_nll_closed_form_and_invalid_sigma_rejected():
    assert gaussian_nll_score_scale([1.0], [1.0], [2.0]) == pytest.approx(math.log(2.0))
    expected = 0.5 * ((3.0 - 1.0) / 2.0) ** 2 + math.log(2.0) + 0.5 * math.log(2 * math.pi)
    assert gaussian_nll_score_scale([3.0], [1.0], [2.0], include_constant=True) == pytest.approx(expected)
    with pytest.raises(ValueError):
        nonconformity_scores([1.0], [1.0], [0.0])


def test_uncertainty_error_correlation_detects_association():
    sigma = np.linspace(0.1, 1.0, 50)
    assert uncertainty_error_correlation(sigma, sigma * 2) == pytest.approx(1.0)
    assert uncertainty_error_correlation(sigma, -sigma) == pytest.approx(-1.0)
    assert [t["coverage"] for t in coverage_by_sigma_tercile(np.zeros(9), np.zeros(9), np.ones(9), 1.0)] == [1.0] * 3
