"""Predictive-distribution and prediction-interval evaluation for Experiment C.

Primary calibration method (predeclared): SPLIT CONFORMAL on the held-out calibration partition.

    Scale conversion (model outputs live in normalized [0, 1] space, labels in [0, 4]):
        mu_i    = SCORE_RANGE * mean_norm_i
        sigma_i = SCORE_RANGE * exp(logvar_norm_i / 2)          (equivalently log-var shift of 2*ln(SCORE_RANGE))
    Nonconformity score:   r_i = |y_i - mu_i| / sigma_i          (y_i in raw [0, 4] units)
    Rank:                  k = ceil((n + 1) * coverage)           (coverage = 1 - alpha, fixed in config beforehand)
    Quantile:              q = k-th smallest r_i;  if k > n the interval is UNBOUNDED (q = +inf)
    Interval:              [mu_i - q*sigma_i, mu_i + q*sigma_i]   (unclipped), and its clip to [0, 4]

Guarantee (marginal coverage >= coverage) holds under exchangeability of calibration and evaluation points, which
this project cannot establish (image-level split, no patient grouping). The method is never fitted on official
test labels. Clipping to [0, 4] shrinks width but cannot change whether a label inside [0, 4] is covered.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.stats import spearmanr

SCORE_RANGE = 4.0           # raw label range [0, 4]; must match ct_iqa.data.ldct_iqac.SCORE_MAX - SCORE_MIN
SCORE_LOW, SCORE_HIGH = 0.0, 4.0


def to_score_scale(mean_norm, log_var_norm, score_range: float = SCORE_RANGE):
    """Convert normalized-space (mean, log-variance) to [0, 4]-scale (mu, sigma)."""
    mean_norm, log_var_norm = np.asarray(mean_norm, dtype=np.float64), np.asarray(log_var_norm, dtype=np.float64)
    return score_range * mean_norm, score_range * np.exp(0.5 * log_var_norm)


def gaussian_nll_score_scale(y_raw, mu, sigma, include_constant: bool = False) -> float:
    """Mean Gaussian NLL of raw labels under N(mu, sigma^2), in the [0, 4] scale (lower is better)."""
    y_raw, mu, sigma = (np.asarray(a, dtype=np.float64) for a in (y_raw, mu, sigma))
    nll = 0.5 * (((y_raw - mu) / sigma) ** 2 + 2.0 * np.log(sigma))
    if include_constant:
        nll = nll + 0.5 * math.log(2.0 * math.pi)
    return float(np.mean(nll))


def conformal_rank(n: int, coverage: float) -> int:
    """k = ceil((n + 1) * coverage), robust to floating-point noise such as 100 * 0.9 = 90.00000000000001."""
    if n < 1 or not 0.0 < coverage < 1.0:
        raise ValueError("need n >= 1 and 0 < coverage < 1")
    return math.ceil(round((n + 1) * coverage, 9))


def conformal_quantile(scores, coverage: float) -> tuple[float, int]:
    """Return (q, k). q = k-th smallest score, or +inf when k > n (unbounded interval)."""
    scores = np.sort(np.asarray(scores, dtype=np.float64))
    if not np.isfinite(scores).all():
        raise ValueError("nonconformity scores must be finite")
    k = conformal_rank(len(scores), coverage)
    return (math.inf if k > len(scores) else float(scores[k - 1])), k


def nonconformity_scores(y_raw, mu, sigma) -> np.ndarray:
    y_raw, mu, sigma = (np.asarray(a, dtype=np.float64) for a in (y_raw, mu, sigma))
    if np.any(sigma <= 0) or not np.isfinite(sigma).all():
        raise ValueError("sigma must be positive and finite")
    return np.abs(y_raw - mu) / sigma


def build_intervals(mu, sigma, q: float) -> dict[str, np.ndarray]:
    mu, sigma = np.asarray(mu, dtype=np.float64), np.asarray(sigma, dtype=np.float64)
    half = q * sigma                                   # +inf if q is +inf
    lo, hi = mu - half, mu + half
    return {"lower": lo, "upper": hi,
            "lower_clipped": np.clip(lo, SCORE_LOW, SCORE_HIGH), "upper_clipped": np.clip(hi, SCORE_LOW, SCORE_HIGH)}


def interval_metrics(y_raw, lower, upper) -> dict[str, float]:
    y_raw, lower, upper = (np.asarray(a, dtype=np.float64) for a in (y_raw, lower, upper))
    covered = (y_raw >= lower) & (y_raw <= upper)
    width = upper - lower
    finite = np.isfinite(width)
    return {
        "coverage": float(covered.mean()),
        "mean_width": float(width.mean()) if finite.all() else math.inf,
        "median_width": float(np.median(width)) if finite.all() else math.inf,
        "n": int(len(y_raw)),
    }


def clipped_fraction(lower, upper) -> float:
    lower, upper = np.asarray(lower, dtype=np.float64), np.asarray(upper, dtype=np.float64)
    return float(np.mean((lower < SCORE_LOW) | (upper > SCORE_HIGH)))


def uncertainty_error_correlation(sigma, abs_error) -> float:
    """Spearman correlation between predicted sigma and absolute error (nan if either is constant)."""
    return float(spearmanr(np.asarray(sigma), np.asarray(abs_error))[0])


def coverage_by_sigma_tercile(y_raw, mu, sigma, q: float) -> list[dict[str, float]]:
    """Coverage of mu +/- q*sigma within terciles of predicted sigma (a conditional-coverage diagnostic)."""
    y_raw, mu, sigma = (np.asarray(a, dtype=np.float64) for a in (y_raw, mu, sigma))
    order = np.argsort(sigma, kind="stable")
    out = []
    for t, idx in enumerate(np.array_split(order, 3), start=1):
        covered = np.abs(y_raw[idx] - mu[idx]) <= q * sigma[idx]
        out.append({"tercile": t, "n": int(len(idx)), "mean_sigma": float(sigma[idx].mean()),
                    "coverage": float(covered.mean())})
    return out


def evaluate_intervals(eval_y, eval_mean_norm, eval_logvar_norm, q: float) -> dict:
    """Evaluate mu +/- q*sigma intervals (unclipped and clipped) and uncertainty diagnostics for a given q."""
    ev_mu, ev_sigma = to_score_scale(eval_mean_norm, eval_logvar_norm)
    iv = build_intervals(ev_mu, ev_sigma, q)
    return {
        "unclipped": interval_metrics(eval_y, iv["lower"], iv["upper"]),
        "clipped_to_0_4": {**interval_metrics(eval_y, iv["lower_clipped"], iv["upper_clipped"]),
                           "fraction_intervals_clipped": clipped_fraction(iv["lower"], iv["upper"])},
        "nll_score_scale": gaussian_nll_score_scale(eval_y, ev_mu, ev_sigma),
        "sigma_error_spearman": uncertainty_error_correlation(ev_sigma, np.abs(np.asarray(eval_y) - ev_mu)),
        "coverage_by_sigma_tercile": coverage_by_sigma_tercile(eval_y, ev_mu, ev_sigma, q),
        "mean_sigma": float(ev_sigma.mean()),
    }


def calibrate_and_evaluate(cal_y, cal_mean_norm, cal_logvar_norm, eval_y, eval_mean_norm, eval_logvar_norm,
                           coverage: float) -> dict:
    """Fit split-conformal on the calibration arrays and evaluate the intervals on the evaluation arrays.

    Both partitions are passed as normalized-space model outputs plus raw [0, 4] labels. The evaluation
    partition must NOT be the calibration partition when a coverage estimate is wanted; the caller chooses
    (validation partition for diagnostics; official test only inside the guarded final evaluation).
    """
    cal_mu, cal_sigma = to_score_scale(cal_mean_norm, cal_logvar_norm)
    q, k = conformal_quantile(nonconformity_scores(cal_y, cal_mu, cal_sigma), coverage)
    return {
        "method": "split_conformal", "target_coverage": coverage, "n_calibration": int(len(np.asarray(cal_y))),
        "rank_k": k, "quantile_q": q, "unbounded": bool(math.isinf(q)),
        **evaluate_intervals(eval_y, eval_mean_norm, eval_logvar_norm, q),
    }
