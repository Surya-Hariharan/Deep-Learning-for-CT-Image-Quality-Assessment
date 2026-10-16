"""Paired bootstrap for comparing two models' predictions on the same images.

Sampling unit: the IMAGE (rows are resampled with replacement, jointly for both models, so the pairing is kept).
Patient grouping is unknown for LDCT-IQAC, so images from one patient may be correlated; the resulting intervals
are then optimistically narrow. This is a limitation, not something the method corrects.
"""

from __future__ import annotations

import numpy as np

from ct_iqa.evaluation.metrics import compute_point_metrics

METRICS = ("plcc", "srocc", "krocc", "mse", "mae", "rmse")


def paired_bootstrap_difference(y_true, pred_a, pred_b, n_boot: int = 2000, seed: int = 0,
                                confidence: float = 0.95) -> dict[str, dict[str, float]]:
    """Bootstrap distribution of metric(B) - metric(A) (positive PLCC/SROCC/KROCC difference favors B;
    positive MSE/MAE/RMSE difference favors A). Returns point difference, percentile CI and the bootstrap
    fraction of resamples in which the difference is > 0."""
    y, a, b = (np.asarray(v, dtype=np.float64) for v in (y_true, pred_a, pred_b))
    if not (len(y) == len(a) == len(b)):
        raise ValueError("y_true, pred_a and pred_b must have equal length")
    rng = np.random.default_rng(seed)
    n = len(y)
    point_a, point_b = compute_point_metrics(y, a), compute_point_metrics(y, b)
    diffs = {m: [] for m in METRICS}
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if np.ptp(y[idx]) == 0:
            continue                                   # degenerate resample: correlations undefined
        ma, mb = compute_point_metrics(y[idx], a[idx]), compute_point_metrics(y[idx], b[idx])
        for m in METRICS:
            diffs[m].append(mb[m] - ma[m])
    lo_q, hi_q = (1 - confidence) / 2 * 100, (1 + confidence) / 2 * 100
    out = {}
    for m in METRICS:
        d = np.asarray(diffs[m])
        d = d[np.isfinite(d)]
        out[m] = {
            "difference_b_minus_a": point_b[m] - point_a[m],
            "ci_low": float(np.percentile(d, lo_q)), "ci_high": float(np.percentile(d, hi_q)),
            "fraction_positive": float(np.mean(d > 0)), "n_boot_used": int(len(d)),
        }
    return out
