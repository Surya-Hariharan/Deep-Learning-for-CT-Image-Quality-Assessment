"""Unit tests for the notebook table/figure helpers. All inputs are SYNTHETIC dictionaries; no run or data is read."""

from __future__ import annotations

import math

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402

from ct_iqa.experiment import report  # noqa: E402


def _history(n=4):
    epochs = [{"epoch": i, "train_loss": 0.1 / i, "val_loss": 0.12 / i, "val_plcc": 0.5 + 0.1 * i, "val_srocc": 0.4 + 0.1 * i,
               "val_krocc": 0.3, "val_mse": 0.5, "val_mae": 0.5, "val_rmse": 0.7, "selected": i == n,
               "train_loss_parts": {"mse": 0.1, "rank": 0.2, "n_valid_pairs": 900.0}} for i in range(1, n + 1)]
    return {"best_epoch": n, "epochs": epochs}


def test_tables_have_expected_shape_and_index():
    h = _history()
    df = report.history_frame(h)
    assert list(df.index) == [1, 2, 3, 4] and bool(df.loc[4, "selected"]) and "val_plcc" in df
    assert list(report.loss_parts_frame(h).columns) == ["mse", "rank", "n_valid_pairs"]
    assert list(report.metrics_frame({"plcc": 0.9, "srocc": 0.8, "n": 100}).index) == ["plcc", "srocc"]


def test_figures_build_without_error():
    fig = report.plot_history(_history(), "t")
    assert len(fig.axes) == 2
    assert len(report.plot_prediction_scatter([0, 1, 2], [0.1, 1.2, 1.9]).axes) == 1
    res = {m: {"difference_b_minus_a": 0.01, "ci_low": -0.02, "ci_high": 0.04} for m in report.POINT_METRICS}
    assert len(report.plot_bootstrap_differences(res, "B - A").axes) == len(report.POINT_METRICS)


def test_interval_plot_handles_finite_and_unbounded_q():
    rng = np.random.default_rng(0)
    y, pred, sigma = rng.uniform(0, 4, 20), rng.uniform(0, 4, 20), rng.uniform(0.1, 0.5, 20)
    terciles = report.calibration_frames({
        "target_coverage": 0.9, "n_calibration": 100, "rank_k": 91, "quantile_q": 2.0, "unbounded": False,
        "nll_score_scale": 0.1, "sigma_error_spearman": 0.2, "mean_sigma": 0.3,
        "unclipped": {"coverage": 0.9, "mean_width": 1.0}, "clipped_to_0_4": {"coverage": 0.9, "mean_width": 0.9},
        "coverage_by_sigma_tercile": [{"tercile": t, "n": 33, "mean_sigma": 0.2, "coverage": 0.9} for t in (1, 2, 3)],
    })["terciles"]
    assert len(report.plot_intervals(y, pred, sigma, 2.0, 0.9, terciles).axes) == 2
    assert len(report.plot_intervals(y, pred, sigma, math.inf, 0.9).axes) == 1
