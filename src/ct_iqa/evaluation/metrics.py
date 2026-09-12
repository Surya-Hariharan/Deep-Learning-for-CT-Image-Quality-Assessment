"""General evaluation-metric orchestration for the CT-IQA regression task.

`compute_metrics` operates on RAW predicted/ground-truth scores (no
nonlinear mapping) and reports MSE plus the three correlation coefficients
(`ct_iqa.evaluation.correlation`). The Ohashi paper additionally applies a
five-parameter logistic (5PL) regression (`ct_iqa.evaluation.calibration`)
before computing PLCC/SROCC, per common IQA evaluation methodology
(VQEG/ITU-T style). That mapping is NEVER applied silently --
`compute_metrics_with_logistic_mapping` must be called explicitly, and
results from the two paths must not be conflated when reporting.
"""

from __future__ import annotations

import numpy as np

from ct_iqa.evaluation.calibration import five_parameter_logistic_fit
from ct_iqa.evaluation.correlation import krocc, plcc, srocc, to_1d_array


def mse(y_true, y_pred) -> float:
    y_true, y_pred = to_1d_array(y_true), to_1d_array(y_pred)
    return float(np.mean((y_true - y_pred) ** 2))


def compute_metrics(y_true, y_pred) -> dict[str, float]:
    """Raw-space MSE, PLCC, SROCC, KROCC. No logistic mapping is applied."""
    return {
        "mse": mse(y_true, y_pred),
        "plcc": plcc(y_true, y_pred),
        "srocc": srocc(y_true, y_pred),
        "krocc": krocc(y_true, y_pred),
    }


def compute_metrics_with_logistic_mapping(y_true, y_pred) -> dict[str, float]:
    """MSE/PLCC/SROCC/KROCC computed AFTER a 5-parameter logistic mapping of y_pred onto y_true's scale.

    Distinct from `compute_metrics` (raw correlation) -- report both,
    labeled, rather than only one.
    """
    y_true_arr = to_1d_array(y_true)
    y_pred_mapped = five_parameter_logistic_fit(y_pred, y_true_arr)
    return compute_metrics(y_true_arr, y_pred_mapped)
