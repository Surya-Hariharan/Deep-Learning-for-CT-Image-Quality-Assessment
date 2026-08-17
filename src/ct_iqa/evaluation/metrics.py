"""Evaluation metrics for the CT-IQA regression task.

`compute_metrics` operates on RAW predicted/ground-truth scores (no
nonlinear mapping). The Ohashi paper additionally applies a five-parameter
logistic (5PL) regression before computing PLCC/SROCC, per common IQA
evaluation methodology (VQEG/ITU-T style). That mapping is implemented
separately in `five_parameter_logistic_fit` /
`compute_metrics_with_logistic_mapping` and is NEVER applied silently --
callers must explicitly opt into it, and results from the two paths must
not be conflated when reporting.
"""

from __future__ import annotations

import numpy as np
from scipy import optimize
from scipy.stats import kendalltau, pearsonr, spearmanr


def _to_1d_array(values) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.ndim != 1:
        raise ValueError(f"Expected 1-D scores, got shape {arr.shape}")
    return arr


def mse(y_true, y_pred) -> float:
    y_true, y_pred = _to_1d_array(y_true), _to_1d_array(y_pred)
    return float(np.mean((y_true - y_pred) ** 2))


def plcc(y_true, y_pred) -> float:
    """Pearson linear correlation coefficient."""
    y_true, y_pred = _to_1d_array(y_true), _to_1d_array(y_pred)
    return float(pearsonr(y_true, y_pred)[0])


def srocc(y_true, y_pred) -> float:
    """Spearman rank-order correlation coefficient."""
    y_true, y_pred = _to_1d_array(y_true), _to_1d_array(y_pred)
    return float(spearmanr(y_true, y_pred)[0])


def krocc(y_true, y_pred) -> float:
    """Kendall rank-order correlation coefficient."""
    y_true, y_pred = _to_1d_array(y_true), _to_1d_array(y_pred)
    return float(kendalltau(y_true, y_pred)[0])


def compute_metrics(y_true, y_pred) -> dict[str, float]:
    """Raw-space MSE, PLCC, SROCC, KROCC. No logistic mapping is applied."""
    return {
        "mse": mse(y_true, y_pred),
        "plcc": plcc(y_true, y_pred),
        "srocc": srocc(y_true, y_pred),
        "krocc": krocc(y_true, y_pred),
    }


def _five_param_logistic(x: np.ndarray, b1: float, b2: float, b3: float, b4: float, b5: float) -> np.ndarray:
    # Standard VQEG-style 5-parameter logistic used for IQA score mapping.
    return b1 * (0.5 - 1.0 / (1.0 + np.exp(b2 * (x - b3)))) + b4 * x + b5


def five_parameter_logistic_fit(y_pred, y_true) -> np.ndarray:
    """Fit a 5-parameter logistic mapping predicted scores onto the ground-truth scale.

    Mirrors the evaluation methodology described in the Ohashi paper
    (5PL fit before PLCC/SROCC). This is an OPTIONAL, explicitly-invoked
    step -- it is not applied inside `compute_metrics`.

    Returns the mapped predictions (same shape as `y_pred`).
    """
    y_pred, y_true = _to_1d_array(y_pred), _to_1d_array(y_true)

    b1_init = float(np.max(y_true) - np.min(y_true))
    b2_init = 1.0
    b3_init = float(np.mean(y_pred))
    b4_init = 0.0
    b5_init = float(np.mean(y_true))
    initial_params = [b1_init, b2_init, b3_init, b4_init, b5_init]

    params, _ = optimize.curve_fit(
        _five_param_logistic, y_pred, y_true, p0=initial_params, maxfev=20000
    )
    return _five_param_logistic(y_pred, *params)


def compute_metrics_with_logistic_mapping(y_true, y_pred) -> dict[str, float]:
    """MSE/PLCC/SROCC/KROCC computed AFTER a 5-parameter logistic mapping of y_pred onto y_true's scale.

    Distinct from `compute_metrics` (raw correlation) -- report both,
    labeled, rather than only one.
    """
    y_true_arr = _to_1d_array(y_true)
    y_pred_mapped = five_parameter_logistic_fit(y_pred, y_true_arr)
    return compute_metrics(y_true_arr, y_pred_mapped)
