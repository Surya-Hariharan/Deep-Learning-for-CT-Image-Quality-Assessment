"""Five-parameter-logistic (5PL) score calibration for IQA evaluation.

Extracted from `ct_iqa.evaluation.metrics` with no change in numerical
behavior. Mirrors the evaluation methodology described in the Ohashi paper
(5PL fit before PLCC/SROCC), per common IQA evaluation methodology
(VQEG/ITU-T style).
"""

from __future__ import annotations

import numpy as np
from scipy import optimize

from ct_iqa.evaluation.correlation import to_1d_array


def _five_param_logistic(
    x: np.ndarray, b1: float, b2: float, b3: float, b4: float, b5: float
) -> np.ndarray:
    # Standard VQEG-style 5-parameter logistic used for IQA score mapping.
    return b1 * (0.5 - 1.0 / (1.0 + np.exp(b2 * (x - b3)))) + b4 * x + b5


def five_parameter_logistic_fit(y_pred, y_true) -> np.ndarray:
    """Fit a 5-parameter logistic mapping predicted scores onto the ground-truth scale.

    This is an OPTIONAL, explicitly-invoked calibration step -- it is not
    applied inside `ct_iqa.evaluation.metrics.compute_metrics`.

    Returns the mapped predictions (same shape as `y_pred`).
    """
    y_pred, y_true = to_1d_array(y_pred), to_1d_array(y_true)

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
