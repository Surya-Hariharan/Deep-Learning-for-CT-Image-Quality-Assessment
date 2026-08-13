"""Ohashi five-parameter logistic calibration of raw quality scores.

OHASHI-SPECIFIED:

    Y_L = b1 * (1/2 - 1 / (1 + exp(b2 * (Y - b3)))) + b4 * Y + b5

Fitted by nonlinear least squares, applied before PLCC. ``predicted_score`` and
``calibrated_score`` (see ``ct_iqa.data.manifest.DegradedRecord``) are kept as
separate fields throughout the pipeline -- never overwritten in place.

NOT SPECIFIED BY OHASHI: initial parameter values or bounds for the optimiser.
``default_initial_guess`` derives them from the data itself (a documented
PROJECT ADAPTATION) instead of hardcoding unexplained constants, and the
winning start is returned in the fit result for the record.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def five_param_logistic(y: np.ndarray, b1: float, b2: float, b3: float, b4: float, b5: float) -> np.ndarray:
    """The Ohashi calibration function."""
    y = np.asarray(y, dtype=np.float64)
    # np.exp overflows for large |b2*(y-b3)|; clipping the exponent keeps the
    # optimiser numerically stable without changing the fitted function.
    # PROJECT ADAPTATION: numerical safeguard only.
    z = np.clip(b2 * (y - b3), -500.0, 500.0)
    return b1 * (0.5 - 1.0 / (1.0 + np.exp(z))) + b4 * y + b5


@dataclass
class CalibrationFit:
    """Fitted calibration parameters and the settings that produced them."""

    params: tuple[float, float, float, float, float]
    initial_guess: tuple[float, float, float, float, float]
    converged: bool
    n_samples: int
    message: str = ""

    def apply(self, y: np.ndarray) -> np.ndarray:
        return five_param_logistic(y, *self.params)


def default_initial_guess(
    predicted: np.ndarray, target: np.ndarray
) -> tuple[float, float, float, float, float]:
    """Data-driven starting point for the optimiser.

    PROJECT ADAPTATION (the source does not specify initialisation):
      b1 <- target range, the amplitude the logistic term has to cover
      b2 <- 1 / std(predicted), setting the slope to the data's own scale
      b3 <- mean(predicted), centring the sigmoid on the data
      b4 <- 0, no initial linear term
      b5 <- mean(target), placing the curve at the right offset
    """
    predicted = np.asarray(predicted, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    spread = float(np.std(predicted))
    b2 = 1.0 / spread if spread > 1e-12 else 1.0
    return (
        float(np.ptp(target)) or 1.0,
        b2,
        float(np.mean(predicted)),
        0.0,
        float(np.mean(target)),
    )


def fit_calibration(
    predicted: np.ndarray,
    target: np.ndarray,
    initial_guess: tuple[float, float, float, float, float] | None = None,
    max_iterations: int = 10000,
) -> CalibrationFit:
    """Fit the five-parameter logistic by nonlinear least squares.

    Args:
        predicted: raw model outputs Y.
        target: values to calibrate towards (VIF for the synthetic stage;
            expert_score for the subjective-evaluation stage -- these are two
            separate calibration fits, never the same fit reused).
        initial_guess: optional explicit start. When given it is used alone;
            when omitted, the multi-start below is used.

    PROJECT ADAPTATION -- multi-start: a single starting point fixes the sign
    of b1 and therefore the orientation of the sigmoid term, so a predictor
    with inverse-S curvature fails to converge from a positive-b1 start. We
    try multiple orientations and keep the lowest sum of squared errors. This
    changes only the optimiser's robustness, never the model form, which is
    exactly as specified.

    The fit must be performed on a held-out or training partition and then
    applied unchanged to the test partition -- fitting on test would leak.
    """
    from scipy.optimize import curve_fit

    predicted = np.asarray(predicted, dtype=np.float64).ravel()
    target = np.asarray(target, dtype=np.float64).ravel()
    if predicted.shape != target.shape:
        raise ValueError(f"shape mismatch: {predicted.shape} vs {target.shape}")
    if predicted.size < 5:
        raise ValueError("need at least 5 samples to fit 5 parameters")

    base = initial_guess or default_initial_guess(predicted, target)
    if initial_guess is not None:
        starts = [base]
    else:
        b1, b2, b3, b4, b5 = base
        slope = float(np.polyfit(predicted, target, 1)[0]) if np.std(predicted) > 1e-12 else 0.0
        starts = [
            base,
            (-b1, b2, b3, b4, b5),          # inverse-S orientation
            (b1, -b2, b3, b4, b5),          # mirrored slope
            (0.0, b2, b3, slope, b5),       # degenerate: pure linear
        ]

    best: CalibrationFit | None = None
    best_sse = np.inf
    last_error = ""

    for start in starts:
        try:
            params, _ = curve_fit(
                five_param_logistic, predicted, target, p0=start, maxfev=max_iterations
            )
        except (RuntimeError, ValueError) as exc:
            last_error = str(exc)
            continue
        residual = target - five_param_logistic(predicted, *params)
        sse = float(np.sum(residual**2))
        if np.isfinite(sse) and sse < best_sse:
            best_sse = sse
            best = CalibrationFit(
                tuple(map(float, params)), start, True, predicted.size,
                f"converged (sse={sse:.6g})",
            )

    if best is not None:
        return best
    return CalibrationFit(tuple(base), base, False, predicted.size, f"did not converge: {last_error}")
