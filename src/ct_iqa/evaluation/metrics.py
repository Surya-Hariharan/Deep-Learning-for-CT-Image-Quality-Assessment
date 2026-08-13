"""OHASHI-SPECIFIED evaluation metrics: MSE, PLCC, SROCC."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def mean_squared_error(predicted: np.ndarray, target: np.ndarray) -> float:
    predicted = np.asarray(predicted, dtype=np.float64).ravel()
    target = np.asarray(target, dtype=np.float64).ravel()
    if predicted.shape != target.shape:
        raise ValueError(f"shape mismatch: {predicted.shape} vs {target.shape}")
    return float(np.mean((predicted - target) ** 2))


@dataclass
class CorrelationResult:
    plcc: float
    srocc: float
    n: int

    def as_dict(self) -> dict[str, float | int]:
        return {"plcc": self.plcc, "srocc": self.srocc, "n": self.n}


def evaluate_correlation(predicted: np.ndarray, target: np.ndarray) -> CorrelationResult:
    """Pearson (PLCC) and Spearman (SROCC) correlation.

    PLCC is conventionally reported *after* the five-parameter logistic
    calibration, since it measures linear agreement; SROCC is rank-based and so
    is unchanged by any monotonic calibration. Pass calibrated predictions for
    PLCC and either for SROCC.
    """
    from scipy.stats import pearsonr, spearmanr

    predicted = np.asarray(predicted, dtype=np.float64).ravel()
    target = np.asarray(target, dtype=np.float64).ravel()
    if predicted.shape != target.shape:
        raise ValueError(f"shape mismatch: {predicted.shape} vs {target.shape}")

    finite = np.isfinite(predicted) & np.isfinite(target)
    predicted, target = predicted[finite], target[finite]
    if predicted.size < 2:
        raise ValueError("need at least 2 finite pairs")

    return CorrelationResult(
        plcc=float(pearsonr(predicted, target)[0]),
        srocc=float(spearmanr(predicted, target)[0]),
        n=int(predicted.size),
    )
