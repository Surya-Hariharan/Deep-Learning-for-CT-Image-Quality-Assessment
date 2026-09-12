"""Rank/linear correlation coefficients for the CT-IQA regression task.

Extracted from `ct_iqa.evaluation.metrics` with no change in numerical
behavior.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import kendalltau, pearsonr, spearmanr


def to_1d_array(values) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.ndim != 1:
        raise ValueError(f"Expected 1-D scores, got shape {arr.shape}")
    return arr


def plcc(y_true, y_pred) -> float:
    """Pearson linear correlation coefficient."""
    y_true, y_pred = to_1d_array(y_true), to_1d_array(y_pred)
    return float(pearsonr(y_true, y_pred)[0])


def srocc(y_true, y_pred) -> float:
    """Spearman rank-order correlation coefficient."""
    y_true, y_pred = to_1d_array(y_true), to_1d_array(y_pred)
    return float(spearmanr(y_true, y_pred)[0])


def krocc(y_true, y_pred) -> float:
    """Kendall rank-order correlation coefficient."""
    y_true, y_pred = to_1d_array(y_true), to_1d_array(y_pred)
    return float(kendalltau(y_true, y_pred)[0])
