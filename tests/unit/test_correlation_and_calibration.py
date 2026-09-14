"""Tests for `ct_iqa.evaluation.correlation` and `ct_iqa.evaluation.calibration`,
extracted from `ct_iqa.evaluation.metrics` (see
docs/internal/repository_architecture_audit.md, section 3/7). `test_metrics.py`
already covers `compute_metrics`/`compute_metrics_with_logistic_mapping`
end-to-end; these tests exercise the extracted modules directly.
"""

from __future__ import annotations

import numpy as np
import pytest

from ct_iqa.evaluation.calibration import five_parameter_logistic_fit
from ct_iqa.evaluation.correlation import krocc, plcc, srocc, to_1d_array


def test_to_1d_array_flattens():
    arr = to_1d_array([[1, 2], [3, 4]])
    assert arr.shape == (4,)


def test_to_1d_array_flattens_higher_dimensional_input():
    # `to_1d_array` reshapes with -1, which always yields ndim == 1 -- this
    # is pre-existing, preserved behavior (see docs/replication note in
    # ct_iqa.evaluation.correlation), not something introduced here.
    result = to_1d_array(np.zeros((2, 2, 2)))
    assert result.shape == (8,)


def test_plcc_perfect_correlation():
    assert plcc([0, 1, 2, 3], [0, 1, 2, 3]) == pytest.approx(1.0)


def test_srocc_perfect_inverse_correlation():
    assert srocc([0, 1, 2, 3], [3, 2, 1, 0]) == pytest.approx(-1.0)


def test_krocc_perfect_correlation():
    assert krocc([0, 1, 2, 3], [0, 1, 2, 3]) == pytest.approx(1.0)


def test_five_parameter_logistic_fit_recovers_monotonic_relationship():
    rng = np.random.default_rng(0)
    y_true = np.linspace(0, 4, 40)
    y_pred = 1.0 / (1.0 + np.exp(-(y_true - 2.0))) + rng.normal(0, 0.01, size=40)

    mapped = five_parameter_logistic_fit(y_pred, y_true)
    # after a good 5PL fit, mapped predictions should correlate near-perfectly
    # with the ground truth in raw space
    assert plcc(y_true, mapped) > 0.99
