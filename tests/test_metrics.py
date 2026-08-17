import numpy as np
import pytest

from ct_iqa.evaluation.metrics import compute_metrics, compute_metrics_with_logistic_mapping


def test_perfect_correlation():
    y_true = [0.0, 1.0, 2.0, 3.0, 4.0]
    y_pred = [0.0, 1.0, 2.0, 3.0, 4.0]
    metrics = compute_metrics(y_true, y_pred)
    assert metrics["mse"] == pytest.approx(0.0)
    assert metrics["plcc"] == pytest.approx(1.0)
    assert metrics["srocc"] == pytest.approx(1.0)
    assert metrics["krocc"] == pytest.approx(1.0)


def test_inverse_correlation():
    y_true = [0.0, 1.0, 2.0, 3.0, 4.0]
    y_pred = [4.0, 3.0, 2.0, 1.0, 0.0]
    metrics = compute_metrics(y_true, y_pred)
    assert metrics["plcc"] == pytest.approx(-1.0)
    assert metrics["srocc"] == pytest.approx(-1.0)
    assert metrics["krocc"] == pytest.approx(-1.0)


def test_mse_known_value():
    y_true = [0.0, 0.0, 0.0]
    y_pred = [1.0, 1.0, 1.0]
    metrics = compute_metrics(y_true, y_pred)
    assert metrics["mse"] == pytest.approx(1.0)


def test_metrics_dict_keys():
    metrics = compute_metrics([0, 1, 2], [0, 1, 2])
    assert set(metrics.keys()) == {"mse", "plcc", "srocc", "krocc"}


def test_logistic_mapping_improves_or_preserves_monotonic_fit():
    rng = np.random.default_rng(0)
    y_true = np.linspace(0, 4, 40)
    # monotonic nonlinear compression of the true score, as a synthetic predictor
    y_pred = 1.0 / (1.0 + np.exp(-(y_true - 2.0))) + rng.normal(0, 0.01, size=40)

    raw = compute_metrics(y_true, y_pred)
    mapped = compute_metrics_with_logistic_mapping(y_true, y_pred)

    # rank correlation is invariant to a monotonic mapping
    assert mapped["srocc"] == pytest.approx(raw["srocc"], abs=1e-6)
    # but MSE against the raw (unmapped) scale should improve substantially
    assert mapped["mse"] < raw["mse"]
