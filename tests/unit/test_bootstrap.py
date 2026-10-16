"""Unit tests for the paired bootstrap (SYNTHETIC predictions)."""

from __future__ import annotations

import numpy as np
import pytest

from ct_iqa.evaluation.bootstrap import paired_bootstrap_difference


def _data(n=300, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.uniform(0, 4, n)
    return y, rng


def test_identical_models_have_zero_difference_and_degenerate_ci():
    y, rng = _data()
    p = y + rng.normal(0, 0.5, len(y))
    r = paired_bootstrap_difference(y, p, p.copy(), n_boot=200, seed=1)
    for m in r.values():
        assert m["difference_b_minus_a"] == 0 and m["ci_low"] == 0 == m["ci_high"]


def test_clearly_better_model_has_ci_excluding_zero_and_is_deterministic():
    y, rng = _data()
    a = y + rng.normal(0, 1.5, len(y))
    b = y + rng.normal(0, 0.3, len(y))
    r1 = paired_bootstrap_difference(y, a, b, n_boot=300, seed=2)
    r2 = paired_bootstrap_difference(y, a, b, n_boot=300, seed=2)
    assert r1 == r2
    assert r1["plcc"]["ci_low"] > 0 and r1["rmse"]["ci_high"] < 0
    assert r1["plcc"]["fraction_positive"] > 0.99


def test_length_mismatch_rejected():
    with pytest.raises(ValueError):
        paired_bootstrap_difference([1, 2, 3], [1, 2], [1, 2, 3])
