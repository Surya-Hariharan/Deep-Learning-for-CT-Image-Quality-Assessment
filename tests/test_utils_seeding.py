"""Tests for ct_iqa.utils.reproducibility."""

from __future__ import annotations

import random

import numpy as np

from ct_iqa.utils.reproducibility import seed_everything


def test_seed_everything_makes_random_reproducible():
    seed_everything(123)
    a = [random.random() for _ in range(5)]
    seed_everything(123)
    b = [random.random() for _ in range(5)]
    assert a == b


def test_seed_everything_makes_numpy_reproducible():
    seed_everything(123)
    a = np.random.rand(5)
    seed_everything(123)
    b = np.random.rand(5)
    np.testing.assert_array_equal(a, b)


def test_different_seeds_diverge():
    seed_everything(1)
    a = np.random.rand(5)
    seed_everything(2)
    b = np.random.rand(5)
    assert not np.array_equal(a, b)
