"""Tests for `ct_iqa.data.splits` and `ct_iqa.data.loader`, extracted from
`ct_iqa.training.trainer` (see docs/repository_architecture_audit.md,
section 3/6).
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from PIL import Image

from ct_iqa.config import ExperimentConfig
from ct_iqa.data.loader import build_dataloaders, build_test_dataloader
from ct_iqa.data.splits import train_val_split


class _ListDataset:
    def __init__(self, n):
        self._n = n

    def __len__(self):
        return self._n

    def __getitem__(self, i):
        return i


def test_train_val_split_sizes():
    ds = _ListDataset(100)
    train, val = train_val_split(ds, val_fraction=0.1, seed=0)
    assert len(train) + len(val) == 100
    assert len(val) == 10


def test_train_val_split_is_reproducible_given_same_seed():
    ds = _ListDataset(100)
    train_a, val_a = train_val_split(ds, val_fraction=0.2, seed=42)
    train_b, val_b = train_val_split(ds, val_fraction=0.2, seed=42)
    assert list(train_a.indices) == list(train_b.indices)
    assert list(val_a.indices) == list(val_b.indices)


def test_train_val_split_differs_across_seeds():
    ds = _ListDataset(100)
    _, val_a = train_val_split(ds, val_fraction=0.2, seed=1)
    _, val_b = train_val_split(ds, val_fraction=0.2, seed=2)
    assert list(val_a.indices) != list(val_b.indices)


@pytest.fixture
def dataset_config(tmp_path):
    def _write_float_tiff(path, value, size=(32, 32)):
        array = np.full(size, value, dtype=np.float32)
        Image.fromarray(array, mode="F").save(path)

    train_dir = tmp_path / "train_images"
    train_dir.mkdir()
    train_labels = {}
    for i in range(20):
        fn = f"{i:04d}.tif"
        _write_float_tiff(train_dir / fn, value=i / 20.0)
        train_labels[fn] = float(i % 5)
    train_json = tmp_path / "train.json"
    train_json.write_text(json.dumps(train_labels))

    test_dir = tmp_path / "test_images"
    test_dir.mkdir()
    test_labels = {}
    for i in range(6):
        fn = f"test{i:04d}.tif"
        _write_float_tiff(test_dir / fn, value=i / 6.0)
        test_labels[fn] = float(i % 5)
    test_json = tmp_path / "test.json"
    test_json.write_text(json.dumps(test_labels))

    return ExperimentConfig(
        train_image_dir=str(train_dir),
        train_json_path=str(train_json),
        test_image_dir=str(test_dir),
        test_json_path=str(test_json),
        image_size=32,
        batch_size=4,
        val_fraction=0.25,
        seed=0,
    )


def test_build_dataloaders_sizes(dataset_config):
    train_loader, val_loader = build_dataloaders(dataset_config)
    assert len(train_loader.dataset) + len(val_loader.dataset) == 20
    assert len(val_loader.dataset) == 5  # 0.25 * 20


def test_build_test_dataloader_size(dataset_config):
    test_loader = build_test_dataloader(dataset_config)
    assert len(test_loader.dataset) == 6
