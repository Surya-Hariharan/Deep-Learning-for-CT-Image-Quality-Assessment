import json

import numpy as np
import pytest
import torch
from PIL import Image

from ct_iqa.data.ldct_iqac import (
    LDCTIQACDataset,
    LDCTIQACLabelError,
    denormalize_score,
    normalize_score,
)


def _write_float_tiff(path, value=0.5, size=(16, 16)):
    array = np.full(size, value, dtype=np.float32)
    Image.fromarray(array, mode="F").save(path)


@pytest.fixture
def clean_dataset_dir(tmp_path):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    labels = {}
    for i in range(5):
        fn = f"{i:04d}.tif"
        _write_float_tiff(image_dir / fn, value=i / 10.0)
        labels[fn] = float(i)  # scores 0..4, matches real dataset's [0,4] range
    json_path = tmp_path / "labels.json"
    json_path.write_text(json.dumps(labels))
    return image_dir, json_path


def test_dataset_length_matches_json(clean_dataset_dir):
    image_dir, json_path = clean_dataset_dir
    ds = LDCTIQACDataset(image_dir, json_path, image_size=32)
    assert len(ds) == 5


def test_image_label_matching(clean_dataset_dir):
    image_dir, json_path = clean_dataset_dir
    ds = LDCTIQACDataset(image_dir, json_path, image_size=32)
    scores = sorted(ds.raw_scores)
    assert scores == [0.0, 1.0, 2.0, 3.0, 4.0]


def test_image_tensor_shape(clean_dataset_dir):
    image_dir, json_path = clean_dataset_dir
    ds = LDCTIQACDataset(image_dir, json_path, image_size=32)
    image, score = ds[0]
    assert image.shape == (1, 32, 32)
    assert image.dtype == torch.float32
    assert score.ndim == 0


def test_label_range(clean_dataset_dir):
    image_dir, json_path = clean_dataset_dir
    ds = LDCTIQACDataset(image_dir, json_path, image_size=32)
    for score in ds.raw_scores:
        assert 0.0 <= score <= 4.0


def test_missing_label_raises(tmp_path):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    _write_float_tiff(image_dir / "0000.tif")
    _write_float_tiff(image_dir / "0001.tif")  # no matching label
    json_path = tmp_path / "labels.json"
    json_path.write_text(json.dumps({"0000.tif": 1.0}))

    with pytest.raises(LDCTIQACLabelError, match="0001.tif"):
        LDCTIQACDataset(image_dir, json_path)


def test_missing_image_raises(tmp_path):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    _write_float_tiff(image_dir / "0000.tif")
    json_path = tmp_path / "labels.json"
    json_path.write_text(json.dumps({"0000.tif": 1.0, "0001.tif": 2.0}))

    with pytest.raises(LDCTIQACLabelError, match="0001.tif"):
        LDCTIQACDataset(image_dir, json_path)


def test_malformed_json_raises(tmp_path):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    json_path = tmp_path / "labels.json"
    json_path.write_text("{not valid json")

    with pytest.raises(LDCTIQACLabelError):
        LDCTIQACDataset(image_dir, json_path)


def test_non_object_json_raises(tmp_path):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    json_path = tmp_path / "labels.json"
    json_path.write_text(json.dumps([1, 2, 3]))

    with pytest.raises(LDCTIQACLabelError):
        LDCTIQACDataset(image_dir, json_path)


def test_ds_store_is_ignored(clean_dataset_dir):
    image_dir, json_path = clean_dataset_dir
    (image_dir / ".DS_Store").write_bytes(b"\x00")
    ds = LDCTIQACDataset(image_dir, json_path, image_size=32)
    assert len(ds) == 5


def test_normalize_denormalize_roundtrip():
    for raw in [0.0, 1.6, 2.8333333333333335, 4.0]:
        normalized = normalize_score(raw)
        assert 0.0 <= normalized <= 1.0
        assert denormalize_score(normalized) == pytest.approx(raw)
