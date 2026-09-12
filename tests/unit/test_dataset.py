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
        _write_float_tiff(image_dir / fn, value=i / 10.0, size=(64, 64))
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


# --- Center crop (not resize) ---


def _write_gradient_tiff(path, size=(64, 64)):
    """Write a TIFF whose pixel value at (row, col) == row (0..size[0]-1), scaled to [0,1].

    Lets a test assert exactly which rows survived a center crop -- a resize
    would instead interpolate/blend rows, so this also distinguishes crop
    from resize behaviorally, not just by output shape.
    """
    h, w = size
    rows = np.arange(h, dtype=np.float32) / (h - 1)
    array = np.repeat(rows[:, None], w, axis=1)
    Image.fromarray(array, mode="F").save(path)


def test_getitem_center_crops_not_resizes(tmp_path):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    _write_gradient_tiff(image_dir / "0000.tif", size=(64, 64))
    json_path = tmp_path / "labels.json"
    json_path.write_text(json.dumps({"0000.tif": 2.0}))

    crop_size = 32
    ds = LDCTIQACDataset(image_dir, json_path, image_size=crop_size)
    image, _ = ds[0]

    assert image.shape == (1, crop_size, crop_size)

    # The source is a 64x64 row-gradient in [0,1]; a center crop of 32
    # keeps rows [16:48) untouched (mapped to [-1,1]), so the first row of
    # the crop must equal the source's row 16, exactly -- no interpolation.
    expected_first_row_raw = 16 / 63.0
    expected_first_row_normalized = 2.0 * expected_first_row_raw - 1.0
    assert image[0, 0, 0].item() == pytest.approx(expected_first_row_normalized, abs=1e-5)

    # A resize (the old behavior) would instead compress all 64 source rows
    # into 32 output rows, so row 0 of the output would come from source
    # row 0 (value 0.0), not source row 16. Guard against a regression back
    # to resize by checking the crop's rows differ from a naive resize's.
    assert image[0, 0, 0].item() != pytest.approx(2.0 * 0.0 - 1.0, abs=1e-5)


def test_center_crop_is_exactly_centered(tmp_path):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    _write_gradient_tiff(image_dir / "0000.tif", size=(64, 64))
    json_path = tmp_path / "labels.json"
    json_path.write_text(json.dumps({"0000.tif": 2.0}))

    ds = LDCTIQACDataset(image_dir, json_path, image_size=32)
    image, _ = ds[0]

    # Last row of a centered 32-crop out of 64 rows is source row 47.
    expected_last_row_raw = 47 / 63.0
    expected_last_row_normalized = 2.0 * expected_last_row_raw - 1.0
    assert image[0, -1, 0].item() == pytest.approx(expected_last_row_normalized, abs=1e-5)


def test_center_crop_matches_real_dataset_dimensions(tmp_path):
    """512x512 source -> 224x224 center crop, matching the real LDCT-IQAC/Ohashi setup."""
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    _write_float_tiff(image_dir / "0000.tif", value=0.5, size=(512, 512))
    json_path = tmp_path / "labels.json"
    json_path.write_text(json.dumps({"0000.tif": 2.0}))

    ds = LDCTIQACDataset(image_dir, json_path, image_size=224)
    image, _ = ds[0]
    assert image.shape == (1, 224, 224)


def test_crop_size_larger_than_source_raises(tmp_path):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    _write_float_tiff(image_dir / "0000.tif", value=0.5, size=(16, 16))
    json_path = tmp_path / "labels.json"
    json_path.write_text(json.dumps({"0000.tif": 2.0}))

    ds = LDCTIQACDataset(image_dir, json_path, image_size=32)
    with pytest.raises(LDCTIQACLabelError, match="smaller than"):
        ds[0]


# --- RadImageNet [-1, 1] normalization ---


def test_getitem_normalizes_to_minus_one_one_range(clean_dataset_dir):
    image_dir, json_path = clean_dataset_dir
    ds = LDCTIQACDataset(image_dir, json_path, image_size=32)
    for i in range(len(ds)):
        image, _ = ds[i]
        assert torch.all(image >= -1.0)
        assert torch.all(image <= 1.0)


def test_getitem_normalization_formula_is_2x_minus_1(tmp_path):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    json_path = tmp_path / "labels.json"
    labels = {}
    # Known raw pixel values spanning the documented [0,1] source range.
    raw_values = [0.0, 0.25, 0.5, 0.75, 1.0]
    for i, v in enumerate(raw_values):
        fn = f"{i:04d}.tif"
        _write_float_tiff(image_dir / fn, value=v, size=(32, 32))
        labels[fn] = 2.0
    json_path.write_text(json.dumps(labels))

    ds = LDCTIQACDataset(image_dir, json_path, image_size=32)
    for i, v in enumerate(raw_values):
        image, _ = ds[i]
        expected = 2.0 * v - 1.0
        assert torch.allclose(image, torch.full_like(image, expected), atol=1e-6)


def test_raw_score_unaffected_by_image_normalization(clean_dataset_dir):
    """Image pixel normalization must not touch the label/score space."""
    image_dir, json_path = clean_dataset_dir
    ds = LDCTIQACDataset(image_dir, json_path, image_size=32)
    _, score = ds[0]
    assert 0.0 <= score.item() <= 4.0
