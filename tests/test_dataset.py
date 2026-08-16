"""PNG dataset path resolution and manifest handling, using synthetic
fixtures -- the full dataset is never required for these tests."""

import json

import numpy as np
import pytest
import SimpleITK as sitk

from quality_aware_lung_ct.data.png.metadata import (
    get_split,
    load_manifest,
    resolve_series_path,
)
from quality_aware_lung_ct.data.png.dataset import PNGDataset, split_data_list


def _write_volume(path, shape=(8, 8, 8)):
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.random.randint(-1000, 1000, size=shape).astype(np.int16)
    sitk.WriteImage(sitk.GetImageFromArray(arr), str(path))


@pytest.fixture
def png_root(tmp_path):
    root = tmp_path / "png"
    # Manifest references `.nii.gz`; the file actually shipped is `.nii`,
    # reproducing the real PNG dataset's documented mismatch.
    _write_volume(root / "images" / "P1" / "01" / "t0_01_img.nii")
    _write_volume(root / "masks" / "P1" / "01" / "t0_01_msk.nii")

    manifest = {
        "training": [{
            "info": {"PatientId": "P1", "NoduleId": 1, "t0": 20200101,
                     "t1": 20200601, "t2": 20201101, "Months1": 5, "Months2": 5},
            "series": [{"image": "images/P1/01/t0_01_img.nii.gz",
                       "label": "masks/P1/01/t0_01_msk.nii.gz"}],
        }],
        "test": [],
    }
    (root / "NGP3T.json").write_text(json.dumps(manifest))
    return root


def test_load_manifest_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_manifest(tmp_path, "NGP3T.json")


def test_get_split_missing_raises():
    with pytest.raises(ValueError):
        get_split({"training": []}, "training")


def test_resolve_series_path_handles_nii_gz_mismatch(png_root):
    resolved = resolve_series_path(png_root, "images/P1/01/t0_01_img.nii.gz")
    assert resolved.name == "t0_01_img.nii"
    assert resolved.is_file()


def test_resolve_series_path_missing_raises(png_root):
    with pytest.raises(FileNotFoundError):
        resolve_series_path(png_root, "images/P1/01/does_not_exist.nii.gz")


def test_split_data_list_is_disjoint_and_complete():
    data = [{"id": i} for i in range(10)]
    train, val = split_data_list(data, num_folds=5, fold=0)
    assert len(train) + len(val) == len(data)
    assert not (set(d["id"] for d in train) & set(d["id"] for d in val))


def test_png_dataset_getitem_resolves_and_loads(png_root):
    manifest = load_manifest(png_root, "NGP3T.json")
    entries = get_split(manifest, "training")

    def passthrough(images, labels):
        return images, labels

    dataset = PNGDataset(png_root, entries, transforms=passthrough)
    assert len(dataset) == 1
    # single series in this fixture, so `*images`/`*labels` each unpack to one item
    image, label, month1, month2, info = dataset[0]
    assert image.endswith("t0_01_img.nii")
    assert label.endswith("t0_01_msk.nii")
    assert (month1, month2) == (5, 5)
    assert info["PatientId"] == "P1"
