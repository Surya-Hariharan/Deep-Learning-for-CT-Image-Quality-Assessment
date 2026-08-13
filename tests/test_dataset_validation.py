"""Dataset structure/validation tests.

Tests that depend on a dataset being downloaded skip (rather than fail) when it
is absent, so the suite is meaningful both before and after acquisition.
"""

from __future__ import annotations

import json

import pytest

from ct_iqa.config.loader import dataset_config, manifests_dir, project_root
from ct_iqa.data.validation import ScanResult, scan_directory, sniff_format
from ct_iqa.utils.paths import resolve

DATASETS = ["png", "lidc_idri", "luna16", "lndb", "deeplesion", "cq500", "ldctiqa"]


# ------------------------------------------------------------------ structure

def test_project_root_found():
    assert (project_root() / "configs" / "paths.yaml").is_file()


@pytest.mark.parametrize("name", DATASETS)
def test_dataset_config_exists_and_declares_role(name):
    cfg = dataset_config(name)
    assert cfg["name"] == name
    assert cfg.get("role"), f"{name} must declare a research role"
    assert cfg.get("raw_root", "").startswith("data/raw/")


@pytest.mark.parametrize("name", DATASETS)
def test_raw_directory_exists(name):
    assert resolve(dataset_config(name)["raw_root"]).is_dir()


def test_dataset_roles_are_distinct_per_dataset():
    """Datasets must keep separate identities, not be merged into one pool."""
    roots = [dataset_config(n)["raw_root"] for n in DATASETS]
    assert len(set(roots)) == len(roots)


# ---------------------------------------------------------------- validation

def test_sniff_format_detects_png(tmp_path):
    png = tmp_path / "x.png"
    png.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 64)
    assert sniff_format(png) == "png"


def test_sniff_format_detects_dicom(tmp_path):
    dcm = tmp_path / "x.dcm"
    dcm.write_bytes(b"\x00" * 128 + b"DICM" + b"\x00" * 32)
    assert sniff_format(dcm) == "dicom"


def test_sniff_format_ignores_misleading_suffix(tmp_path):
    """A file named .png that is really a zip must be reported as a zip."""
    fake = tmp_path / "archive.png"
    fake.write_bytes(b"PK\x03\x04" + b"\x00" * 64)
    assert sniff_format(fake) == "zip"


def test_scan_missing_directory_is_not_an_error(tmp_path):
    result = scan_directory(tmp_path / "does_not_exist")
    assert isinstance(result, ScanResult)
    assert not result.exists and result.n_files == 0


def test_scan_flags_zero_byte_files(tmp_path):
    (tmp_path / "empty.png").touch()
    result = scan_directory(tmp_path)
    assert any("zero-byte" in detail for _, detail in result.corrupt)


def test_scan_ignores_gitkeep(tmp_path):
    (tmp_path / ".gitkeep").touch()
    assert scan_directory(tmp_path).n_files == 0


# -------------------------------------------------------------------- ldctiqa

def _ldctiqa_available() -> bool:
    return resolve(dataset_config("ldctiqa")["images_dir"]).is_dir()


@pytest.mark.skipif(not _ldctiqa_available(), reason="ldctiqa not present")
def test_ldctiqa_image_count_matches_config():
    cfg = dataset_config("ldctiqa")
    n = len(list(resolve(cfg["images_dir"]).glob("*.png")))
    assert n == cfg["format"]["n_images"]


@pytest.mark.skipif(not _ldctiqa_available(), reason="ldctiqa not present")
def test_ldctiqa_labels_cover_every_image():
    cfg = dataset_config("ldctiqa")
    labels = json.loads(resolve(cfg["labels_file"]).read_text(encoding="utf-8"))
    files = {p.name for p in resolve(cfg["images_dir"]).glob("*.png")}
    assert set(labels) == files


@pytest.mark.skipif(not _ldctiqa_available(), reason="ldctiqa not present")
def test_ldctiqa_labels_within_declared_scale():
    cfg = dataset_config("ldctiqa")
    lo, hi = cfg["labels"]["scale"]
    labels = json.loads(resolve(cfg["labels_file"]).read_text(encoding="utf-8"))
    assert all(lo <= v <= hi for v in labels.values())


@pytest.mark.skipif(not _ldctiqa_available(), reason="ldctiqa not present")
def test_ldctiqa_dimensions_are_uniform():
    from PIL import Image

    cfg = dataset_config("ldctiqa")
    expected = (cfg["format"]["width"], cfg["format"]["height"])
    for path in sorted(resolve(cfg["images_dir"]).glob("*.png"))[:50]:
        with Image.open(path) as im:
            assert im.size == expected


@pytest.mark.skipif(not _ldctiqa_available(), reason="ldctiqa not present")
def test_ldctiqa_patient_level_split_stays_blocked():
    """No patient ids means patient-level splitting stays impossible."""
    cfg = dataset_config("ldctiqa")
    assert cfg["identifiers"]["patient_id"] is None
    assert "BLOCKED" in cfg["splits"]["patient_level"]["strategy"]


@pytest.mark.skipif(not _ldctiqa_available(), reason="ldctiqa not present")
def test_ldctiqa_reference_level_split_is_enabled():
    """Reference-level splitting (the Ohashi split unit) does not need patient ids."""
    cfg = dataset_config("ldctiqa")
    ref_split = cfg["splits"]["reference_level"]
    assert ref_split["strategy"] == "reference_level"
    assert ref_split["unit"] == "image_id"
    assert ref_split["ratios"] == {"train": 0.6, "val": 0.2, "test": 0.2}


def _manifest_available() -> bool:
    return (manifests_dir() / "ldctiqa_manifest.csv").exists()


@pytest.mark.skipif(not _manifest_available(), reason="manifests not built yet")
def test_manifest_paths_are_relative():
    """Manifests must never contain machine-specific absolute paths."""
    for name in DATASETS:
        path = manifests_dir() / f"{name}_manifest.csv"
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        assert ":\\" not in text and "C:/" not in text, f"{name} manifest has absolute paths"


@pytest.mark.skipif(not _manifest_available(), reason="manifests not built yet")
def test_manifest_header_records_provenance():
    header_path = manifests_dir() / "ldctiqa_manifest.header.json"
    header = json.loads(header_path.read_text(encoding="utf-8"))
    assert header["dataset_name"] == "ldctiqa"
    assert header["generated_at"] and header["generator"]
    assert header["seed"] is not None, "manifests must record the seed used"
