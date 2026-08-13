"""Tests for ct_iqa.data.manifest: the manifest parser/writer, and the
guarantee that missing fields stay ``None`` rather than being invented."""

from __future__ import annotations

import json

from ct_iqa.data.manifest import DegradedRecord, ImageRecord, ManifestHeader, read_manifest_csv, write_manifest


def test_write_manifest_roundtrip(tmp_path):
    records = [
        ImageRecord(dataset_name="demo", image_id="a", filename="a.png", source_path="data/raw/demo/a.png"),
        ImageRecord(dataset_name="demo", image_id="b", filename="b.png", source_path="data/raw/demo/b.png", label=1.0),
    ]
    written = write_manifest(records, ManifestHeader(dataset_name="demo"), "demo", out_dir=tmp_path)
    assert written["csv"].is_file() and written["header"].is_file()

    header = json.loads(written["header"].read_text(encoding="utf-8"))
    assert header["n_records"] == 2
    assert header["dataset_name"] == "demo"

    rows = read_manifest_csv(written["csv"])
    assert len(rows) == 2
    assert rows[0]["image_id"] == "a"
    assert rows[1]["label"] == "1.0"


def test_unknown_values_stay_none():
    """An unknown field must remain None, never a plausible substitute."""
    record = ImageRecord(dataset_name="demo", image_id="x", filename="x.png", source_path="p.png")
    assert record.spacing is None
    assert record.patient_id is None
    assert record.slice_thickness is None


def test_manifest_header_records_seed_and_generator(tmp_path):
    header = ManifestHeader(dataset_name="demo", generator="scripts/build_manifest.py", seed=42)
    written = write_manifest([], header, "empty", out_dir=tmp_path, record_type=ImageRecord)
    saved = json.loads(written["header"].read_text(encoding="utf-8"))
    assert saved["generator"] == "scripts/build_manifest.py"
    assert saved["seed"] == 42
    assert saved["n_records"] == 0


def test_degraded_record_keeps_expert_score_and_vif_score_separate():
    """The core dataset-adaptation guarantee: two distinct label fields."""
    record = DegradedRecord(
        reference_id="0000",
        degraded_id="0000_noise_6",
        degradation_type="noise",
        noise_sigma=6.0,
        expert_score=2.8,
        vif_score=None,
    )
    assert record.expert_score == 2.8
    assert record.vif_score is None
    assert record.predicted_score is None
    assert record.calibrated_score is None


def test_degraded_record_manifest_roundtrip_preserves_both_scores(tmp_path):
    records = [
        DegradedRecord(
            reference_id="0000", degraded_id="0000_clean", degradation_type="clean",
            expert_score=2.8, vif_score=1.0, split="train",
        ),
    ]
    written = write_manifest(
        records, ManifestHeader(dataset_name="quality_iqa"), "quality_iqa", out_dir=tmp_path,
        record_type=DegradedRecord,
    )
    rows = read_manifest_csv(written["csv"])
    assert rows[0]["expert_score"] == "2.8"
    assert rows[0]["vif_score"] == "1.0"


def test_source_paths_in_manifest_never_contain_absolute_windows_paths(tmp_path):
    records = [ImageRecord(dataset_name="demo", image_id="a", filename="a.png", source_path="data/raw/demo/a.png")]
    written = write_manifest(records, ManifestHeader(dataset_name="demo"), "demo", out_dir=tmp_path)
    text = written["csv"].read_text(encoding="utf-8")
    assert ":\\" not in text and "C:/" not in text
