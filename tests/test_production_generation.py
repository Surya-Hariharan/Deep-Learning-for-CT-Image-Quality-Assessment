"""Tests for the production dataset generator's config-driven, checkpointed
pipeline (``scripts/quality/generate_production_dataset.py``).

Exercises real logic (grid construction, one-reference generation, manifest
assembly) against tiny synthetic fixtures -- never the real 1,000-image
LDCT-IQAC dataset, and never a full 169,000-image run.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

_SPEC = importlib.util.spec_from_file_location(
    "generate_production_dataset", REPO_ROOT / "scripts" / "quality" / "generate_production_dataset.py"
)
gpd = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(gpd)  # type: ignore[union-attr]

from ct_iqa.data.checkpoint import CheckpointManager  # noqa: E402


def test_build_conditions_has_169_entries_with_one_clean():
    conditions = gpd.build_conditions()
    assert len(conditions) == 169
    assert sum(1 for c in conditions if c.degradation_type == "clean") == 1
    assert sum(1 for c in conditions if c.degradation_type == "noise") == 12
    assert sum(1 for c in conditions if c.degradation_type == "blur") == 12
    assert sum(1 for c in conditions if c.degradation_type == "noise_blur") == 144


# Smallest image size the steerable pyramid used by vif_wavelet(profile="project")
# accepts (see src/ct_iqa/vif/wavelet.py) -- keeps the per-image VIF computation
# fast enough for a unit test while still exercising the real code path, not a
# stub. Real LDCT-IQAC references are 512x512; that scale is covered by
# scripts/preflight_generation.py and scripts/quality/generate_production_dataset.py
# --dry-run, not repeated here.
_MIN_VIF_SIZE = 184


@pytest.fixture
def synthetic_reference_row(tmp_path) -> dict:
    from PIL import Image

    img = np.random.default_rng(0).integers(0, 255, size=(_MIN_VIF_SIZE, _MIN_VIF_SIZE), dtype=np.uint8)
    src_path = tmp_path / "REF_0000.png"
    Image.fromarray(img).save(src_path)
    return {
        "image_id": "REF_0000", "filename": "REF_0000.png",
        "source_path": str(src_path), "label": "2.5",
    }


def test_process_reference_generates_169_records_and_writes_pixels(tmp_path, synthetic_reference_row):
    conditions = gpd.build_conditions()
    out_root = tmp_path / "output"

    records = gpd.process_reference(
        synthetic_reference_row,
        output_root=str(out_root).replace("\\", "/"), references_dir="references",
        degraded_dirs={"noise": "degraded/noise", "blur": "degraded/blur", "noise_blur": "degraded/noise_blur"},
        order="blur_then_noise", sigma_scale=1.0, clip_after=False,
        blur_truncate=4.0, blur_mode="nearest", output_clip_range=(0.0, 255.0),
        vif_variant="vif_wavelet", vif_profile="project", clean_reference_label=1.0,
        seed=20260813, split="train", conditions=conditions,
    )

    assert len(records) == 169
    clean_records = [r for r in records if r["degradation_type"] == "clean"]
    assert len(clean_records) == 1
    assert clean_records[0]["vif_score"] == 1.0
    assert clean_records[0]["diagnostic_status"] == "clean_reference_identity_not_computed"

    degraded_records = [r for r in records if r["degradation_type"] != "clean"]
    assert len(degraded_records) == 168
    assert all(r["diagnostic_status"] for r in degraded_records)
    assert all(np.isfinite(r["vif_score"]) for r in degraded_records)

    assert (out_root / "references" / "REF_0000.png").exists()
    assert len(list((out_root / "degraded" / "noise").glob("*.png"))) == 12
    assert len(list((out_root / "degraded" / "blur").glob("*.png"))) == 12
    assert len(list((out_root / "degraded" / "noise_blur").glob("*.png"))) == 144


def test_process_reference_is_deterministic(tmp_path, synthetic_reference_row):
    # A handful of conditions is enough to prove determinism; the full-169
    # code path is already covered by test_process_reference_generates_*.
    conditions = [c for c in gpd.build_conditions() if c.degradation_type != "noise_blur"][:6]
    kwargs = dict(
        output_root=str((tmp_path / "out1")).replace("\\", "/"), references_dir="references",
        degraded_dirs={"noise": "degraded/noise", "blur": "degraded/blur", "noise_blur": "degraded/noise_blur"},
        order="blur_then_noise", sigma_scale=1.0, clip_after=False,
        blur_truncate=4.0, blur_mode="nearest", output_clip_range=(0.0, 255.0),
        vif_variant="vif_wavelet", vif_profile="project", clean_reference_label=1.0,
        seed=20260813, split="train", conditions=conditions,
    )
    records_a = gpd.process_reference(synthetic_reference_row, **kwargs)

    kwargs["output_root"] = str((tmp_path / "out2")).replace("\\", "/")
    records_b = gpd.process_reference(synthetic_reference_row, **kwargs)

    vif_a = [r["vif_score"] for r in records_a]
    vif_b = [r["vif_score"] for r in records_b]
    assert vif_a == vif_b


def test_checkpoint_skips_already_completed_reference(tmp_path, synthetic_reference_row):
    conditions = [c for c in gpd.build_conditions() if c.degradation_type in ("clean", "noise")][:4]
    out_root = tmp_path / "output"
    records = gpd.process_reference(
        synthetic_reference_row,
        output_root=str(out_root).replace("\\", "/"), references_dir="references",
        degraded_dirs={"noise": "degraded/noise", "blur": "degraded/blur", "noise_blur": "degraded/noise_blur"},
        order="blur_then_noise", sigma_scale=1.0, clip_after=False,
        blur_truncate=4.0, blur_mode="nearest", output_clip_range=(0.0, 255.0),
        vif_variant="vif_wavelet", vif_profile="project", clean_reference_label=1.0,
        seed=20260813, split="train", conditions=conditions,
    )

    mgr = CheckpointManager(
        tmp_path / "checkpoint", config_hash_value="h", code_commit_hash_value=None,
        seed=20260813, total_references=1,
    )
    mgr.write_shard_and_mark_complete("REF_0000", records)

    mgr2 = CheckpointManager(
        tmp_path / "checkpoint", config_hash_value="h", code_commit_hash_value=None,
        seed=20260813, total_references=1,
    )
    assert mgr2.is_reference_complete("REF_0000", expected_n_records=len(conditions)) is True


def test_assemble_final_manifest_streams_all_records(tmp_path):
    mgr = CheckpointManager(
        tmp_path / "checkpoint", config_hash_value="h", code_commit_hash_value=None,
        seed=1, total_references=2,
    )
    for ref_id in ("REF_A", "REF_B"):
        rows = [
            {
                "reference_id": ref_id, "source_filename": f"{ref_id}.png", "expert_score": "2.0",
                "degraded_id": f"{ref_id}_c{i}", "degradation_type": "noise", "noise_sigma": "1.0",
                "blur_sigma": "", "vif_score": "0.7", "diagnostic_status": "ok",
                "max_channel_gain": "1.0", "covariance_condition_number": "5.0", "split": "train",
            }
            for i in range(3)
        ]
        mgr.write_shard_and_mark_complete(ref_id, rows)

    deg_cfg = {"output": {"root": "data/processed/quality_iqa"}}
    import ct_iqa.config.loader as loader_mod

    original_manifests_dir = loader_mod.manifests_dir
    loader_mod.manifests_dir = lambda: tmp_path / "manifests"
    gpd.manifests_dir = loader_mod.manifests_dir
    try:
        n = gpd.assemble_final_manifest(mgr, deg_cfg, seed=1, cfg_hash="h", commit_hash=None)
    finally:
        loader_mod.manifests_dir = original_manifests_dir

    assert n == 6
    out_csv = tmp_path / "manifests" / "quality_iqa_manifest.csv"
    assert out_csv.exists()
    content = out_csv.read_text(encoding="utf-8")
    assert content.count("\n") == 7  # header + 6 rows (+ trailing newline)
