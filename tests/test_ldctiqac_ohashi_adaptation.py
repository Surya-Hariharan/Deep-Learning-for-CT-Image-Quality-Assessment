"""Tests for the LDCT-IQAC adaptation of the Ohashi synthetic pipeline.

These exercise scripts/quality/build_ohashi_dataset.py's planning functions
directly (no subprocess, no pixel generation) against the real 1,000-image
LDCT-IQAC manifest, and assert the project's two non-negotiable rules for this
adaptation:

1. The dataset total and split sizes match the 1,000-reference arithmetic
   (169,000 / 101,400 / 33,800 / 33,800), not the original 105-reference one.
2. expert_score and vif_score never collapse into each other.

Skips if the manifest has not been built yet.
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import pytest

from ct_iqa.config.loader import load_config, manifests_dir
from ct_iqa.data.splitting import check_group_leakage

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "quality"))
import build_ohashi_dataset as bod  # noqa: E402

CONFIG_PATH = "configs/quality/ohashi_ldctiqac.yaml"


def _manifest_available() -> bool:
    return (manifests_dir() / "ldctiqa_manifest.csv").exists()


pytestmark = pytest.mark.skipif(not _manifest_available(), reason="ldctiqa_manifest.csv not built yet")


@pytest.fixture(scope="module")
def cfg():
    return load_config(CONFIG_PATH)


@pytest.fixture(scope="module")
def references(cfg):
    return bod.load_references(cfg)


@pytest.fixture(scope="module")
def records(cfg, references):
    return bod.plan_records(cfg, references)


def test_config_declares_the_adapted_counts(cfg):
    assert cfg["references"]["n_total"] == 1000
    assert cfg["counts"]["dataset_total"] == 169000
    assert cfg["splits"]["reference_counts"] == {"train": 600, "val": 200, "test": 200}
    assert cfg["splits"]["image_counts"] == {"train": 101400, "val": 33800, "test": 33800}


def test_all_1000_real_references_are_loaded(references):
    assert len(references) == 1000
    assert len({r["image_id"] for r in references}) == 1000


def test_planned_dataset_matches_1000_reference_arithmetic(records):
    assert len(records) == 169000
    assert len({r.reference_id for r in records}) == 1000

    by_type = Counter(r.degradation_type for r in records)
    assert by_type == {"clean": 1000, "noise": 12000, "blur": 12000, "noise_blur": 144000}


def test_planned_split_sizes_are_exact(records):
    by_split = Counter(r.split for r in records)
    assert by_split == {"train": 101400, "val": 33800, "test": 33800}


def test_planned_split_has_zero_reference_leakage(records):
    report = check_group_leakage([r.reference_id for r in records], [r.split for r in records])
    assert report.ok
    assert report.n_groups == 1000


def test_expert_score_is_carried_and_constant_per_reference(records):
    """Every one of a reference's 169 variants must carry the same expert_score."""
    by_ref: dict[str, set] = {}
    for r in records:
        by_ref.setdefault(r.reference_id, set()).add(r.expert_score)
    assert all(len(scores) == 1 for scores in by_ref.values())


def test_expert_score_and_vif_score_never_conflated(records):
    """vif_score must stay null at planning time; expert_score must not be null
    for any reference whose label was present in the source manifest."""
    assert all(r.vif_score is None for r in records)
    labelled = sum(1 for r in records if r.expert_score is not None)
    assert labelled == 169000  # LDCT-IQAC has 100% label coverage (verified in docs/dataset_audit.md)


def test_expert_score_stays_within_ldctiqac_scale(records):
    assert all(0.0 <= r.expert_score <= 4.0 for r in records if r.expert_score is not None)


def test_verify_plan_reports_no_problems(cfg, records):
    assert bod.verify_plan(cfg, records) == []


def test_preflight_blockers_are_all_methodology_not_data(cfg, references):
    """Data must not be a blocker (it's present); only NOT SPECIFIED BY OHASHI
    methodology choices should remain."""
    blockers = bod.preflight(cfg, references)
    assert not any(b.startswith("DATA:") for b in blockers)
    assert all(b.startswith("NOT SPECIFIED BY OHASHI") for b in blockers)


def test_methodology_blockers_are_now_resolved(cfg, references):
    """As of the 2026-08-14 production decision (docs/research_decisions.md,
    decisions A-16/A-17/A-18), sigma_units/combination-order/vif_variant are
    resolved PROJECT ADAPTATION values, not open blockers -- this legacy
    planning script's preflight() must reflect that. Real production
    generation itself now lives in
    scripts/quality/generate_production_dataset.py (with checkpointing),
    not this skeleton's --execute path, which remains an unimplemented
    placeholder -- see that module's docstring."""
    assert bod.preflight(cfg, references) == []
