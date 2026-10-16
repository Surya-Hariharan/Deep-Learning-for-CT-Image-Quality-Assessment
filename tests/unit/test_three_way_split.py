"""Unit tests for the study-v2 800/100/100 split (SYNTHETIC filenames and labels)."""

from __future__ import annotations

import copy

import pytest

from ct_iqa.data.splits import (
    SplitError, build_manifest, load_manifest, partition_indices, save_manifest,
    three_way_split, verify_manifest,
)

LABELS = {f"{i:04d}.tif": (i % 21) / 5.0 for i in range(1000)}


def test_sizes_cover_and_disjoint():
    p = three_way_split(LABELS.keys(), 100, 100, seed=7)
    assert [len(p[k]) for k in ("train", "validation", "calibration")] == [800, 100, 100]
    allnames = p["train"] + p["validation"] + p["calibration"]
    assert len(set(allnames)) == 1000 == len(allnames) and set(allnames) == set(LABELS)


def test_deterministic_and_seed_sensitive_and_order_independent():
    a = three_way_split(LABELS.keys(), 100, 100, seed=7)
    assert a == three_way_split(list(reversed(list(LABELS))), 100, 100, seed=7)
    assert a != three_way_split(LABELS.keys(), 100, 100, seed=8)


def test_manifest_roundtrip_and_never_overwritten(tmp_path):
    m = build_manifest(LABELS, 100, 100, 7, "unit test")
    path = tmp_path / "split.json"
    save_manifest(m, path)
    assert load_manifest(path, LABELS)["manifest_sha256"] == m["manifest_sha256"]
    with pytest.raises(FileExistsError):
        save_manifest(m, path)


def test_verify_detects_overlap_tampering_and_missing_images():
    m = build_manifest(LABELS, 100, 100, 7, "unit test")
    bad = copy.deepcopy(m); bad["validation"][0] = bad["train"][0]
    with pytest.raises(SplitError, match="overlap|duplicate"):
        verify_manifest(bad, LABELS)
    bad = copy.deepcopy(m); bad["calibration"][0], bad["calibration"][1] = bad["calibration"][1], "x.tif"
    with pytest.raises(SplitError):
        verify_manifest(bad, LABELS)
    with pytest.raises(SplitError, match="cover"):
        verify_manifest(m, {**LABELS, "extra.tif": 1.0})


def test_partition_indices_follow_dataset_order():
    m = build_manifest(LABELS, 100, 100, 7, "unit test")
    names = sorted(LABELS)
    idx = partition_indices(names, m, "calibration")
    assert [names[i] for i in idx] == sorted(m["calibration"]) and idx == sorted(idx)
    with pytest.raises(SplitError):
        partition_indices(names[:-50], m, "train")
