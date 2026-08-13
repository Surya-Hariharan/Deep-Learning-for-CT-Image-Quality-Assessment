"""Tests for leakage-safe splitting.

Leakage prevention is a first-class research requirement, so it is tested as
one: these tests assert both that correct splits pass and that deliberately
leaky ones are caught. This includes the specific requirement (Section 8 of
the project brief): reference IDs must not overlap between train/val/test, and
generated descendants (degraded images) must inherit their reference's split.
"""

from __future__ import annotations

from collections import Counter

import pytest

from ct_iqa.data.splitting import (
    LeakageError,
    assign_splits,
    check_cross_dataset_overlap,
    check_group_leakage,
)

RATIOS = {"train": 0.6, "val": 0.2, "test": 0.2}


def test_assignment_is_deterministic():
    groups = [f"REF_{i:03d}" for i in range(105)]
    a = assign_splits(groups, RATIOS, seed=20260813)
    b = assign_splits(groups, RATIOS, seed=20260813)
    assert a == b


def test_different_seeds_give_different_partitions():
    groups = [f"REF_{i:03d}" for i in range(105)]
    assert assign_splits(groups, RATIOS, seed=1) != assign_splits(groups, RATIOS, seed=2)


def test_every_group_assigned_exactly_once():
    groups = [f"P{i}" for i in range(103)]
    assignment = assign_splits(groups, RATIOS, seed=0)
    assert set(assignment) == set(groups)
    assert len(assignment) == len(groups)


def test_split_proportions_are_approximately_correct():
    groups = [f"P{i}" for i in range(1000)]
    assignment = assign_splits(groups, RATIOS, seed=0)
    counts = Counter(assignment.values())
    assert counts["train"] == 600
    assert counts["val"] == 200
    assert counts["test"] == 200


def test_explicit_counts_reproduce_ldctiqac_split():
    """1,000 LDCT-IQAC references, 60/20/20 -> 600/200/200 exactly."""
    refs = [f"{i:04d}" for i in range(1000)]
    assignment = assign_splits(
        refs, RATIOS, seed=20260813, counts={"train": 600, "val": 200, "test": 200}
    )
    counts = Counter(assignment.values())
    assert counts == {"train": 600, "val": 200, "test": 200}


def test_explicit_counts_must_cover_all_groups():
    with pytest.raises(ValueError):
        assign_splits([f"R{i}" for i in range(10)], RATIOS, seed=0,
                      counts={"train": 5, "val": 2, "test": 2})


def test_ratios_must_sum_to_one():
    with pytest.raises(ValueError):
        assign_splits(["a", "b"], {"train": 0.5, "test": 0.2}, seed=0)


def test_reference_ids_do_not_overlap_between_partitions():
    """Direct assertion of the non-negotiable rule: no reference id appears in
    more than one of train/validation/test."""
    refs = [f"{i:04d}" for i in range(1000)]
    assignment = assign_splits(refs, RATIOS, seed=20260813, counts={"train": 600, "val": 200, "test": 200})
    train = {r for r, s in assignment.items() if s == "train"}
    val = {r for r, s in assignment.items() if s == "val"}
    test = {r for r, s in assignment.items() if s == "test"}
    assert not (train & val)
    assert not (train & test)
    assert not (val & test)


def test_degraded_variants_inherit_reference_split_without_leakage():
    """The real anti-leakage guarantee: 169 variants follow their reference."""
    refs = [f"REF_{i:03d}" for i in range(105)]
    assignment = assign_splits(refs, RATIOS, seed=20260813,
                               counts={"train": 63, "val": 21, "test": 21})
    group_ids, split_names = [], []
    for ref in refs:
        for variant in range(169):
            group_ids.append(ref)
            split_names.append(assignment[ref])

    report = check_group_leakage(group_ids, split_names)
    assert report.ok
    assert report.n_groups == 105
    assert len(group_ids) == 17745


def test_leaky_split_is_detected():
    """Randomly splitting degraded images -- the mistake this guards against."""
    import random

    rng = random.Random(0)
    group_ids, split_names = [], []
    for ref in [f"REF_{i:03d}" for i in range(105)]:
        for _ in range(169):
            group_ids.append(ref)
            split_names.append(rng.choice(["train", "val", "test"]))

    report = check_group_leakage(group_ids, split_names)
    assert not report.ok
    assert len(report.violations) > 100
    with pytest.raises(LeakageError):
        report.raise_if_leaky()


def test_patient_timepoints_stay_together():
    """Longitudinal rule: all timepoints of a patient share one partition."""
    patients = [f"PNG_{i:03d}" for i in range(103)]
    assignment = assign_splits(patients, RATIOS, seed=20260813)
    group_ids, split_names = [], []
    for i, patient in enumerate(patients):
        for _ in range(3 + i % 4):  # irregular number of timepoints per patient
            group_ids.append(patient)
            split_names.append(assignment[patient])
    assert check_group_leakage(group_ids, split_names).ok


def test_scan_level_split_leaks_patients():
    """The wrong way: splitting per scan puts one patient in several partitions."""
    group_ids, split_names = [], []
    for patient in [f"PNG_{i:03d}" for i in range(10)]:
        for tp, split in enumerate(["train", "val", "test"]):
            group_ids.append(patient)
            split_names.append(split)
    report = check_group_leakage(group_ids, split_names)
    assert not report.ok
    assert len(report.violations) == 10


def test_mismatched_lengths_rejected():
    with pytest.raises(ValueError):
        check_group_leakage(["a", "b"], ["train"])


def test_cross_dataset_overlap():
    assert check_cross_dataset_overlap(["A", "B", "C"], ["B", "C", "D"]) == ["B", "C"]
    assert check_cross_dataset_overlap(["A"], ["B"]) == []
