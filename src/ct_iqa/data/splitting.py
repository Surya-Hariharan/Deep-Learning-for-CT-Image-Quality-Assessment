"""Deterministic, leakage-safe splitting.

Two rules govern every split in this project:

1. The split unit is an *identity*, never a sample. For the IQA dataset the
   identity is the reference image (``image_id``); for any longitudinal or
   patient-level dataset it is the patient. Degraded variants follow their
   reference, and all timepoints follow their patient.
2. Splits are reproducible from a seed recorded in config.

``assign_splits`` produces the partition; ``check_group_leakage`` proves it.
The dataset validator calls both, so a leaky split fails loudly rather than
quietly inflating a metric.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence


class LeakageError(AssertionError):
    """Raised when one group identity appears in more than one partition."""


def _stable_order(groups: Iterable[str]) -> list[str]:
    """Sort unique group ids by a seed-independent stable hash.

    Plain ``sorted()`` would work too, but hashing decorrelates the partition
    from any ordering already present in the ids (e.g. all DeepLesion ids
    sorting before all CQ500 ids).
    """
    uniq = sorted(set(groups))
    return sorted(uniq, key=lambda g: hashlib.sha256(g.encode("utf-8")).hexdigest())


def assign_splits(
    groups: Iterable[str],
    ratios: Mapping[str, float],
    seed: int,
    counts: Mapping[str, int] | None = None,
) -> dict[str, str]:
    """Map each group id to a partition name.

    Args:
        groups: group identities (reference ids, patient ids, ...).
        ratios: partition name -> fraction, must sum to 1.
        seed: reproducibility seed; the same seed and inputs always give the
            same assignment.
        counts: optional exact partition sizes in groups. When given it
            overrides ``ratios`` -- used where the target counts are known
            exactly (the Ohashi/LDCT-IQAC 600/200/200 reference split) so that
            rounding can never drift.

    Returns:
        ``{group_id: partition_name}``.
    """
    import random

    ordered = _stable_order(groups)
    rng = random.Random(seed)
    rng.shuffle(ordered)
    n = len(ordered)

    if counts is not None:
        if sum(counts.values()) != n:
            raise ValueError(
                f"explicit split counts sum to {sum(counts.values())} but there are {n} groups"
            )
        sizes = dict(counts)
    else:
        total = sum(ratios.values())
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"split ratios must sum to 1.0, got {total}")
        names = list(ratios)
        sizes = {name: int(n * ratios[name]) for name in names}
        # Give any remainder from truncation to the largest partition, keeping
        # the assignment deterministic rather than dependent on float rounding.
        remainder = n - sum(sizes.values())
        if remainder:
            sizes[max(names, key=lambda k: ratios[k])] += remainder

    assignment: dict[str, str] = {}
    cursor = 0
    for name, size in sizes.items():
        for group in ordered[cursor : cursor + size]:
            assignment[group] = name
        cursor += size
    return assignment


@dataclass
class LeakageReport:
    """Outcome of a leakage check."""

    ok: bool
    n_groups: int
    violations: dict[str, list[str]]  # group id -> partitions it appears in

    def raise_if_leaky(self) -> None:
        if not self.ok:
            detail = "; ".join(f"{g} in {p}" for g, p in list(self.violations.items())[:10])
            raise LeakageError(
                f"{len(self.violations)} group(s) span multiple partitions: {detail}"
            )


def check_group_leakage(group_ids: Sequence[str], split_names: Sequence[str]) -> LeakageReport:
    """Verify that no group identity appears in more than one partition.

    Args:
        group_ids: per-sample group identity (patient id, reference id, ...).
        split_names: per-sample partition name, parallel to ``group_ids``.
    """
    if len(group_ids) != len(split_names):
        raise ValueError("group_ids and split_names must be the same length")

    seen: dict[str, set[str]] = defaultdict(set)
    for group, split in zip(group_ids, split_names):
        seen[group].add(split)

    violations = {g: sorted(s) for g, s in seen.items() if len(s) > 1}
    return LeakageReport(ok=not violations, n_groups=len(seen), violations=violations)


def check_cross_dataset_overlap(
    ids_a: Iterable[str], ids_b: Iterable[str]
) -> list[str]:
    """Return identities shared between two datasets.

    Needed because LUNA16 is derived from LIDC-IDRI: training on LIDC and
    evaluating on LUNA16 leaks patients unless the intersection is removed.
    """
    return sorted(set(ids_a) & set(ids_b))
