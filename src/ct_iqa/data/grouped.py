"""Study v3: recovered image-similarity groups and grouped cross-validation folds.

IMPORTANT: the groups are connected components of an image-similarity graph (thumbnail Pearson correlation above a
threshold). They are NOT patient, scan or slice identifiers and must never be described as such. The point of grouping is
to keep near-identical images (for example the same slice rendered at several quality levels, or adjacent slices) together
inside one partition.

Pure functions only (numpy); the tools under `tools/` do the file I/O.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict

import numpy as np

GROUP_PARTITIONS = ("train", "validation", "calibration", "evaluation")
N_FOLDS = 5


class GroupedSplitError(ValueError):
    """Raised when a grouped manifest is inconsistent."""


def thumbnail_correlation(thumbs: np.ndarray) -> np.ndarray:
    """Pearson correlation matrix between flattened thumbnails (rows)."""
    t = np.asarray(thumbs, dtype=np.float64)
    t = (t - t.mean(1, keepdims=True)) / (t.std(1, keepdims=True) + 1e-12)
    return t @ t.T / t.shape[1]


def similarity_components(corr: np.ndarray, threshold: float) -> list[list[int]]:
    """Connected components of the graph with an edge wherever corr > threshold (diagonal ignored)."""
    n = corr.shape[0]
    adj = corr > threshold
    np.fill_diagonal(adj, False)
    seen = np.zeros(n, dtype=bool)
    comps = []
    for i in range(n):
        if seen[i]:
            continue
        stack, comp = [i], []
        while stack:
            k = stack.pop()
            if seen[k]:
                continue
            seen[k] = True
            comp.append(k)
            stack.extend(np.flatnonzero(adj[k] & ~seen).tolist())
        comps.append(sorted(comp))
    return comps


def name_groups(filenames: list[str], comps: list[list[int]]) -> dict[str, list[str]]:
    """Deterministic ids G000.. ordered by each group's smallest filename."""
    groups = [sorted(filenames[i] for i in c) for c in comps]
    groups.sort(key=lambda g: g[0])
    return {f"G{i:03d}": g for i, g in enumerate(groups)}


def groups_hash(groups: dict[str, list[str]], params: dict) -> str:
    canon = json.dumps({"params": params, "groups": groups}, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()


def _greedy_bins(group_sizes: dict[str, int], ids: list[str], n_bins: int, seed: int) -> list[list[str]]:
    """Greedy balancing of `ids` into `n_bins` bins by image count: groups by decreasing size (ties in seeded random
    order), each to the bin with the fewest images so far (ties: lowest index)."""
    ids = sorted(ids)
    perm = np.random.default_rng(seed).permutation(len(ids))
    order = sorted((ids[i] for i in perm), key=lambda g: -group_sizes[g])  # stable sort keeps the seeded tie order
    bins: list[list[str]] = [[] for _ in range(n_bins)]
    totals = [0] * n_bins
    for g in order:
        k = min(range(n_bins), key=lambda i: (totals[i], i))
        bins[k].append(g)
        totals[k] += group_sizes[g]
    return [sorted(b) for b in bins]


def _take_until(group_sizes: dict[str, int], pool: list[str], target: int, rng) -> list[str]:
    chosen, total = [], 0
    for i in rng.permutation(len(pool)):
        if total >= target:
            break
        chosen.append(pool[i])
        total += group_sizes[pool[i]]
    return sorted(chosen)


def make_grouped_folds(group_sizes: dict[str, int], seed: int, n_folds: int = N_FOLDS, role_target: int = 100,
                       max_role_group: int = 40) -> list[dict[str, list[str]]]:
    """Grouped K-fold with validation and calibration partitions.

    1. Evaluation partitions: groups greedily balanced into `n_folds` bins by image count (each group is evaluated once).
    2. For fold k, from the groups not in evaluation: groups larger than `max_role_group` images always stay in train
       (so no validation/calibration set is a single huge group); validation, then calibration, are drawn from the rest
       in a seeded random order (seed + k) until each holds at least `role_target` images; everything else is train."""
    ids = sorted(group_sizes)
    eval_bins = _greedy_bins(group_sizes, ids, n_folds, seed)
    folds = []
    for k in range(n_folds):
        rest = [g for g in ids if g not in set(eval_bins[k])]
        eligible = [g for g in rest if group_sizes[g] <= max_role_group]
        rng = np.random.default_rng(seed + 1000 + k)
        val = _take_until(group_sizes, eligible, role_target, rng)
        cal = _take_until(group_sizes, [g for g in eligible if g not in set(val)], role_target, rng)
        used = set(val) | set(cal) | set(eval_bins[k])
        folds.append({"evaluation": eval_bins[k], "validation": val, "calibration": cal,
                      "train": sorted(g for g in ids if g not in used)})
    return folds


def fold_manifest_hash(seed: int, group_manifest_sha256: str, folds: list[dict]) -> str:
    canon = json.dumps({"seed": seed, "group_manifest_sha256": group_manifest_sha256, "folds": folds},
                       sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()


def expand_fold(fold: dict, groups: dict[str, list[str]]) -> dict[str, list[str]]:
    return {p: sorted(f for g in fold[p] for f in groups[g]) for p in GROUP_PARTITIONS}


def verify_grouped_manifest(fold_manifest: dict, group_manifest: dict, labels: dict[str, float]) -> None:
    """Integrity checks: hashes, groups cover the labeled images exactly once, within every fold the four partitions
    are group-disjoint and file-disjoint and cover everything, every group is evaluated in exactly one fold."""
    groups = group_manifest["groups"]
    if groups_hash(groups, group_manifest["params"]) != group_manifest["manifest_sha256"]:
        raise GroupedSplitError("group manifest hash mismatch")
    files = [f for g in groups.values() for f in g]
    if len(files) != len(set(files)) or set(files) != set(labels):
        raise GroupedSplitError("groups do not partition exactly the labeled images")
    if fold_manifest["group_manifest_sha256"] != group_manifest["manifest_sha256"]:
        raise GroupedSplitError("fold manifest refers to a different group manifest")
    folds = [{p: f[p] for p in GROUP_PARTITIONS} for f in fold_manifest["folds"]]
    if fold_manifest_hash(fold_manifest["seed"], group_manifest["manifest_sha256"], folds) != fold_manifest["manifest_sha256"]:
        raise GroupedSplitError("fold manifest hash mismatch")
    eval_count: dict[str, int] = defaultdict(int)
    for k, f in enumerate(folds):
        seen: dict[str, str] = {}
        for p in GROUP_PARTITIONS:
            for g in f[p]:
                if g not in groups:
                    raise GroupedSplitError(f"fold {k}: unknown group {g}")
                if g in seen:
                    raise GroupedSplitError(f"fold {k}: group {g} in both {seen[g]} and {p}")
                seen[g] = p
        if set(seen) != set(groups):
            raise GroupedSplitError(f"fold {k}: partitions do not cover all groups")
        exp = expand_fold(f, groups)
        sets = {p: set(v) for p, v in exp.items()}
        for i, a in enumerate(GROUP_PARTITIONS):
            for b in GROUP_PARTITIONS[i + 1:]:
                if sets[a] & sets[b]:
                    raise GroupedSplitError(f"fold {k}: file overlap between {a} and {b}")
        for g in f["evaluation"]:
            eval_count[g] += 1
    if set(eval_count) != set(groups) or any(v != 1 for v in eval_count.values()):
        raise GroupedSplitError("every group must be in the evaluation partition of exactly one fold")
