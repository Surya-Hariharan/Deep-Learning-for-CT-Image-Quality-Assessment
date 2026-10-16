"""Train/validation split logic for the LDCT-IQAC training set.

Extracted from `ct_iqa.training.trainer.build_dataloaders` with no change in
behavior. The official LDCT-IQAC test set is never involved in this split --
see `ct_iqa.data.loader.build_test_dataloader` for the separate, independent
test-set loader.
"""

from __future__ import annotations

import torch
from torch.utils.data import Dataset, Subset, random_split


def train_val_split(
    dataset: Dataset, val_fraction: float, seed: int
) -> tuple[Subset, Subset]:
    """Split `dataset` into (train_subset, val_subset) via a seeded random_split.

    Reproducible given the same `seed` and `val_fraction`.
    """
    n_val = int(round(len(dataset) * val_fraction))
    n_train = len(dataset) - n_val
    generator = torch.Generator().manual_seed(seed)
    train_subset, val_subset = random_split(dataset, [n_train, n_val], generator=generator)
    return train_subset, val_subset


# ---------------------------------------------------------------------------
# Study-v2 three-way split (train / model-selection validation / calibration)
# ---------------------------------------------------------------------------
# The historical `train_val_split` above is unchanged (experiments 001-003 depend on it).
# The functions below implement the frozen study-v2 protocol: an IMAGE-LEVEL split of the
# 1000 labeled training images into 800 / 100 / 100. The official test set never enters.

import hashlib
import json
from pathlib import Path

import numpy as np

SPLIT_ALGORITHM = "numpy.random.default_rng(seed).permutation(sorted(filenames)); first n_train -> train, next n_val -> validation, rest -> calibration"
PARTITIONS = ("train", "validation", "calibration")


class SplitError(ValueError):
    """Raised when a split manifest is inconsistent with the labels or with itself."""


def three_way_split(filenames, n_val: int, n_cal: int, seed: int) -> dict[str, list[str]]:
    names = sorted(filenames)
    if len(set(names)) != len(names):
        raise SplitError("duplicate filenames")
    if n_val < 1 or n_cal < 1 or n_val + n_cal >= len(names):
        raise SplitError(f"invalid sizes n_val={n_val}, n_cal={n_cal} for {len(names)} images")
    order = np.random.default_rng(seed).permutation(len(names))
    shuffled = [names[i] for i in order]
    n_train = len(names) - n_val - n_cal
    return {
        "train": sorted(shuffled[:n_train]),
        "validation": sorted(shuffled[n_train:n_train + n_val]),
        "calibration": sorted(shuffled[n_train + n_val:]),
    }


def manifest_hash(seed: int, parts: dict[str, list[str]]) -> str:
    canonical = json.dumps({"seed": seed, "algorithm": SPLIT_ALGORITHM,
                            **{k: parts[k] for k in PARTITIONS}}, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def build_manifest(labels: dict[str, float], n_val: int, n_cal: int, seed: int, rationale: str) -> dict:
    parts = three_way_split(labels.keys(), n_val, n_cal, seed)
    stats = {}
    for k, names in parts.items():
        v = np.array([labels[n] for n in names], dtype=np.float64)
        stats[k] = {"n": len(names), "label_mean": round(float(v.mean()), 4), "label_std": round(float(v.std()), 4),
                    "label_min": float(v.min()), "label_max": float(v.max())}
    manifest = {
        "protocol": "study_v2 image-level 800/100/100 split of the labeled LDCT-IQAC training set",
        "split_level": "image (patient identifiers are not available; patient-independence is NOT claimed)",
        "seed": seed,
        "algorithm": SPLIT_ALGORITHM,
        "roles": {
            "train": "optimizer updates only",
            "validation": "model selection / checkpoint selection / hyperparameter tuning",
            "calibration": "uncertainty calibration (Experiment C) only; never used for training or selection in any experiment",
        },
        "official_test_set": "300 images, excluded from this manifest; previously used to evaluate historical experiments 001 and 003, so any new evaluation on it is retrospective",
        "rationale": rationale,
        "label_statistics_by_partition": stats,
        "manifest_sha256": manifest_hash(seed, parts),
        **parts,
    }
    return manifest


def verify_manifest(manifest: dict, labels: dict[str, float] | None = None) -> None:
    parts = {k: manifest[k] for k in PARTITIONS}
    sets = {k: set(v) for k, v in parts.items()}
    for k in PARTITIONS:
        if len(sets[k]) != len(parts[k]):
            raise SplitError(f"duplicate filenames inside partition {k!r}")
    for i, a in enumerate(PARTITIONS):
        for b in PARTITIONS[i + 1:]:
            overlap = sets[a] & sets[b]
            if overlap:
                raise SplitError(f"partitions {a!r} and {b!r} overlap: {sorted(overlap)[:5]}")
    if labels is not None and set().union(*sets.values()) != set(labels):
        raise SplitError("manifest partitions do not cover exactly the labeled training images")
    if manifest_hash(manifest["seed"], parts) != manifest["manifest_sha256"]:
        raise SplitError("manifest_sha256 does not match manifest contents")


def save_manifest(manifest: dict, path: str | Path) -> None:
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"{path} already exists; the split protocol is frozen and is never overwritten")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=1), encoding="utf-8")


def load_manifest(path: str | Path, labels: dict[str, float] | None = None) -> dict:
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    verify_manifest(manifest, labels)
    return manifest


def partition_indices(dataset_filenames: list[str], manifest: dict, partition: str) -> list[int]:
    """Indices into a dataset (ordered as `dataset_filenames`) belonging to `partition`."""
    if partition not in PARTITIONS:
        raise SplitError(f"unknown partition {partition!r}")
    wanted = set(manifest[partition])
    idx = [i for i, n in enumerate(dataset_filenames) if n in wanted]
    if len(idx) != len(wanted):
        raise SplitError(f"dataset is missing {len(wanted) - len(idx)} image(s) of partition {partition!r}")
    return idx
