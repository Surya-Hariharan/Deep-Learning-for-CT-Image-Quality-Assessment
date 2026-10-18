"""Create the study-v3 group manifest and grouped 5-fold manifest (Phase 1/2; no model involved).

Groups = connected components of the image-similarity graph (64x64 bilinear thumbnails, Pearson r > 0.99). They are NOT
patient/scan/slice identifiers. Refuses to overwrite existing files. Test images are read only to report descriptive
similarity statistics (test vs labeled); they never enter a group, a fold, or any decision.

    python tools/create_study_v3_manifests.py
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import PIL
from PIL import Image

from ct_iqa.data.grouped import (
    expand_fold, fold_manifest_hash, groups_hash, make_grouped_folds, name_groups, similarity_components,
    thumbnail_correlation, verify_grouped_manifest)

THRESHOLD = 0.99          # declared in docs/methodology/STUDY_V3_PROTOCOL.md before any model run
THUMB = 64
BLOCK_SEED = 3030
OUT = Path("experiments/study_v3/protocol")
LABELS = Path("data/labels/ldct_iqac/train.json")
TEST_LABELS = Path("data/labels/ldct_iqac/test.json")
TRAIN_DIR = Path("data/raw/ldct_iqac/train/image")
TEST_DIR = Path("data/raw/ldct_iqac/test/images")


def thumbs(directory: Path, names: list[str]) -> np.ndarray:
    return np.array([np.asarray(Image.open(directory / n).resize((THUMB, THUMB), Image.BILINEAR), dtype=np.float64).ravel()
                     for n in names])


def stats(v) -> dict:
    v = np.asarray(v, dtype=float)
    return {"n": int(len(v)), "label_mean": round(float(v.mean()), 4), "label_std": round(float(v.std()), 4)}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    gpath, fpath = OUT / "group_manifest.json", OUT / "fold_manifest.json"
    for p in (gpath, fpath):
        if p.exists():
            raise SystemExit(f"{p} already exists; study-v3 manifests are frozen and never overwritten")
    labels = json.loads(LABELS.read_text(encoding="utf-8"))
    names = sorted(labels)
    if {p.name for p in TRAIN_DIR.iterdir()} != set(names):
        raise SystemExit("image directory and labels disagree")
    T = thumbs(TRAIN_DIR, names)
    C = thumbnail_correlation(T)
    comps = similarity_components(C, THRESHOLD)
    groups = name_groups(names, comps)
    params = {"method": "connected components of thumbnail-correlation graph", "threshold_r_greater_than": THRESHOLD,
              "thumbnail": f"{THUMB}x{THUMB} PIL BILINEAR of the full 512x512 float image, flattened, Pearson r",
              "pillow_version": PIL.__version__, "numpy_version": np.__version__}
    # --- integrity / audit statistics ---------------------------------------------------------------------
    file_to_g = {f: g for g, fs in groups.items() for f in fs}
    idx = {n: i for i, n in enumerate(names)}
    gid = np.array([int(file_to_g[n][1:]) for n in names])
    cross_max = float(np.max(np.where(gid[:, None] != gid[None, :], C, -1)))
    nearest = [float(np.max(np.where(gid != int(g[1:]), C[[idx[f] for f in fs]], -1).max())) for g, fs in groups.items()]
    tn = sorted(json.loads(TEST_LABELS.read_text(encoding="utf-8")))
    Tt = thumbs(TEST_DIR, tn)
    Tz = (Tt - Tt.mean(1, keepdims=True)) / (Tt.std(1, keepdims=True) + 1e-12)
    Lz = (T - T.mean(1, keepdims=True)) / (T.std(1, keepdims=True) + 1e-12)
    test_max = (Tz @ Lz.T / T.shape[1]).max(1)
    sizes = Counter(len(v) for v in groups.values())
    group_manifest = {
        "version": "study_v3", "params": params, "n_images": len(names), "n_groups": len(groups),
        "WARNING": "groups are image-similarity components, NOT patient, scan or slice identifiers",
        "group_size_histogram": {str(k): v for k, v in sorted(sizes.items())},
        "audit": {"max_correlation_between_different_groups": cross_max,
                  "nearest_other_group_corr_quantiles_0_25_50_75_100": [round(float(q), 4) for q in np.quantile(nearest, [0, .25, .5, .75, 1])],
                  "official_test_max_corr_to_any_labeled_image_quantiles_0_50_100": [round(float(q), 4) for q in np.quantile(test_max, [0, .5, 1])],
                  "note": "the official test set is more separated from the labeled images (max r about 0.94) than any two groups here (max r < 0.99): v3 held-out data remain closer to training data than the official test set"},
        "groups": groups, "manifest_sha256": groups_hash(groups, params),
    }
    sizes_by_g = {g: len(v) for g, v in groups.items()}
    folds = make_grouped_folds(sizes_by_g, BLOCK_SEED)
    fold_manifest = {
        "version": "study_v3", "seed": BLOCK_SEED, "group_manifest_sha256": group_manifest["manifest_sha256"],
        "algorithm": "evaluation: groups sorted by decreasing size (ties in seeded random order) greedily assigned to the bin with fewest images (5 bins). For fold k, groups with more than 40 images always stay in train unless evaluated; validation then calibration are drawn from the remaining groups in a seeded random order (seed+1000+k) until each has at least 100 images; the rest is train",
        "roles": {"train": "optimizer updates only", "validation": "checkpoint and hyperparameter selection only",
                  "calibration": "Experiment C conformal fitting only; never used for selection", "evaluation": "held-out groups, touched only after training for reporting; never used for selection or calibration of the same fold"},
        "official_test_set": "not part of any fold; descriptive retrospective evaluation only (previously examined)",
        "folds": folds,
        "manifest_sha256": fold_manifest_hash(BLOCK_SEED, group_manifest["manifest_sha256"], folds),
    }
    verify_grouped_manifest(fold_manifest, group_manifest, labels)
    fold_manifest["summary"] = []
    for k, f in enumerate(folds):
        exp = expand_fold(f, groups)
        fold_manifest["summary"].append({"fold": k, **{p: {"n_groups": len(f[p]), **stats([labels[n] for n in exp[p]])} for p in exp}})
    fpath.write_text(json.dumps(fold_manifest, indent=1), encoding="utf-8")
    gpath.write_text(json.dumps(group_manifest, indent=1), encoding="utf-8")
    print("group manifest sha256", group_manifest["manifest_sha256"])
    print("fold manifest  sha256", fold_manifest["manifest_sha256"])
    print("groups", len(groups), "size histogram", group_manifest["group_size_histogram"])
    print("audit", json.dumps(group_manifest["audit"], indent=1))
    for s in fold_manifest["summary"]:
        print(s)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
