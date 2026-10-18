"""Phase-2 verification of the study-v3 manifests (run before the freeze marker is created; read-only).

Independently rebuilds the groups from the raw images, checks hashes, zero group/file overlap in every fold, that each group is evaluated once,
quantifies held-out-vs-train image similarity per fold (against the official test set's separation), and checks the smoke run's partitions.

    python tools/verify_study_v3.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image

from ct_iqa.data.grouped import (
    expand_fold, groups_hash, name_groups, similarity_components, thumbnail_correlation, verify_grouped_manifest)

P = Path("experiments/study_v3/protocol")
OUT = Path("reports/study_v3")


def thumbs(d: Path, names):
    return np.array([np.asarray(Image.open(d / n).resize((64, 64), Image.BILINEAR), dtype=np.float64).ravel() for n in names])


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    labels = json.loads(Path("data/labels/ldct_iqac/train.json").read_text(encoding="utf-8"))
    gm = json.loads((P / "group_manifest.json").read_text(encoding="utf-8"))
    fm = json.loads((P / "fold_manifest.json").read_text(encoding="utf-8"))
    checks: dict = {}
    verify_grouped_manifest(fm, gm, labels)
    checks["manifest_integrity_verified"] = True
    names = sorted(labels)
    T = thumbs(Path("data/raw/ldct_iqac/train/image"), names)
    C = thumbnail_correlation(T)
    rebuilt = name_groups(names, similarity_components(C, gm["params"]["threshold_r_greater_than"]))
    checks["groups_rebuilt_from_raw_images_identical"] = rebuilt == gm["groups"]
    checks["group_hash_recomputed_matches"] = groups_hash(rebuilt, gm["params"]) == gm["manifest_sha256"]
    file_to_g = {f: g for g, fs in gm["groups"].items() for f in fs}
    idx = {n: i for i, n in enumerate(names)}
    # each group's members are mutually reachable and no cross-group edge exceeds the threshold
    gid = np.array([int(file_to_g[n][1:]) for n in names])
    cross = np.where(gid[:, None] != gid[None, :], C, -1.0)
    checks["max_cross_group_correlation"] = float(cross.max())
    checks["no_cross_group_edge_above_threshold"] = bool(cross.max() <= gm["params"]["threshold_r_greater_than"])
    test_names = sorted(json.loads(Path("data/labels/ldct_iqac/test.json").read_text(encoding="utf-8")))
    Tt = thumbs(Path("data/raw/ldct_iqac/test/images"), test_names)
    Tz = (Tt - Tt.mean(1, keepdims=True)) / (Tt.std(1, keepdims=True) + 1e-12)
    Lz = (T - T.mean(1, keepdims=True)) / (T.std(1, keepdims=True) + 1e-12)
    test_max = (Tz @ Lz.T / T.shape[1]).max(1)
    per_fold = []
    for k, f in enumerate(fm["folds"]):
        exp = expand_fold(f, gm["groups"])
        sets = {p: set(v) for p, v in exp.items()}
        pairs = {f"{a}&{b}": len(sets[a] & sets[b]) for i, a in enumerate(sets) for b in list(sets)[i + 1:]}
        gsets = {p: set(f[p]) for p in f}
        gpairs = {f"{a}&{b}": len(gsets[a] & gsets[b]) for i, a in enumerate(gsets) for b in list(gsets)[i + 1:]}
        tr = [idx[n] for n in exp["train"]]
        row = {"fold": k, "n_images": {p: len(v) for p, v in exp.items()}, "n_groups": {p: len(f[p]) for p in f},
               "file_overlap_between_partitions": pairs, "group_overlap_between_partitions": gpairs,
               "covers_all_images": sum(len(v) for v in exp.values()) == len(names)}
        for p in ("validation", "calibration", "evaluation"):
            m = C[np.ix_([idx[n] for n in exp[p]], tr)].max(1)
            row[f"max_corr_to_train__{p}__quantiles_0_50_100"] = [round(float(q), 4) for q in np.quantile(m, [0, .5, 1])]
        per_fold.append(row)
    checks["per_fold"] = per_fold
    checks["zero_overlap_everywhere"] = all(sum(r["file_overlap_between_partitions"].values()) == 0 and
                                            sum(r["group_overlap_between_partitions"].values()) == 0 and r["covers_all_images"] for r in per_fold)
    checks["official_test_max_corr_to_labeled__quantiles_0_50_100"] = [round(float(q), 4) for q in np.quantile(test_max, [0, .5, 1])]
    # smoke run partition check (if present)
    smoke = sorted(Path("experiments/study_v3_smoke").glob("*/runs/*/run_manifest.json"))
    sm = []
    for m in smoke:
        man = json.loads(m.read_text(encoding="utf-8"))
        d = m.parent
        parts = expand_fold(fm["folds"][man["fold"]], gm["groups"])
        ev = list(csv.DictReader((d / "predictions_evaluation.csv").open(encoding="utf-8")))
        sm.append({"run": d.name, "status": man["status"], "fold": man["fold"],
                   "evaluation_files_equal_fold_evaluation": sorted(r["filename"] for r in ev) == parts["evaluation"],
                   "evaluation_disjoint_from_train": not (set(r["filename"] for r in ev) & set(parts["train"])),
                   "hashes_recorded": man["group_manifest_sha256"] == gm["manifest_sha256"] and man["fold_manifest_sha256"] == fm["manifest_sha256"],
                   "official_test_set_used": man["official_test_set_used"]})
    checks["smoke_runs"] = sm
    checks["PASS"] = bool(checks["manifest_integrity_verified"] and checks["groups_rebuilt_from_raw_images_identical"] and
                          checks["group_hash_recomputed_matches"] and checks["no_cross_group_edge_above_threshold"] and
                          checks["zero_overlap_everywhere"] and all(s["status"] == "completed" and s["evaluation_files_equal_fold_evaluation"] and
                                                                      s["evaluation_disjoint_from_train"] and s["hashes_recorded"] and not s["official_test_set_used"] for s in sm))
    (OUT / "phase2_verification.json").write_text(json.dumps(checks, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in checks.items() if k != "per_fold"}, indent=1))
    for r in per_fold:
        print(r["fold"], r["n_groups"], r["max_corr_to_train__evaluation__quantiles_0_50_100"], r["group_overlap_between_partitions"])
    return 0 if checks["PASS"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
