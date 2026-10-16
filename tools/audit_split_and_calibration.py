"""Independent audit of (1) the split/grouping structure and (2) Experiment C's conformal calibration.

Re-implements the conformal quantile and coverage with plain numpy (no ct_iqa evaluation code) from the saved prediction
CSVs, compares with `calibration.json`, and reports covered counts. Read-only; writes `reports/audit/split_and_calibration_audit.json`.

    python tools/audit_split_and_calibration.py
"""

from __future__ import annotations

import csv
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path("experiments/study_v2")
OUT = Path("reports/audit")
MANIFEST = json.loads((ROOT / "protocol/split_manifest.json").read_text(encoding="utf-8"))


def read(path: Path):
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    d = {"filename": [r["filename"] for r in rows], "y": np.array([float(r["label_0_4"]) for r in rows]),
         "mu": np.array([float(r["prediction_0_4"]) for r in rows])}
    d["sigma"] = 4.0 * np.exp(np.array([float(r["log_var_normalized"]) for r in rows]) / 2.0)
    d["mu_from_norm"] = 4.0 * np.array([float(r["prediction_normalized"]) for r in rows])
    return d


def conformal(cal, coverage=0.90):
    n = len(cal["y"])
    k = math.ceil((n + 1) * coverage)
    r = np.sort(np.abs(cal["y"] - cal["mu"]) / cal["sigma"])
    return (float(r[k - 1]) if k <= n else math.inf), k


def cover(ev, q):
    ok = np.abs(ev["y"] - ev["mu"]) <= q * ev["sigma"]
    return int(ok.sum()), int(len(ok))


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    report: dict = {}

    # --- 1. grouping structure ----------------------------------------------------------------------------
    groups = list(csv.DictReader((Path("reports/eda/anatomy_groups_thr0.999.csv")).open(encoding="utf-8")))
    by_g: dict[str, list[dict]] = {}
    for r in groups:
        by_g.setdefault(r["group"], []).append(r)
    sizes = Counter(len(v) for v in by_g.values())
    test_g = [v for v in by_g.values() if any(x["split"] == "official_test" for x in v)]
    lab_g = [v for v in by_g.values() if all(x["split"] != "official_test" for x in v)]
    part_of = {n: p for p in ("train", "validation", "calibration") for n in MANIFEST[p]}
    n_cross = sum(1 for v in lab_g if len({x["split"] for x in v}) > 1)
    report["grouping"] = {
        "identifiers_available": "none: TIFF headers contain only tifffile shape/resolution tags; file names are shuffled indices; no patient/scan/slice IDs anywhere in the repository",
        "recovered_groups_method": "connected components of 64x64-thumbnail Pearson r > 0.999 (image-similarity, NOT patient identity)",
        "group_size_histogram": dict(sorted(sizes.items())),
        "n_groups_labeled": len(lab_g), "n_groups_test": len(test_g),
        "labeled_groups_spanning_more_than_one_partition": n_cross,
        "test_groups_sharing_a_group_with_labeled_images": sum(1 for v in test_g if any(x["split"] != "official_test" for x in v)),
        "interpretation": "Groups of exactly 10 images are consistent with one source slice rendered at 10 quality levels (100 sources in the labeled set, 30 in the test set); larger groups (20, 40) are merged similar slices. Whether different sources come from the same patient is unknowable here.",
        "validation_images_with_group_mate_in_train": float(np.mean([any(x["split"] == "train" for x in by_g[g]) for g in by_g for x in by_g[g] if x["split"] == "validation"])),
    }

    # --- 2. calibration audit -----------------------------------------------------------------------------
    runs = []
    for line in (ROOT / "run_registry.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if r["exp"] == "C" and r["purpose"] == "final":
            runs.append(Path(r["run_dir"]))
    cal_files = set(MANIFEST["calibration"])
    val_files = set(MANIFEST["validation"])
    per_seed = []
    for d in runs:
        cal, val = read(d / "predictions_calibration.csv"), read(d / "predictions_validation.csv")
        test = read(d / "test_retrospective" / "predictions_test.csv")
        stored_cal = json.loads((d / "calibration.json").read_text(encoding="utf-8"))
        stored_test = json.loads((d / "test_retrospective" / "metrics_test.json").read_text(encoding="utf-8"))["intervals"]
        q, k = conformal(cal)
        cv_val, n_val = cover(val, q)
        cv_test, n_test = cover(test, q)
        r_cal = np.abs(cal["y"] - cal["mu"]) / cal["sigma"]
        r_val = np.abs(val["y"] - val["mu"]) / val["sigma"]
        r_test = np.abs(test["y"] - test["mu"]) / test["sigma"]
        # oracle: what q would give 90% on test (descriptive diagnostic only; never used for anything)
        q_test_90 = float(np.quantile(r_test, 0.90, method="higher"))
        per_seed.append({
            "run": d.name, "n_cal": len(cal["y"]), "k": k, "q_recomputed": q, "q_stored": stored_cal["quantile_q"],
            "q_matches": bool(abs(q - float(stored_cal["quantile_q"])) < 1e-6),
            "cal_files_are_calibration_partition": set(cal["filename"]) == cal_files,
            "val_files_are_validation_partition": set(val["filename"]) == val_files,
            "cal_val_disjoint": not (set(cal["filename"]) & set(val["filename"])),
            "mu_equals_4x_normalized": bool(np.allclose(cal["mu"], cal["mu_from_norm"], atol=1e-6)),
            "val_covered": f"{cv_val}/{n_val}", "val_coverage_recomputed": cv_val / n_val,
            "val_coverage_stored": stored_cal["unclipped"]["coverage"],
            "test_covered": f"{cv_test}/{n_test}", "test_coverage_recomputed": cv_test / n_test,
            "test_coverage_stored": stored_test["unclipped"]["coverage"],
            "coverage_matches_stored": bool(abs(cv_val / n_val - stored_cal["unclipped"]["coverage"]) < 1e-6
                                            and abs(cv_test / n_test - stored_test["unclipped"]["coverage"]) < 1e-6),
            "median_normalized_residual": {"calibration": float(np.median(r_cal)), "validation": float(np.median(r_val)), "test": float(np.median(r_test))},
            "rmse_original_scale": {"calibration": float(np.sqrt(np.mean((cal["y"] - cal["mu"]) ** 2))),
                                    "validation": float(np.sqrt(np.mean((val["y"] - val["mu"]) ** 2))),
                                    "test": float(np.sqrt(np.mean((test["y"] - test["mu"]) ** 2)))},
            "mean_sigma": {"calibration": float(cal["sigma"].mean()), "validation": float(val["sigma"].mean()), "test": float(test["sigma"].mean())},
            "q_that_would_give_90pct_on_test_DIAGNOSTIC_ONLY": q_test_90,
        })
    report["experiment_c_calibration"] = {
        "target_coverage": 0.90, "formula": "k = ceil((n+1)*0.9); q = k-th smallest |y-mu|/sigma on the 100 calibration images; interval mu +/- q*sigma; sigma = 4*exp(s/2), mu = 4*sigmoid output",
        "per_seed": per_seed,
        "pooled_test_covered": f"{sum(int(p['test_covered'].split('/')[0]) for p in per_seed)}/{sum(int(p['test_covered'].split('/')[1]) for p in per_seed)} (3 seeds x 300 images; the three seeds evaluate the SAME 300 images, so this is not 900 independent observations)",
        "all_checks_pass": all(p["q_matches"] and p["coverage_matches_stored"] and p["cal_files_are_calibration_partition"]
                               and p["val_files_are_validation_partition"] and p["cal_val_disjoint"] and p["mu_equals_4x_normalized"] for p in per_seed),
    }
    # exact conformal property sanity check on synthetic exchangeable data (formula correctness, not about the data)
    rng = np.random.default_rng(0)
    hits = []
    for _ in range(2000):
        s = np.abs(rng.normal(size=101))
        q = np.sort(s[:100])[math.ceil(101 * 0.9) - 1]
        hits.append(s[100] <= q)
    report["synthetic_formula_check"] = {"exchangeable_normal_coverage_over_2000_trials": float(np.mean(hits)), "expected_at_least": 0.90, "note": "Monte Carlo SE about 0.007, so 0.8965 is consistent with 0.90"}
    (OUT / "split_and_calibration_audit.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps({"grouping": report["grouping"], "all_checks_pass": report["experiment_c_calibration"]["all_checks_pass"],
                      "pooled_test": report["experiment_c_calibration"]["pooled_test_covered"],
                      "synthetic": report["synthetic_formula_check"]}, indent=1))
    for p in per_seed:
        print(p["run"][-22:], "q", round(p["q_recomputed"], 3), "val", p["val_covered"], "test", p["test_covered"], "medianR", {k: round(v, 3) for k, v in p["median_normalized_residual"].items()},
              "rmse", {k: round(v, 3) for k, v in p["rmse_original_scale"].items()}, "q90test", round(p["q_that_would_give_90pct_on_test_DIAGNOSTIC_ONLY"], 2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
