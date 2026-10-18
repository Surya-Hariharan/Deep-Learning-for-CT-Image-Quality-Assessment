"""Apply the study-v3 Experiment B selection rule (docs/methodology/STUDY_V3_PROTOCOL.md section 5) and write the record.

Reads only fold-0 / seed-42 / purpose=selection runs: A (reference) and the 7 declared grid points; validation partition only.
The evaluation partition and the official test set are never read. Writes experiments/study_v3/B_selection_record.json (never overwritten).

    python tools/select_b_v3.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from ct_iqa.evaluation.metrics import compute_point_metrics
from ct_iqa.experiment.grid import b_grid_stage1, b_grid_stage2, RMSE_TOLERANCE
from ct_iqa.experiment.metadata import utc_now

ROOT = Path("experiments/study_v3")
SEED, FOLD = 42, 0
N_BOOT = 2000


def mean_corr(m):
    return float(np.mean([m["plcc"], m["srocc"], m["krocc"]]))


def runs():
    out = {}
    for line in (ROOT / "run_registry.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if r["purpose"] != "selection" or r["fold"] != FOLD or r["seed"] != SEED:
            continue
        d = Path(r["run_dir"])
        m = json.loads((d / "run_manifest.json").read_text(encoding="utf-8"))
        if m["status"] != "completed":
            continue
        rk = r["overrides"].get("loss", {}).get("ranking")
        key = "A" if r["exp"] == "A" else (rk["lambda_rank"], rk["margin"], rk["min_score_diff"])
        out.setdefault(key, []).append(d)
    return out


def read(d):
    rows = list(csv.DictReader((d / "predictions_validation.csv").open(encoding="utf-8")))
    return [r["filename"] for r in rows], np.array([float(r["label_0_4"]) for r in rows]), np.array([float(r["prediction_0_4"]) for r in rows])


def group_boot(names, y, pa, pb, groups_of, seed=0, level=0.95):
    gm = json.loads((ROOT / "protocol/group_manifest.json").read_text(encoding="utf-8"))["groups"]
    f2g = {f: g for g, fs in gm.items() for f in fs}
    gl = sorted({f2g[n] for n in names})
    members = {g: [i for i, n in enumerate(names) if f2g[n] == g] for g in gl}
    rng = np.random.default_rng(seed)
    d = []
    for _ in range(N_BOOT):
        idx = np.concatenate([members[gl[j]] for j in rng.integers(0, len(gl), len(gl))])
        if np.ptp(y[idx]) == 0:
            continue
        d.append(mean_corr(compute_point_metrics(y[idx], pb[idx])) - mean_corr(compute_point_metrics(y[idx], pa[idx])))
    d = np.array(d)
    lo, hi = np.percentile(d, [(1 - level) / 2 * 100, (1 + level) / 2 * 100])
    return {"difference": mean_corr(compute_point_metrics(y, pb)) - mean_corr(compute_point_metrics(y, pa)),
            "ci_low": float(lo), "ci_high": float(hi), "half_width": float((hi - lo) / 2), "n_boot_used": int(len(d)), "n_groups": len(gl)}


def main() -> int:
    import sys
    stage1_only = "--stage1-only" in sys.argv
    out = ROOT / "B_selection_record.json"
    if out.exists() and not stage1_only:
        raise SystemExit(f"{out} exists; never overwritten")
    r = runs()
    if "A" not in r:
        raise SystemExit("missing A reference run (fold 0, seed 42)")
    s1 = b_grid_stage1()
    keys1 = [(p["lambda_rank"], p["margin"], p["min_score_diff"]) for p in s1]
    missing = [k for k in keys1 if k not in r]
    dup = [k for k in list(r) if len(r[k]) > 1]
    if missing or dup:
        raise SystemExit(f"stage-1 incomplete or ambiguous: missing={missing} duplicates={dup}")
    # stage-1 lambda: guarded + best mean correlation among the 4 stage-1 points (same rule as v2 stage 1)
    a_names, ya, pa = read(r["A"][0])
    a_m = compute_point_metrics(ya, pa)
    rows = {}
    def row(k):
        names, y, p = read(r[k][0])
        m = compute_point_metrics(y, p)
        return {"lambda_rank": k[0], "margin": k[1], "min_score_diff": k[2], "run": r[k][0].name, **{x: m[x] for x in ("plcc", "srocc", "krocc", "mse", "mae", "rmse")},
                "mean_corr": mean_corr(m), "rmse_within_5pct_of_A": m["rmse"] <= (1 + RMSE_TOLERANCE) * a_m["rmse"]}
    for k in keys1:
        rows[k] = row(k)
    best1 = max([v for v in rows.values() if v["rmse_within_5pct_of_A"]], key=lambda v: v["mean_corr"], default=None)
    lam = best1["lambda_rank"] if best1 else 0.3
    stage2_note = "stage-1 lambda from the declared rule" if best1 else "no stage-1 point passed the guard: stage-2 lambda defaults to 0.3 (declared deviation)"
    if stage1_only:
        print(f"LAMBDA {lam}")
        return 0
    keys2 = [(p["lambda_rank"], p["margin"], p["min_score_diff"]) for p in b_grid_stage2(lam)]
    miss2 = [k for k in keys2 if k not in r]
    if miss2:
        raise SystemExit(f"stage-2 incomplete: {miss2}")
    for k in keys2:
        rows[k] = row(k)
    eligible = [v for v in rows.values() if v["rmse_within_5pct_of_A"]]
    sel = max(eligible, key=lambda v: v["mean_corr"]) if eligible else None
    record = {"created_utc": utc_now(), "protocol": "study_v3 section 5", "fold": FOLD, "seed": SEED, "partition": "fold-0 validation groups only",
              "reference_run": r["A"][0].name, "reference_rmse": a_m["rmse"], "reference_mean_corr": mean_corr(a_m), "stage1_lambda": lam,
              "stage1_note": stage2_note, "candidates": list(rows.values())}
    if sel is None:
        record.update(selected=None, retrain_point={"lambda_rank": 0.3, "margin": 0.05, "min_score_diff": 0.25},
                      outcome="no point passes the RMSE guard; final B runs use the declared fallback point (deviation)")
    else:
        names, y, pb = read(r[(sel["lambda_rank"], sel["margin"], sel["min_score_diff"])][0])
        assert names == a_names
        boot = group_boot(names, y, pa, pb, None)
        improved = boot["difference"] > boot["half_width"]
        record.update(selected=sel, bootstrap_vs_A=boot, improvement_demonstrated=bool(improved),
                      retrain_point={k: sel[k] for k in ("lambda_rank", "margin", "min_score_diff")},
                      outcome=("selected point exceeds A by more than the group-bootstrap half-width" if improved else
                               "no demonstrated improvement; the best guarded point is retrained anyway (declared in the protocol)"))
    out.write_text(json.dumps(record, indent=1, default=str), encoding="utf-8")
    for v in rows.values():
        print({k: (round(x, 4) if isinstance(x, float) else x) for k, x in v.items() if k != "run"})
    print("A:", round(a_m["rmse"], 4), round(mean_corr(a_m), 4), "| outcome:", record["outcome"], "| retrain:", record["retrain_point"])
    print("bootstrap:", record.get("bootstrap_vs_A"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
