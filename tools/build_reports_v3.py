"""Study-v3 analysis (rules: docs/methodology/STUDY_V3_PROTOCOL.md sections 6-8). Regenerates everything in `reports/study_v3/`
from raw run records; nothing is typed by hand.

    python tools/build_reports_v3.py

Pooled per seed over the 5 folds' held-out evaluation groups (1000 images, each predicted once by a model that never saw its group);
group-level bootstrap over the 54 similarity groups (seeds are NOT resampled and NOT treated as independent samples).
"""

from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy import stats as sps  # noqa: E402

from ct_iqa.evaluation.metrics import compute_point_metrics  # noqa: E402
from ct_iqa.experiment.runs import verify_artifacts  # noqa: E402

ROOT = Path(os.environ.get("CT_IQA_V3_ROOT", "experiments/study_v3"))      # env overrides are for the synthetic test only
OUT = Path(os.environ.get("CT_IQA_V3_OUT", "reports/study_v3"))
MET = ("plcc", "srocc", "krocc", "mse", "mae", "rmse")
EXPS = ("A", "B", "C")
SEEDS = (1, 2, 3)
FOLDS = range(5)
N_BOOT, BOOT_SEED, COVERAGE = 2000, 0, 0.90


def write_csv(path: Path, rows: list[dict]) -> None:
    if rows:
        keys = list(dict.fromkeys(k for r in rows for k in r))
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)


def read_pred(path: Path) -> dict:
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    d = {"filename": [r["filename"] for r in rows], "y": np.array([float(r["label_0_4"]) for r in rows]),
         "mu": np.array([float(r["prediction_0_4"]) for r in rows])}
    if "log_var_normalized" in rows[0]:
        d["sigma"] = 4.0 * np.exp(np.array([float(r["log_var_normalized"]) for r in rows]) / 2.0)
    return d


def mean_corr(m: dict) -> float:
    return float(np.mean([m["plcc"], m["srocc"], m["krocc"]]))


def kth(values: np.ndarray, coverage: float) -> float:
    n = len(values)
    k = math.ceil((n + 1) * coverage)
    return float(np.sort(values)[k - 1]) if k <= n else math.inf


def load_final_runs() -> dict:
    runs = {}
    for line in (ROOT / "run_registry.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if r["purpose"] != "final":
            continue
        d = Path(r["run_dir"])
        m = json.loads((d / "run_manifest.json").read_text(encoding="utf-8"))
        if m["status"] != "completed":
            continue
        key = (r["exp"], r["fold"], r["seed"])
        if key in runs:
            raise SystemExit(f"ambiguous: several completed final runs for {key}")
        if verify_artifacts(d):
            raise SystemExit(f"artifact verification failed for {d}")
        runs[key] = d
    missing = [(e, f, s) for e in EXPS for f in FOLDS for s in SEEDS if (e, f, s) not in runs]
    if missing:
        raise SystemExit(f"incomplete final runs: {missing}")
    return runs


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    labels = json.loads(Path("data/labels/ldct_iqac/train.json").read_text(encoding="utf-8"))
    gm = json.loads((ROOT / "protocol/group_manifest.json").read_text(encoding="utf-8"))
    fm = json.loads((ROOT / "protocol/fold_manifest.json").read_text(encoding="utf-8"))
    f2g = {f: g for g, fs in gm["groups"].items() for f in fs}
    names = sorted(labels)
    pos = {n: i for i, n in enumerate(names)}
    runs = load_final_runs()
    summary: dict = {"protocol": "docs/methodology/STUDY_V3_PROTOCOL.md", "n_final_runs": len(runs),
                     "group_manifest_sha256": gm["manifest_sha256"], "fold_manifest_sha256": fm["manifest_sha256"]}

    # ---- pooled predictions per (exp, seed) in sorted-filename order -------------------------------------------
    pooled, fold_of = {}, np.zeros(len(names), int)
    per_fold_rows, per_seed_rows = [], []
    for e in EXPS:
        for s in SEEDS:
            y, mu, sg = np.full(len(names), np.nan), np.full(len(names), np.nan), np.full(len(names), np.nan)
            for f in FOLDS:
                p = read_pred(runs[(e, f, s)] / "predictions_evaluation.csv")
                ix = [pos[n] for n in p["filename"]]
                if not np.isnan(mu[ix]).all():
                    raise SystemExit("an image was evaluated twice")
                y[ix], mu[ix], fold_of[ix] = p["y"], p["mu"], f
                if "sigma" in p:
                    sg[ix] = p["sigma"]
                m = json.loads((runs[(e, f, s)] / "metrics_evaluation.json").read_text(encoding="utf-8"))
                vm = json.loads((runs[(e, f, s)] / "metrics_validation.json").read_text(encoding="utf-8"))
                per_fold_rows.append({"exp": e, "fold": f, "seed": s, "best_epoch": m["epoch_1based"], "n_eval": m["n"],
                                      **{k: m[k] for k in MET}, "val_plcc": vm["plcc"], "val_rmse": vm["rmse"]})
            if np.isnan(mu).any():
                raise SystemExit("some images were not evaluated")
            pooled[(e, s)] = {"y": y, "mu": mu, "sigma": sg}
            pm = compute_point_metrics(y, mu)
            per_seed_rows.append({"exp": e, "seed": s, "partition": "held-out evaluation groups, pooled over 5 folds", "n": len(y),
                                  **{k: pm[k] for k in MET}, "mean_corr": mean_corr(pm)})
    write_csv(OUT / "per_fold_per_seed_evaluation.csv", per_fold_rows)
    write_csv(OUT / "pooled_per_seed_evaluation.csv", per_seed_rows)

    # ---- group bootstrap draws (shared by every analysis) ------------------------------------------------------
    gl = sorted(gm["groups"])
    members = {g: np.array([pos[n] for n in gm["groups"][g]]) for g in gl}
    rng = np.random.default_rng(BOOT_SEED)
    draws = [np.concatenate([members[gl[j]] for j in rng.integers(0, len(gl), len(gl))]) for _ in range(N_BOOT)]
    y0 = pooled[("A", SEEDS[0])]["y"]
    bm = {}  # (exp, seed) -> (N_BOOT, 6) metrics, and mean_corr
    for key, d in pooled.items():
        arr = np.full((N_BOOT, 7), np.nan)
        for b, ix in enumerate(draws):
            if np.ptp(y0[ix]) == 0:
                continue
            m = compute_point_metrics(d["y"][ix], d["mu"][ix])
            arr[b] = [m[k] for k in MET] + [mean_corr(m)]
        bm[key] = arr
    cols = MET + ("mean_corr",)
    seedmean = {e: np.mean([bm[(e, s)] for s in SEEDS], axis=0) for e in EXPS}    # (N_BOOT, 7): metrics averaged over seeds in each draw
    agg_rows = []
    for e in EXPS:
        rows = [r for r in per_seed_rows if r["exp"] == e]
        row = {"exp": e, "n_seeds": 3}
        for k in cols:
            v = np.array([r[k] for r in rows])
            lo, hi = np.nanpercentile(seedmean[e][:, cols.index(k)], [2.5, 97.5])
            row.update({f"{k}_mean": float(v.mean()), f"{k}_seed_sd": float(v.std(ddof=1)), f"{k}_group_boot_ci95_low": float(lo), f"{k}_group_boot_ci95_high": float(hi)})
        agg_rows.append(row)
    write_csv(OUT / "aggregate_evaluation.csv", agg_rows)
    summary["aggregate_evaluation"] = agg_rows

    diffs = []
    for e in ("B", "C"):
        for k in cols:
            d = seedmean[e][:, cols.index(k)] - seedmean["A"][:, cols.index(k)]
            pt = float(np.mean([r[k] for r in per_seed_rows if r["exp"] == e]) - np.mean([r[k] for r in per_seed_rows if r["exp"] == "A"]))
            lo95, hi95 = np.nanpercentile(d, [2.5, 97.5])
            lo975, hi975 = np.nanpercentile(d, [1.25, 98.75])
            per_seed_diff = [float([r[k] for r in per_seed_rows if r["exp"] == e and r["seed"] == s][0] - [r[k] for r in per_seed_rows if r["exp"] == "A" and r["seed"] == s][0]) for s in SEEDS]
            diffs.append({"comparison": f"{e} - A", "metric": k, "difference": pt, "group_boot_ci95": [float(lo95), float(hi95)],
                          "group_boot_ci97_5_bonferroni": [float(lo975), float(hi975)],
                          "excludes_zero_at_97_5": bool(lo975 > 0 or hi975 < 0), "per_seed_differences": per_seed_diff})
    summary["paired_differences_group_bootstrap"] = diffs
    write_csv(OUT / "paired_differences.csv", [{**d, "group_boot_ci95": json.dumps(d["group_boot_ci95"]), "group_boot_ci97_5_bonferroni": json.dumps(d["group_boot_ci97_5_bonferroni"]),
                                              "per_seed_differences": json.dumps(d["per_seed_differences"])} for d in diffs])

    # ---- Experiment C: conformal coverage and width ------------------------------------------------------------
    cov_rows, per_img = [], {}
    for s in SEEDS:
        P = pooled[("C", s)]
        r_all = np.abs(P["y"] - P["mu"]) / P["sigma"]
        ae = np.abs(P["y"] - P["mu"])
        cP, wP, wPc, cS, wS, wSc, cCP, wCP, cCS, wCS = (np.zeros(len(names)) for _ in range(10))
        for f in FOLDS:
            ix = np.flatnonzero(fold_of == f)
            other = np.flatnonzero(fold_of != f)
            cal = json.loads((runs[("C", f, s)] / "calibration.json").read_text(encoding="utf-8"))
            qP = math.inf if cal["quantile_q"] in ("inf", math.inf) else float(cal["quantile_q"])
            calp = read_pred(runs[("C", f, s)] / "predictions_calibration.csv")
            qCP = kth(np.abs(calp["y"] - calp["mu"]), COVERAGE)              # constant-width comparator (P)
            qS = kth(r_all[other], COVERAGE)                                 # grouped cross-conformal (S)
            qCS = kth(ae[other], COVERAGE)                                   # constant-width comparator (S)
            for q, c, w, wc in ((qP, cP, wP, wPc), (qS, cS, wS, wSc)):
                lo, hi = P["mu"][ix] - q * P["sigma"][ix], P["mu"][ix] + q * P["sigma"][ix]
                c[ix] = (P["y"][ix] >= lo) & (P["y"][ix] <= hi)
                w[ix] = hi - lo
                wc[ix] = np.clip(hi, 0, 4) - np.clip(lo, 0, 4)
            for q, c, w in ((qCP, cCP, wCP), (qCS, cCS, wCS)):
                c[ix] = np.abs(P["y"][ix] - P["mu"][ix]) <= q
                w[ix] = 2 * q
            cov_rows.append({"seed": s, "fold": f, "n_cal_images": len(calp["y"]), "q_P": qP, "q_S": qS, "q_const_P": qCP, "q_const_S": qCS,
                             "P_covered": f"{int(cP[ix].sum())}/{len(ix)}", "P_coverage": float(cP[ix].mean()), "S_covered": f"{int(cS[ix].sum())}/{len(ix)}", "S_coverage": float(cS[ix].mean()),
                             "constP_coverage": float(cCP[ix].mean()), "constS_coverage": float(cCS[ix].mean()),
                             "P_mean_width": float(wP[ix].mean()), "S_mean_width": float(wS[ix].mean())})
        per_img[s] = dict(cP=cP, wP=wP, wPc=wPc, cS=cS, wS=wS, wSc=wSc, cCP=cCP, wCP=wCP, cCS=cCS, wCS=wCS)
    write_csv(OUT / "exp_c_coverage_per_fold_seed.csv", cov_rows)

    def pooled_stat(arr_key: str) -> tuple[list[float], float, float, float]:
        vals = [float(per_img[s][arr_key].mean()) for s in SEEDS]
        bs = np.mean([[per_img[s][arr_key][ix].mean() for ix in draws] for s in SEEDS], axis=0)
        lo, hi = np.percentile(bs, [2.5, 97.5])
        return vals, float(np.mean(vals)), float(lo), float(hi)

    cov_summary = []
    for label, ck, wk, wck in (("P grouped split conformal (primary)", "cP", "wP", "wPc"), ("S grouped cross-conformal (secondary)", "cS", "wS", "wSc"),
                               ("constant-width comparator (P)", "cCP", "wCP", None), ("constant-width comparator (S)", "cCS", "wCS", None)):
        vals, m, lo, hi = pooled_stat(ck)
        wvals = [float(np.median(per_img[s][wk])) for s in SEEDS]
        cov_summary.append({"method": label, "target": COVERAGE, "coverage_per_seed": vals, "coverage_mean": m, "coverage_group_boot_ci95": [lo, hi],
                            "covered_counts_per_seed": [f"{int(per_img[s][ck].sum())}/{len(names)}" for s in SEEDS],
                            "mean_width_per_seed": [float(per_img[s][wk].mean()) for s in SEEDS], "median_width_per_seed": wvals,
                            "mean_width_clipped_per_seed": None if wck is None else [float(per_img[s][wck].mean()) for s in SEEDS]})
    unc = []
    for s in SEEDS:
        P = pooled[("C", s)]
        ae = np.abs(P["y"] - P["mu"])
        tercile = np.argsort(P["sigma"], kind="stable")
        cov_t = [float(per_img[s]["cP"][t].mean()) for t in np.array_split(tercile, 3)]
        nll = float(np.mean(0.5 * (np.log(P["sigma"] ** 2) + (P["y"] - P["mu"]) ** 2 / P["sigma"] ** 2)))
        unc.append({"seed": s, "spearman_sigma_abs_error": float(sps.spearmanr(P["sigma"], ae)[0]), "nll_score_scale_no_const": nll,
                    "mean_sigma": float(P["sigma"].mean()), "P_coverage_by_sigma_tercile_low_mid_high": cov_t})
    summary["exp_c"] = {"coverage": cov_summary, "uncertainty_diagnostics": unc, "assumptions": "see STUDY_V3_PROTOCOL section 7"}
    write_csv(OUT / "exp_c_uncertainty_diagnostics.csv", [{**u, "P_coverage_by_sigma_tercile_low_mid_high": json.dumps(u["P_coverage_by_sigma_tercile_low_mid_high"])} for u in unc])

    # ---- official test (descriptive, retrospective) ------------------------------------------------------------
    test_rows, test_cov = [], []
    for (e, f, s), d in sorted(runs.items()):
        t = d / "test_retrospective" / "metrics_test.json"
        if t.is_file():
            m = json.loads(t.read_text(encoding="utf-8"))
            test_rows.append({"exp": e, "fold": f, "seed": s, **{k: m[k] for k in MET}})
            if e == "C":
                test_cov.append({"fold": f, "seed": s, "coverage_P": m["intervals"]["unclipped"]["coverage"], "mean_width": m["intervals"]["unclipped"]["mean_width"],
                                 "covered": round(m["intervals"]["unclipped"]["coverage"] * m["n"]), "n": m["n"]})
    write_csv(OUT / "official_test_descriptive_per_run.csv", test_rows)
    if test_rows:
        summary["official_test_descriptive"] = {e: {k: {"mean": float(np.mean([r[k] for r in test_rows if r["exp"] == e])),
                                                       "sd_over_seed_means": float(np.std([np.mean([r[k] for r in test_rows if r["exp"] == e and r["seed"] == s]) for s in SEEDS], ddof=1))}
                                                   for k in MET} for e in EXPS}
        if test_cov:
            summary["official_test_descriptive"]["C_coverage_P"] = {"mean": float(np.mean([r["coverage_P"] for r in test_cov])), "mean_width": float(np.mean([r["mean_width"] for r in test_cov])),
                                                                     "per_run": test_cov}

    # ---- historical comparison inputs ---------------------------------------------------------------------------
    v2 = json.loads(Path("reports/experiments/summary.json").read_text(encoding="utf-8"))
    summary["historical_reference"] = {"study_v2_validation": {e: {k: v2["final_validation_aggregate"][e][k]["mean"] for k in MET} for e in EXPS},
                                       "study_v2_test_retrospective": {e: {k: v2["final_test_aggregate"][e][k]["mean"] for k in MET} for e in EXPS},
                                       "experiment_001_test": {"plcc": 0.8667, "srocc": 0.8699, "krocc": 0.6816, "mse": 0.3291, "mae": 0.4815, "rmse": 0.5736},
                                       "experiment_003_test": {"plcc": 0.8490, "srocc": 0.8525, "krocc": 0.6611, "mse": 0.4474, "mae": 0.5195, "rmse": 0.6689}}

    # ---- figures -------------------------------------------------------------------------------------------------
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.8), sharex=True, sharey=True)
    for ax, e in zip(axs, EXPS):
        yy = pooled[(e, 1)]["y"]
        ax.scatter(yy, np.mean([pooled[(e, s)]["mu"] for s in SEEDS], axis=0), s=6, alpha=.5)
        ax.plot([0, 4], [0, 4], "k--", lw=.8)
        ax.set(title=f"{e}: held-out groups (seed-mean prediction)", xlabel="label")
    axs[0].set_ylabel("prediction")
    fig.tight_layout(); fig.savefig(OUT / "scatter_heldout.png", dpi=150); plt.close(fig)
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    for i, e in enumerate(EXPS):
        for s in SEEDS:
            ax.scatter([i + (s - 2) * 0.12] * 5, [r["plcc"] for r in per_fold_rows if r["exp"] == e and r["seed"] == s], s=12, alpha=.7)
    ax.set_xticks(range(3)); ax.set_xticklabels(EXPS); ax.set(ylabel="per-fold PLCC (held-out groups)", title="Per-fold PLCC, 3 seeds")
    fig.tight_layout(); fig.savefig(OUT / "per_fold_plcc.png", dpi=150); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 3.8))
    labels_ = [c["method"].split(" (")[0].replace("constant-width comparator", "const-width") + (" P" if "(P)" in c["method"] else " S" if "(S)" in c["method"] else "") for c in cov_summary]
    ax.bar(range(len(cov_summary)), [c["coverage_mean"] for c in cov_summary],
           yerr=[[c["coverage_mean"] - c["coverage_group_boot_ci95"][0] for c in cov_summary], [c["coverage_group_boot_ci95"][1] - c["coverage_mean"] for c in cov_summary]], capsize=3)
    ax.axhline(COVERAGE, ls="--", c="k"); ax.set_xticks(range(len(cov_summary))); ax.set_xticklabels(labels_, rotation=20, fontsize=7)
    ax.set(ylabel="pooled coverage (held-out groups)", ylim=(0, 1), title="Experiment C coverage, target 0.90 (group-bootstrap 95%)")
    fig.tight_layout(); fig.savefig(OUT / "exp_c_coverage.png", dpi=150); plt.close(fig)

    (OUT / "summary.json").write_text(json.dumps(summary, indent=1, default=str), encoding="utf-8")
    print("pooled per seed:"); [print(r["exp"], r["seed"], {k: round(r[k], 4) for k in MET}) for r in per_seed_rows]
    for r in agg_rows:
        print(r["exp"], {k: (round(r[f"{k}_mean"], 4), round(r[f"{k}_seed_sd"], 4), [round(r[f"{k}_group_boot_ci95_low"], 4), round(r[f"{k}_group_boot_ci95_high"], 4)]) for k in ("plcc", "mean_corr", "rmse")})
    for d in diffs:
        if d["metric"] in ("mean_corr", "rmse"):
            print(d["comparison"], d["metric"], round(d["difference"], 4), [round(x, 4) for x in d["group_boot_ci97_5_bonferroni"]], d["excludes_zero_at_97_5"], [round(x, 4) for x in d["per_seed_differences"]])
    for c in cov_summary:
        print(c["method"], round(c["coverage_mean"], 3), [round(x, 3) for x in c["coverage_group_boot_ci95"]], c["covered_counts_per_seed"], "width", [round(x, 2) for x in c["mean_width_per_seed"]])
    for u in unc:
        print(u)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
