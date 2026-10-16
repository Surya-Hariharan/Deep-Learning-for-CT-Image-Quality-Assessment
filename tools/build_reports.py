"""Regenerate every table and figure under `reports/experiments/` from the raw run records (nothing is hand-typed).

    python tools/build_reports.py

Inputs: `experiments/study_v2/run_registry.jsonl`, each run's `run_manifest.json`, `metrics_validation.json`,
`predictions_validation.csv`, `calibration.json`, `test_retrospective/*` (only if it exists), and the B selection records.
Roles: purpose `final` = primary comparison (seeds 1-3); `selection` = exploratory (seed 42); `smoke` = never reported.
Validation numbers share anatomy with the training set (see docs/methodology/DATASET_AND_EDA.md) and are optimistic;
test numbers (if present) are retrospective.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy import stats as sps  # noqa: E402

from ct_iqa.evaluation.metrics import compute_point_metrics  # noqa: E402
from ct_iqa.experiment.runs import verify_artifacts  # noqa: E402

ROOT = Path("experiments/study_v2")
OUT = Path("reports/experiments")
METRICS = ("plcc", "srocc", "krocc", "mse", "mae", "rmse")
EXPS = {"A": "A_reference_resnet50", "B": "B_ranking_aware_resnet50", "C": "C_uncertainty_aware_resnet50"}
N_BOOT, BOOT_SEED = 2000, 0


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def read_pred(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    out = {"filename": [r["filename"] for r in rows],
           "y": np.array([float(r["label_0_4"]) for r in rows]),
           "pred": np.array([float(r["prediction_0_4"]) for r in rows])}
    if "log_var_normalized" in rows[0]:
        out["sigma"] = 4.0 * np.exp(np.array([float(r["log_var_normalized"]) for r in rows]) / 2.0)
    return out


def load_runs() -> list[dict]:
    purposes = {}
    for line in (ROOT / "run_registry.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        purposes[Path(r["run_dir"]).name] = r["purpose"]
    runs = []
    for exp, eid in EXPS.items():
        for d in sorted((ROOT / eid / "runs").glob("*")):
            m = json.loads((d / "run_manifest.json").read_text(encoding="utf-8"))
            rk = (m.get("overrides") or {}).get("loss", {}).get("ranking", {})
            row = {"exp": exp, "run_id": d.name, "dir": d, "seed": m["seed"], "status": m["status"],
                   "purpose": purposes.get(d.name, (m.get("overrides") or {}).get("purpose", "unregistered")),
                   "lambda_rank": rk.get("lambda_rank"), "margin": rk.get("margin"), "min_score_diff": rk.get("min_score_diff"),
                   "best_epoch": m.get("best_epoch_1based"), "git_dirty": m["git"]["dirty"], "error": (m.get("error") or {}).get("type")}
            if m["status"] == "completed":
                row["artifacts_verified"] = not verify_artifacts(d)
                row.update(json.loads((d / "metrics_validation.json").read_text(encoding="utf-8")))
                row["has_test"] = (d / "test_retrospective" / "metrics_test.json").is_file()
            runs.append(row)
    return runs


def agg(rows: list[dict], metrics=METRICS) -> dict:
    out = {}
    for k in metrics:
        v = np.array([r[k] for r in rows], dtype=float)
        out[k] = {"mean": float(v.mean()), "sd": float(v.std(ddof=1)) if len(v) > 1 else float("nan"), "n": int(len(v))}
    return out


def mean_corr(m: dict) -> float:
    return float(np.mean([m["plcc"], m["srocc"], m["krocc"]]))


def seed_avg_stat(y, preds: np.ndarray, idx, stat: str) -> float:
    vals = []
    for p in preds:
        m = compute_point_metrics(y[idx], p[idx])
        vals.append(mean_corr(m) if stat == "mean_corr" else m[stat])
    return float(np.mean(vals))


def paired_boot(y, pa: np.ndarray, pb: np.ndarray, stat: str, levels=(0.95, 0.975)) -> dict:
    """Bootstrap (images resampled, same resample for both models) of  stat(B) - stat(A), averaged over seeds."""
    rng = np.random.default_rng(BOOT_SEED)
    d = []
    for _ in range(N_BOOT):
        i = rng.integers(0, len(y), len(y))
        if np.ptp(y[i]) == 0:
            continue
        d.append(seed_avg_stat(y, pb, i, stat) - seed_avg_stat(y, pa, i, stat))
    d = np.asarray(d)
    full = np.arange(len(y))
    res = {"difference": seed_avg_stat(y, pb, full, stat) - seed_avg_stat(y, pa, full, stat), "n_boot": int(len(d))}
    for lv in levels:
        lo, hi = np.percentile(d, [(1 - lv) / 2 * 100, (1 + lv) / 2 * 100])
        res[f"ci{int(lv * 1000) / 10:g}"] = [float(lo), float(hi)]
    return res


def stack_preds(rows: list[dict], name: str):
    ps = [read_pred(r["dir"] / name) for r in rows]
    assert all(p["filename"] == ps[0]["filename"] for p in ps)
    return ps[0]["y"], np.stack([p["pred"] for p in ps]), ps


def const_width_interval(cal: dict, ev: dict, coverage: float = 0.90) -> dict:
    n = len(cal["y"])
    k = math.ceil((n + 1) * coverage)
    q = float(np.sort(np.abs(cal["y"] - cal["pred"]))[k - 1]) if k <= n else float("inf")
    cov = np.abs(ev["y"] - ev["pred"]) <= q
    lo, hi = ev["pred"] - q, ev["pred"] + q
    return {"half_width": q, "coverage": float(cov.mean()), "mean_width": float(2 * q),
            "mean_width_clipped": float(np.mean(np.clip(hi, 0, 4) - np.clip(lo, 0, 4))), "n_eval": int(len(ev["y"]))}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    runs = load_runs()
    write_csv(OUT / "run_table.csv", [{k: v for k, v in r.items() if k != "dir"} for r in runs])
    done = [r for r in runs if r["status"] == "completed"]
    summary: dict = {"n_runs": len(runs), "n_completed": len(done),
                     "n_failed": sum(r["status"] == "failed" for r in runs),
                     "by_purpose": {p: sum(r["purpose"] == p for r in runs) for p in sorted({r["purpose"] for r in runs})}}

    # --- B grid (selection seed) -------------------------------------------------------------------------------
    grid = [r for r in done if r["exp"] == "B" and r["purpose"] == "selection"]
    a_ref = [r for r in done if r["exp"] == "A" and r["purpose"] == "selection"]
    rows = [{"config": "A (lambda=0 reference)", **{k: a_ref[0][k] for k in METRICS}, "mean_corr": mean_corr(a_ref[0])}] if a_ref else []
    for r in sorted(grid, key=lambda r: (r["lambda_rank"], r["margin"], r["min_score_diff"])):
        rows.append({"config": f"B lambda={r['lambda_rank']} margin={r['margin']} delta={r['min_score_diff']}",
                     **{k: r[k] for k in METRICS}, "mean_corr": mean_corr(r), "best_epoch": r["best_epoch"], "run_id": r["run_id"]})
    write_csv(OUT / "b_grid_selection_seed42.csv", rows)
    if grid:
        fig, ax = plt.subplots(figsize=(5.5, 4))
        ax.scatter([r["rmse"] for r in grid], [mean_corr(r) for r in grid], c="C1", label="B grid points")
        if a_ref:
            ax.scatter([a_ref[0]["rmse"]], [mean_corr(a_ref[0])], c="C0", marker="s", label="A")
            ax.axvline(1.05 * a_ref[0]["rmse"], ls="--", c="gray", label="RMSE guard (1.05 x A)")
        ax.set(xlabel="validation RMSE", ylabel="mean(PLCC, SROCC, KROCC)", title="B grid vs A, seed 42 (validation, optimistic)")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(OUT / "b_grid_tradeoff.png", dpi=150)
        plt.close(fig)

    # --- final runs: per seed table + aggregate ----------------------------------------------------------------
    final = {e: [r for r in done if r["exp"] == e and r["purpose"] == "final"] for e in EXPS}
    per_seed = [{"exp": e, "seed": r["seed"], "run_id": r["run_id"], "best_epoch": r["best_epoch"], "partition": "validation",
                 **{k: r[k] for k in METRICS}} for e in EXPS for r in sorted(final[e], key=lambda r: r["seed"])]
    write_csv(OUT / "final_per_seed_validation.csv", per_seed)
    aggs = {e: agg(final[e]) for e in EXPS if final[e]}
    write_csv(OUT / "final_aggregate_validation.csv",
              [{"exp": e, "n_seeds": aggs[e]["plcc"]["n"], **{f"{k}_{s}": aggs[e][k][s] for k in METRICS for s in ("mean", "sd")}} for e in aggs])
    summary["final_validation_aggregate"] = aggs

    # --- test (retrospective) ----------------------------------------------------------------------------------
    test_rows = []
    for e in EXPS:
        for r in sorted(final[e], key=lambda r: r["seed"]):
            f = r["dir"] / "test_retrospective" / "metrics_test.json"
            if f.is_file():
                t = json.loads(f.read_text(encoding="utf-8"))
                test_rows.append({"exp": e, "seed": r["seed"], "partition": "official_test (retrospective)", "n": t["n"],
                                  **{k: t[k] for k in METRICS}})
    write_csv(OUT / "final_per_seed_test_retrospective.csv", test_rows)
    if test_rows:
        summary["final_test_aggregate"] = {e: agg([x for x in test_rows if x["exp"] == e]) for e in EXPS if any(x["exp"] == e for x in test_rows)}

    # --- paired comparisons on common images -------------------------------------------------------------------
    comps = {}
    for part, fname, sel in (("validation", "predictions_validation.csv", lambda r: r),
                             ("test_retrospective", "test_retrospective/predictions_test.csv", lambda r: r if r["has_test"] else None)):
        data = {}
        for e in EXPS:
            rr = [r for r in final[e] if sel(r) is not None]
            if len(rr) >= 2:
                data[e] = stack_preds(rr, fname)[:2]
        for other in ("B", "C"):
            if "A" in data and other in data and np.array_equal(data["A"][0], data[other][0]):
                y = data["A"][0]
                for stat in ("mean_corr", "rmse"):
                    comps[f"{part}: {other} - A [{stat}]"] = paired_boot(y, data["A"][1], data[other][1], stat)
    summary["paired_bootstrap_vs_A"] = {"note": "seed-averaged statistic; images resampled (same resample for both models); ci97.5 is the "
                                        "Bonferroni-adjusted interval for the two pre-declared comparisons; intervals are optimistic "
                                        "(same-anatomy images are treated as independent)", "results": comps}

    # --- scatter plots + per-score-bin errors ------------------------------------------------------------------
    bins = [(0, 1), (1, 2), (2, 3), (3, 4.01)]
    bin_rows, worst = [], []
    for part, fname in (("validation", "predictions_validation.csv"), ("test_retrospective", "test_retrospective/predictions_test.csv")):
        fig, axs = plt.subplots(1, 3, figsize=(12, 3.8), sharex=True, sharey=True)
        any_ = False
        for ax, e in zip(axs, EXPS):
            rr = [r for r in final[e] if (r["dir"] / fname).is_file()]
            if not rr:
                ax.set_title(f"{e}: n/a")
                continue
            y, P, _ = stack_preds(rr, fname)
            p = P.mean(0)
            any_ = True
            ax.scatter(y, p, s=8, alpha=.6)
            ax.plot([0, 4], [0, 4], "k--", lw=.8)
            ax.set(title=f"{e}: seed-mean prediction ({part})", xlabel="label")
            for lo, hi in bins:
                m = (y >= lo) & (y < hi)
                if m.sum():
                    err = P[:, m] - y[m]
                    bin_rows.append({"exp": e, "partition": part, "score_bin": f"[{lo},{min(hi, 4)}{']' if hi > 4 else ')'}", "n_images": int(m.sum()),
                                     "rmse": float(np.sqrt((err ** 2).mean())), "bias": float(err.mean())})
            ae = np.abs(P - y).mean(0)
            names = read_pred(rr[0]["dir"] / fname)["filename"]
            for i in np.argsort(-ae)[:10]:
                worst.append({"exp": e, "partition": part, "filename": names[i], "label": float(y[i]), "mean_prediction": float(p[i]),
                              "mean_abs_error_over_seeds": float(ae[i])})
        axs[0].set_ylabel("prediction")
        fig.tight_layout()
        if any_:
            fig.savefig(OUT / f"scatter_{part}.png", dpi=150)
        plt.close(fig)
    write_csv(OUT / "error_by_score_bin.csv", bin_rows)
    write_csv(OUT / "worst_cases.csv", worst)

    # --- Experiment C: uncertainty ------------------------------------------------------------------------------
    unc = []
    for r in sorted(final["C"], key=lambda r: r["seed"]):
        cal = json.loads((r["dir"] / "calibration.json").read_text(encoding="utf-8"))
        val_p, cal_p = read_pred(r["dir"] / "predictions_validation.csv"), read_pred(r["dir"] / "predictions_calibration.csv")
        row = {"seed": r["seed"], "n_calibration": cal["n_calibration"], "rank_k": cal["rank_k"], "q": cal["quantile_q"],
               "val_n": len(val_p["y"]), "val_coverage_unclipped": cal["unclipped"]["coverage"],
               "val_mean_width_unclipped": cal["unclipped"]["mean_width"],
               "val_coverage_clipped": cal["clipped_to_0_4"]["coverage"], "val_mean_width_clipped": cal["clipped_to_0_4"]["mean_width"],
               "val_sigma_error_spearman": cal["sigma_error_spearman"], "val_nll_score_scale": cal["nll_score_scale"],
               "val_mean_sigma": cal["mean_sigma"],
               "val_tercile_coverage": json.dumps([round(t["coverage"], 3) for t in cal["coverage_by_sigma_tercile"]])}
        cw = const_width_interval(cal_p, val_p)
        row.update({"constwidth_val_coverage": cw["coverage"], "constwidth_val_mean_width": cw["mean_width"],
                    "constwidth_val_mean_width_clipped": cw["mean_width_clipped"]})
        tf = r["dir"] / "test_retrospective" / "metrics_test.json"
        if tf.is_file():
            t = json.loads(tf.read_text(encoding="utf-8"))["intervals"]
            test_p = read_pred(r["dir"] / "test_retrospective" / "predictions_test.csv")
            cwt = const_width_interval(cal_p, test_p)
            row.update({"test_n": len(test_p["y"]), "test_coverage_unclipped": t["unclipped"]["coverage"],
                        "test_mean_width_unclipped": t["unclipped"]["mean_width"], "test_coverage_clipped": t["clipped_to_0_4"]["coverage"],
                        "test_mean_width_clipped": t["clipped_to_0_4"]["mean_width"], "test_sigma_error_spearman": t["sigma_error_spearman"],
                        "constwidth_test_coverage": cwt["coverage"], "constwidth_test_mean_width": cwt["mean_width"]})
        unc.append(row)
    write_csv(OUT / "exp_c_uncertainty_per_seed.csv", unc)
    if unc:
        summary["exp_c_uncertainty_mean"] = {k: float(np.mean([u[k] for u in unc])) for k in unc[0]
                                              if isinstance(unc[0][k], (int, float)) and k != "seed" and all(k in u for u in unc)}
        fig, axs = plt.subplots(1, 2, figsize=(10, 3.8))
        for r in final["C"]:
            vp = read_pred(r["dir"] / "predictions_validation.csv")
            axs[0].scatter(vp["sigma"], np.abs(vp["y"] - vp["pred"]), s=8, alpha=.5, label=f"seed {r['seed']}")
        axs[0].set(xlabel="predicted sigma", ylabel="|error|", title="Experiment C: sigma vs error (validation)")
        axs[0].legend(fontsize=8)
        axs[1].bar(["val unclipped", "val clipped"] + (["test unclipped"] if "test_coverage_unclipped" in unc[0] else []),
                   [np.mean([u["val_coverage_unclipped"] for u in unc]), np.mean([u["val_coverage_clipped"] for u in unc])]
                   + ([np.mean([u["test_coverage_unclipped"] for u in unc])] if "test_coverage_unclipped" in unc[0] else []))
        axs[1].axhline(0.90, ls="--", c="k")
        axs[1].set(ylabel="mean coverage over seeds", ylim=(0, 1), title="Interval coverage (target 0.90)")
        fig.tight_layout()
        fig.savefig(OUT / "exp_c_uncertainty.png", dpi=150)
        plt.close(fig)

    # --- training curves ----------------------------------------------------------------------------------------
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.6), sharey=True)
    for ax, e in zip(axs, EXPS):
        for r in final[e]:
            h = json.loads((r["dir"] / "history.json").read_text(encoding="utf-8"))["epochs"]
            ax.plot([x["epoch"] for x in h], [x["val_plcc"] for x in h], label=f"seed {r['seed']}")
        ax.set(title=f"{e}: validation PLCC", xlabel="epoch")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "training_curves_validation_plcc.png", dpi=150)
    plt.close(fig)

    (OUT / "summary.json").write_text(json.dumps(summary, indent=1, default=str), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "paired_bootstrap_vs_A"}, indent=1, default=str)[:3000])
    print(json.dumps(summary["paired_bootstrap_vs_A"]["results"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
