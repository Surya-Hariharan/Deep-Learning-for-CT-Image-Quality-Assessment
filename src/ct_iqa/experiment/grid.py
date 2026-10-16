"""Experiment B hyperparameter grid: the declared values, run identity, and the declared selection rule.

The grid and rule are transcribed from docs/experiments/b-ranking-aware.md section 7, which was written before any
B run; `tests/unit/test_runs_and_grid.py` checks that this module and the document agree. Nothing here invents
values, and only train / validation artifacts are read: calibration and official-test data are never touched.

Every (grid point, seed) is an independent `run_experiment` call, so each gets its own immutable run directory
whose `run_manifest.json` records the overrides. Selection reads only completed runs and refuses ambiguity
(a missing point, or two completed runs of the same point and seed).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ct_iqa.config.loader import deep_merge, study_config_from_dict
from ct_iqa.config.schema import StudyConfig
from ct_iqa.evaluation.metrics import compute_point_metrics
from ct_iqa.experiment.metadata import utc_now
from ct_iqa.experiment.runs import read_predictions, run_summary, runs_table

B_EXPERIMENT_ID = "B_ranking_aware_resnet50"
GRID_DOC = "docs/experiments/b-ranking-aware.md"

# Stage 1: lambda sweep at fixed margin / delta.
STAGE1_LAMBDAS = (0.03, 0.1, 0.3, 1.0)
STAGE1_MARGIN = 0.05            # normalized [0, 1] score space
STAGE1_MIN_SCORE_DIFF = 0.25    # raw [0, 4] score units
# Stage 2: one-at-a-time sensitivity at the lambda selected in stage 1.
STAGE2_MARGINS = (0.02, 0.1)
STAGE2_MIN_SCORE_DIFFS = (0.5,)
# Selection rule constants.
RMSE_TOLERANCE = 0.05           # validation RMSE within 5% of A's
CORRELATIONS = ("plcc", "srocc", "krocc")


def b_grid_stage1() -> list[dict]:
    return [{"lambda_rank": lam, "margin": STAGE1_MARGIN, "min_score_diff": STAGE1_MIN_SCORE_DIFF}
            for lam in STAGE1_LAMBDAS]


def b_grid_stage2(lambda_selected: float) -> list[dict]:
    if lambda_selected not in STAGE1_LAMBDAS:
        raise ValueError(f"lambda {lambda_selected} is not a stage-1 grid value {STAGE1_LAMBDAS}")
    return ([{"lambda_rank": lambda_selected, "margin": m, "min_score_diff": STAGE1_MIN_SCORE_DIFF}
             for m in STAGE2_MARGINS]
            + [{"lambda_rank": lambda_selected, "margin": STAGE1_MARGIN, "min_score_diff": d}
               for d in STAGE2_MIN_SCORE_DIFFS])


def apply_overrides(cfg: StudyConfig, overrides: dict) -> StudyConfig:
    """Return a new validated config with `overrides` (nested dict) merged in; unknown keys are rejected."""
    return study_config_from_dict(deep_merge(cfg.to_dict(), overrides))


def grid_overrides(point: dict, seed: int, device: str) -> dict:
    return {"loss": {"name": "mse_ranking", "ranking": dict(point)}, "train": {"seed": seed, "device": device}}


def _point_key(row: dict) -> tuple:
    return (row["lambda_rank"], row["margin"], row["min_score_diff"], row["seed"])


def completed_point_runs(points: list[dict], seed: int, output_root: str | Path = "experiments/study_v2") -> dict:
    """{index in `points`: [run_ids of completed runs with exactly that point and seed]}."""
    done: dict[tuple, list[str]] = {}
    for r in runs_table(B_EXPERIMENT_ID, output_root):
        if r.get("status") == "completed" and r.get("lambda_rank") is not None:
            done.setdefault(_point_key(r), []).append(r["run_id"])
    return {i: done.get((p["lambda_rank"], p["margin"], p["min_score_diff"], seed), []) for i, p in enumerate(points)}


def pending_points(points: list[dict], seed: int, output_root: str | Path = "experiments/study_v2") -> list[dict]:
    """Grid points that have no completed run at `seed` yet (so a re-run never duplicates finished points)."""
    done = completed_point_runs(points, seed, output_root)
    return [p for i, p in enumerate(points) if not done[i]]


def run_grid(cfg: StudyConfig, points: list[dict], seeds: list[int], device: str, config_path: str | Path) -> list[Path]:
    """Run each (point, seed) as its own experiment run, in order. The first failure is re-raised (its run
    directory is marked failed); runs completed before it are kept."""
    from ct_iqa.experiment.run import run_experiment

    out = []
    for seed in seeds:
        for point in points:
            ov = grid_overrides(point, seed, device)
            print(f"--- grid run: {point} seed={seed} device={device}", flush=True)
            out.append(run_experiment(apply_overrides(cfg, ov), config_path=config_path, overrides=ov))
    return out


def _mean_corr(m: dict) -> float:
    return float(np.mean([m[k] for k in CORRELATIONS]))


def bootstrap_mean_corr_difference(y, pred_a, pred_b, n_boot: int = 2000, seed: int = 0,
                                   confidence: float = 0.95) -> dict:
    """Paired image-level bootstrap of  mean(PLCC, SROCC, KROCC)[B] - mean(...)[A]  (same resampling scheme as
    `ct_iqa.evaluation.bootstrap.paired_bootstrap_difference`; optimistic because patient grouping is unknown)."""
    y, a, b = (np.asarray(v, dtype=np.float64) for v in (y, pred_a, pred_b))
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if np.ptp(y[idx]) == 0:
            continue
        diffs.append(_mean_corr(compute_point_metrics(y[idx], b[idx])) - _mean_corr(compute_point_metrics(y[idx], a[idx])))
    d = np.asarray(diffs)
    d = d[np.isfinite(d)]
    lo, hi = np.percentile(d, [(1 - confidence) / 2 * 100, (1 + confidence) / 2 * 100])
    return {"difference": _mean_corr(compute_point_metrics(y, b)) - _mean_corr(compute_point_metrics(y, a)),
            "ci_low": float(lo), "ci_high": float(hi), "half_width": float((hi - lo) / 2), "n_boot_used": int(len(d))}


def select_stage1(a_run_dir: str | Path, seed: int, output_root: str | Path = "experiments/study_v2") -> dict:
    """Apply the declared stage-1 rule to completed validation results; return a selection record (not written).

    Rule (b-ranking-aware.md section 7): among grid points whose validation RMSE is within 5% of A's, take the
    highest mean of validation (PLCC, SROCC, KROCC); if it does not exceed A by more than the paired-bootstrap 95%
    half-width, the outcome is "no demonstrated improvement" and A stays the recommended model.
    Operationalization used here: the bootstrap is of the mean-of-three-correlations difference (B - A).
    """
    a = run_summary(a_run_dir)
    if a["manifest"]["loss"] != "mse":
        raise ValueError(f"{a_run_dir} is not an Experiment A (mse) run")
    points = b_grid_stage1()
    done = completed_point_runs(points, seed, output_root)
    missing = [points[i] for i, ids in done.items() if not ids]
    dupes = {i: ids for i, ids in done.items() if len(ids) > 1}
    if missing:
        raise ValueError(f"stage 1 is incomplete at seed {seed}; no completed run for {missing}")
    if dupes:
        raise ValueError(f"ambiguous: several completed runs for one grid point at seed {seed}: {dupes}")
    root = Path(output_root) / B_EXPERIMENT_ID / "runs"
    a_rmse, a_corr = a["metrics"]["rmse"], _mean_corr(a["metrics"])
    rows = []
    for i, p in enumerate(points):
        s = run_summary(root / done[i][0])
        m = s["metrics"]
        rows.append({**p, "seed": seed, "run_id": done[i][0], "best_epoch": s["manifest"]["best_epoch_1based"],
                     **{k: m[k] for k in ("plcc", "srocc", "krocc", "mse", "mae", "rmse")},
                     "mean_corr": _mean_corr(m), "rmse_within_5pct_of_A": m["rmse"] <= (1 + RMSE_TOLERANCE) * a_rmse})
    eligible = [r for r in rows if r["rmse_within_5pct_of_A"]]
    record = {"created_utc": utc_now(), "experiment_id": B_EXPERIMENT_ID, "stage": 1, "seed": seed,
              "partition": "validation (also used for checkpoint selection: selection-biased, exploratory)",
              "rule_source": GRID_DOC + " section 7",
              "rule": "RMSE within 5% of A; highest mean(PLCC, SROCC, KROCC); improvement only if the paired-bootstrap "
                      "mean-correlation difference exceeds its 95% half-width",
              "reference_run": a["run_dir"].name, "reference_rmse": a_rmse, "reference_mean_corr": a_corr,
              "candidates": rows}
    if not eligible:
        record.update(selected=None, outcome="no grid point passes the RMSE guard; A remains the recommended model")
        return record
    best = max(eligible, key=lambda r: r["mean_corr"])           # ties keep the earlier (smaller-lambda) point
    pa = read_predictions(a["run_dir"])
    pb = read_predictions(root / best["run_id"])
    if pa["filename"] != pb["filename"]:
        raise ValueError("A and B validation predictions are not on the same images in the same order")
    boot = bootstrap_mean_corr_difference(pa["label_0_4"], pa["prediction_0_4"], pb["prediction_0_4"])
    improved = boot["difference"] > boot["half_width"]
    record.update(selected=best, bootstrap_vs_A=boot, improvement_demonstrated=bool(improved),
                  outcome=("selected configuration exceeds A by more than the bootstrap half-width" if improved else
                           "no demonstrated improvement: A remains the recommended model"))
    return record


def write_selection_record(record: dict, output_root: str | Path = "experiments/study_v2") -> Path:
    """Write `<output_root>/B_ranking_aware_resnet50/selection_record_stage<k>.json`; never overwrites."""
    path = Path(output_root) / B_EXPERIMENT_ID / f"selection_record_stage{record['stage']}.json"
    if path.exists():
        raise FileExistsError(f"{path} already exists; a selection record is never overwritten")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as f:
        json.dump(record, f, indent=1, default=str)
    return path
