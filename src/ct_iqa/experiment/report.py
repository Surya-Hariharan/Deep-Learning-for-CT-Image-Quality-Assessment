"""Tables and figures for finished runs, used by the study notebooks.

Read-only presentation helpers: they take the dictionaries a run directory contains (`history.json`,
`metrics_validation.json`, `calibration.json`, `predictions_validation.csv`) and return pandas frames or matplotlib
figures. pandas and matplotlib are imported lazily so that importing `ct_iqa` never requires them.
"""

from __future__ import annotations

import math

import numpy as np

POINT_METRICS = ("plcc", "srocc", "krocc", "mse", "mae", "rmse")
_EPOCH_COLUMNS = ("epoch", "train_loss", "val_loss", "val_plcc", "val_srocc", "val_krocc", "val_mse", "val_mae",
                  "val_rmse", "selected")


def _flatten(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(_flatten(v, f"{prefix}{k}."))
        else:
            out[f"{prefix}{k}"] = v
    return out


def config_frame(**configs: dict):
    """Side-by-side table of one or more config dicts (`StudyConfig.to_dict()`), one row per `section.key`."""
    import pandas as pd

    return pd.DataFrame({name: pd.Series(_flatten(c), dtype=object) for name, c in configs.items()})


def reproducibility_frame(summary: dict):
    """What identifies a run: IDs, config, seed, device, split and weights hashes, selected epoch, artifact paths
    (relative to the repository root, as recorded); `summary` is `ct_iqa.experiment.runs.run_summary(...)`."""
    import pandas as pd

    m, run_dir = summary["manifest"], summary["run_dir"]
    w = m.get("pretrained_weights") or {}
    rows = {
        "run_id": m["run_id"], "experiment_id": m["experiment_id"], "status": m["status"],
        "started / completed (UTC)": f"{m.get('started_utc')} / {m.get('completed_utc')}",
        "git commit (dirty)": f"{(m.get('git') or {}).get('commit')} ({(m.get('git') or {}).get('dirty')})",
        "model / loss / optimizer": f"{m['model']} / {m['loss']} / {m['optimizer']}",
        "seed": m["seed"], "device requested": m["environment"].get("device_requested"),
        "cuda device": m["environment"].get("cuda_device_name", "-"), "torch": m["environment"].get("torch"),
        "bn_eps": m["bn_eps"], "split manifest": m["split_manifest"],
        "split manifest SHA-256": m["split_manifest_sha256"],
        "pretrained weights SHA-256 verified": w.get("sha256_verified"),
        "overrides on top of the YAML": m.get("overrides", "(not recorded: run predates override logging)"),
        "selection rule": m["selection_rule"], "selected epoch (1-based)": m.get("best_epoch_1based"),
        "official test set used": m["official_test_set_used"],
        "run directory": run_dir.as_posix(),
        **{f"artifact: {a}": f"{(run_dir / a).as_posix()} (sha256 {h[:12]}...)"
           for a, h in m.get("artifact_sha256", {}).items()},
    }
    return pd.DataFrame({"value": pd.Series(rows, dtype=object)})


def metrics_frame(metrics: dict):
    """One-column table of the point metrics found in a metrics dict (raw [0, 4] scale)."""
    import pandas as pd

    rows = {k: metrics[k] for k in POINT_METRICS if k in metrics}
    return pd.DataFrame({"value": rows}).round(4)


def history_frame(history: dict):
    """Per-epoch table (index = 1-based epoch) of losses, validation metrics and the selected flag."""
    import pandas as pd

    return pd.DataFrame([{k: e[k] for k in _EPOCH_COLUMNS} for e in history["epochs"]]).set_index("epoch")


def loss_parts_frame(history: dict, which: str = "train_loss_parts"):
    """Per-epoch table of the loss components (e.g. mse / rank / n_valid_pairs for Experiment B)."""
    import pandas as pd

    return pd.DataFrame([{"epoch": e["epoch"], **e[which]} for e in history["epochs"]]).set_index("epoch")


def plot_history(history: dict, title: str = ""):
    """Loss curves (left) and validation PLCC / SROCC (right); the dashed line marks the selected epoch."""
    import matplotlib.pyplot as plt

    df, best = history_frame(history), history["best_epoch"]
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(11, 3.8))
    a0.plot(df.index, df["train_loss"], label="train loss")
    a0.plot(df.index, df["val_loss"], label="validation loss")
    a0.set(xlabel="epoch (1-based)", ylabel="loss (normalized score space)", title="Loss")
    a1.plot(df.index, df["val_plcc"], label="PLCC")
    a1.plot(df.index, df["val_srocc"], label="SROCC")
    a1.set(xlabel="epoch (1-based)", ylabel="correlation", title="Validation correlation")
    for ax in (a0, a1):
        ax.axvline(best, ls="--", color="gray", label=f"selected epoch {best}")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def plot_prediction_scatter(y, pred, title: str = ""):
    """Predicted vs. true score with the identity line (both on the [0, 4] scale)."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4.6, 4.4))
    ax.scatter(y, pred, s=14, alpha=0.7)
    ax.plot([0, 4], [0, 4], "k--", lw=1, label="identity")
    ax.set(xlabel="label (0-4)", ylabel="prediction (0-4)", title=title, xlim=(-0.1, 4.1), ylim=(-0.1, 4.1))
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig


def calibration_frames(cal: dict) -> dict:
    """Split `calibration.json` into display tables: summary, unclipped / clipped interval metrics, terciles."""
    import pandas as pd

    summary = pd.DataFrame({"value": {k: cal[k] for k in ("target_coverage", "n_calibration", "rank_k", "quantile_q",
                                                         "unbounded", "nll_score_scale", "sigma_error_spearman",
                                                         "mean_sigma")}})
    intervals = pd.DataFrame({"unclipped": cal["unclipped"], "clipped to [0, 4]": cal["clipped_to_0_4"]})
    return {"summary": summary, "intervals": intervals, "terciles": pd.DataFrame(cal["coverage_by_sigma_tercile"])}


def plot_intervals(y, pred, sigma, q, target_coverage: float, terciles=None, title: str = ""):
    """Validation labels with mu +/- q*sigma (clipped to [0, 4]), sorted by label; optional tercile coverage bars."""
    import matplotlib.pyplot as plt

    y, pred, sigma = (np.asarray(a, dtype=float) for a in (y, pred, sigma))
    order = np.argsort(y, kind="stable")
    half = np.full_like(sigma, math.inf) if math.isinf(q) else q * sigma
    lo, hi = np.clip(pred - half, 0, 4)[order], np.clip(pred + half, 0, 4)[order]
    x = np.arange(len(y))
    ncols = 2 if terciles is not None else 1
    fig, axes = plt.subplots(1, ncols, figsize=(6 + 4 * (ncols - 1), 4), squeeze=False)
    ax = axes[0][0]
    ax.vlines(x, lo, hi, color="tab:blue", alpha=0.45, lw=2, label=f"interval (target {target_coverage:.0%})")
    ax.scatter(x, y[order], s=10, color="k", label="label", zorder=3)
    ax.scatter(x, pred[order], s=10, color="tab:orange", label="prediction", zorder=3)
    ax.set(xlabel="validation image (sorted by label)", ylabel="score (0-4)", title="Prediction intervals")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    if terciles is not None:
        bx = axes[0][1]
        bx.bar(terciles["tercile"].astype(str), terciles["coverage"], color="tab:blue", alpha=0.8)
        bx.axhline(target_coverage, color="k", ls="--", label="target")
        bx.set(xlabel="predicted-sigma tercile (1 = smallest)", ylabel="empirical coverage", ylim=(0, 1.05),
               title="Coverage by tercile (n per bar is small)")
        bx.legend()
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def plot_loss_parts(history: dict, title: str = ""):
    """Per-epoch loss components (train solid, validation dashed), e.g. mse / rank (B) or nll / mse (C), plus the
    total objective; a second panel shows the fraction of eligible in-batch pairs when the run logged pair counts."""
    import matplotlib.pyplot as plt

    tr, va = loss_parts_frame(history, "train_loss_parts"), loss_parts_frame(history, "val_loss_parts")
    total = history_frame(history)
    parts = [c for c in tr.columns if not c.startswith("n_")]
    pairs = "n_valid_pairs" in tr.columns and "n_possible_pairs" in tr.columns
    fig, axes = plt.subplots(1, 2 if pairs else 1, figsize=(11 if pairs else 6, 3.8), squeeze=False)
    ax = axes[0][0]
    ax.plot(total.index, total["train_loss"], color="k", label="train total")
    ax.plot(total.index, total["val_loss"], color="k", ls="--", label="validation total")
    for i, c in enumerate(parts):
        ax.plot(tr.index, tr[c], color=f"C{i}", label=f"train {c}")
        if c in va.columns:
            ax.plot(va.index, va[c], color=f"C{i}", ls="--", label=f"validation {c}")
    ax.axvline(history["best_epoch"], ls=":", color="gray", label=f"selected epoch {history['best_epoch']}")
    ax.set(xlabel="epoch (1-based)", ylabel="value (normalized score space)", title="Objective and its components")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    if pairs:
        bx = axes[0][1]
        bx.plot(tr.index, tr["n_valid_pairs"] / tr["n_possible_pairs"], label="train")
        bx.plot(va.index, va["n_valid_pairs"] / va["n_possible_pairs"], ls="--", label="validation")
        bx.set(xlabel="epoch (1-based)", ylabel="eligible / possible pairs", ylim=(0, 1.05),
               title="Eligible in-batch pairs (sample-weighted mean)")
        bx.grid(alpha=0.3)
        bx.legend()
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def plot_residuals(y, pred, title: str = ""):
    """Residual (prediction - label) histogram and residual vs label, both on the [0, 4] scale, with n."""
    import matplotlib.pyplot as plt

    y, pred = np.asarray(y, dtype=float), np.asarray(pred, dtype=float)
    r = pred - y
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(10, 3.8))
    a0.hist(r, bins=20, color="tab:blue", alpha=0.8)
    a0.axvline(0, color="k", ls="--", lw=1)
    a0.set(xlabel="residual = prediction - label (score units)", ylabel="images",
           title=f"Residual distribution (n = {len(r)}, mean = {r.mean():+.3f})")
    a1.scatter(y, r, s=12, alpha=0.7)
    a1.axhline(0, color="k", ls="--", lw=1)
    a1.set(xlabel="label (0-4)", ylabel="residual (score units)", title="Residual vs label")
    for a in (a0, a1):
        a.grid(alpha=0.3)
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def error_by_score_bin(y, pred, edges=(0.0, 1.0, 2.0, 3.0, 4.0)):
    """Error summary per label range (last bin closed on the right), with the number of images per bin."""
    import pandas as pd

    y, pred = np.asarray(y, dtype=float), np.asarray(pred, dtype=float)
    r = pred - y
    rows = []
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        last = i == len(edges) - 2
        m = (y >= lo) & ((y <= hi) if last else (y < hi))
        n = int(m.sum())
        rows.append({"label range": f"[{lo:g}, {hi:g}{']' if last else ')'}", "n": n,
                     "mean residual": float(r[m].mean()) if n else math.nan,
                     "MAE": float(np.abs(r[m]).mean()) if n else math.nan,
                     "RMSE": float(np.sqrt((r[m] ** 2).mean())) if n else math.nan})
    return pd.DataFrame(rows).set_index("label range")


def plot_sigma_vs_error(sigma, abs_error, title: str = ""):
    """Predicted sigma vs absolute error (both in score units); the dashed line is |error| = sigma."""
    import matplotlib.pyplot as plt

    sigma, abs_error = np.asarray(sigma, dtype=float), np.asarray(abs_error, dtype=float)
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    ax.scatter(sigma, abs_error, s=12, alpha=0.7)
    top = float(max(sigma.max(), abs_error.max())) * 1.05
    ax.plot([0, top], [0, top], "k--", lw=1, label="|error| = sigma")
    ax.set(xlabel="predicted sigma (score units)", ylabel="|prediction - label| (score units)",
           title=title or f"Uncertainty vs error (n = {len(sigma)})")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig


def binned_sigma_calibration(sigma, error, n_bins: int = 5):
    """Equal-count bins of predicted sigma: mean sigma vs observed RMSE of the error in each bin.

    For a well-calibrated Gaussian predictor RMSE ~ mean sigma (ratio ~ 1); with ~20 images per bin the
    estimate is noisy, so this is a diagnostic, not a test.
    """
    import pandas as pd

    sigma, error = np.asarray(sigma, dtype=float), np.asarray(error, dtype=float)
    order = np.argsort(sigma, kind="stable")
    rows = []
    for b, idx in enumerate(np.array_split(order, n_bins), start=1):
        rmse = float(np.sqrt(np.mean(error[idx] ** 2)))
        ms = float(sigma[idx].mean())
        rows.append({"sigma bin": b, "n": int(len(idx)), "mean sigma": ms, "RMSE of error": rmse,
                     "RMSE / mean sigma": rmse / ms})
    return pd.DataFrame(rows).set_index("sigma bin")


def clamp_hits(log_var_norm) -> dict:
    """How many predicted log-variances sit exactly on the clamp bounds of `ct_iqa.models.heads`."""
    from ct_iqa.models.heads import LOGVAR_MAX, LOGVAR_MIN

    s = np.asarray(log_var_norm, dtype=float)
    return {"n": int(len(s)), f"at lower bound {LOGVAR_MIN:g}": int(np.sum(s <= LOGVAR_MIN)),
            f"at upper bound {LOGVAR_MAX:g}": int(np.sum(s >= LOGVAR_MAX)),
            "min log-var": float(s.min()), "max log-var": float(s.max())}


def seed_aggregate(rows, metrics=POINT_METRICS):
    """Across-seed mean, SD (ddof = 1) and n of each metric, from one metrics dict per run.

    This is training-randomness variability for a fixed configuration; it is a different quantity from an
    image-bootstrap interval and the two must not be pooled.
    """
    import pandas as pd

    df = pd.DataFrame([{m: r[m] for m in metrics} for r in rows])
    return pd.DataFrame({"mean": df.mean(), "SD across seeds": df.std(ddof=1), "n seeds": len(df)})


def plot_bootstrap_differences(results: dict, label: str, metrics=POINT_METRICS):
    """Forest plot of paired-bootstrap differences (model minus reference) with their 95% intervals."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(metrics), figsize=(2.4 * len(metrics), 2.6))
    for ax, m in zip(np.atleast_1d(axes), metrics):
        r = results[m]
        ax.errorbar([r["difference_b_minus_a"]], [0], xerr=[[r["difference_b_minus_a"] - r["ci_low"]],
                    [r["ci_high"] - r["difference_b_minus_a"]]], fmt="o", capsize=4)
        ax.axvline(0, color="gray", ls="--")
        ax.set(title=m.upper(), yticks=[])
        ax.grid(alpha=0.3)
    fig.suptitle(label)
    fig.tight_layout()
    return fig
