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
