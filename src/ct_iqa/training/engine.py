"""Study-v2 training engine: one loop for Experiments A, B and C.

Model, loss, optimizer and selection policy are injected, so no experiment needs its own loop.
The legacy `ct_iqa.training.trainer.train` is kept unchanged for the historical notebooks.

Access rules enforced here: `fit` receives a train loader and a validation loader only. Calibration data and the
official test set are never passed in, so they cannot influence updates or checkpoint selection.

Scales: the model and loss work in the NORMALIZED [0, 1] score space; every reported metric (PLCC, SROCC, KROCC,
MSE, MAE, RMSE) is computed on the original [0, 4] scale after `denormalize_score`. `val_loss` is the training
objective on the validation set (normalized space) and is only a diagnostic unless selected explicitly.
"""

from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from ct_iqa.data.ldct_iqac import denormalize_score, normalize_score
from ct_iqa.evaluation.metrics import compute_point_metrics
from ct_iqa.models.registry import ModelOutput
from ct_iqa.training.checkpointing import save_run_checkpoint
from ct_iqa.training.optimizers import build_study_optimizer
from ct_iqa.training.selection import Selector

logger = logging.getLogger(__name__)


@dataclass
class EvalResult:
    loss: float
    y_raw: np.ndarray             # labels, [0, 4]
    mean_norm: np.ndarray         # predicted mean, [0, 1]
    log_var_norm: np.ndarray | None
    loss_parts: dict[str, float]

    @property
    def pred_raw(self) -> np.ndarray:
        return np.asarray(denormalize_score(self.mean_norm), dtype=np.float64)


def _accumulate(total: dict[str, float], parts: dict[str, float], weight: float) -> None:
    for k, v in parts.items():
        total[k] = total.get(k, 0.0) + v * weight


@torch.no_grad()
def evaluate_loader(model: nn.Module, loader: DataLoader, loss_fn, device: str) -> EvalResult:
    model.eval()
    ys, means, logvars = [], [], []
    loss_sum, n, parts_sum = 0.0, 0, {}
    for images, raw in loader:
        images, raw = images.to(device), raw.to(device)
        out: ModelOutput = model(images)
        res = loss_fn(out, normalize_score(raw), raw)
        b = images.shape[0]
        loss_sum += float(res.total) * b
        n += b
        _accumulate(parts_sum, res.parts, b)
        ys.append(raw.cpu().numpy()); means.append(out.mean.cpu().numpy())
        if out.log_var is not None:
            logvars.append(out.log_var.cpu().numpy())
    return EvalResult(
        loss=loss_sum / n, y_raw=np.concatenate(ys).astype(np.float64),
        mean_norm=np.concatenate(means).astype(np.float64),
        log_var_norm=np.concatenate(logvars).astype(np.float64) if logvars else None,
        loss_parts={k: v / n for k, v in parts_sum.items()},
    )


def _train_one_epoch(model, loader, loss_fn, optimizer, device) -> tuple[float, dict[str, float]]:
    model.train()
    loss_sum, n, parts_sum = 0.0, 0, {}
    for images, raw in loader:
        images, raw = images.to(device), raw.to(device)
        optimizer.zero_grad(set_to_none=True)
        res = loss_fn(model(images), normalize_score(raw), raw)
        if not torch.isfinite(res.total):
            raise FloatingPointError(f"non-finite training loss: {res.parts}")
        res.total.backward()
        optimizer.step()
        b = images.shape[0]
        loss_sum += float(res.total.detach()) * b
        n += b
        _accumulate(parts_sum, res.parts, b)
    return loss_sum / n, {k: v / n for k, v in parts_sum.items()}


def fit(model: nn.Module, loss_fn, train_loader: DataLoader, val_loader: DataLoader, cfg, run_dir: Path,
        run_id: str) -> dict:
    """Train for `cfg.train.epochs` epochs; save the validation-selected checkpoint to `<run_dir>/checkpoints/best.pt`.

    Returns the history dict (also written to `<run_dir>/history.json`).
    """
    device = cfg.train.device
    model.to(device)
    optimizer = build_study_optimizer(model, cfg.optim)
    selector = Selector(cfg.train.selection_metric)
    epochs: list[dict] = []
    ckpt_path = Path(run_dir) / "checkpoints" / "best.pt"

    for epoch in range(1, cfg.train.epochs + 1):
        train_loss, train_parts = _train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        ev = evaluate_loader(model, val_loader, loss_fn, device)
        pm = compute_point_metrics(ev.y_raw, ev.pred_raw)
        record = {
            "epoch": epoch, "train_loss": train_loss, "val_loss": ev.loss,
            "val_plcc": pm["plcc"], "val_srocc": pm["srocc"], "val_krocc": pm["krocc"],
            "val_mse": pm["mse"], "val_mae": pm["mae"], "val_rmse": pm["rmse"],
            "train_loss_parts": train_parts, "val_loss_parts": ev.loss_parts,
        }
        selected = selector.update(epoch, record)
        record["selected"] = selected
        epochs.append(record)
        if selected:
            save_run_checkpoint(ckpt_path, model, epoch_1based=epoch, metrics={
                k: record[k] for k in ("val_loss", "val_plcc", "val_srocc", "val_krocc", "val_mse", "val_mae", "val_rmse")},
                config=cfg.to_dict(), run_id=run_id, selection_policy=cfg.train.selection_metric)
        logger.info("epoch %d/%d train_loss=%.5f val_loss=%.5f val_plcc=%.4f%s", epoch, cfg.train.epochs,
                    train_loss, ev.loss, pm["plcc"], " *" if selected else "")
        _write_history(run_dir, cfg, selector, epochs, complete=False)   # survives a crash in a later epoch

    if selector.best_epoch == 0:
        raise RuntimeError("no checkpoint was selected (all selection metrics were NaN)")
    return _write_history(run_dir, cfg, selector, epochs, complete=True)


def _write_history(run_dir: Path, cfg, selector: Selector, epochs: list[dict], complete: bool) -> dict:
    """Atomically (temp file + os.replace) write `<run_dir>/history.json`; `complete` is False until training ends."""
    history = {
        "epoch_indexing": "1-based", "selection_policy": cfg.train.selection_metric, "complete": complete,
        "best_epoch": selector.best_epoch,
        "best_value": None if math.isnan(selector.best_value) else selector.best_value,
        "epochs": epochs,
    }
    path = Path(run_dir) / "history.json"
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(history, indent=1), encoding="utf-8")
    os.replace(tmp, path)
    return history
