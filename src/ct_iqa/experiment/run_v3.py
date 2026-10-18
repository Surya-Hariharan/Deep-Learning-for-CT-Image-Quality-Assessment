"""Study v3 orchestration: grouped 5-fold runs (train / validation / calibration / evaluation) in isolated run directories.

Differences from `ct_iqa.experiment.run` (study v2, untouched): the split comes from the frozen group+fold manifests, every run
also predicts the held-out *evaluation* groups, and Experiment C's conformal quantile (fitted on the fold's calibration groups) is
evaluated on those held-out evaluation groups. The official test set is reachable only through `final_test_v3` (marker + flag).
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from ct_iqa.config.schema import StudyConfig
from ct_iqa.data.grouped import GROUP_PARTITIONS, expand_fold, verify_grouped_manifest
from ct_iqa.data.ldct_iqac import LDCTIQACDataset
from ct_iqa.evaluation.intervals import calibrate_and_evaluate, evaluate_intervals
from ct_iqa.evaluation.metrics import compute_point_metrics
from ct_iqa.experiment.metadata import config_hash, environment_info, git_info, sha256_path, utc_now
from ct_iqa.experiment.run import _dump, _write_predictions
from ct_iqa.losses.registry import build_loss
from ct_iqa.models.registry import build_model
from ct_iqa.training.checkpointing import load_run_checkpoint
from ct_iqa.training.engine import evaluate_loader, fit
from ct_iqa.utils.seed import set_seed

V3_ROOT = Path("experiments/study_v3")
GROUP_MANIFEST = V3_ROOT / "protocol/group_manifest.json"
FOLD_MANIFEST = V3_ROOT / "protocol/fold_manifest.json"
FROZEN_MARKER = V3_ROOT / "protocol/FROZEN"
logger = logging.getLogger(__name__)


def load_v3_manifests(labels: dict[str, float]) -> tuple[dict, dict]:
    gm = json.loads(GROUP_MANIFEST.read_text(encoding="utf-8"))
    fm = json.loads(FOLD_MANIFEST.read_text(encoding="utf-8"))
    verify_grouped_manifest(fm, gm, labels)
    return gm, fm


def make_run_dir_v3(cfg: StudyConfig) -> tuple[Path, str]:
    stamp = utc_now().replace("-", "").replace(":", "").replace("+0000", "Z").replace("+00:00", "Z")
    run_id = f"{cfg.run.experiment_id}_{stamp}_s{cfg.train.seed}_{config_hash(cfg.to_dict())[:6]}"
    run_dir = Path(cfg.run.output_root) / cfg.run.experiment_id / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir, run_id


def run_v3_experiment(cfg: StudyConfig, fold: int, overrides: dict | None = None, require_frozen: bool = True) -> Path:
    cfg.validate()
    labels = json.loads(Path(cfg.data.train_json_path).read_text(encoding="utf-8"))
    gm, fm = load_v3_manifests(labels)
    if require_frozen and not FROZEN_MARKER.exists():
        raise PermissionError("study v3 protocol is not frozen (experiments/study_v3/protocol/FROZEN missing)")
    parts = expand_fold(fm["folds"][fold], gm["groups"])
    run_dir, run_id = make_run_dir_v3(cfg)
    _dump(run_dir / "config.snapshot.json", cfg.to_dict())
    manifest = {
        "run_id": run_id, "experiment_id": cfg.run.experiment_id, "study": "v3_grouped", "fold": fold, "status": "started",
        "started_utc": utc_now(), "git": git_info("."), "environment": environment_info(cfg.train.device),
        "seed": cfg.train.seed, "group_manifest_sha256": gm["manifest_sha256"], "fold_manifest_sha256": fm["manifest_sha256"],
        "partition_sizes": {p: len(v) for p, v in parts.items()}, "model": cfg.model.name, "loss": cfg.loss.name,
        "bn_eps": cfg.model.bn_eps, "selection_rule": cfg.train.selection_metric, "epoch_indexing": "1-based",
        "official_test_set_used": False, "overrides": dict(overrides or {}),
    }
    mpath = run_dir / "run_manifest.json"
    _dump(mpath, manifest)
    try:
        _train_eval(cfg, run_dir, run_id, parts, mpath, manifest)
    except BaseException as e:
        manifest.update({"status": "failed", "failed_utc": utc_now(), "error": {"type": type(e).__name__, "message": str(e)}})
        _dump(mpath, manifest)
        raise
    return run_dir


def _loaders(cfg: StudyConfig, parts: dict[str, list[str]]):
    full = LDCTIQACDataset(image_dir=cfg.data.train_image_dir, json_path=cfg.data.train_json_path, image_size=cfg.data.image_size)
    names = [s.filename for s in full.samples]
    pos = {n: i for i, n in enumerate(names)}
    idx = {p: [pos[n] for n in names if n in set(parts[p])] for p in GROUP_PARTITIONS}
    gen = torch.Generator().manual_seed(cfg.train.seed)
    bs = cfg.train.batch_size
    loaders = {"train": DataLoader(Subset(full, idx["train"]), batch_size=bs, shuffle=True, drop_last=False, generator=gen)}
    for p in ("validation", "calibration", "evaluation"):
        loaders[p] = DataLoader(Subset(full, idx[p]), batch_size=bs, shuffle=False)
    return loaders, {p: [names[i] for i in idx[p]] for p in GROUP_PARTITIONS}


def _train_eval(cfg, run_dir: Path, run_id: str, parts, mpath: Path, manifest: dict) -> None:
    set_seed(cfg.train.seed)
    loaders, names = _loaders(cfg, parts)
    model, report = build_model(cfg.model, cfg.data.image_size)
    manifest["pretrained_weights"] = None if report is None else {"path": report.weights_path, "sha256": report.sha256,
                                                                    "sha256_verified": report.sha256_verified}
    loss_fn = build_loss(cfg.loss)
    history = fit(model, loss_fn, loaders["train"], loaders["validation"], cfg, run_dir, run_id)
    meta = load_run_checkpoint(model, run_dir / "checkpoints" / "best.pt", map_location=cfg.train.device)
    model.to(cfg.train.device)
    gaussian = cfg.model.name == "resnet50_gaussian"
    artifacts = ["config.snapshot.json", "history.json", "checkpoints/best.pt"]
    evs = {}
    for p in ("validation", "evaluation") + (("calibration",) if gaussian else ()):
        ev = evaluate_loader(model, loaders[p], loss_fn, cfg.train.device)
        evs[p] = ev
        _write_predictions(run_dir / f"predictions_{p}.csv", names[p], ev, gaussian)
        artifacts.append(f"predictions_{p}.csv")
        if p != "calibration":
            _dump(run_dir / f"metrics_{p}.json", {"partition": p, "n": len(ev.y_raw), "scale": "raw [0, 4]",
                                                  "epoch_1based": meta["epoch"], **compute_point_metrics(ev.y_raw, ev.pred_raw)})
            artifacts.append(f"metrics_{p}.json")
    if gaussian:
        cal = calibrate_and_evaluate(evs["calibration"].y_raw, evs["calibration"].mean_norm, evs["calibration"].log_var_norm,
                                     evs["evaluation"].y_raw, evs["evaluation"].mean_norm, evs["evaluation"].log_var_norm,
                                     cfg.calibration.target_coverage)
        cal["fitted_on"] = "calibration groups of this fold"
        cal["evaluated_on"] = "evaluation groups of this fold (disjoint from train, validation and calibration groups)"
        _dump(run_dir / "calibration.json", cal)
        artifacts.append("calibration.json")
    manifest.update({"status": "completed", "completed_utc": utc_now(), "best_epoch_1based": history["best_epoch"],
                     "validation_metrics": json.loads((run_dir / "metrics_validation.json").read_text(encoding="utf-8")),
                     "evaluation_metrics": json.loads((run_dir / "metrics_evaluation.json").read_text(encoding="utf-8")),
                     "artifact_sha256": {a: sha256_path(run_dir / a) for a in artifacts}})
    _dump(mpath, manifest)


def final_test_v3(run_dir: str | Path, acknowledge_retrospective: bool) -> Path:
    """Descriptive evaluation of a finished v3 run on the previously examined official test set (never confirmatory)."""
    if not acknowledge_retrospective:
        raise PermissionError("acknowledge_retrospective=True required: the official test set was examined before study v3")
    if not FROZEN_MARKER.is_file():
        raise PermissionError("study v3 protocol is not frozen")
    run_dir = Path(run_dir)
    from ct_iqa.config.loader import study_config_from_dict
    cfg = study_config_from_dict(json.loads((run_dir / "config.snapshot.json").read_text(encoding="utf-8")))
    out = run_dir / "test_retrospective"
    out.mkdir(exist_ok=False)
    model, _ = build_model(cfg.model, cfg.data.image_size)
    load_run_checkpoint(model, run_dir / "checkpoints" / "best.pt", map_location=cfg.train.device)
    model.to(cfg.train.device)
    ds = LDCTIQACDataset(cfg.data.test_image_dir, cfg.data.test_json_path, cfg.data.image_size)
    ev = evaluate_loader(model, DataLoader(ds, batch_size=cfg.train.batch_size, shuffle=False), build_loss(cfg.loss), cfg.train.device)
    res = {"partition": "official_test", "evaluation_status": "descriptive, retrospective (set examined before study v3)",
           "n": len(ev.y_raw), "scale": "raw [0, 4]", **compute_point_metrics(ev.y_raw, ev.pred_raw)}
    gaussian = ev.log_var_norm is not None
    _write_predictions(out / "predictions_test.csv", [s.filename for s in ds.samples], ev, gaussian)
    if gaussian:
        cal = json.loads((run_dir / "calibration.json").read_text(encoding="utf-8"))
        q = math.inf if cal["quantile_q"] in ("inf", math.inf) else float(cal["quantile_q"])
        res["intervals"] = {"quantile_q_from_fold_calibration_groups": cal["quantile_q"], **evaluate_intervals(ev.y_raw, ev.mean_norm, ev.log_var_norm, q)}
    _dump(out / "metrics_test.json", res)
    return out
