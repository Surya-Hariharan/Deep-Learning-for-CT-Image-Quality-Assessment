"""Experiment orchestration for study v2: train + validate (+ calibrate for Experiment C) in an immutable run directory.

`run_experiment` never loads the official test set. The test set is reachable only through
`final_test_evaluation`, which requires (1) the frozen-protocol marker file and (2) an explicit
acknowledgement that the evaluation is retrospective on a previously examined benchmark.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import torch

from ct_iqa.config.loader import study_config_from_dict
from ct_iqa.config.schema import StudyConfig
from ct_iqa.data.ldct_iqac import LDCTIQACDataset
from ct_iqa.data.loader import build_study_loaders
from ct_iqa.data.splits import load_manifest
from ct_iqa.evaluation.intervals import calibrate_and_evaluate, evaluate_intervals
from ct_iqa.evaluation.metrics import compute_point_metrics
from ct_iqa.experiment.metadata import config_hash, environment_info, git_info, sha256_path, utc_now
from ct_iqa.losses.registry import build_loss
from ct_iqa.models.registry import build_model
from ct_iqa.training.checkpointing import load_run_checkpoint
from ct_iqa.training.engine import evaluate_loader, fit
from ct_iqa.utils.seed import set_seed

FROZEN_MARKER = Path("experiments/study_v2/protocol/FROZEN")


def _json_default(o):
    if isinstance(o, float) and math.isinf(o):
        return "inf"
    raise TypeError(f"not serializable: {type(o)}")


def _dump(path: Path, obj) -> None:
    text = json.dumps(obj, indent=1, default=_json_default, allow_nan=True)
    path.write_text(text, encoding="utf-8")


def make_run_dir(cfg: StudyConfig) -> tuple[Path, str]:
    stamp = utc_now().replace("-", "").replace(":", "").replace("+0000", "Z").replace("+00:00", "Z")
    run_id = f"{cfg.run.experiment_id}_{stamp}_s{cfg.train.seed}_{config_hash(cfg.to_dict())[:6]}"
    run_dir = Path(cfg.run.output_root) / cfg.run.experiment_id / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)          # a run directory is never reused or overwritten
    return run_dir, run_id


def _write_predictions(path: Path, names: list[str], ev, extra_sigma: bool) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["filename", "label_0_4", "prediction_0_4", "prediction_normalized"]
                   + (["log_var_normalized"] if extra_sigma else []))
        for i, n in enumerate(names):
            row = [n, f"{ev.y_raw[i]:.10g}", f"{ev.pred_raw[i]:.10g}", f"{ev.mean_norm[i]:.10g}"]
            if extra_sigma:
                row.append(f"{ev.log_var_norm[i]:.10g}")
            w.writerow(row)


def run_experiment(cfg: StudyConfig, config_path: str | Path | None = None, repo_root: str | Path = ".") -> Path:
    cfg.validate()
    labels = json.loads(Path(cfg.data.train_json_path).read_text(encoding="utf-8"))
    manifest = load_manifest(cfg.data.split_manifest, labels)       # verifies non-overlap, coverage, hash

    run_dir, run_id = make_run_dir(cfg)
    manifest_path = run_dir / "run_manifest.json"
    _dump(run_dir / "config.snapshot.json", cfg.to_dict())
    if config_path is not None:
        (run_dir / "config.source.yaml").write_text(Path(config_path).read_text(encoding="utf-8"), encoding="utf-8")
    run_manifest = {
        "run_id": run_id, "experiment_id": cfg.run.experiment_id, "status": "started", "started_utc": utc_now(),
        "git": git_info(repo_root), "environment": environment_info(cfg.train.device),
        "seed": cfg.train.seed, "split_manifest": cfg.data.split_manifest,
        "split_manifest_sha256": manifest["manifest_sha256"], "split_seed": manifest["seed"],
        "model": cfg.model.name, "loss": cfg.loss.name, "optimizer": cfg.optim.name,
        "bn_eps": cfg.model.bn_eps, "selection_rule": cfg.train.selection_metric,
        "epoch_indexing": "1-based", "official_test_set_used": False,
    }
    _dump(manifest_path, run_manifest)

    set_seed(cfg.train.seed)
    loaders, names = build_study_loaders(cfg.data, cfg.train, manifest)
    model, report = build_model(cfg.model, cfg.data.image_size)
    run_manifest["pretrained_weights"] = None if report is None else {
        "path": report.weights_path, "sha256": report.sha256, "sha256_verified": report.sha256_verified,
        "matched_keys": len(report.matched_keys), "missing_learned_keys": len(report.missing_learned_keys)}
    loss_fn = build_loss(cfg.loss)

    history = fit(model, loss_fn, loaders["train"], loaders["validation"], cfg, run_dir, run_id)
    best_path = run_dir / "checkpoints" / "best.pt"
    ckpt_meta = load_run_checkpoint(model, best_path, map_location=cfg.train.device)
    model.to(cfg.train.device)

    gaussian = cfg.model.name == "resnet50_gaussian"
    val_ev = evaluate_loader(model, loaders["validation"], loss_fn, cfg.train.device)
    val_metrics = {"partition": "validation", "n": len(val_ev.y_raw), "scale": "raw [0, 4]",
                   "epoch_1based": ckpt_meta["epoch"], **compute_point_metrics(val_ev.y_raw, val_ev.pred_raw)}
    _dump(run_dir / "metrics_validation.json", val_metrics)
    _write_predictions(run_dir / "predictions_validation.csv", names["validation"], val_ev, gaussian)

    artifacts = ["config.snapshot.json", "history.json", "metrics_validation.json",
                 "predictions_validation.csv", "checkpoints/best.pt"]
    if gaussian:
        cal_ev = evaluate_loader(model, loaders["calibration"], loss_fn, cfg.train.device)
        calibration = calibrate_and_evaluate(
            cal_ev.y_raw, cal_ev.mean_norm, cal_ev.log_var_norm,
            val_ev.y_raw, val_ev.mean_norm, val_ev.log_var_norm, cfg.calibration.target_coverage)
        calibration["fitted_on"] = "calibration partition"
        calibration["evaluated_on"] = "validation partition (independent of the calibration partition)"
        _dump(run_dir / "calibration.json", calibration)
        artifacts.append("calibration.json")

    run_manifest.update({
        "status": "completed", "completed_utc": utc_now(), "best_epoch_1based": history["best_epoch"],
        "validation_metrics": val_metrics,
        "artifact_sha256": {a: sha256_path(run_dir / a) for a in artifacts},
    })
    _dump(manifest_path, run_manifest)
    return run_dir


def final_test_evaluation(run_dir: str | Path, acknowledge_retrospective: bool,
                          marker: str | Path = FROZEN_MARKER) -> Path:
    """Retrospective evaluation of a finished run on the official test set. Guarded; never called by training.

    Preconditions: the frozen-protocol marker exists, and the caller acknowledges that the official test set was
    previously examined (experiments 001 and 003). For Experiment C the conformal quantile stored in
    `calibration.json` is applied as-is; nothing is re-fitted on test labels.
    """
    if not acknowledge_retrospective:
        raise PermissionError("official-test evaluation requires acknowledge_retrospective=True: the test set was "
                              "previously used for historical experiments, so this is a retrospective evaluation")
    if not Path(marker).is_file():
        raise PermissionError(f"protocol is not frozen: marker file {marker} does not exist")
    run_dir = Path(run_dir)
    cfg = study_config_from_dict(json.loads((run_dir / "config.snapshot.json").read_text(encoding="utf-8")))
    out_dir = run_dir / "test_retrospective"
    out_dir.mkdir(exist_ok=False)                       # a second final evaluation of the same run is refused

    model, _ = build_model(cfg.model, cfg.data.image_size)
    load_run_checkpoint(model, run_dir / "checkpoints" / "best.pt", map_location=cfg.train.device)
    model.to(cfg.train.device)
    ds = LDCTIQACDataset(cfg.data.test_image_dir, cfg.data.test_json_path, cfg.data.image_size)
    loader = torch.utils.data.DataLoader(ds, batch_size=cfg.train.batch_size, shuffle=False)
    ev = evaluate_loader(model, loader, build_loss(cfg.loss), cfg.train.device)
    result = {"partition": "official_test", "evaluation_status": "retrospective evaluation on a previously examined benchmark",
              "n": len(ev.y_raw), "scale": "raw [0, 4]", **compute_point_metrics(ev.y_raw, ev.pred_raw)}
    _write_predictions(out_dir / "predictions_test.csv", [s.filename for s in ds.samples], ev, ev.log_var_norm is not None)
    if ev.log_var_norm is not None:
        cal = json.loads((run_dir / "calibration.json").read_text(encoding="utf-8"))
        q = math.inf if cal["quantile_q"] in ("inf", math.inf) else float(cal["quantile_q"])
        result["intervals"] = {"quantile_q_from_calibration_partition": cal["quantile_q"],
                               **evaluate_intervals(ev.y_raw, ev.mean_norm, ev.log_var_norm, q)}
    _dump(out_dir / "metrics_test.json", result)
    return out_dir
