"""Locate and read finished run directories (used by the study notebooks)."""

from __future__ import annotations

import csv
import json
from pathlib import Path


def list_runs(experiment_id: str, output_root: str | Path = "experiments/study_v2") -> list[Path]:
    """Run directories of an experiment, oldest first (run IDs start with a UTC timestamp)."""
    runs = Path(output_root) / experiment_id / "runs"
    return sorted(p for p in runs.iterdir() if p.is_dir()) if runs.is_dir() else []


def completed_runs(experiment_id: str, output_root: str | Path = "experiments/study_v2") -> list[Path]:
    out = []
    for p in list_runs(experiment_id, output_root):
        m = p / "run_manifest.json"
        if m.is_file() and json.loads(m.read_text(encoding="utf-8")).get("status") == "completed":
            out.append(p)
    return out


def latest_completed_run(experiment_id: str, output_root: str | Path = "experiments/study_v2") -> Path | None:
    runs = completed_runs(experiment_id, output_root)
    return runs[-1] if runs else None


def select_run(experiment_id: str, run_id: str | None = None,
               output_root: str | Path = "experiments/study_v2") -> Path | None:
    """Pick one completed run explicitly.

    With `run_id`, return that run (error if it is not a completed run of the experiment). Without it, return the
    only completed run, `None` if there is none, and raise if several exist, so a notebook can never silently
    use an arbitrary hyperparameter-grid or repeated run.
    """
    runs = completed_runs(experiment_id, output_root)
    names = [p.name for p in runs]
    if run_id is not None:
        if run_id not in names:
            raise ValueError(f"{run_id!r} is not a completed run of {experiment_id}; completed runs: {names}")
        return runs[names.index(run_id)]
    if len(runs) > 1:
        raise ValueError(f"{len(runs)} completed runs of {experiment_id}; set RUN_ID to one of: {names}")
    return runs[0] if runs else None


def read_predictions(run_dir: str | Path, name: str = "predictions_validation.csv") -> dict[str, list]:
    """Columns of a predictions CSV as lists (strings for filename, floats otherwise)."""
    with (Path(run_dir) / name).open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return {k: ([r[k] for r in rows] if k == "filename" else [float(r[k]) for r in rows])
            for k in (rows[0] if rows else {})}


class RunArtifactError(RuntimeError):
    """A run directory is incomplete, failed, tampered with, or incompatible with another run."""


def read_manifest(run_dir: str | Path) -> dict:
    path = Path(run_dir) / "run_manifest.json"
    if not path.is_file():
        raise RunArtifactError(f"{path} does not exist: not a study-v2 run directory")
    return json.loads(path.read_text(encoding="utf-8"))


def verify_artifacts(run_dir: str | Path) -> dict[str, str]:
    """Recompute the SHA-256 of every artifact listed in the manifest; return {artifact: problem} (empty = all ok)."""
    from ct_iqa.experiment.metadata import sha256_path

    run_dir = Path(run_dir)
    problems = {}
    for name, expected in read_manifest(run_dir).get("artifact_sha256", {}).items():
        p = run_dir / name
        if not p.is_file():
            problems[name] = "missing"
        elif sha256_path(p) != expected:
            problems[name] = "SHA-256 differs from run_manifest.json"
    return problems


def run_summary(run_dir: str | Path, verify: bool = True) -> dict:
    """Manifest, config snapshot, history and validation metrics of ONE completed run.

    Raises `RunArtifactError` if the run is not completed (started / failed), an expected file is missing, or
    (with `verify=True`) an artifact no longer matches its recorded SHA-256.
    """
    run_dir = Path(run_dir)
    manifest = read_manifest(run_dir)
    status = manifest.get("status")
    if status != "completed":
        err = manifest.get("error")
        raise RunArtifactError(f"{run_dir.name}: status is {status!r}, not 'completed'"
                               + (f" ({err['type']}: {err['message']})" if err else ""))
    if verify:
        problems = verify_artifacts(run_dir)
        if problems:
            raise RunArtifactError(f"{run_dir.name}: artifact check failed: {problems}")
    out = {"run_dir": run_dir, "manifest": manifest}
    for key, name in (("config", "config.snapshot.json"), ("history", "history.json"),
                      ("metrics", "metrics_validation.json")):
        p = run_dir / name
        if not p.is_file():
            raise RunArtifactError(f"{run_dir.name}: {name} is missing")
        out[key] = json.loads(p.read_text(encoding="utf-8"))
    return out


def runs_table(experiment_id: str, output_root: str | Path = "experiments/study_v2") -> list[dict]:
    """One row per run directory of an experiment, ANY status (failed and started runs are listed, not hidden)."""
    rows = []
    for p in list_runs(experiment_id, output_root):
        try:
            m = read_manifest(p)
        except RunArtifactError:
            rows.append({"run_id": p.name, "status": "no manifest"})
            continue
        snap = p / "config.snapshot.json"
        cfg = json.loads(snap.read_text(encoding="utf-8")) if snap.is_file() else {}
        loss = cfg.get("loss", {})
        r = loss.get("ranking", {}) if loss.get("name") == "mse_ranking" else {}
        vm = m.get("validation_metrics", {})
        rows.append({"run_id": p.name, "status": m.get("status"), "loss": m.get("loss"),
                     "lambda_rank": r.get("lambda_rank"), "margin": r.get("margin"),
                     "min_score_diff": r.get("min_score_diff"), "seed": m.get("seed"),
                     "device": m.get("environment", {}).get("device_requested"),
                     "best_epoch": m.get("best_epoch_1based"),
                     **{k: vm.get(k) for k in ("plcc", "srocc", "krocc", "mse", "mae", "rmse")},
                     "error": (m.get("error") or {}).get("type")})
    return rows


def check_compatible(run_dirs: dict[str, str | Path]) -> dict[str, dict]:
    """Verify that runs can be compared on the validation partition; return {label: run_summary}.

    Every run must be completed with intact artifacts, and all runs must share the split manifest (SHA-256),
    bn_eps, input image size, selection rule and the exact validation filenames in the same order.
    Raises `RunArtifactError` listing every problem found.
    """
    summaries, problems = {}, []
    for label, d in run_dirs.items():
        try:
            summaries[label] = run_summary(d)
        except RunArtifactError as e:
            problems.append(f"{label}: {e}")
    if problems:
        raise RunArtifactError("; ".join(problems))
    checks = {
        "split manifest SHA-256": lambda s: s["manifest"]["split_manifest_sha256"],
        "bn_eps": lambda s: s["manifest"]["bn_eps"],
        "image size": lambda s: s["config"]["data"]["image_size"],
        "selection rule": lambda s: s["manifest"]["selection_rule"],
        "validation filenames": lambda s: tuple(read_predictions(s["run_dir"])["filename"]),
    }
    for what, get in checks.items():
        values = {label: get(s) for label, s in summaries.items()}
        if len(set(values.values())) > 1:
            shown = {k: (f"{len(v)} files" if isinstance(v, tuple) else v) for k, v in values.items()}
            problems.append(f"{what} differs: {shown}")
    if problems:
        raise RunArtifactError("incompatible runs: " + "; ".join(problems))
    return summaries
