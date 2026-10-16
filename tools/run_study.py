"""Driver for study-v2 runs with explicit purpose labels (nothing trains on import).

    python tools/run_study.py --exp A --seed 42 --purpose selection
    python tools/run_study.py --exp B --seed 42 --purpose selection --lam 0.1 --margin 0.05 --delta 0.25
    python tools/run_study.py --exp C --seed 1 --purpose final
    python tools/run_study.py --exp A --seed 42 --purpose smoke --epochs 1

Every run goes through `ct_iqa.experiment.run.run_experiment` (immutable run directory, run manifest). The purpose
(`smoke`, `selection`, `final`) is recorded in the run manifest's `overrides` and appended to
`experiments/study_v2/run_registry.jsonl`, so exploratory/pilot runs cannot be mistaken for final replicates.
Epochs, batch size and learning rate are not overridable except `--epochs` for purpose `smoke` only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ct_iqa.config.loader import load_study_config
from ct_iqa.experiment.grid import apply_overrides, grid_overrides
from ct_iqa.experiment.metadata import utc_now
from ct_iqa.experiment.run import run_experiment

CONFIGS = {"A": "configs/experiments/a_reference.yaml", "B": "configs/experiments/b_ranking_aware.yaml",
           "C": "configs/experiments/c_uncertainty_aware.yaml"}
REGISTRY = Path("experiments/study_v2/run_registry.jsonl")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", choices=list(CONFIGS), required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--purpose", choices=["smoke", "selection", "final"], required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--lam", type=float)
    ap.add_argument("--margin", type=float)
    ap.add_argument("--delta", type=float)
    ap.add_argument("--epochs", type=int, help="smoke runs only")
    a = ap.parse_args()

    cfg_path = CONFIGS[a.exp]
    cfg = load_study_config(cfg_path, "configs/base.yaml")
    if a.exp == "B":
        if None in (a.lam, a.margin, a.delta):
            raise SystemExit("Experiment B needs --lam --margin --delta (a declared grid point)")
        ov = grid_overrides({"lambda_rank": a.lam, "margin": a.margin, "min_score_diff": a.delta}, a.seed, a.device)
    else:
        ov = {"train": {"seed": a.seed, "device": a.device}}
    if a.epochs is not None:
        if a.purpose != "smoke":
            raise SystemExit("--epochs is only allowed for purpose=smoke (the protocol fixes 30 epochs)")
        ov["train"]["epochs"] = a.epochs
    cfg = apply_overrides(cfg, ov)
    recorded = {**ov, "purpose": a.purpose}
    run_dir = run_experiment(cfg, config_path=cfg_path, overrides=recorded)
    REGISTRY.parent.mkdir(parents=True, exist_ok=True)
    with REGISTRY.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"utc": utc_now(), "run_dir": run_dir.as_posix(), "exp": a.exp, "seed": a.seed,
                            "purpose": a.purpose, "overrides": ov}) + "\n")
    print(f"run complete: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
