"""Driver for study-v3 grouped runs (nothing trains on import).

    python tools/run_study_v3.py --exp A --fold 0 --seed 42 --purpose selection
    python tools/run_study_v3.py --exp B --fold 0 --seed 42 --purpose selection --lam 0.3 --margin 0.05 --delta 0.5
    python tools/run_study_v3.py --exp C --fold 2 --seed 1 --purpose final
    python tools/run_study_v3.py --exp A --fold 0 --seed 0 --purpose smoke --epochs 1     # writes to experiments/study_v3_smoke

Purposes: smoke (pipeline check, separate output tree, allowed before the freeze), selection (B grid, fold 0, seed 42), final
(folds 0-4, seeds 1-3; requires the FROZEN marker). Epochs are overridable for smoke runs only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ct_iqa.config.loader import load_study_config
from ct_iqa.experiment.grid import apply_overrides
from ct_iqa.experiment.metadata import utc_now
from ct_iqa.experiment.run_v3 import run_v3_experiment

CONFIGS = {"A": "configs/experiments/a_reference.yaml", "B": "configs/experiments/b_ranking_aware.yaml",
           "C": "configs/experiments/c_uncertainty_aware.yaml"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", choices=list(CONFIGS), required=True)
    ap.add_argument("--fold", type=int, choices=range(5), required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--purpose", choices=["smoke", "selection", "final"], required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--lam", type=float)
    ap.add_argument("--margin", type=float)
    ap.add_argument("--delta", type=float)
    ap.add_argument("--epochs", type=int)
    a = ap.parse_args()
    cfg = load_study_config(CONFIGS[a.exp], "configs/base.yaml")
    root = "experiments/study_v3_smoke" if a.purpose == "smoke" else "experiments/study_v3"
    ov = {"train": {"seed": a.seed, "device": a.device}, "run": {"output_root": root, "experiment_id": f"{cfg.run.experiment_id}__fold{a.fold}"},
          "data": {"split_manifest": "experiments/study_v3/protocol/fold_manifest.json"}}
    if a.exp == "B":
        if None in (a.lam, a.margin, a.delta):
            raise SystemExit("B needs --lam --margin --delta")
        ov["loss"] = {"name": "mse_ranking", "ranking": {"lambda_rank": a.lam, "margin": a.margin, "min_score_diff": a.delta}}
    if a.epochs is not None:
        if a.purpose != "smoke":
            raise SystemExit("--epochs only for smoke runs")
        ov["train"]["epochs"] = a.epochs
    cfg = apply_overrides(cfg, ov)
    run_dir = run_v3_experiment(cfg, a.fold, overrides={**ov, "purpose": a.purpose}, require_frozen=a.purpose != "smoke")
    reg = Path(root) / "run_registry.jsonl"
    reg.parent.mkdir(parents=True, exist_ok=True)
    with reg.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"utc": utc_now(), "run_dir": run_dir.as_posix(), "exp": a.exp, "fold": a.fold, "seed": a.seed,
                            "purpose": a.purpose, "overrides": ov}) + "\n")
    print(f"run complete: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
