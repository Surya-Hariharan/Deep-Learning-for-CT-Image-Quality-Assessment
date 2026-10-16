"""Command-line interface for study v2.

    python -m ct_iqa.cli validate-config --config configs/experiments/a_reference.yaml
    python -m ct_iqa.cli train           --config configs/experiments/a_reference.yaml [--device cuda]
    python -m ct_iqa.cli evaluate        --run-dir <run_dir> --split validation|calibration [--device cuda]
    python -m ct_iqa.cli final-test      --run-dir <run_dir> --acknowledge-retrospective [--device cuda]

`train` trains for real (30 epochs by default); do not run it without an approved compute budget.
`final-test` evaluates the official test set and refuses to run unless the protocol marker exists.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import sys
from pathlib import Path

from config.loader import load_study_config, study_config_from_dict
from config.schema import ConfigError, StudyConfig


def _load(args) -> StudyConfig:
    cfg = load_study_config(args.config, args.base)
    if getattr(args, "device", None):
        cfg = dataclasses.replace(cfg, train=dataclasses.replace(cfg.train, device=args.device))
        cfg.validate()
    return cfg


def cmd_validate(args) -> int:
    from data.splits import load_manifest
    from models.weights import sha256_file

    cfg = _load(args)
    problems = []
    labels = json.loads(Path(cfg.data.train_json_path).read_text(encoding="utf-8")) if Path(cfg.data.train_json_path).is_file() else None
    if labels is None:
        problems.append(f"missing labels: {cfg.data.train_json_path}")
    try:
        manifest = load_manifest(cfg.data.split_manifest, labels)
        print(f"split manifest OK: {cfg.data.split_manifest} sha256={manifest['manifest_sha256'][:16]}...")
    except Exception as exc:  # report, do not hide
        problems.append(f"split manifest: {exc}")
    if cfg.model.pretrained_weights:
        w = Path(cfg.model.pretrained_weights)
        if not w.is_file():
            problems.append(f"missing pretrained weights: {w}")
        elif sha256_file(w) != cfg.model.pretrained_sha256.lower():
            problems.append(f"pretrained weights SHA-256 mismatch: {w}")
        else:
            print(f"pretrained weights OK: {w.name} sha256 verified")
    for p in (cfg.data.train_image_dir,):
        if not Path(p).is_dir():
            problems.append(f"missing image directory: {p}")
    print(f"config OK: experiment={cfg.run.experiment_id} model={cfg.model.name} loss={cfg.loss.name} "
          f"bn_eps={cfg.model.bn_eps} epochs={cfg.train.epochs} device={cfg.train.device}")
    for p in problems:
        print("PROBLEM:", p)
    return 1 if problems else 0


def cmd_train(args) -> int:
    from experiment.run import run_experiment

    cfg = _load(args)
    run_dir = run_experiment(cfg, config_path=args.config)
    print(f"run complete: {run_dir}")
    return 0


def _evaluate(args) -> int:
    from data.loader import build_study_loaders
    from data.splits import load_manifest
    from evaluation.metrics import compute_point_metrics
    from experiment.metadata import utc_now
    from losses.registry import build_loss
    from models.registry import build_model
    from training.checkpointing import load_run_checkpoint
    from training.engine import evaluate_loader

    run_dir = Path(args.run_dir)
    cfg = study_config_from_dict(json.loads((run_dir / "config.snapshot.json").read_text(encoding="utf-8")))
    device = args.device or cfg.train.device
    labels = json.loads(Path(cfg.data.train_json_path).read_text(encoding="utf-8"))
    manifest = load_manifest(cfg.data.split_manifest, labels)
    loaders, _ = build_study_loaders(cfg.data, cfg.train, manifest)
    model, _ = build_model(cfg.model, cfg.data.image_size)
    load_run_checkpoint(model, run_dir / "checkpoints" / "best.pt", map_location=device)
    model.to(device)
    ev = evaluate_loader(model, loaders[args.split], build_loss(cfg.loss), device)
    result = {"partition": args.split, "n": len(ev.y_raw), "scale": "raw [0, 4]",
              **compute_point_metrics(ev.y_raw, ev.pred_raw)}
    out = run_dir / f"eval_{args.split}_{utc_now().replace(':', '').replace('+0000', 'Z')}.json"
    out.write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(json.dumps(result, indent=1))
    return 0


def cmd_final_test(args) -> int:
    from experiment.run import final_test_evaluation

    print(f"wrote {final_test_evaluation(args.run_dir, args.acknowledge_retrospective)}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ct_iqa", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    for name, fn in (("validate-config", cmd_validate), ("train", cmd_train)):
        p = sub.add_parser(name)
        p.add_argument("--config", required=True)
        p.add_argument("--base", default="configs/base.yaml")
        p.add_argument("--device")
        p.set_defaults(fn=fn)
    p = sub.add_parser("evaluate")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--split", choices=["validation", "calibration"], required=True)
    p.add_argument("--device")
    p.set_defaults(fn=_evaluate)
    p = sub.add_parser("final-test")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--acknowledge-retrospective", action="store_true")
    p.set_defaults(fn=cmd_final_test)

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        return args.fn(args)
    except (ConfigError, PermissionError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
