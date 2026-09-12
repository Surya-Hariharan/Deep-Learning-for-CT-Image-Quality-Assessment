# Experiment 001: ResNet50 baseline

**Status: scaffolding only -- no training run has been executed yet.**

## Purpose

The first Ohashi-ResNet50 replication baseline on LDCT-IQAC, per
`docs/replication/replication_scope.md`. Orchestrated by
`notebooks/03_training/01_resnet50_baseline.ipynb` /
`notebooks/04_evaluation/01_test_set_evaluation.ipynb`.

## Configuration

See `config.yaml` in this directory (mirrors `configs/*.yaml` at the time
this experiment is run) and, once a run has been executed, `config.json`
(the exact resolved `ExperimentConfig` for that specific run, saved by the
training notebook).

## Layout

```
001_resnet50_baseline/
├── README.md       <- this file
├── config.yaml      <- planned configuration (this experiment's intent)
├── config.json      <- (after a run) resolved ExperimentConfig for that run
└── checkpoint/
    └── best.pt      <- (after a run) best-validation-loss model weights
```

`checkpoint/` is gitignored (model weights are binary and experiment-run
artifacts, not source) -- see the root `.gitignore`.

## Result

**No result yet.** RadImageNet weights are not currently confirmed loaded
(see `weights/pretrained/radimagenet/resnet50/README.md`), so even once run,
this experiment's first pass will be a random-backbone-initialization
result, reported as such -- not a full paper replication -- per
`docs/replication/deviations.md`.
