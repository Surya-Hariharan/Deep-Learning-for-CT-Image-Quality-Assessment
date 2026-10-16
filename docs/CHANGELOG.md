# Changelog

## 2026-10-16: study-v2 rebuild (branch `study-v2-rebuild`, not committed)

Stage 1 - preservation
- Branched from `main` at `821461e`; backed up 137 files (6 checkpoints, 2 weight files, notebooks, labels, results,
  experiments, docs, tools, configs, tests, source) to `C:\Users\surya\ct_iqa_backup`; all SHA-256 verified.
- Added `docs/audit/{INITIAL_AUDIT,CLEANUP_MANIFEST,BACKUP_MANIFEST.*}`.

Stage 2 - cleanup
- Removed generated `src/ct_iqa.egg-info/`, `__pycache__/`, empty `.benchmarks/`.
- `.gitignore` header corrected (no rule weakened); note added to the superseded `experiments/001_*/config.yaml`.

Stage 3 - scientific correctness
- BatchNorm epsilon is now a constructor property (`bn_eps`); canonical 1.001e-5 from the weights file's own recorded
  configuration (53/53 layers) and Keras source. Weight loading no longer changes it. Legacy class unchanged. ADR-002.
- Weight checksums recorded (`CHECKSUMS.sha256`) and verified at load.

Stage 4 - architecture
- Model registry (`resnet50_regression`, `resnet50_gaussian`), head layers, `ModelOutput`; strict typed config
  (`StudyConfig`). `ct_iqa/config.py` became a package with the old code at `config/legacy.py` (re-exported).
- Checkpoint compatibility matrix: all six historical checkpoints load strictly (structure only). ADR-001.

Protocol freeze
- Frozen 800/100/100 image-level split (seed 2026) with manifest hash; ADR-003; `experiment-protocol.md`.

Stage 5 - training infrastructure
- Losses (MSE, margin ranking, Gaussian NLL), in-batch pair mask, selection policies, atomic run checkpoints,
  training engine, run metadata, `ct_iqa.cli`, guarded `final_test_evaluation`.

Stages 6-8 - experiments (implementation only)
- Configs and specifications for A, B, C; split-conformal intervals (`evaluation/intervals.py`); paired bootstrap.
  No experiment was run.

Stage 9 - documentation and notebooks
- Documentation set, Mermaid diagrams, study notebooks 00-07 (inspection-oriented; no cell trains unless a flag is set).

Behavior changes to be aware of
- New models use `bn_eps = 1.001e-5`; historical runs used 1e-3 (train) and 1e-5 (test).
- New runs use 1-based epoch numbering; historical files use 0-based indices.
- Study-v2 checkpoints omit optimizer state (resume not implemented).
