# Study v3 - Phase 2 review report (before any v3 training)

Protocol: [STUDY_V3_PROTOCOL](STUDY_V3_PROTOCOL.md) (v3.0). Marker: `experiments/study_v3/protocol/FROZEN` (SHA-256 of protocol, manifests, configs and result-producing code). Verification output: `reports/study_v3/phase2_verification.json`.

## Verified
- Group manifest rebuilt independently from the raw images: identical groups and hash (`fab04204...5d79`); 54 image-similarity groups; no cross-group correlation above 0.99 (max 0.9898).
- Fold manifest (`9c0fe67b...33a5`): zero file and zero group overlap between train / validation / calibration / evaluation in all 5 folds; every image covered; every group evaluated exactly once.
- Smoke run (1 epoch, separate tree `experiments/study_v3_smoke/`): predictions on exactly the fold's evaluation files, disjoint from train; hashes recorded; official test unused. Test suite: 185 passed, 0 failed, including grouped-split mechanics and a synthetic dry run of the analysis tool.
- Feasibility: 54 groups give validation 5-9 groups and calibration 6-8 groups per fold, evaluation 8-12 groups per fold. Enough to run the planned design and to report group-level intervals, but **weak** for B selection (fold-0 validation has 8 groups) and for per-fold conformal calibration (6-8 groups). Both are disclosed; no stronger claims are planned. Groups are image-similarity components, not patients.
- Known residual risk: held-out images still have training neighbours at median r 0.95-0.97 (official test: 0.896); v3 held-out performance is expected to lie between v2 validation and v2 test.

## What will be trained (all with 30 epochs, batch 64, Adam 1e-3, seeds as stated)
1. Selection (fold 0, seed 42): A reference + 7 B grid points (lambda 0.03/0.1/0.3/1.0; then margin 0.02/0.1 and delta 0.5 at the stage-1 lambda) = 8 runs; `tools/select_b_v3.py` applies the declared rule (RMSE guard 1.05 x A, mean-correlation, group bootstrap) and fixes the B point to retrain.
2. Final: A, B (selected point), C on folds 0-4 x seeds 1, 2, 3 = 45 runs, each in its own immutable directory under `experiments/study_v3/`.
3. After training: descriptive retrospective evaluation of each final model on the official test set (not part of any decision).

## How results are evaluated
Pooled per seed over all held-out evaluation groups (1000 images); mean and SD over 3 seeds; group-level bootstrap (54 groups, 2000 draws) for intervals and paired B - A, C - A differences with Bonferroni 97.5% intervals; seeds are never treated as independent samples. Experiment C: grouped split conformal (primary) and grouped cross-conformal (secondary) at target 0.90, coverage with counts and group-bootstrap intervals, widths, constant-width comparators; q is never tuned on evaluation or test coverage.
