# Experiment protocol - study v2 (frozen before any A/B/C training run)

Declared: 2026-10-16, before the first non-smoke run. Supersedes `docs/experiment-protocol.md` (now a pointer). Per-experiment details:
[A](../experiments/a-reference.md), [B](../experiments/b-ranking-aware.md), [C](../experiments/c-uncertainty-aware.md). Decisions:
[ADR-002](../decisions/0002-batchnorm-and-preprocessing.md), [ADR-003](../decisions/0003-split-and-test-discipline.md).
Historical experiments 001-003 are immutable and never overwritten; their IDs are not reused.

## 1. Research questions and hypotheses
| ID | Question | Hypothesis |
|---|---|---|
| A (001) | How accurately can a ResNet50 + sigmoid regression head predict the continuous [0, 4] quality score? | Reference only; no hypothesis. |
| B (002) | Does adding a pairwise ranking loss improve rank correlation (SROCC/KROCC/PLCC) without unacceptable RMSE loss? | Mean(PLCC, SROCC, KROCC) increases beyond bootstrap noise at RMSE within 5% of A. |
| C (003) | Does a heteroscedastic Gaussian head give informative uncertainty and usable conformal intervals? | sigma correlates positively with absolute error; conformal intervals reach about 90% coverage with finite, informative width. |

## 2. Data and partitions
Frozen manifest `experiments/study_v2/protocol/split_manifest.json` (800 train / 100 validation / 100 calibration, seed 2026, canonical hash
`be5831c5...0a16`, verified in `reports/eda/dataset_audit.json`). Roles: train = optimizer updates only; validation = checkpoint and hyperparameter
selection; calibration = Experiment C conformal fitting only; official test (300) = retrospective evaluation only, after freezing.
**Known limitation (EDA):** every validation and calibration image has a same-anatomy partner in train; test images have none. Validation/calibration
metrics are therefore optimistic interpolation estimates, and image-level bootstrap intervals are too narrow (see [DATASET_AND_EDA](DATASET_AND_EDA.md)).
The test set was examined by historical 001 and 003, so new test numbers are retrospective, not confirmatory.

## 3. Common configuration (identical across A, B, C; only the loss/head differ)
ResNet50 (custom v1 stride, `bn_eps = 1.001e-5`), RadImageNet weights (SHA-256 `45cba0d5...`), 224 center crop, `2x-1`
([PREPROCESSING](PREPROCESSING.md)), gray -> 3 channels in-model, dropout 0.5, Adam lr 1e-3, no weight decay/scheduler/augmentation, batch 64, 30 epochs, targets
`y/4` in [0, 1], sigmoid output, inverse scaling to [0, 4]. Checkpoint selection: highest validation PLCC (raw scale, evaluated every epoch, ties keep earlier epoch).
Epochs are 1-based in all new artifacts. Training uses seeded RNGs and cuDNN deterministic mode; residual CUDA nondeterminism is possible and recorded.
No setting is altered after seeing results. A resource limit (e.g. out-of-memory) stops the stage and is documented before any change.

## 4. Seeds and run roles
| Role | Seeds | Used for |
|---|---|---|
| smoke | 0 | pipeline check only (1 epoch); excluded from every table |
| selection | 42 | Experiment A reference run and all B grid runs (paired, same seed); selection-biased, exploratory |
| final | 1, 2, 3 | independent replicates of A, selected-B and C; the primary comparison. Seed 42 runs are **not** reused as final replicates. |

Every run is a separate immutable directory; `experiments/study_v2/run_registry.jsonl` and `run_manifest.json` record the purpose.

## 5. Experiment B grid and selection rule (identical to `ct_iqa.experiment.grid`)
Stage 1: `lambda_rank in {0.03, 0.1, 0.3, 1.0}`, margin 0.05 (normalized space), `min_score_diff` 0.25 (raw units). Stage 2 at the stage-1 lambda:
margin in {0.02, 0.1} and `min_score_diff` 0.5 (one factor at a time). 7 configurations, each a unique (lambda, margin, delta, seed) run. Selection (seed 42, validation only):
1. Guard (lower is better): validation RMSE <= 1.05 x RMSE of the A seed-42 run.
2. Among guarded configs choose the highest mean(PLCC, SROCC, KROCC) (higher is better); ties keep the earlier grid point.
3. Paired image-level bootstrap of the mean-correlation difference B - A: 2000 resamples, seed 0, percentile 95% interval; resamples with constant labels skipped.
   Improvement is declared only if the point difference exceeds the interval half-width (`grid.py`); whether the CI excludes zero is reported alongside.
4. If no config passes the guard, or no improvement is demonstrated, the outcome is "no demonstrated improvement"; A remains the recommended model and the
   guard is never relaxed. Failed or incomplete runs are listed, never replaced silently; a failed point is rerun as a new run and both are recorded
   (selection refuses ambiguous duplicates). The lambda = 0 objective equals A (unit-tested in `tests/unit/test_losses.py`).
5. If a configuration is selected it is retrained at final seeds 1, 2, 3. If none is, B's final runs are not performed (reported as such); the grid results stay as the B result.

## 6. Experiment C calibration (target coverage 0.90, set before any result)
Gaussian NLL, `s = log sigma^2` clamped to [-10, 2]; aleatoric uncertainty only (no epistemic estimate). Split-conformal on the 100 calibration images with
`r_i = |y_i - mu_i| / sigma_i`, `k = ceil((n+1)(1-alpha)) = 91`, interval `mu +/- q sigma`; clipped and unclipped intervals both reported with coverage, mean/median
width and n. Coverage is evaluated on the validation partition (shares anatomy with train and calibration; not independent of checkpoint selection) and, retrospectively, on
the official test set with the stored quantile (no refit on test labels). Calibration images are never used for checkpoint selection and their own coverage is not evidence.
Also reported: Spearman(sigma, |error|), coverage by sigma tercile, Gaussian NLL (scale labelled). Exchangeability is not established, especially for the anatomy-shifted test set.
A constant-width conformal baseline built from A's residuals on the same calibration images is the comparator for "useful" intervals (computed in the analysis stage).

## 7. Metrics and statistics
PLCC, SROCC, KROCC, MSE, MAE, RMSE on raw [0, 4] scores (no logistic mapping). Per-seed values plus mean and SD over the 3 final seeds (SD with n = 3 is a rough spread, not a
confidence statement). Between-model comparisons use paired image-level bootstrap on a common partition (2000 resamples, seed 0) of seed-averaged predictions, reported as effect sizes with intervals
and treated as descriptive. A difference is called significant only if the 95% interval of the primary metric (mean of the three correlations) excludes zero after Bonferroni over the A-vs-B and A-vs-C
comparisons (two-sided alpha 0.025 per comparison, i.e. 97.5% intervals). Image-level resampling ignores same-anatomy dependence, so intervals are optimistic.

## 8. Freeze, test boundary, and stopping conditions
- The official test set is read only by `final_test_evaluation` after `experiments/study_v2/protocol/FROZEN` exists, once per final model; test metrics are labelled retrospective.
- The marker is created only after all selection and calibration runs are complete; it records the repository HEAD, dirty state, and SHA-256 of this document, the configs, the manifest and the selection records.
- Stop conditions: any failed gate (hash mismatch, non-finite loss, run marked failed, artifact verification failure) halts the pipeline until diagnosed and documented.
- Expected artifacts per run: `config.snapshot.json`, `run_manifest.json`, `history.json`, `checkpoints/best.pt`, `metrics_validation.json`, `predictions_validation.csv` (plus `calibration.json`, `predictions_calibration.csv` for C).
- Budget: about 6-8 minutes per 30-epoch run on the RTX 4060 (measured about 24 s for a 1-epoch smoke run including start-up); about 20 runs planned.

## 9. Clarification recorded after the B selection (before any B final run)
Stage-2 selection (seed 42, `selection_record_stage2.json`) chose the best guarded configuration lambda 0.3, margin 0.05, delta 0.5 (validation mean-corr 0.9528 vs A 0.9510;
difference +0.0018, 95% CI [-0.0036, +0.0080], half-width 0.0058) and reported **"no demonstrated improvement: A remains the recommended model"**.
Section 5 item 5 was ambiguous about whether a configuration that is the best guarded point but not a demonstrated improvement is retrained. Interpretation adopted now,
before any B final run exists and independent of any final-seed result: the best guarded point **is** retrained at the final seeds 1, 2, 3, so that the A-vs-B comparison rests on
independent replicates rather than on one selection-biased seed. This does not change the selection outcome; B is not recommended on the selection evidence alone.
Stage 1 note: the first lambda = 0.03 run failed with a Windows `PermissionError` on an atomic file rename (OneDrive-synced folder); the run is kept as `failed`, a retry
helper was added to the atomic writes (`ct_iqa/utils/fs.py`), and the point was rerun as a new run with identical configuration and seed. Selection used the completed rerun.
