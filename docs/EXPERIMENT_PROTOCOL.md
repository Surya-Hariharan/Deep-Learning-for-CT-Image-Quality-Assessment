# Experiment protocol (study v2)

Status: **protocol drafted; split frozen; no experiment of this study has been run.** Frozen-protocol marker for
test evaluation: not yet created (see "Freezing"). Decision records: [ADR-002](decisions/ADR-002-batchnorm-and-preprocessing.md),
[ADR-003](decisions/ADR-003-split-and-test-discipline.md).

## Experiments

| ID | Directory | Model | Loss | Extra stage |
|---|---|---|---|---|
| A | `experiments/study_v2/A_reference_resnet50/` | `resnet50_regression` | `mse` | none |
| B | `experiments/study_v2/B_ranking_aware_resnet50/` | `resnet50_regression` | `mse_ranking` (`lambda_rank` > 0) | none |
| C | `experiments/study_v2/C_uncertainty_aware_resnet50/` | `resnet50_gaussian` | `gaussian_nll` | split-conformal calibration |

Historical experiments `001`-`003` are separate and immutable; their IDs are never reused.

## Data

Frozen manifest: `experiments/study_v2/protocol/split_manifest.json` (800 train / 100 validation / 100 calibration,
seed 2026, image-level). Official test set: 300 images, retrospective only. Preprocessing: center crop 224x224,
`2x - 1`, grayscale replicated to 3 channels inside the model (see [PREPROCESSING_AND_WEIGHTS](PREPROCESSING_AND_WEIGHTS.md)).
Score scale: raw [0, 4]; training targets normalized to [0, 1]; all reported errors are in the [0, 4] scale.

## Common settings (identical across A, B, C)

Backbone: custom ResNet50, `bn_eps = 1.001e-5`, RadImageNet weights (SHA-256 verified), dropout 0.5, Adam,
learning rate 1e-3, batch size 64, 30 epochs, no scheduler, no weight decay, no augmentation, `train.seed` from config.
Checkpoint selection: highest validation PLCC (raw [0, 4] scale) evaluated each epoch; ties keep the earlier epoch.
Epoch numbering in all new artifacts is **1-based**. These are the historical starting recipe, not tuned values.

## Hyperparameter selection

Only train and validation data may inform any choice. Experiment B tunes `lambda_rank`, `margin` and
`min_score_diff` over a small declared grid on validation PLCC/SROCC/KROCC with MSE/MAE/RMSE as a guard; the grid and
the selection rule must be written down before the first B run. `lambda_rank = 0` is exactly A's objective
(unit-tested), so A doubles as B's ablation row. Learning rate, epochs and batch size are not re-tuned.

## Experiment C calibration (split-conformal, predeclared)

Target coverage `1 - alpha = 0.90` is fixed in `configs/base.yaml` before any calibration result is viewed.
Model outputs `(mu_n, s_n)` are in normalized [0, 1] space; convert to the [0, 4] label scale
`mu = 4 * mu_n`, `sigma = 4 * exp(s_n / 2)`. On the 100 calibration images compute
`r_i = |y_i - mu_i| / sigma_i`. Let `k = ceil((n + 1)(1 - alpha))` with `n = 100`; if `k > n` the interval is
unbounded (reported as such, before any clipping). Otherwise `q` is the k-th smallest `r_i` and the prediction
interval is `[mu - q * sigma, mu + q * sigma]`. Both the unclipped interval and the interval clipped to [0, 4] are
reported (coverage, mean/median width, fraction clipped); clipping changes the interval and can affect the nominal
guarantee. Guarantee assumptions: exchangeability of calibration and evaluation images, which is not established
(image-level split, no patient grouping). The method is never tuned on official test labels.

## Metrics

PLCC, SROCC, KROCC, MSE, MAE, RMSE on raw [0, 4] scores (no logistic mapping; any 5PL mapping is reported
separately and fitted on validation data only). For C additionally Gaussian NLL (reported in the normalized and
[0, 4] scale, labeled), empirical coverage, interval width, and Spearman correlation between `sigma` and absolute error.
Definitions: [EVALUATION_METRICS](EVALUATION_METRICS.md).

## Comparing A, B, C

Primary comparison on the **validation** partition with paired bootstrap confidence intervals over images
(resampling unit: image; patient grouping unknown, so intervals are likely optimistic). A winner is not declared on
differences smaller than the bootstrap interval. The official test set is evaluated once per final model, after the
protocol is frozen, and is reported as retrospective.

## Run artifacts

Each run writes to `experiments/study_v2/<EXPERIMENT_ID>/runs/<run_id>/` (never overwritten): `config.snapshot.yaml`,
`run_manifest.json` (git SHA and dirty flag, seed, split hash, weights SHA-256, versions, device, timestamps, selection
rule, best epoch, artifact hashes), `history.json`, `checkpoints/best.pt`, `metrics_validation.json`,
`predictions_validation.csv`. Large checkpoints stay out of Git.

## Freezing

Before any official-test evaluation: the grid for B, all configs, and this document are committed (by the project
owner) and the file `experiments/study_v2/protocol/FROZEN` is created containing the commit SHA. Code that loads the
test set refuses to run without it.

## Commands

See [REPRODUCIBILITY](REPRODUCIBILITY.md) (generated once the CLI exists).
