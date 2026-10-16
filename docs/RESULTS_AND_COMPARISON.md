# Results and comparison

Sections are kept separate on purpose. Numbers in section 1 are copied from the historical result files and were not
regenerated. Sections 2-5 contain **no results**: nothing in study v2 has been trained or evaluated. All errors are in
the [0, 4] score scale; correlations are raw (no logistic mapping).

## 1. Historical experiments 001-003 (immutable)

Test evaluation is on the official 300-image set; checkpoint-selection and BatchNorm context differ per row.
Source (files removed from the working tree, recoverable from git history, see [ARCHIVED_ARTIFACTS](ARCHIVED_ARTIFACTS.md)): `results/metrics/001_resnet50_baseline_test_metrics.json`, `results/metrics/003_final_training_test_metrics.json`,
`results/tables/*.csv`.

| Experiment | Run / checkpoint | Training data | Selection rule | Split evaluated | PLCC | SROCC | KROCC | MSE | MAE | RMSE |
|---|---|---|---|---|---|---|---|---|---|---|
| 001 | `001_resnet50_baseline/checkpoint/best.pt`, epoch index 20 (0-based) | 900 | best validation PLCC | official test (300) | 0.8667 | 0.8699 | 0.6816 | 0.3291 | 0.4815 | 0.5736 |
| 003 | `003_final_training/checkpoint/final.pt`, epoch index 29 (0-based) | 1000 | final epoch | official test (300) | 0.8490 | 0.8525 | 0.6611 | 0.4474 | 0.5195 | 0.6689 |
| 002 | four learning-rate runs | 900 | lowest validation loss per run | validation (100) only | see below | | | | | |

Experiment 002 (validation only, 900/100 split): PLCC 0.9464 (lr 1e-2), 0.9750 (1e-3), 0.9722 (1e-4), 0.9434 (1e-5);
lr 1e-3 was selected by lowest validation loss. 001's best validation PLCC was 0.9756 (epoch index 20).

Caveats for every row: BatchNorm epsilon was 1e-3 during training/validation and 1e-5 during the recorded test
evaluation (effect **unquantified**); 001 ran on CPU, 002/003 on CUDA; the test set was examined for 001 and 003, so any
further use of it is retrospective; validation PLCC (0.96-0.975) versus test PLCC (about 0.85-0.87) shows a gap whose
causes (eps mismatch, image-level leakage within the training pool, distribution shift) are not separated.

## 2. Experiment A (new)

| Run ID | Split | Selection | Epoch (1-based) | PLCC | SROCC | KROCC | MSE | MAE | RMSE | Status |
|---|---|---|---|---|---|---|---|---|---|---|
| - | validation (100) | val PLCC | - | - | - | - | - | - | - | **NOT RUN** |

## 3. Experiment B (new)

| Run ID | lambda / margin / delta | Split | PLCC | SROCC | KROCC | MSE | MAE | RMSE | Status |
|---|---|---|---|---|---|---|---|---|---|
| - | 0 (= A) | validation (100) | - | - | - | - | - | - | **NOT RUN** |
| - | grid per [EXPERIMENT_B](experiments/EXPERIMENT_B_RANKING_AWARE.md) | validation (100) | - | - | - | - | - | - | **NOT RUN** |

## 4. Experiment C (new)

| Run ID | Split | PLCC | SROCC | KROCC | RMSE | NLL (0-4) | Coverage | Mean width | Spearman(sigma,err) | Status |
|---|---|---|---|---|---|---|---|---|---|---|
| - | validation (100) | - | - | - | - | - | - | - | - | **NOT RUN** |

## 5. Ablations, sensitivity and final comparison

| Item | Status |
|---|---|
| B: lambda grid, margin and delta sensitivity | NOT RUN |
| Paired bootstrap B-A and C-A on validation | NOT RUN |
| Constant-width conformal baseline for C | NOT RUN |
| Retrospective official-test evaluation of frozen A/B/C | NOT RUN (protocol not frozen; requires approval) |
| Historical 001 versus A | NOT RUN; not directly comparable (different split, epsilon) |
