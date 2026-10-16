# Experiment A: reference ResNet50 regression

ID `A_reference_resnet50` | config [a_reference.yaml](../../configs/experiments/a_reference.yaml) | notebook
`notebooks/03_reference_baseline.ipynb` | status: **NOT RUN** (implementation and synthetic tests only)

## 1. Objective
Establish the common reference model for study v2 on the shared infrastructure, so that B and C are compared against
a baseline built and evaluated identically.

## 2. Research question and hypothesis
RQ: what does a RadImageNet-initialized ResNet50 with a sigmoid regression head achieve on LDCT-IQAC when trained and
selected on the frozen 800/100 partitions? No directional hypothesis; A is a baseline. Hypothesis H-A (descriptive):
validation PLCC/SROCC will be in the range seen for historical 001 (validation PLCC about 0.96-0.975 on a different
100-image split); this is an expectation, not a claim.

## 3. Scientific rationale
Reproduces the historical starting recipe (Ohashi-style head, MSE) so improvements from B and C can be attributed to
objective/uncertainty changes, not to a different backbone or pipeline.

## 4. What is reproduced and what is adapted
| Aspect | Historical 001 | Experiment A | Reason |
|---|---|---|---|
| Backbone | custom ResNet50, v1 strides | same (`models.backbone`) | RadImageNet Keras layout |
| Head | Dropout 0.5, Linear(2048,1), Sigmoid | same | reference |
| Pretrained weights | converted RadImageNet, no checksum recorded | same file, SHA-256 verified at load | provenance |
| BatchNorm eps | train/validate 1e-3 (set on weight load), test 1e-5 (fresh model) | **1.001e-5 throughout** | [ADR-002](../decisions/0002-batchnorm-and-preprocessing.md) |
| Split | 900 train / 100 validation (`random_split`, seed 42) | 800 / 100 / 100 frozen manifest | [ADR-003](../decisions/0003-split-and-test-discipline.md) |
| Checkpoint selection | val PLCC, 0-based epoch index in files | val PLCC, **1-based** epochs | consistency |
| Device | CPU (001) | set at run time, recorded | hardware |
| Optimizer, loss, lr, batch, epochs, dropout, seed | Adam, MSE, 1e-3, 64, 30, 0.5, 42 | same | historical starting recipe |

A is therefore "the historical recipe re-implemented on corrected infrastructure", not a numerical reproduction of
001. Whether A matches 001's validation behavior is itself an empirical question (PENDING).

## 5. Architecture and outputs
See [MODEL_ARCHITECTURE](../model-architecture.md) section 2. Output: `ModelOutput.mean` in [0, 1]; evaluation uses
`denormalize_score` (x4).

## 6. Loss
`L_A = (1/N) sum_i (f_i - y_i / 4)^2`, computed in the normalized space. Code: `losses/mse.py`.

## 7. Dataset and split
Frozen manifest, train 800 / validation 100 (calibration 100 unused by A). Official test: retrospective only.

## 8. Preprocessing
Center crop 224, `2x - 1`, channel replication inside the model ([PREPROCESSING_AND_WEIGHTS](../preprocessing-and-weights.md)).

## 9. Hyperparameters
Adam lr 1e-3, no weight decay, batch 64, 30 epochs, dropout 0.5, seed 42. Not tuned in this study.

## 10. Training and validation procedure
`ct_iqa.training.engine.fit`: per epoch train, then evaluate on validation; metrics on the [0, 4] scale.

## 11. Checkpoint selection
Highest validation PLCC; strict improvement; 1-based epoch; saved atomically to `checkpoints/best.pt`.

## 12. Metrics
PLCC, SROCC, KROCC, MSE, MAE, RMSE (raw, [0, 4]) - [EVALUATION_METRICS](../evaluation-metrics.md).

## 13. Baselines and ablations
A is the baseline. It also serves as B's `lambda = 0` ablation row. Optional seed repeats change initialization and
data order only.

## 14. Reproducibility requirements
Run manifest (git SHA/dirty flag, versions, weights hash, split hash, seed), config snapshot, no run-directory reuse.

## 15. Expected artifacts
`experiments/study_v2/A_reference_resnet50/runs/<run_id>/`: `config.snapshot.json`, `config.source.yaml`,
`run_manifest.json`, `history.json`, `checkpoints/best.pt`, `metrics_validation.json`, `predictions_validation.csv`.

## 16. Results
| Run ID | Split | Selection | Epoch | PLCC | SROCC | KROCC | MSE | MAE | RMSE | Status |
|---|---|---|---|---|---|---|---|---|---|---|
| - | validation (100) | val PLCC | - | - | - | - | - | - | - | **PENDING** |

## 17. Interpretation rules
Validation numbers from 100 images carry sampling error of roughly 0.01-0.02 in PLCC; do not rank variants on smaller
differences. Do not compare A's validation numbers with historical test numbers; the partitions and epsilon differ.
Any statement about the official test set must say "retrospective".

## 18. Limitations
Image-level split; BatchNorm train/eval inconsistency of the historical runs is unquantified, so A-versus-001 gaps
cannot be attributed to a single cause.

## 19. Completion checklist
- [ ] `validate-config` passes on the target machine
- [ ] run completed, manifest status `completed`
- [ ] validation metrics recorded in section 16 with run ID
- [ ] deviations from 001 re-checked against the actual run config
