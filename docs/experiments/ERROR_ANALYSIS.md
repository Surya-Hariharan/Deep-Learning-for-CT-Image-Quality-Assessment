# Error analysis

Source: `reports/experiments/error_by_score_bin.csv`, `worst_cases.csv`, `scatter_validation.png`, `scatter_test_retrospective.png`, `exp_c_uncertainty.png`
(seed-mean predictions of the final seeds 1-3). Test results are retrospective.

## 1. Validation versus test
All three models fit validation almost perfectly (RMSE about 0.24-0.26 per score bin, no bin worse than 0.29 for A) but lose roughly 0.11 PLCC and more than double their RMSE on the test set. This gap is the main
error pattern and is consistent with the EDA finding that validation images share anatomy with training images while test images do not. It is an interpretation supported by the group analysis; it was not isolated by a controlled experiment (no group-aware re-split was run, by protocol).

## 2. Errors by score bin (test, RMSE / mean bias = prediction - label)
| Bin | n | A RMSE / bias | B RMSE / bias | C RMSE / bias |
|---|---|---|---|---|
| [0, 1) | 46 | 0.66 / +0.40 | 0.70 / +0.48 | 0.64 / +0.39 |
| [1, 2) | 86 | 0.61 / +0.20 | 0.62 / +0.26 | 0.63 / +0.22 |
| [2, 3) | 79 | 0.64 / +0.33 | 0.74 / +0.44 | 0.61 / +0.28 |
| [3, 4] | 89 | 0.50 / -0.01 | 0.51 / +0.13 | 0.52 / -0.06 |

Low- and mid-quality test images are systematically **over-rated** (positive bias up to +0.4 to +0.5), high-quality images are unbiased: the models are too optimistic about poor images from unseen anatomy. B shows the largest positive bias (also visible on validation, +0.04 to +0.11 vs about -0.05 to +0.01 for A), which accounts for its higher RMSE without a correlation loss, i.e. the ranking term appears to shift the score level while preserving order.

## 3. Worst cases (A, test, mean absolute error over seeds)
`test235` (label 0.83, predicted 2.60), `test248` (2.17 vs 3.79), `test145` (0.67 vs 2.22), `test006` (2.17 vs 3.72), `test228` (0.33 vs 1.84): mostly poor images predicted as acceptable and a few mid-quality images predicted as good; two high-label cases are under-rated (`test180` 4.0 vs 2.67, `test046`). Visual inspection of these images was not performed, so no statement is made about their content or artifacts.

## 4. Uncertainty versus error (Experiment C)
Validation Spearman(sigma, |error|) = 0.24 / 0.03 / 0.12 (seeds 1-3) and test 0.15 / 0.01 / 0.11: sigma is a weak error indicator. The low-sigma tercile under-covers on validation (0.82-0.94) while the other terciles over-cover (0.97-1.0), i.e. sigma separates difficulty only partly. On test the intervals cover only 0.62-0.66 of labels, concentrated failures being the over-rated poor-quality images above, whose errors (up to 1.8 score units) exceed even the widest intervals (mean width about 1.1).

## 5. Limits of this analysis
Seed-mean predictions hide seed variance; 100/300 images give noisy bin statistics; no per-pathology or per-scanner metadata exists; patient/anatomy identifiers are unavailable, so group-level error attribution is not possible.
