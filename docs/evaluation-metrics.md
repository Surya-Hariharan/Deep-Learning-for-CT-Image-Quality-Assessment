# Evaluation metrics

Unless stated otherwise, every metric is computed on the **original [0, 4] score scale** (`denormalize_score` applied to
model outputs) and **without** any logistic mapping. Training losses are in the normalized [0, 1] scale and are labeled
as such; they are not comparable with the reported errors.

## Point-prediction metrics (`ct_iqa.evaluation.metrics.compute_point_metrics`)

| Metric | Definition | Better | Scale |
|---|---|---|---|
| PLCC | Pearson correlation between label and prediction (`scipy.stats.pearsonr`) | higher | unit-free |
| SROCC | Spearman rank correlation (`spearmanr`) | higher | unit-free |
| KROCC | Kendall tau (`kendalltau`, tau-b) | higher | unit-free |
| MSE | mean of squared error | lower | score units squared, [0, 4] |
| MAE | mean absolute error | lower | score units |
| RMSE | square root of MSE | lower | score units |

PLCC is sensitive to non-linear monotonic distortion; SROCC/KROCC are not. Constant predictions give undefined
(NaN) correlations; the selector treats NaN as "not an improvement".

## Logistic mapping (separate)

`ct_iqa.evaluation.calibration.five_parameter_logistic_fit` fits a five-parameter logistic mapping of predictions onto
the labels before correlation, as is common in IQA evaluation. It is **never** applied inside `compute_point_metrics`.
If used, the mapping must be fitted on validation (not test) data, and mapped results must be labeled "5PL-mapped" and
reported next to, not instead of, raw results. Study v2 reports raw metrics only.

## Challenge aggregate score

The LDCTIQAC 2023 overall score is described in a later paper using the dataset as the **sum** of PLCC, SROCC and
KROCC ([Celik et al., arXiv:2511.07298](https://arxiv.org/html/2511.07298v1), which cites the challenge). The official
challenge evaluation page could not be retrieved (HTTP 404), so the exact official definition is **unverified**. If an
aggregate is reported it will be labeled `PLCC+SROCC+KROCC (sum, per arXiv:2511.07298; unverified against the official page)`.
No aggregate is used for model selection in study v2 (selection is by validation PLCC).

## Experiment C metrics (`ct_iqa.evaluation.intervals`)

| Metric | Definition |
|---|---|
| Gaussian NLL, [0, 4] scale | `mean( 0.5 * ((y - mu)/sigma)^2 + log sigma )`, constant `0.5 log 2 pi` omitted unless stated; `mu = 4 mu_n`, `sigma = 4 exp(s_n/2)`. Lower is better. |
| Training NLL | same expression in the normalized space (`s_n` is log-variance there). Differs from the [0, 4] value by `log 4` per sample. |
| Empirical coverage | fraction of labels inside `[lower, upper]` |
| Mean / median width | of `upper - lower` (infinite for an unbounded interval) |
| Clipped fraction | share of intervals extending outside [0, 4] |
| Spearman(sigma, abs error) | rank association between predicted sigma and absolute error |
| Coverage by sigma tercile | coverage within the lowest, middle and highest third of predicted sigma |

Interval coverage and width are properties of the calibrated interval; NLL is a property of the Gaussian predictive
distribution; they measure different things and are not combined into one number.

## Comparing models (`ct_iqa.evaluation.bootstrap`)

Paired bootstrap over images (default 2000 resamples, seed fixed): reports metric(B) minus metric(A), the percentile
95% interval and the fraction of resamples with a positive difference. The interval assumes images are independent,
which is unverified (no patient IDs), so it is likely too narrow.
