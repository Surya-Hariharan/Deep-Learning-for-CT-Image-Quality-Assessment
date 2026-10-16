# Experiment B: ranking-aware ResNet50 regression

ID `B_ranking_aware_resnet50` | config [b_ranking_aware.yaml](../../configs/experiments/b_ranking_aware.yaml) | notebook
`notebooks/04_ranking_aware_experiment.ipynb` | status: **NOT RUN**; config values are untuned placeholders

## 1. Objective
Test whether adding an explicit pairwise ranking term to the MSE objective changes the ordering quality of predictions
(PLCC/SROCC/KROCC) without materially worsening absolute error (MSE/MAE/RMSE).

## 2. Research question and hypothesis
RQ: does `L_B = L_MSE + lambda * L_rank` change rank correlation relative to A on the same data and backbone?
H-B (to be tested, not assumed): SROCC/KROCC increase for some `lambda > 0`, with RMSE within 5% of A. The null (no
detectable change) is equally plausible, because MSE already induces ordering on this near-continuous score.

## 3. Scientific rationale
Quality assessment is judged by rank and linear correlation, while MSE penalizes absolute error. A margin ranking term
adds direct supervision on pair order. Pairwise ranking losses are an established idea in learning-to-rank; this study
makes no novelty claim for the loss itself.

## 4. Architecture and outputs
Same model as A (`resnet50_regression`). The ranking branch exists in training only
([MODEL_ARCHITECTURE](../model-architecture.md) section 3).

## 5. Loss
Within each mini-batch, with predicted means `f` (normalized) and raw scores `y`:
`P = {(i, j): y_i - y_j >= delta}` (i is the higher-quality image),
`L_rank = mean_{(i,j) in P} max(0, m - (f_i - f_j))`, `L_B = mean (f - y/4)^2 + lambda * L_rank`.
`delta = min_score_diff` is in raw [0, 4] units, `m = margin` in the normalized space. Ties and pairs closer than
`delta` are excluded; the diagonal is excluded; an empty `P` gives `L_rank = 0` with zero gradient. The pair tensor is
B x B (4096 for B = 64); no dataset-wide pair set exists. There is no random pair subsampling, so the only source of
pair variation is batch composition, controlled by the shuffle generator seeded with `train.seed`. Per-batch counts of
valid and possible pairs are logged in `history.json` (`n_valid_pairs`, `n_possible_pairs`). Code: `losses/ranking.py`,
`data/sampling.py`. Tests: `tests/unit/test_losses.py`.

## 6. Data, preprocessing, selection
Identical to A: frozen 800/100 partitions, same preprocessing, selection by validation PLCC.

## 7. Hyperparameter selection (declared before any B run)
Only train and validation data are used. Grid: `lambda_rank in {0.03, 0.1, 0.3, 1.0}` with `margin = 0.05` and
`min_score_diff = 0.25`; then, at the selected `lambda` only, one-at-a-time sensitivity on `margin in {0.02, 0.1}` and
`min_score_diff in {0.5}`. Everything else equals A (including seed 42). Selection rule: among configurations whose
validation RMSE is within 5% of A's, choose the highest mean of validation (PLCC, SROCC, KROCC); if none exceeds A by
more than the paired-bootstrap 95% interval half-width, report "no demonstrated improvement" and keep A as the
recommended model. The `lambda = 0` ablation is A itself (the equality of the objectives is unit-tested).

## 8. Evaluation and comparison
Metrics as A. Comparison with A on the validation partition via `evaluation.bootstrap.paired_bootstrap_difference`
(unit: image, optimistic because patient grouping is unknown). Seed repeats, if affordable, report the spread.

## 9. Reproducibility and artifacts
As A, plus `n_valid_pairs` logging. Grid runs are separate run directories; the chosen configuration and the
grid results are all kept.

## 10. Results
| Run ID | lambda | margin | delta | Epoch | PLCC | SROCC | KROCC | MSE | MAE | RMSE | Status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| - | 0 (= A) | - | - | - | - | - | - | - | - | - | **PENDING** |
| - | 0.03 / 0.1 / 0.3 / 1.0 | 0.05 | 0.25 | - | - | - | - | - | - | - | **PENDING** |

## 11. Interpretation rules
No claim that ranking loss improves rank correlation unless the paired interval excludes zero on validation and the
RMSE guard holds. Differences within seed-to-seed variation are not findings. Test-set numbers, if ever produced,
are retrospective and never used to choose among grid points.

## 12. Limitations
In-batch pairs depend on batch composition (64 images drawn from 800); 100 validation images give noisy rank
correlations; the near-continuous 21-level score has many ties (delta handles them but discards pairs).

## 13. Completion checklist
- [ ] grid and selection rule unchanged from section 7
- [ ] all grid runs completed and listed in section 10
- [ ] paired bootstrap against A recorded
- [ ] `lambda = 0` equality test passing at run time
