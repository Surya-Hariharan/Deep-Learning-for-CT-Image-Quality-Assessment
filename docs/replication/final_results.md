# Final results

**OUR IMPLEMENTATION.** This document summarizes the completed
experimental pipeline -- experiments 001 through 003, the final
independent test-set evaluation, and the subsequent analysis-only
prediction-behavior study (`notebooks/04_evaluation/02_prediction_analysis.ipynb`)
-- and states what it does and does not show. It introduces no new
number computed outside those two evaluation/analysis notebooks; every
value here is measured and persisted in `experiments/*/README.md` and
`results/`, collected here for a single-document overview. Statements are
explicitly labeled **PAPER FACT**, **OUR IMPLEMENTATION**,
**IMPLEMENTATION DECISION**, **OBSERVED RESULT**, or **HYPOTHESIS**
throughout.

## A. Experimental protocol

**IMPLEMENTATION DECISION**, not prescribed verbatim by the paper (the
paper states its final training hyperparameters directly and does not
describe a validation-driven LR-search-then-retrain-on-all-data
procedure):

1. **Experiment 001 (baseline)**: 900/100 train/validation split of the
   1000 labeled LDCT-IQAC training images, RadImageNet-initialized
   `OhashiResNet50`, Adam/MSE/batch 64/30 epochs/lr 1e-3
   (**PAPER FACT**, not searched -- see `docs/paper/training.md`),
   checkpoint selected by **highest validation PLCC**
   (`config.selection_metric="plcc"`, updated 2026-09-14 -- see
   `docs/internal/decisions/README.md`; previously lowest validation loss,
   which does not guarantee maximum PLCC/SROCC). Evaluated once on the
   untouched 300-image test set.
2. **Experiment 002 (learning-rate search)**: the same 900/100 split used
   to search the paper's own four candidate learning rates
   (1e-2, 1e-3, 1e-4, 1e-5 -- **PAPER FACT** as the candidate set; the
   search procedure itself is an **IMPLEMENTATION DECISION**), selecting
   **1e-3** by lowest best-validation-loss. The test set was never
   touched.
3. **Experiment 003 (final training)**: retrained from a **fresh**
   RadImageNet initialization (not experiment 001's or 002's weights),
   using the selected LR (1e-3), on **all 1000** labeled training images
   -- no validation split. Checkpoint is the unconditional final epoch
   (epoch 30, `checkpoint/final.pt`), not a validation-selected one.
4. **Final independent test-set evaluation**:
   `experiments/003_final_training/checkpoint/final.pt` loaded fresh and
   evaluated exactly once on the same untouched 300-image test set used by
   experiment 001.
5. **Analysis-only prediction-behavior study** (this update): the
   already-generated test-set prediction CSVs and training histories from
   steps 1-4 were analyzed for distributional, residual, quality-bin,
   compression, and error-case behavior. No model was retrained,
   fine-tuned, or re-evaluated against the test set in this step -- see
   `notebooks/04_evaluation/02_prediction_analysis.ipynb`.

See `docs/replication/reproducibility.md` for the full seeding/provenance
detail behind each step, and `docs/internal/decisions/README.md` for the
process/structural decisions made along the way (e.g. why `final.pt` is
named differently from `best.pt`).

## B. Experiment 001 baseline -- test-set result

**OBSERVED RESULT.** RAW test metrics (`[0,4]` scale, n=300, no
calibration), from the checkpoint selected by highest validation PLCC
(epoch 20; retrained 2026-09-14 -- see section A and
`docs/internal/decisions/README.md`):

| PLCC | SROCC | KROCC | MSE | MAE | RMSE |
|---|---|---|---|---|---|
| 0.8667 | 0.8699 | 0.6816 | 0.3291 | 0.4815 | 0.5736 |

Systematic **overprediction**: mean prediction (2.317) exceeded mean
ground truth (2.131) by **+0.1862**. Prediction std (0.9219) was somewhat
below ground-truth std (1.0871) -- std ratio **0.8480**, i.e. mild
compression toward the mean. Full detail:
`experiments/001_resnet50_baseline/README.md`.

(For reference: the pre-2026-09-14 run, whose checkpoint was selected by
lowest validation loss instead, epoch 21, scored PLCC 0.8804 / SROCC
0.8793 / KROCC 0.6952 / MSE 0.3765 / MAE 0.5024 / RMSE 0.6136 with bias
+0.3049 on the same test set -- see section F for what changing the
selection criterion did and did not change.)

## C. Experiment 002 LR selection

**OBSERVED RESULT.** Searched 1e-2, 1e-3, 1e-4, 1e-5 on the validation
split only; **1e-3** selected by lowest best-validation-loss (0.003540).
The test set was never touched during this search, and experiment 002 has
no test-set result -- none is reported or invented here (see
`results/tables/final_results_summary.csv`). Full detail:
`experiments/002_learning_rate_search/README.md`.

## D. Experiment 003 final training

**OBSERVED RESULT.** All 1000 labeled training images, LR 1e-3, 30
epochs, no validation split, final-epoch checkpoint (`checkpoint/final.pt`,
epoch 29 0-indexed). RadImageNet initialization verified (265/318
backbone keys matched, 0 unexpected). Final training loss: **0.001783**
-- lower than experiment 001's final (epoch 30) training loss (0.002992),
despite experiment 003 performing worse on the test set (see F below). No
test data was used at any point in this stage. Full detail:
`experiments/003_final_training/README.md`.

## E. Final test performance

**OBSERVED RESULT.** Loaded `checkpoint/final.pt` into a fresh
`OhashiResNet50` instance, ran inference once on all 300 test images
(`model.eval()` / `torch.no_grad()`), no calibration fit on the test set.

RAW test metrics (`[0,4]` scale, n=300):

| PLCC | SROCC | KROCC | MSE | MAE | RMSE |
|---|---|---|---|---|---|
| 0.8490 | 0.8525 | 0.6611 | 0.4474 | 0.5195 | 0.6689 |

Prediction distribution vs. ground truth (`[0,4]` scale):

| | min | max | mean | median | std |
|---|---|---|---|---|---|
| ground truth | 0.000 | 4.000 | 2.131 | 2.167 | 1.087 |
| prediction | 0.024 | 3.689 | 1.791 | 1.752 | 0.878 |

Systematic **underprediction**: mean prediction is **-0.3396** below mean
ground truth -- the opposite direction of experiment 001's bias.
Prediction std (0.878) is below ground-truth std (1.087) -- std ratio
**0.8076** (descriptive only, not a formal calibration metric), i.e. more
compression toward the mean than experiment 001 showed.

Artifacts: `results/predictions/003_final_training_test_predictions.csv`,
`results/metrics/003_final_training_test_metrics.json`,
`results/figures/003_final_training_test_*.png`.

**TEST SET USED ONLY FOR FINAL EVALUATION** -- no test data, metric, or
result influenced training, LR selection, checkpoint selection,
preprocessing, or architecture at any stage.

## F. Experiment 001 vs. Experiment 003 comparison

**OBSERVED RESULT.** Both models were evaluated on the identical,
untouched 300-image test set -- same dataset, same architecture, same
preprocessing, different training protocol (900/1000 +
validation-PLCC-selected checkpoint vs. all 1000/1000 + final-epoch
checkpoint).

| Metric | Experiment 001 | Experiment 003 | Difference (003-001) | Relative change |
|---|---|---|---|---|
| PLCC | 0.8667 | 0.8490 | -0.0178 | -2.05% |
| SROCC | 0.8699 | 0.8525 | -0.0174 | -2.00% |
| KROCC | 0.6816 | 0.6611 | -0.0206 | -3.01% |
| MSE | 0.3291 | 0.4474 | +0.1184 | +35.97% |
| MAE | 0.4815 | 0.5195 | +0.0379 | +7.88% |
| RMSE | 0.5736 | 0.6689 | +0.0953 | +16.61% |

(machine-readable: `results/tables/001_vs_003_analysis.csv`,
`results/tables/001_vs_003_final_comparison.csv`)

On this test set, the final (all-data, final-epoch) model performed
empirically worse than the validation-selected baseline on every metric
above, and its prediction bias flipped sign (+0.1862 vs. -0.3396). No
statistical significance test is implemented in this project -- these are
measured differences, not a claim of significance.

**Quality-bin breakdown** (`results/tables/quality_bin_error_analysis.csv`,
quartiles of the ground-truth score distribution):

| Experiment | Quartile (GT range) | n | mean GT | mean pred | mean residual | MAE | RMSE |
|---|---|---|---|---|---|---|---|
| 001 | Q1 [0.00, 1.17] | 82 | 0.774 | 1.318 | +0.544 | 0.621 | 0.702 |
| 001 | Q2 [1.17, 2.17] | 76 | 1.757 | 2.008 | +0.251 | 0.457 | 0.555 |
| 001 | Q3 [2.17, 3.00] | 77 | 2.706 | 2.848 | +0.143 | 0.401 | 0.453 |
| 001 | Q4 [3.00, 4.00] | 65 | 3.600 | 3.311 | -0.289 | 0.430 | 0.542 |
| 003 | Q1 [0.00, 1.17] | 82 | 0.774 | 0.882 | +0.108 | 0.315 | 0.385 |
| 003 | Q2 [1.17, 2.17] | 76 | 1.757 | 1.452 | -0.304 | 0.485 | 0.584 |
| 003 | Q3 [2.17, 3.00] | 77 | 2.706 | 2.296 | -0.410 | 0.466 | 0.609 |
| 003 | Q4 [3.00, 4.00] | 65 | 3.600 | 2.737 | -0.863 | 0.881 | 1.019 |

Experiment 001's bias is positive (overprediction) in the lowest three
quartiles and turns negative (-0.289, mild underprediction) in the
highest quartile -- unlike the pre-2026-09-14 (loss-selected) checkpoint,
which stayed positive across all four (see section B). Experiment 003's
bias is positive only at the lowest quartile (+0.108) and becomes
increasingly negative at higher quartiles, reaching -0.863 (and its worst
per-bin MAE/RMSE, 0.881/1.019) in the top quartile -- unchanged from
before, since experiment 003's checkpoint was not retrained (its training
loop never called `trainer.train`; see section A).

### OBSERVED / PLAUSIBLE HYPOTHESIS / NOT DETERMINED

**OBSERVED:**
- Experiment 001 outperforms experiment 003 on every reported test
  metric, both before and after the 2026-09-14 checkpoint-selection fix.
- Experiment 001's bias is +0.1862 (overprediction); experiment 003's is
  -0.3396 (underprediction) -- signs are opposite.
- Experiment 003's underprediction grows monotonically more negative from
  the lowest to the highest ground-truth quartile.
- Experiment 003's final *training* loss (0.001783) is lower than
  experiment 001's final training loss (0.002992).
- Experiment 001's checkpoint is selected by highest validation PLCC
  (epoch 20/30); experiment 003's checkpoint has no such selection (it is
  unconditionally epoch 30/30).
- Experiment 003 trained on all 1000 labeled images; experiment 001 on
  900 of the 1000.
- Changing experiment 001's selection criterion from lowest validation
  loss (epoch 21) to highest validation PLCC (epoch 20) *did not* change
  which experiment wins the overall comparison, but it did change
  experiment 001's own numbers: test PLCC/SROCC/KROCC each *decreased*
  slightly (e.g. PLCC 0.8804 -> 0.8667, worse) while MSE/MAE/RMSE each
  *decreased* too (e.g. RMSE 0.6136 -> 0.5736, better, since lower is
  better for these) and mean bias shrank (+0.3049 -> +0.1862). Validation-
  set PLCC did not transfer to a higher test-set PLCC than validation-loss
  selection did, on this run.

**PLAUSIBLE HYPOTHESIS** (consistent with, but not proven by, the above):
- Experiment 001's validation-based checkpoint selection (on either
  metric) may have acted as an implicit regularizer/early-stopping
  mechanism that experiment 003's unconditional final-epoch checkpoint
  lacked -- this holds regardless of which validation metric was used to
  pick the epoch, since both selected checkpoints substantially
  outperform experiment 003.
- That validation PLCC and test PLCC diverged slightly for experiment 001
  (previous point) is consistent with the 100-image validation split
  being small enough that its epoch-to-epoch PLCC is noisy (see
  `history.json`'s `val_plcc` swings, e.g. epoch 25's 0.922 vs.
  neighboring epochs' ~0.97) -- one seed/split's validation PLCC ranking
  need not perfectly predict the test-set ranking of nearby epochs.
- The bias-sign flip (001 vs. 003) may reflect where each specific
  selected/stopped checkpoint's Sigmoid output happened to sit, rather
  than a general property of "all-data training."
- Training-set-size difference (900 vs. 1000 images) could plausibly
  affect the model; this project's data does not isolate this from the
  checkpoint-selection explanation above.

**NOT DETERMINED:**
- Whether checkpoint-selection method, training-set size, or a
  combination is the dominant cause of the 001-vs-003 gap -- no
  controlled ablation (e.g. a final-epoch checkpoint from a 900/100 run,
  or a validation-selected checkpoint from an all-1000 run) exists in
  this project.
- Whether an earlier (non-final) epoch of experiment 003's own run would
  have matched or exceeded experiment 001 -- untestable without either a
  validation signal for experiment 003 (none exists) or evaluating
  intermediate checkpoints against the test set (out of scope; would
  contaminate the test set).
- Whether PLCC-based selection is generally better or worse than
  loss-based selection for this model/dataset -- this project has one
  seed's worth of evidence (this run), not a multi-seed comparison, so no
  general claim about which selection metric is "better" is made here.
- Any causal mechanism for the bias-sign flip specifically.

## G. Prediction behavior

**OBSERVED RESULT.** Descriptive-only linear fits (`prediction = a *
ground_truth + b`, fit by ordinary least squares on the existing
predictions; **not** used to recalibrate either model or to produce any
reported performance metric):

| Experiment | slope (a) | intercept (b) | R² | std(pred)/std(GT) |
|---|---|---|---|---|
| 001 | 0.7350 | 0.7509 | 0.7513 | 0.8480 |
| 003 | 0.6856 | 0.3303 | 0.7207 | 0.8076 |

An ideal, uncompressed 1:1 relationship would have slope 1.0 and
intercept 0.0. Experiment 003's descriptive slope (0.686) is further from
1.0 than experiment 001's (0.735), and its std ratio (0.808) is close to
(but still slightly further from 1.0 than) experiment 001's (0.848) --
both consistent with a similar, moderate degree of compression of
predictions toward the mean in both experiments (unlike the
pre-2026-09-14 checkpoint, whose std ratio of 1.003 showed essentially no
compression at all -- see section B), though neither experiment shows a
collapse (both retain most of the ground-truth spread and R² > 0.7).
Neither experiment shows hard saturation at the score boundaries: no
prediction reaches exactly 0 or 4 in either experiment's test set (min/max
in section B/E above).

Figures: `results/figures/00{1,3}_prediction_vs_gt_analysis.png` (identity
line vs. descriptive-only regression line, clearly distinguished) --
distinct from, and does not overwrite, the original evaluation notebook's
`00{1,3}_*_test_pred_vs_gt.png`.

## H. Error behavior

**OBSERVED RESULT.** Absolute-error-vs-ground-truth-quality correlation
(`corr(ground_truth, |prediction - ground_truth|)`):

| Experiment | corr(GT, \|error\|) | mean \|error\| lowest GT quartile | mean \|error\| highest GT quartile |
|---|---|---|---|
| 001 | -0.195 | 0.621 | 0.431 |
| 003 | +0.462 | 0.315 | 0.783 |

Experiment 001's absolute error is (weakly) *higher* for low-quality
images and lower for high-quality images; experiment 003 shows the
opposite pattern, more strongly -- error grows toward high-quality images.
This is reported as an **observed relationship**, not a causal claim (no
mechanism is established by this data for why either pattern holds).

**Top-error cases** (`results/tables/00{1,3}_top_error_cases.csv`, top 10
by absolute error each): experiment 001's worst cases are dominated by
strong *overprediction* on low/mid ground-truth images (e.g.
`test228.tiff`, GT 0.33 -> prediction 1.80, error 1.46); experiment 003's
worst cases are dominated by strong *underprediction* on high
ground-truth images (e.g. `test180.tiff`, GT 4.00 -> prediction 1.79,
error 2.21), consistent with the quality-bin and error-vs-quality findings
above. No medical/diagnostic claim is made about any specific image.

## I. Calibration status

**CALIBRATION STATUS: requires new protocol.**

`src/ct_iqa/evaluation/calibration.py`'s
`five_parameter_logistic_fit(y_pred, y_true)` fits its 5 parameters
directly against whatever pair it receives -- it has no built-in
train/apply split. No calibration mapping has been fit on the test set,
here or anywhere else in this project; final test metrics (sections B, E,
F) remain raw/uncalibrated.

A valid non-test calibration protocol is not currently constructible from
existing artifacts because: (1) experiment 001's 100-image validation
split has only summary metrics persisted (`validation_metrics.json`), not
the per-image `(y_pred, y_true)` pairs a 5PL fit would need; (2)
experiment 003 has no validation split at all, so it has no non-test data
to calibrate against; and (3) no "fit on validation, freeze, apply
unchanged to test" code path exists anywhere in `src/ct_iqa/`. Building
one is a new protocol decision -- deliberately not implemented in this
analysis-only phase.

## J. Observed findings

- Experiment 001 outperforms experiment 003 on every reported test
  metric (PLCC, SROCC, KROCC, MSE, MAE, RMSE) -- section F -- both before
  and after switching experiment 001's checkpoint selection from lowest
  validation loss to highest validation PLCC (2026-09-14).
- Experiment 001 overpredicts on average (+0.1862); experiment 003
  underpredicts on average (-0.3396) -- opposite signs -- sections B, E.
- Experiment 003's underprediction bias grows monotonically more negative
  at higher ground-truth quality (+0.108 to -0.863 across quartiles) --
  section F. Experiment 001's bias is positive in the lowest three
  quartiles and turns slightly negative (-0.289) in the highest quartile
  -- section F.
- Neither experiment shows hard saturation at the score boundaries --
  sections B, E.
- Experiment 003's prediction std/ground-truth-std ratio (0.808) and
  experiment 001's (0.848) are both moderately below 1.0, and both
  experiments' descriptive regression slopes (0.686 and 0.735
  respectively) are similarly below 1.0 -- both experiments show a
  broadly similar, moderate degree of compression toward the mean --
  section G.
- Experiment 003's final training loss is lower than experiment 001's,
  despite performing worse on the test set -- section D.
- PLCC and SROCC are close to each other within each experiment (Exp 001:
  0.8667 vs. 0.8699; Exp 003: 0.8490 vs. 0.8525), consistent with a
  reasonably linear/monotonic prediction-vs-ground-truth relationship in
  both cases; KROCC is consistently lower than both in both experiments,
  expected given its different (more conservative) numeric scale.
- Switching experiment 001's selection criterion from lowest validation
  loss to highest validation PLCC selected a different epoch (20 instead
  of 21) and changed experiment 001's own test-set numbers -- correlation
  metrics slightly worsened (PLCC 0.8804 -> 0.8667) while MSE/MAE/RMSE
  and mean bias all improved -- but did not change which experiment wins
  the 001-vs-003 comparison. See section F for the full breakdown; this is
  evidence that the two selection criteria are not equivalent in
  practice, not evidence that one is generally superior (single-seed
  result).

## K. Plausible hypotheses

See the "OBSERVED / PLAUSIBLE HYPOTHESIS / NOT DETERMINED" breakdown in
section F. In brief: the leading plausible (not proven) hypothesis is
that experiment 001's validation-based checkpoint selection -- on either
metric -- acted as an implicit regularizer that experiment 003's
unconditional final-epoch checkpoint lacked; training-set-size difference
(900 vs. 1000 images) remains a plausible, uncontrolled-for contributing
factor. The small divergence between validation-PLCC and test-PLCC
rankings for nearby epochs of experiment 001 is plausibly attributable to
the 100-image validation split's epoch-to-epoch PLCC noise, not to a flaw
in PLCC as a selection criterion per se.

## L. Limitations

- Only the ResNet50 baseline is implemented; no InceptionResNetV2.
- Results on LDCT-IQAC are not directly comparable to the paper's own
  reported numbers (different dataset, different label semantics).
- Dropout probability (0.5) is this project's choice, not a paper value.
- No calibration protocol (5PL or otherwise) has been validated on
  non-test data; only raw metrics are reported for both experiments (see
  section I).
- Experiment 003 has no validation split, so there is no principled,
  non-test way (within this project's current implementation) to
  diagnose why it underperforms experiment 001 beyond the training-loss
  curve alone.
- No controlled ablation exists to separate "checkpoint-selection method"
  from "training-set size" as causes of the experiment 001 vs. 003 gap
  (section F).
- No statistical significance testing is implemented anywhere in this
  project's evaluation code -- all differences reported here are
  descriptive.
- The choice of PLCC over validation loss as the checkpoint-selection
  metric (section A) is evaluated here on a single seed/run; this project
  does not claim PLCC-based selection is generally superior to loss-based
  selection, only that they are not equivalent in practice (section F).

## M. Reproducibility information

See `docs/replication/reproducibility.md` for seeding
(`ct_iqa.utils.seed.set_seed(42)`), the train/validation split mechanism,
and configuration provenance (`experiments/<NNN_name>/config.json`
alongside each checkpoint). Every checkpoint referenced above
(`experiments/001_resnet50_baseline/checkpoint/best.pt`,
`experiments/003_final_training/checkpoint/final.pt`) was independently
reloaded into a fresh model instance/process and evaluated with
`model.eval()`/`torch.no_grad()` before any number in sections B/E was
recorded. The analysis in sections F-K
(`notebooks/04_evaluation/02_prediction_analysis.ipynb`) reads only the
already-persisted prediction CSVs and training-history JSONs -- it
reloads no checkpoint and reruns no inference, and its recomputed metrics
were cross-checked to match the persisted `results/metrics/*.json` values
exactly before any further analysis was performed.

## Difference from the Ohashi paper

Both experiments 001 and 003 are **Ohashi-style RadImageNet ResNet50
models adapted to LDCT-IQAC**, not exact reproductions of the paper. The
paper regresses a synthetic-degradation-derived VIF score on its own
dataset; this project regresses LDCT-IQAC's real radiologist-assigned
quality score. See `docs/replication/dataset_adaptation.md` and
`docs/replication/deviations.md` for the full, itemized list of paper
facts vs. implementation decisions. The PLCC/SROCC/KROCC values reported
above are never compared numerically against the paper's own reported
values, since the label semantics and datasets differ.
