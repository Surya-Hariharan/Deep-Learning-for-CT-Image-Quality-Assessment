# Preprocessing and checkpoint compatibility

Status: **Part 1 (pre-registration) written before the probe was run.** Part 2 (results) is appended after execution.
Related: [ADR-002](../decisions/0002-batchnorm-and-preprocessing.md), [preprocessing-and-weights](../preprocessing-and-weights.md).

## Part 1 - Pre-registered frozen-backbone probe

### Purpose and limits
The pretraining input convention of the RadImageNet ResNet50 `.h5` release is unverified (ADR-002 E4 vs E6). A
frozen-backbone ridge probe measures how *useful* each candidate input convention is for this task. Downstream probe
performance is evidence of task suitability, **not** proof of the convention used to pretrain the weights.

### Candidate conventions (declared; all use the 224x224 center crop of the [0,1] float image, gray replicated to 3 channels)
| ID | Name | Per-channel input |
|---|---|---|
| P1 | `2x-1` | `2x - 1`, range [-1, 1] (current project convention; RadImageNet PyTorch example) |
| P2 | caffe-style | `x*255 - mean_c`, then `/255`, with BGR channel means (103.939, 116.779, 123.68); the Keras `preprocess_input` reading of the TF fine-tuning script (ADR-002 E6) |
| P3 | ImageNet mean/std | `(x - mean_c) / std_c`, mean (0.485, 0.456, 0.406), std (0.229, 0.224, 0.225) (E5, unverified third-party claim) |

### Procedure (frozen before any validation result was seen)
1. Backbone: study-v2 `ResNet50Backbone`, `bn_eps = 1.001e-5`, SHA-256-verified RadImageNet weights, `eval()` mode, no
   gradient, GAP features (2048-d). Same backbone for all conventions.
2. Fit data: the 800 **training** images only. Target: raw [0, 4] score.
3. Features are standardized with train-only mean/std. Ridge regression (closed form), regularization `alpha` chosen from
   `logspace(-3, 4, 15)` by **5-fold cross-validation on the training set only** (criterion: CV RMSE), separately per convention.
4. Evaluation: the 100 **validation** images. Primary metric: PLCC; also SROCC, KROCC, RMSE. Calibration and the official test
   set are **never** read by the probe.
5. Comparison rule: a convention is reported as "distinguishably better on validation" only if its PLCC exceeds every other
   convention's by more than the 95% half-width of a paired image-level bootstrap (2000 resamples, seed 0) of the PLCC
   difference. Otherwise the result is reported as inconclusive.
6. **Decision rule (declared now):** the probe does not select the project convention. The convention stays `2x-1`
   (provenance-based default, comparable with historical 001-003) regardless of probe ranking. A probe result may only be
   cited as supporting or not supporting that choice, together with the caveat below.
7. Known caveat: validation images share anatomy with training images (see [DATASET_AND_EDA](DATASET_AND_EDA.md)), so absolute
   probe numbers are optimistic. This affects all conventions alike but limits how much a small difference means.

## Part 2 - Probe results (executed; `tools/preprocessing_probe.py`, output `reports/preprocessing_probe/probe_results.json`)

Run on GPU (RTX 4060, torch 2.5.1+cu121) with the SHA-256-verified weights; features finite for all 900 images; ridge alpha chosen by
train-only 5-fold CV. Validation partition, n = 100. Raw [0, 4] scale.

| Convention | alpha (CV) | PLCC | SROCC | KROCC | RMSE |
|---|---|---|---|---|---|
| P1 `2x-1` | 100 | 0.9724 | 0.9735 | 0.8759 | 0.2591 |
| P2 caffe-style | 100 | 0.9709 | 0.9692 | 0.8664 | 0.2665 |
| P3 ImageNet mean/std | 3.16 | 0.8781 | 0.9036 | 0.7462 | 0.5454 |

Paired bootstrap of the PLCC difference (2000 resamples, seed 0): P1 - P2 = +0.0016, 95% CI [-0.0080, +0.0110] (half-width 0.0095) ->
**not distinguishable**; P1 - P3 = +0.094, CI [+0.058, +0.136]; P2 - P3 = +0.093, CI [+0.055, +0.135] -> P3 is worse.

### Interpretation (what this does and does not show)
- P1 and P2 give practically the same downstream task quality; the probe **cannot** tell which convention the weights were
  pretrained under. P3 (ImageNet mean/std, a third-party claim) is clearly less suitable and is not adopted.
- Per the pre-registered decision rule the project convention remains **`2x-1`**: it is the best-supported convention from
  provenance (RadImageNet PyTorch example), it equals the historical pipeline, and the probe gives no reason to change it.
  The residual uncertainty (the TF `.h5` pretraining convention is unverified) is unchanged and is listed as a limitation.
- Validation images share anatomy with training images, so absolute probe numbers are optimistic; relative conclusions
  between conventions are still informative because the same data enter every convention.
- Rejected alternatives: P2 (no evidence of benefit, extra assumption about BGR means on grayscale), P3 (clearly worse, unverified source).

### Checkpoint compatibility
- Weights: `.pt` SHA-256 `45cba0d5...a0f785` verified by the loader; strict load reports all backbone keys matched (only
  `num_batches_tracked` counters are absent by construction). BatchNorm epsilon `1.001e-5` is read from the `.h5` model config (ADR-002 E1).
- Input: 224x224 center crop (crop, not resize), gray replicated to 3 channels inside the model, `2x-1`; label targets normalized to [0, 1]
  with a sigmoid output and inverse-scaled to [0, 4].
- Historical checkpoints 001-003 trained at BN eps 1e-3 and evaluated at 1e-5 (ADR-002); they are not comparable to study-v2 runs
  without that caveat.
