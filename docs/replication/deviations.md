# Deviations from the paper

**OUR IMPLEMENTATION / IMPLEMENTATION DECISION.** Every entry below is
either something the paper doesn't specify (and this project had to choose
something) or something this project deliberately does differently. Nothing
here is a paper fact -- see `docs/paper/` for those. Nothing here was
invented to fill a gap in the paper's own description; where a detail is
simply unknown, that is stated as unknown.

## Dataset: LDCT-IQAC instead of the paper's own dataset

**IMPLEMENTATION DECISION.** See `dataset_adaptation.md` for the full
comparison. Summary: LDCT-IQAC's human radiologist quality score is used as
the regression target, not the paper's VIF metric.

## No synthetic degradation pipeline

**IMPLEMENTATION DECISION.** The paper generates training data by applying
synthetic degradation (noise, blur, etc.) to clean CT images at varying
severities. This project does not implement any degradation pipeline
(`src/ct_iqa/degradation/` does not exist) because LDCT-IQAC already
supplies images with real, naturally-occurring quality variation. Building
an unused degradation pipeline just to match the target directory structure
would be inventing paper methodology this project doesn't actually use --
see `docs/internal/decisions/README.md`.

## No VIF label computation

**IMPLEMENTATION DECISION**, following directly from the above:
`src/ct_iqa/labeling/vif.py` does not exist. LDCT-IQAC's labels are used
as-is (`ct_iqa.data.ldct_iqac.normalize_score`/`denormalize_score` handle
only the linear rescaling to/from the model's `[0, 1]` Sigmoid output
space, not VIF computation).

## Image format: TIFF, not PNG

**IMPLEMENTATION DECISION** (forced by the actual data). An earlier task
brief assumed the dataset would be PNG; the images actually found on disk
are TIFF (`.tif`/`.tiff`, PIL mode `'F'`, 32-bit float, pixel range
`[0, 1]`). `ct_iqa.data.ldct_iqac.LDCTIQACDataset` was written around the
format actually present, not the assumed one.

## Grayscale-to-3-channel replication

**IMPLEMENTATION DECISION**, not specified by the paper. LDCT-IQAC images
are single-channel float TIFFs; `OhashiResNet50` internally replicates them
to 3 channels (`x.repeat(1, 3, 1, 1)`) before the backbone, matching the
common convention for applying RGB-pretrained backbones (including
RadImageNet's own training convention) to grayscale medical images.

## Dropout probability

**IMPLEMENTATION DECISION.** The paper states a Dropout layer precedes the
final Dense layer but never gives its probability.
`ExperimentConfig.dropout_p` has **no built-in default** in
`src/ct_iqa/config.py` -- it must be set explicitly (`configs/model.yaml`
currently sets it to `0.5`, a common default elsewhere in the literature,
not a value taken from the paper) so every experiment run records a
deliberate choice rather than a silently invented "paper value."

## Pixel normalization formula

**IMPLEMENTATION DECISION.** The paper does not state a pixel normalization
formula. This project maps the source `[0, 1]` float range to `[-1, 1]`
(`ct_iqa.preprocessing.normalize.to_radimagenet_range`), matching
RadImageNet's own preprocessing convention as confirmed from RadImageNet's
official reference code (`BMEII-AI/RadImageNet/pytorch_example.ipynb`), on
the reasoning that a RadImageNet-pretrained backbone should see inputs
normalized the way RadImageNet itself was trained on.

## ResNet50 stride placement ("v1" vs. "v1.5")

**IMPLEMENTATION DECISION**, made specifically *for* faithful replication.
`src/ct_iqa/models/resnet50.py`'s bottleneck blocks place the stride-2
downsampling on the **first 1x1 (reduce) conv** of a stage's first block --
the "ResNet v1" placement used by `keras.applications.ResNet50` (and
therefore by the RadImageNet-pretrained weights this project loads) -- NOT
the "v1.5" placement (stride on the 3x3 conv) used by `torchvision`'s
ResNet50. This is a deliberate correction so RadImageNet's Keras-trained
weights map onto this backbone without a stride/receptive-field mismatch;
using the more common torchvision convention here would have made weight
*shapes* match while silently producing a semantically different feature
extractor.

## RadImageNet weight availability

**EXPERIMENTAL STATUS**, not a design decision. **UPDATED 2026-09-13** by
the scientific pipeline audit: the officially-sourced RadImageNet `.h5` has
been obtained and converted, and `OhashiResNet50.load_radimagenet_weights`
has been verified against it (265/318 backbone keys matched, 0 unexpected,
the remaining 53 are non-learned `num_batches_tracked` counters) -- see
`weights/pretrained/radimagenet/resnet50/README.md` for the full
verification record. `configs/model.yaml`'s `radimagenet_weights_path` now
points at the converted checkpoint (previously `null`).

**What is still not true:** no full training run using these weights has
been executed and persisted under `experiments/` as of this audit --
*loading* is verified, a *trained result* using the loaded weights is not.
This project deliberately does **not** substitute unverified third-party
"RadImageNet" checkpoints found elsewhere (e.g. community re-uploads) to
avoid mislabeling a result as a RadImageNet replication when it isn't
verified as one, and `load_radimagenet_weights`/`torch.load` never silently
fall back to ImageNet weights or random initialization on a load failure --
a missing/incompatible file raises loudly instead.

## Checkpoint-selection metric

**IMPLEMENTATION DECISION**, not specified by the paper (the paper states
a fixed 30-epoch training run with no described checkpoint-selection or
early-stopping procedure -- see `docs/paper/training.md`). Since this
project reports and compares experiments primarily on PLCC/SROCC (raw
`[0,4]` scale -- see `docs/replication/final_results.md`), not on training
loss, `ct_iqa.training.trainer.train` selects the saved `best.pt`
checkpoint by the best validation-set PLCC seen during training
(`config.selection_metric`, default `"plcc"` in `configs/training.yaml`),
rather than by the lowest validation loss in normalized `[0,1]` target
space. `"srocc"` and `"val_loss"` remain available as alternatives.

**Updated 2026-09-14**, following an external repository audit's "Model
Selection Mismatch in Training Loop" finding: minimum validation MSE does
not guarantee maximum validation PLCC/SROCC, so a checkpoint selected on
loss alone could be presented as "validation-selected" while not actually
being selected for the metric it's compared on. Experiment 001 was
retrained under the corrected criterion; see
`docs/internal/decisions/README.md` (2026-09-14 entry) for the mechanism
and `docs/replication/final_results.md` section F for what changing the
criterion did and did not change about the measured result (both
selection criteria still make experiment 001 outperform experiment 003).

## Reporting both raw and 5PL-mapped metrics

**IMPLEMENTATION DECISION**, not a deviation from what's measured (see
`docs/paper/evaluation.md`) -- both `compute_metrics` (raw) and
`compute_metrics_with_logistic_mapping` (5PL-mapped, matching the paper's
methodology) are computed and reported side by side, labeled, rather than
only reporting the 5PL-mapped numbers as the paper does.
