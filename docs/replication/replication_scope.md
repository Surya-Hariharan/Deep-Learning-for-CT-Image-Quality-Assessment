# Replication scope

**OUR IMPLEMENTATION.** This document defines what this project does and
does not attempt to replicate from Ohashi et al.

## In scope

- The **ResNet50** baseline only (`src/ct_iqa/models/resnet50.py` +
  `src/ct_iqa/models/ohashi_resnet50.py`).
- RadImageNet pretraining of that backbone (subject to weight
  availability -- see `docs/replication/deviations.md` and
  `weights/pretrained/radimagenet/resnet50/README.md`).
- The regression head (`Dropout -> Dense(1) -> Sigmoid`) and 224x224
  center-crop preprocessing, as specified in `docs/paper/architecture.md`
  and `docs/paper/preprocessing.md`.
- The paper-specified training configuration (Adam, MSE, batch size 64,
  30 epochs, lr 1e-3) in `src/ct_iqa/training/`.
- PLCC/SROCC/KROCC + 5PL-mapped correlation evaluation, in
  `src/ct_iqa/evaluation/`.

## Explicitly out of scope

- **InceptionResNetV2** -- the paper's second backbone. Not implemented
  anywhere in this repository, and not planned.
- **Any project novelty beyond the ResNet50 baseline itself** -- this
  repository's current purpose is a faithful baseline replication, not new
  methodology.
- **Synthetic degradation + VIF labeling**, the paper's own dataset
  construction pipeline. This project instead uses LDCT-IQAC, a dataset
  that already varies in real quality with human-assigned labels -- see
  `dataset_adaptation.md`. There is no `src/ct_iqa/degradation/` or
  `src/ct_iqa/labeling/` package, and none is planned unless this project's
  scope changes to require reproducing the paper's own dataset from
  scratch.

## Current experimental status

As of this migration: the ResNet50 baseline architecture, dataset loader,
training loop, and evaluation pipeline are implemented and unit-tested
end-to-end (`pytest`), and the pipeline has been exercised as a
smoke-test/sanity-check (forward pass, one optimizer step, full evaluation
codepath) against a randomly-initialized backbone. **No full 30-epoch
training run has been executed and persisted in this repository as of this
migration**, and RadImageNet weights are not confirmed loaded in any
existing result -- see `weights/pretrained/radimagenet/resnet50/README.md`
for current status. Do not treat any numbers currently reachable from this
repository's code as a finished replication result.
