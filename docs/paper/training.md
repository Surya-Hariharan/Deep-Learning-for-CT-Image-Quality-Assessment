# Training (paper)

**PAPER FACT**, as replicated in `configs/training.yaml` /
`src/ct_iqa/training/`:

- Optimizer: **Adam**.
- Loss: **MSE** (mean squared error).
- Batch size: **64**.
- Epochs: **30**.
- Learning rate: **1e-3**.

No scheduler, warmup, weight decay, data augmentation, or mixed precision is
mentioned for this baseline configuration; this project's training loop
(`src/ct_iqa/training/trainer.py`) accordingly adds none of these, to stay
close to the paper's stated configuration for the first replication
baseline.

**This does NOT include checkpoint selection.** `trainer.train` saves the
model checkpoint with the best validation-set PLCC seen so far
(`config.selection_metric`, see `docs/replication/deviations.md`) --
functionally a form of validation-based model selection, since only the
epoch that improves on this metric is kept. This is an explicit
**IMPLEMENTATION DECISION**, not a paper-specified detail: the paper's fixed
30-epoch configuration above doesn't call for it, and it was added because
this project reports and compares experiments on PLCC/SROCC, not because
the paper describes early stopping or any other selection rule.

## What the paper does not specify (for this project's regression target)

The regression **target** used during training in the paper is the VIF
score computed from synthetically-degraded images (see
`docs/paper/preprocessing.md`). This project's training loop instead
regresses against LDCT-IQAC's radiologist-assigned quality score -- see
`docs/replication/dataset_adaptation.md` and
`docs/replication/deviations.md`.
