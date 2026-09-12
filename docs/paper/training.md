# Training (paper)

**PAPER FACT**, as replicated in `configs/training.yaml` /
`src/ct_iqa/training/`:

- Optimizer: **Adam**.
- Loss: **MSE** (mean squared error).
- Batch size: **64**.
- Epochs: **30**.
- Learning rate: **1e-3**.

No scheduler, warmup, weight decay, early stopping, data augmentation, or
mixed precision is mentioned for this baseline configuration; this
project's training loop (`src/ct_iqa/training/trainer.py`) accordingly adds
none of these, to stay close to the paper's stated configuration for the
first replication baseline.

## What the paper does not specify (for this project's regression target)

The regression **target** used during training in the paper is the VIF
score computed from synthetically-degraded images (see
`docs/paper/preprocessing.md`). This project's training loop instead
regresses against LDCT-IQAC's radiologist-assigned quality score -- see
`docs/replication/dataset_adaptation.md` and
`docs/replication/deviations.md`.
