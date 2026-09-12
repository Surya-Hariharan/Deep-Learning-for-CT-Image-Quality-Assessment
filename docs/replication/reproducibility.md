# Reproducibility

**OUR IMPLEMENTATION.**

## Seeding

`ct_iqa.utils.seed.set_seed(seed)` seeds Python's `random`, NumPy, and
PyTorch (CPU and CUDA) RNGs, and requests deterministic cuDNN behavior
(`torch.backends.cudnn.deterministic = True`,
`torch.backends.cudnn.benchmark = False`). This is best-effort determinism:
some CUDA ops remain nondeterministic regardless of these flags, but it is
sufficient for reproducible data splits and weight initialization on the
CPU-only baseline this project currently runs.

## Train/validation split

`ct_iqa.data.splits.train_val_split` uses `torch.utils.data.random_split`
with a `torch.Generator` seeded from `ExperimentConfig.seed`, so the same
`seed` + `val_fraction` always produces the same split. The official
LDCT-IQAC test set is never involved in this split -- it is loaded
independently by `ct_iqa.data.loader.build_test_dataloader` and used only
for final held-out evaluation.

## Configuration provenance

Every experiment run should save its resolved `ExperimentConfig` (see
`ExperimentConfig.save`/`.load`) alongside its checkpoint, under
`experiments/<NNN_name>/config.json` -- so that the exact hyperparameters,
paths, and seed used for a given checkpoint/result are always recoverable
from the experiment directory itself, not just from whatever
`configs/*.yaml` happen to say at some later point in time.

## Known reproducibility gaps

- **RadImageNet weight loading is not currently exercised in any persisted
  result** (see `deviations.md`) -- reproducing a *RadImageNet-initialized*
  result is not yet possible until real weights are obtained and a full
  training run is performed and recorded.
- **No full 30-epoch training run has been executed and committed to an
  `experiments/` directory as of this migration** -- see
  `experiments/001_resnet50_baseline/README.md`.
