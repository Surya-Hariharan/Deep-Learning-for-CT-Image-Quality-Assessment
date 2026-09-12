# Reproducibility

**OUR IMPLEMENTATION.**

## Seeding

`ct_iqa.utils.seed.set_seed(seed)` seeds Python's `random`, NumPy, and
PyTorch (CPU and CUDA) RNGs, and requests deterministic cuDNN behavior
(`torch.backends.cudnn.deterministic = True`,
`torch.backends.cudnn.benchmark = False`). This is best-effort determinism:
some CUDA ops remain nondeterministic regardless of these flags.

**Updated 2026-09-13** (experiment 001's first real training run): this
project now runs on **GPU** (`device="cuda"`, NVIDIA GeForce RTX 4060
Laptop GPU, PyTorch 2.5.1+cu121), not the CPU-only baseline this note
previously assumed. `cudnn.benchmark = False` forgoes cuDNN's algorithm
auto-tuning, which has a real (if modest, on the batch sizes/image size
used here) throughput cost on GPU in exchange for run-to-run determinism.
This tradeoff is intentional and is not silently ignored: reproducibility
of the train/validation split and weight initialization is prioritized
over shaving GPU time off a 30-epoch run that already completes in
minutes.

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

## Pipeline validation (2026-09-13 scientific pipeline audit)

Before any full training run, the complete `raw image -> dataset loader ->
preprocessing -> tensor -> target normalization -> split -> DataLoader ->
model -> Sigmoid prediction` path was independently verified against the
real dataset and real RadImageNet weights:

- `notebooks/02_data_preparation/01_dataset_preparation.ipynb` -- dataset
  integrity, score distribution, preprocessing correctness, model-input
  shapes, and train/test leakage, ending in an explicit `DATA PIPELINE
  STATUS: READY` assertion.
- `tests/integration/test_ldct_iqac_pipeline_integration.py` -- the same
  invariants as automated tests (dataset counts, no leakage, score
  normalization bounds, one real training step, RadImageNet loading via
  the actual configured path), skipped gracefully when the real
  dataset/weights aren't present locally.
- A 2-batch smoke test (`notebooks/03_training/01_resnet50_baseline.ipynb`)
  runs a real forward/backward/`optimizer.step()` on real data + the real
  RadImageNet-loaded model before any full-training decision is made, then
  restores pre-smoke-test weights.

## Experiment 001's first real training run (2026-09-13)

The first full, RadImageNet-initialized, 30-epoch training run has been
executed and persisted: `experiments/001_resnet50_baseline/` now contains
`checkpoint/best.pt` (model + optimizer state + epoch + val_loss + config +
seed), `history.json` (full per-epoch train/val loss), and
`validation_metrics.json` (post-hoc PLCC/SROCC/KROCC on the **validation**
split only). See `experiments/001_resnet50_baseline/README.md` for the
actual measured numbers. The checkpoint was independently reloaded in a
fresh model instance/process and verified to produce finite, `[0,1]`-bounded
predictions on a real validation batch before this was considered done.

No NaN/Inf loss occurred at any of the 30 epochs (verified programmatically
over the saved history, not just spot-checked). The validation loss curve
is noticeably noisier than the training loss curve, which is expected given
the validation split is only 100 images (2 batches) -- reported as an
observed property of this run, not treated as a defect.

## Known reproducibility gaps

- **Test-set evaluation has not yet been performed** for this checkpoint --
  the 300-image LDCT-IQAC test set remains untouched by training and by
  checkpoint/model selection, per design. The next step is an independent
  pass with `notebooks/04_evaluation/01_test_set_evaluation.ipynb`.
- **No learning-rate search or further training run exists yet**
  (`experiments/002_learning_rate_search/`,
  `experiments/003_final_replication/` remain scaffolding-only).
