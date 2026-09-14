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

**Updated 2026-09-14** (checkpoint-selection fix, see
`docs/internal/decisions/README.md`): experiment 001 was retrained on
**CPU** (`device="cpu"`, PyTorch 2.14.0+cpu -- no CUDA device was available
in the environment that performed this retrain), not the GPU used for the
original 2026-09-13 run. This is recorded in
`experiments/001_resnet50_baseline/config.json`'s `device` field for that
checkpoint; it does not affect the seeding/determinism guarantees above,
only wall-clock time (a CPU run of this size takes on the order of an
hour, vs. minutes on the GPU noted above).

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

## Experiment 001's first real training run (2026-09-13, retrained 2026-09-14)

The first full, RadImageNet-initialized, 30-epoch training run has been
executed and persisted: `experiments/001_resnet50_baseline/` now contains
`checkpoint/best.pt` (model + optimizer state + epoch + val_loss +
`checkpoint_type` + config + seed), `history.json` (full per-epoch
train/val loss **and** val PLCC/SROCC, added 2026-09-14), and
`validation_metrics.json` (post-hoc PLCC/SROCC/KROCC on the **validation**
split only). See `experiments/001_resnet50_baseline/README.md` for the
actual measured numbers. The checkpoint was independently reloaded in a
fresh model instance/process and verified to produce finite, `[0,1]`-bounded
predictions on a real validation batch before this was considered done.

No NaN/Inf loss occurred at any of the 30 epochs (verified programmatically
over the saved history, not just spot-checked). The validation loss/PLCC
curves are noticeably noisier than the training loss curve, which is
expected given the validation split is only 100 images (2 batches) --
reported as an observed property of this run, not treated as a defect.

**Retrained 2026-09-14**: the checkpoint-selection criterion changed from
lowest validation loss to highest validation PLCC (see
`docs/internal/decisions/README.md`), which selects a different epoch (20,
not 21) and therefore produces a different checkpoint/history/metrics than
the original 2026-09-13 run. The "Test set contamination check" and
"Independent test-set evaluation procedure" sections below describe the
current (PLCC-based) procedure.

## Independent test-set evaluation procedure (2026-09-13; script updated 2026-09-14)

`tools/evaluate_baseline_test_set.py` loads
`experiments/001_resnet50_baseline/checkpoint/best.pt` into a **fresh**
`OhashiResNet50` instance (no dependency on the training notebook's
process/memory), cross-checks the checkpoint's own recorded
`dropout_p`/`in_channels`/`image_size` against the current
`configs/*.yaml` before evaluating, then runs inference over the full
300-image test set exactly once (`model.eval()` / `torch.no_grad()`),
using the identical preprocessing/target-normalization functions training
used. See `experiments/001_resnet50_baseline/README.md` for the actual
measured test metrics.

**Note:** this evaluation originally lived in
`notebooks/04_evaluation/01_test_set_evaluation.ipynb`, but that notebook
was repurposed (2026-09-13) to evaluate experiment 003 instead, using
experiment 001's numbers only as a fixed reference for its comparison
table -- it no longer reproduces them. `tools/evaluate_baseline_test_set.py`
is the current, actual reproduction path for experiment 001's test
metrics; it was used to re-evaluate experiment 001 after the 2026-09-14
checkpoint-selection fix required retraining it (see
`docs/internal/decisions/README.md`).

**Calibration policy:** `ct_iqa.evaluation.calibration.five_parameter_logistic_fit`
is deliberately never called with test-set data. It fits its parameters
directly against whichever `(y_pred, y_true)` pair it receives, so fitting
it on the test set and reporting the resulting test metrics would be
circular -- the calibration would be optimized against the exact data
being evaluated, not an unbiased estimate. No validation-set-based
(fit-on-validation, apply-fixed-mapping-to-test) calibration protocol is
implemented in this project yet; calibrated test metrics are deferred
until one is built, rather than computed in a way that would silently
leak test information into the reported result.

**Test set contamination check:** the checkpoint was selected using
validation-set PLCC only (§ above; validation loss until the 2026-09-14
selection-criterion fix), before this evaluation notebook ran; no
hyperparameter in `configs/*.yaml` was chosen using any number this
notebook produced; this notebook constructs no training or validation
`Dataset`/`DataLoader` at all. **Test set used only for final evaluation.**

## Learning-rate search and final-training protocol (2026-09-13)

**IMPLEMENTATION DECISION**, not prescribed verbatim by the paper (the
paper does not describe a validation-driven LR-search-then-retrain-on-all-data
procedure for its own experiment; it states final training hyperparameters
directly). This project's protocol, in order:

1. **Experiment 002 (LR search)**: the 900/100 train/validation split (same
   split used by experiment 001) was used to search the Ohashi paper's own
   set of four learning rates (1e-2, 1e-3, 1e-4, 1e-5), selecting **1e-3**
   by lowest best-validation-loss. The test set was never touched.
2. **Experiment 003 (final training)**: retrained from a **fresh**
   RadImageNet initialization (not experiment 001's or 002's weights),
   using the selected LR (1e-3), on **all 1000 labeled training images** --
   no train/validation split. This is an implementation decision: once LR
   selection is complete, there is no remaining reason to hold out
   validation data from the final model, so all available labeled training
   data is used. Consequently there is **no validation-based "best epoch"**
   in experiment 003 -- the checkpoint is the epoch-30 (final) model,
   named `checkpoint/final.pt` (not `best.pt`), and `ct_iqa.training.checkpointing`'s
   `save_checkpoint`/`load_checkpoint`/`load_checkpoint_metadata` all accept
   an explicit `filename` parameter (default `best.pt`, unchanged for
   experiments 001/002) specifically to support this without duplicating
   checkpoint I/O logic in the final-training notebook.
3. The test set remains untouched until a separate, independent final
   evaluation of `experiments/003_final_training/checkpoint/final.pt`.

## Final independent test-set evaluation of experiment 003 (2026-09-13)

`notebooks/04_evaluation/01_test_set_evaluation.ipynb` (regenerated by
`tools/build_evaluation_notebook.py`, now targeting experiment 003) loaded
`experiments/003_final_training/checkpoint/final.pt` into a **fresh**
`OhashiResNet50` instance, cross-checked the checkpoint's recorded
`dropout_p`/`in_channels`/`image_size` and confirmed
`checkpoint_type == "final_epoch"`, then ran inference over the full
300-image test set exactly once. RAW test metrics: PLCC 0.8490, SROCC
0.8525, KROCC 0.6611, MSE 0.4474, MAE 0.5195, RMSE 0.6689 (`[0,4]` scale).
See `experiments/003_final_training/README.md` and
`docs/replication/final_results.md` for the full record, including the
baseline-vs-final comparison and observed prediction bias.

**Test set contamination check:** training (experiment 003), LR selection
(experiment 002), and checkpoint creation all completed before this
evaluation ran; the final checkpoint is an unconditional final-epoch
checkpoint with no validation- or test-based selection; no calibration was
fit on the test set; this notebook constructs no training or validation
`Dataset`/`DataLoader`. **Test set used only for final evaluation.**

## Known reproducibility gaps

- **No validation-set-based calibration protocol exists yet** -- only raw
  (uncalibrated) test metrics have been computed for experiments 001 and
  003.
