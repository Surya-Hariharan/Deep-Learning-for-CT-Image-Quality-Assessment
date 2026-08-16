# Reproduction

## Environment

```bash
pip install -e .              # runtime (inference) dependencies only
pip install -e ".[train]"     # + monai, for scripts/train.py --lrschedule warmupcosine
pip install -e ".[dev]"       # + pytest
pip install -e ".[notebook]"  # + matplotlib/jupyter, for notebooks/
```

Requires Python >=3.10. Verified in a CPU-only environment (torch 2.13,
CUDA unavailable) as well as `torch>=2.0` generally; CUDA is used
automatically when available (`NGPNetConfig.resolve_device`).

## Dataset setup

The PNG dataset is not distributed with this repository. Obtain it (see
`docs/upstream/NGP-Net-README.md` for the source) and place it at:

```
data/external/png/
```

or point `--data-root` at wherever you keep it. See `data/README.md`.

## Validate the dataset

```bash
python scripts/validate_dataset.py --data-root data/external/png
```

## Smoke test (no dataset required)

```bash
python scripts/smoke_test.py
```

Imports the package, constructs `NGPNet(img_size=64, feat_dim=8)`, asserts
the parameter count is 1,984,821, and runs one synthetic forward pass on
CPU (and CUDA, if available). Completes in seconds; trains nothing.

## Unit tests

```bash
pytest tests -q
```

No test requires the PNG dataset; dataset-shaped tests use synthetic
fixtures (`tests/test_dataset.py`).

## Visual walkthrough

```bash
jupyter notebook notebooks/01_ngpnet_walkthrough.ipynb
```

Builds the model, loads a real PNG window (or a synthetic one if the
dataset isn't present), runs a forward pass, and plots the input slices,
predicted image/mask, and deformation field. Uses a randomly initialized
model until a checkpoint exists -- see the notebook's own status note and
`notebooks/README.md`.

## Training

```bash
python scripts/train.py --data-root data/external/png
python scripts/train.py --data-root data/external/png --cross-validate
```

Hyperparameter defaults (`NGPNetConfig`) are inherited unchanged from the
author's `utils/config.py` -- see `docs/architecture.md` for the mapping.

**Status: not run in this repository.** No checkpoint exists under
`checkpoints/`. Training on the full 748-window training split for 200
epochs is a multi-hour GPU job; it has not been executed as part of this
cleanup pass, so no trained-model results can be reported yet.

## Evaluation

```bash
python scripts/evaluate.py --data-root data/external/png --trained-model best_model.pth
```

Requires a checkpoint produced by `scripts/train.py`. Reports DSC,
sensitivity (TPR), specificity (TNR), PPV, PSNR, SSIM, MSE_ROI and MSE_PN
on the manifest's `test` split (see `docs/datasets.md` for its overlap
caveat), and writes source/predicted volume pairs under
`--pred-save-dir` (default `outputs/predictions/`).
