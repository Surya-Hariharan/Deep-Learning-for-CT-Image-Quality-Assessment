# Architecture

## Pipeline (target vs. implemented)

```
Longitudinal CT scans
        |
Preprocessing              <- implemented (preprocessing/volumes.py)
        |
Image Quality Assessment   <- planned (quality/)
        |
Longitudinal Nodule Analysis
        |
NGP-Net Growth Prediction  <- implemented (ngpnet/)
        |
Prediction Uncertainty     <- planned (uncertainty/)
        |
Quality + Uncertainty Fusion <- planned (fusion/) -- the project's contribution
        |
Accept / Flag / Reject
        |
Structured Longitudinal Report <- planned (reporting/)
```

Currently implemented foundation:

```
PNG dataset -> NGP-Net -> growth prediction
```

## Package layout

```
src/quality_aware_lung_ct/
    ngpnet/          NGP-Net model, losses, config, inference
        layers.py      generic building blocks (DropPath, LayerNorm, GRN, Down/UpBlock)
        temporal.py    time-conditioned blocks (TEM, STEM, ReSTEBlock, BlockGroup)
        encoder.py     UNetEncoder
        decoder.py     UNetDecoder (shared by shape_net and texture_net)
        heads.py       SpatialTransformer
        model.py       NGPNet (encoder + two decoders + spatial transformer)
        losses.py      training losses (SSIM, Dice+CE, gradient smoothness)
        config.py      NGPNetConfig dataclass + CLI parser (no import-time side effects)
        inference.py   NGPNetPredictor / GrowthPrediction wrapper
    data/png/        PNG dataset manifest, splits, transforms, PyTorch Dataset
    preprocessing/   volume I/O, cropping, CT windowing, lung segmentation
    training/        optimizer/loss construction, epoch loop, checkpointing
    evaluation/      metrics (PSNR, SSIM, MSE, Dice, confusion-matrix-derived) + evaluator
    quality/         planned -- see package docstring
    uncertainty/     planned -- see package docstring
    fusion/          planned -- see package docstring (the project's actual contribution)
    reporting/       planned -- see package docstring
    common/          seeding, CSV run logging
```

## Provenance: what changed vs. the original NGP-Net code

The original author implementation (`nn/`, `utils/`, `train.py`, `test.py`,
`cross_val.py`, `cross_test.py`) has been reorganized into the package
above. The reorganization is **packaging only** -- module boundaries,
import paths, and configuration handling changed; the network's
mathematical behavior did not.

| Original | New location | What changed |
|---|---|---|
| `nn/ngpnet.py` (`NGPnet`) | `ngpnet/model.py` (`NGPNet`) + `encoder.py`/`decoder.py`/`heads.py` | Split into files; renamed class for naming consistency. Forward math unchanged. |
| `nn/net_utils.py` | `ngpnet/layers.py` + `ngpnet/temporal.py` | Split by concern (generic vs. time-conditioned blocks). Math unchanged. |
| `nn/losses.py` | `ngpnet/losses.py` | Verbatim. |
| `nn/metrics.py` | `evaluation/metrics.py` | Verbatim. |
| `utils/config.py` | `ngpnet/config.py` | Was `parser.parse_args()` executed at *import time*, plus `setproctitle` and RNG seeding as import side effects. Now a `NGPNetConfig` dataclass (same defaults) with an explicit `build_arg_parser()`/`config_from_args()` for CLI use. Importing the module does nothing. |
| `utils/dataloader.py` (`DatasetNG3T`) | `data/png/dataset.py` (`PNGDataset`) | Paths resolved through `metadata.resolve_series_path`, which tolerates the manifest's `.nii`/`.nii.gz` mismatch (see docs/datasets.md) instead of a naive `os.path.join`. |
| `utils/transforms.py` | `data/png/transforms.py` | Verbatim transform classes; `get_transforms` now takes `NGPNetConfig` instead of an argparse namespace. |
| `utils/img_utils.py` | `preprocessing/volumes.py` | Verbatim. |
| `utils/common.py` | `training/trainer.py`, `training/checkpoints.py`, `common/logging.py` | Split by concern. |
| `train.py` + `cross_val.py` | `training/trainer.py` (`NGPNetTrainer.fit`) + `scripts/train.py` | These two scripts were a duplicated epoch loop (single split vs. k-fold). Consolidated into one implementation parameterized by an optional `fold` label. |
| `test.py` + `cross_test.py` | `evaluation/evaluator.py` (`evaluate`) + `scripts/evaluate.py` | Same consolidation for the duplicated test-time loop. |

No layer definition, tensor shape, loss formula, or metric formula was
changed. `tests/test_model.py` asserts the parameter count at the paper's
defaults (`img_size=64, feat_dim=8`) is unchanged: **1,984,821**.
