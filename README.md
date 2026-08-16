# Quality-Aware Longitudinal Lung CT Follow-up Assistant

Final-year research project asking whether combining an image-quality
score **Q** with a prediction-uncertainty estimate **U** separates
reliable from unreliable pulmonary-nodule growth predictions better than
either signal alone.

## Pipeline

```
Longitudinal CT scans -> Preprocessing -> Image Quality Assessment ->
Longitudinal Nodule Analysis -> NGP-Net Growth Prediction ->
Prediction Uncertainty -> Quality + Uncertainty Fusion ->
Accept / Flag / Reject -> Structured Longitudinal Report
```

**Currently implemented:** PNG dataset -> NGP-Net -> growth prediction.
Everything downstream of NGP-Net (quality, uncertainty, fusion, reporting)
is a designed architecture boundary, not yet an implementation -- see
`docs/research-status.md` for exactly what exists and what doesn't.

**The novelty is the fusion.** NGP-Net is the baseline, not the contribution.

## Baseline: NGP-Net

Tang, Luo, Liu, Huang & Zou, *NGP-Net: A Lightweight Growth Prediction
Network for Pulmonary Nodules*, IEEE TMI 2026
([doi:10.1109/TMI.2026.3656184](https://doi.org/10.1109/TMI.2026.3656184)) --
official code at <https://github.com/XinKai-Tang/NGP-Net>. This repository
is a **reimplementation/reproduction**, not an independent invention: the
architecture, tensor shapes, losses, and metrics are migrated from that
implementation unchanged, reorganized into an installable package. See
`docs/architecture.md` for exactly what moved where and why, and
`docs/upstream/NGP-Net-README.md` for the authors' own description.

Measured parameter count at the paper's defaults (`img_size=64,
feat_dim=8`) is **1,984,821**, matching the published figure --
`tests/test_model.py` asserts this.

## Dataset: PNG

**PNG = Pulmonary Nodule Growth** (the dataset's name, not the image
format -- volumes are NIfTI). 768 images + 768 masks, all 64<sup>3</sup>,
103 patients / 226 nodules, 748 training + 150 test three-timepoint
windows. Not committed to Git -- see `data/README.md` and
`docs/datasets.md` (which documents the manifest's `.nii.gz`-vs-`.nii`
mismatch and the training/test split overlap, both handled explicitly
rather than silently).

## Install

```bash
pip install -e .              # runtime (inference) dependencies
pip install -e ".[train]"     # + monai, for the warmup-cosine LR schedule
pip install -e ".[dev]"       # + pytest
pip install -e ".[notebook]"  # + matplotlib/jupyter, for notebooks/
```

## Quick start

```bash
python scripts/smoke_test.py                                    # no dataset needed
pytest tests -q
python scripts/validate_dataset.py --data-root data/external/png
python scripts/train.py --data-root data/external/png
python scripts/evaluate.py --data-root data/external/png --trained-model best_model.pth
```

To see it visually: `notebooks/01_ngpnet_walkthrough.ipynb` builds the
model, loads a real PNG window, runs a forward pass, and plots the input
slices, predicted image/mask, and deformation field -- see
[`notebooks/README.md`](notebooks/README.md).

```python
from quality_aware_lung_ct.ngpnet import NGPNet
from quality_aware_lung_ct.ngpnet.inference import NGPNetPredictor

model = NGPNet(img_size=64, feat_dim=8)
predictor = NGPNetPredictor(model, device="cpu")
prediction = predictor.predict(
    earlier_scan=im0, later_scan=im1,
    observed_interval=tm0,   # months t0 -> t1, shape (B,)
    target_interval=tm1,     # months t1 -> prediction target, shape (B,)
)
prediction.image    # (B,1,64,64,64) predicted scan
prediction.mask     # (B,1,64,64,64) hard nodule mask
prediction.logits   # (B,2,64,64,64) mask logits -- the only probabilistic output
prediction.field    # (B,3,64,64,64) deformation field
```

## Layout

```
src/quality_aware_lung_ct/
    ngpnet/          model, losses, config, inference
    data/png/        manifest, splits, transforms, dataset
    preprocessing/   volume I/O, cropping, CT windowing
    training/        optimizer/loss construction, epoch loop, checkpoints
    evaluation/      metrics + test-time evaluator
    quality/ uncertainty/ fusion/ reporting/   planned -- boundary packages only
    common/          seeding, run logging
scripts/    train.py  evaluate.py  smoke_test.py  validate_dataset.py
notebooks/  01_ngpnet_walkthrough.ipynb + its _build_*.py generator source
tests/      docs/
data/       external/png/ (gitignored)  processed/ (gitignored)
checkpoints/  outputs/    (gitignored)
```

## Documentation

Full index: [`docs/README.md`](docs/README.md).

| | |
|---|---|
| `docs/architecture.md` | Package layout and exactly what changed vs. the original NGP-Net code |
| `docs/datasets.md` | PNG dataset: measured properties, the manifest defect, split overlap |
| `docs/reproduction.md` | Environment, dataset setup, validation, training, evaluation |
| `docs/research-status.md` | What's implemented vs. planned, and why |
| `docs/upstream/` | The vendored authors' own README and figures, for attribution |
| `notebooks/README.md` | The visual walkthrough notebook and how to regenerate it |
| `data/README.md` | Dataset location, setup, and preprocessing policy |

## License

See `LICENSE`. This project's own code is MIT. The reimplemented NGP-Net
architecture (`src/quality_aware_lung_ct/ngpnet/`) is migrated from an
upstream repository that publishes no license of its own -- see `LICENSE`
for what that means before any redistribution. The PNG dataset carries its
own terms and is never committed here.

## Citation

```bibtex
@article{tang2026ngpnet,
  author={Tang, Xinkai and Luo, Zhiyao and Liu, Feng and Huang, Wencai and Zou, Jiani},
  journal={IEEE Transactions on Medical Imaging},
  title={NGP-Net: a Lightweight Growth Prediction Network for Pulmonary Nodules},
  year={2026}, volume={45}, number={5}, pages={2468-2480},
  doi={10.1109/TMI.2026.3656184}
}
```
