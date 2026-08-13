# Ohashi Method Specification

What Ohashi et al. (2025), *"Development of a No-Reference CT Image Quality
Assessment Method Using RadImageNet Pre-trained Deep Learning Models,"*
Journal of Imaging Informatics in Medicine, specify — as summarized in the
project brief. This document is the PRIMARY METHODOLOGICAL SOURCE reference
for this project; it does not itself introduce any dataset substitution (see
`dataset_adaptation.md` for that).

Every item below is OHASHI-SPECIFIED unless marked otherwise. Items the brief
does not pin down are marked NOT SPECIFIED BY OHASHI — do not treat their
absence here as permission to invent a value.

## Reference images (original)

- 105 total: 90 from DeepLesion, 15 from CQ500.
- NOT SPECIFIED BY OHASHI: exact selection rule, preprocessing before
  degradation.

## Synthetic degradation

- Gaussian noise, sigma ∈ {1, 1.5, 2, 3, 4.5, 6, 7.5, 9, 14, 21, 30, 50} (12 levels).
- Gaussian blur, sigma ∈ {0.1, 0.15, 0.2, 0.3, 0.45, 0.6, 0.75, 0.9, 1.4, 2.1, 3.0, 5.0} (12 levels).
- Three condition families: noise-only, blur-only, noise+blur combinations.
- No other degradation type in the baseline.
- NOT SPECIFIED BY OHASHI: sigma units (HU vs 8-bit grey levels); combined-
  condition order (blur-then-noise vs noise-then-blur); clipping behaviour.

## Labelling

- VIF (Visual Information Fidelity) of each degraded image against its own
  clean reference is the synthetic-stage training target.
- Formulation: the paper cites Sheikh & Bovik (2006), "Image information and
  visual quality," IEEE Trans Image Process 15(2):430-444 [ref 27] — the
  original **wavelet-domain VIF** (Gaussian Scale Mixture model over wavelet
  subbands), not the simplified pixel-domain **VIFp** from an earlier (2005)
  Sheikh/Bovik paper. This citation is specific enough to rule out VIFp as
  the intended formulation. See `docs/research_decisions.md` → "VIF
  Implementation Status" for the full mismatch writeup.
- NOT SPECIFIED BY OHASHI: the exact MATLAB function/toolbox used, wavelet
  decomposition levels/subbands, σ_nsq, and other implementation-level
  parameters of their wavelet-domain VIF run.
- Original implementation: MATLAB R2024a.

## Dataset splitting

- Reference-level split, 60% train / 20% val / 20% test.
- Split BEFORE generating degradations; every degraded variant of a reference
  inherits that reference's partition. Never split degraded images directly.

## Model architecture

```
Input -> central crop -> 224x224 -> RadImageNet-pretrained ResNet50
       -> Dropout -> Fully Connected(1) -> Sigmoid -> quality score
```

- First backbone in scope: ResNet50 only (DenseNet121, InceptionV3,
  InceptionResNetV2 deferred).
- Backbone is **fine-tuned**, not frozen: the paper states the RadImageNet
  pre-trained models were adapted "By fine-tuning these pre-trained models
  with our IQA dataset." Confirmed from the source text, not inferred.
- NOT SPECIFIED BY OHASHI: dropout probability; exact grayscale-to-3-channel
  implementation; resize/interpolation before the central crop.
- No augmentation beyond the central crop (no random crop/flip/rotation/color).

## Training

- Optimizer: Adam. Batch size: 64. Epochs: 30. Loss: MSE.
- Learning-rate candidates: {1e-2, 1e-3, 1e-4, 1e-5}, selected by validation MSE.
- Reported best LR for ResNet50: 1e-3 — the replication must still run the
  search, not assume this result.
- NOT SPECIFIED BY OHASHI: Adam betas, LR schedule, early stopping,
  checkpoint selection.

## Calibration

Five-parameter logistic, fitted by nonlinear least squares, applied to raw
predictions before PLCC:

```
Y_L = b1 * (1/2 - 1/(1 + exp(b2*(Y - b3)))) + b4*Y + b5
```

- NOT SPECIFIED BY OHASHI: initial parameter values, bounds.

## Evaluation (three stages)

1. **Synthetic test set**: predicted score vs VIF score. Metrics: MSE, PLCC, SROCC.
2. **Subjective evaluation**: predicted score vs human expert quality score. Metrics: PLCC, SROCC.
3. **Real-image evaluation**: generalisation to real clinical-quality images.

PLCC is computed on calibrated predictions; SROCC is rank-based and reported
on either (calibration does not change it, since it is monotonic).
