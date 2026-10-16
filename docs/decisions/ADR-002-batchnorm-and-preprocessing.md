# ADR-002: BatchNorm epsilon and input preprocessing

Status: accepted for the study-v2 framework. Date: 2026-10-16.

## Context

The Stage 0 audit found that `OhashiResNet50.load_radimagenet_weights` set every BatchNorm `eps` to 1e-3 as a
side effect of loading weights, on the stated belief that 1e-3 is "Keras's default". Because `eps` is not part of a
`state_dict`, any model rebuilt for evaluation without calling that method ran at PyTorch's default 1e-5. The
recorded history is therefore: training and validation at 1e-3, test evaluation at 1e-5. The audit also found that
the `2x - 1` pixel mapping was attributed to RadImageNet's reference code without any evidence stored in the repo.

## Evidence (collected 2026-10-16)

| # | Source | Observation |
|---|---|---|
| E1 | The weights file itself, `RadImageNet-ResNet50_notop.h5` (SHA-256 `1f1d1c9f...d184f159`), `model_config` attribute, read with h5py | All **53** `BatchNormalization` layers have `epsilon = 1.001e-05`, `momentum = 0.99`. File attrs: `keras_version 2.4.0`, `backend tensorflow`. Input shape `[None, 224, 224, 3]`. Stem uses `ZeroPadding2D((3,3))` + 7x7 valid conv; `ZeroPadding2D((1,1))` + 3x3 valid max-pool. |
| E2 | Keras source, [keras/src/applications/resnet.py](https://raw.githubusercontent.com/keras-team/keras/master/keras/src/applications/resnet.py) | Every `BatchNormalization` call in the ResNet builder uses `epsilon=1.001e-5`. (Keras's bare-layer default of 1e-3 is not used.) |
| E3 | RadImageNet repository, [github.com/BMEII-AI/RadImageNet](https://github.com/BMEII-AI/RadImageNet) README | Lists ResNet50, DenseNet121, InceptionResNetV2, InceptionV3 for TensorFlow and PyTorch. States nothing about preprocessing, mean/std, or how the models were built. |
| E4 | RadImageNet [pytorch_example.ipynb](https://raw.githubusercontent.com/BMEII-AI/RadImageNet/main/pytorch_example.ipynb) | For the **PyTorch** release (`resnet50_torch.pt`, built on torchvision `resnet50`): `image = cv2.imread(path); image = (image-127.5)*2 / 255; image = cv2.resize(image,(224,224))`; no mean/std normalization. |
| E5 | Third-party mirror, Hugging Face `Lab-Rasool/RadImageNet` (found by web search; not fetched or verified here) | Reported to recommend ImageNet mean/std. This conflicts with E4 and is **not** relied on. |

## Decisions

1. **Canonical `bn_eps = 1.001e-5`** for all study-v2 models. Basis: E1 (the weights file's own recorded
   configuration) corroborated by E2. This is a measured property of the file, not an assumption. Caveat: the
   file records the epsilon of the *exported* Keras model; RadImageNet's training code was not found, so we infer
   that this is the epsilon the running statistics were accumulated under.
2. **Epsilon is a construction-time property.** `ResNet50Backbone(bn_eps=...)` applies it to all 53 BatchNorm
   layers when the module is built; `ModelConfig.bn_eps` has no default and must be written in the YAML;
   `ct_iqa.models.weights.load_backbone_weights` never touches it. Training, validation, checkpoint reload and
   evaluation build the model from the same config, so they agree by construction (unit-tested in
   `tests/unit/test_registry_and_bn_eps.py`).
3. **Input range `[-1, 1]` via `2x - 1` is retained** for pixels stored in `[0, 1]`. Basis: E4's formula
   `(p - 127.5) * 2 / 255` equals `2(p/255) - 1`, i.e. exactly `2x - 1` for `x = p / 255`.
   Status: **partially verified**. E4 documents it for the PyTorch release; no source states the convention for the
   TensorFlow `.h5` file used here, and the two releases may differ. It is adopted as the best-supported option
   and flagged as an open assumption in `docs/PREPROCESSING_AND_WEIGHTS.md`.
4. **Center crop (not resize) is retained**, as in the historical pipeline. E4 resizes to 224x224; the Ohashi
   method crops. This is a documented deviation from RadImageNet's example, chosen to keep native resolution.
5. **Historical behavior is preserved, not rewritten.** `ct_iqa.models.ohashi_resnet50.OhashiResNet50` and
   `ct_iqa.config.legacy.ExperimentConfig` are unchanged, so experiments 001-003 remain reproducible exactly as
   run (training eps 1e-3 after weight load, evaluation eps 1e-5). Historical checkpoints are not modified.

## Consequences

- Experiment A uses `bn_eps = 1.001e-5` and therefore **differs from historical 001** (1e-3 train / 1e-5 eval).
  The difference is recorded in `docs/experiments/EXPERIMENT_A_REFERENCE.md`.
- The effect of the historical mismatch on 001-003 predictions is **unquantified**: no inference was run.
  Historical metrics must not be re-labelled as produced at a single epsilon.
- Applying a checkpoint trained at a different epsilon with `bn_eps = 1.001e-5` (e.g. loading 001 into the new
  model) is a different numerical function from what was evaluated. Any future use must say so.

## Open items

- Confirm the TensorFlow `.h5` input convention (E4 covers the PyTorch release only).
- Whether channel order (BGR in E4 via `cv2.imread`) matters: not for this project, because the grayscale image is
  replicated identically into all three channels.
