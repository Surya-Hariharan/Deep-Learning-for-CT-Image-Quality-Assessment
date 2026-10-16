# Preprocessing and pretrained weights

Decision record: [ADR-002](decisions/ADR-002-batchnorm-and-preprocessing.md). Code: `src/ct_iqa/preprocessing/`,
`src/ct_iqa/models/{backbone,weights,radimagenet_weights}.py`.

## Input pipeline (per image)

| Step | Behavior | Code | Status |
|---|---|---|---|
| Read | TIFF, PIL mode `F` (32-bit float), 512x512, values in [0, 1]; other modes raise | `data/ldct_iqac.py` | observed on disk (audit); mode enforced by the loader |
| Crop | center crop to 224x224, never a resize; error if smaller | `preprocessing/crop.py` | implemented; deviates from RadImageNet's example (resize) |
| Intensity | `2x - 1`, mapping [0, 1] to [-1, 1] | `preprocessing/normalize.py` | partially verified (ADR-002, decision 3) |
| Tensor | float32, shape `(1, 224, 224)` per image | `data/ldct_iqac.py` | tested |
| Channels | the model repeats the single channel to 3 inside `forward` (`x.repeat(1, 3, 1, 1)`) | `models/registry.py` | tested: 1-channel input equals explicit replication |
| Score | raw [0, 4] -> `(y - 0) / 4` in [0, 1] for training; inverse for evaluation | `data/ldct_iqac.py` | tested |

## Backbone

Custom ResNet50 (`ResNet50Backbone`), not torchvision. Stride-2 is on the first 1x1 convolution of each stage
("v1" placement), matching Keras and the RadImageNet `.h5` file. 53 BatchNorm layers; `bn_eps` is set at
construction. Evidence that the file matches this layout: layer names `convX_blockY_Z_{conv,bn}`, zero-padding then
valid convolution/pooling (ADR-002, E1). Zero-padded max-pool equals PyTorch's padded max-pool here because the
inputs to it are post-ReLU (non-negative).

## Pretrained weights

| File | SHA-256 | Role |
|---|---|---|
| `weights/pretrained/radimagenet/resnet50/RadImageNet-ResNet50_notop.h5` | `1f1d1c9fe6ab6980daa5e528145fe0db249c55415f8b3e1f3e3b1624d184f159` | source, as obtained from the RadImageNet authors |
| `weights/pretrained/radimagenet/resnet50/radimagenet_resnet50_backbone.pt` | `45cba0d53f1d0ec1e1e2ae569a985b56a21da71c4a7271328b01ed5ea9a0f785` | converted state-dict used by all experiments |

Checksums are in `weights/pretrained/radimagenet/resnet50/CHECKSUMS.sha256`; `load_backbone_weights` verifies
the converted file against `model.pretrained_sha256` and refuses to load on a mismatch. Conversion
(`python -m ct_iqa.models.radimagenet_weights`): HWIO -> OIHW kernel transpose and folding each convolution bias into
the following BatchNorm `running_mean` (`mean' = mean - bias`). The fold is exact for a bias-free conv followed by
BatchNorm. Loading reports 265 matched keys, 0 unexpected, 53 `num_batches_tracked` counters left at 0.

The weights themselves are not distributed with this repository. Provenance beyond the file hash (RadImageNet's own
training procedure) has not been independently verified.

## BatchNorm epsilon

Canonical: **1.001e-5** (all 53 layers of the weights file's recorded configuration; Keras source agrees).
Historical experiments 001-003 trained and validated at 1e-3 and evaluated on the test set at 1e-5; the
numerical effect is **unquantified** (no inference was run for this study). New models take `bn_eps` from the
config and keep it for the whole lifecycle.

## Unresolved

- TensorFlow-release input convention not stated by any source found (only the PyTorch release is documented).
- Training-time epsilon of RadImageNet's own training is inferred from the exported model, not observed.
- Effect of the historical eps mismatch on experiments 001-003 predictions: not measured.
