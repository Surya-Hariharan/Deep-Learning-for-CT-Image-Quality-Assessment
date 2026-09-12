# weights/pretrained/radimagenet/resnet50/

Local-only storage for the RadImageNet-pretrained ResNet50 backbone.
Nothing under `weights/` is committed to git (`*.h5`, `*.pt`, `*.pth`, etc.
are all ignored -- see `.gitignore`) because these files are large binaries
best fetched/converted locally rather than duplicated in the repo.
`weights/pretrained/` holds only externally-sourced pretrained weights --
never experiment-generated checkpoints, which live under
`experiments/<NNN_name>/checkpoint/` instead (see
`docs/decisions/README.md`).

## Source

RadImageNet-pretrained ResNet50 backbone, "notop" (no classification head),
distributed by the RadImageNet authors
(https://github.com/BMEII-AI/RadImageNet, access requested via their form).
This project has not independently re-verified the authors' training
procedure -- only that the released weight file's architecture matches
`keras.applications.ResNet50` (53 Conv2D + 53 BatchNormalization layers,
standard `convX_blockY_Z_{conv,bn}` naming, no classification head).

## Expected files

| File | Format | Produced by |
|---|---|---|
| `RadImageNet-ResNet50_notop.h5` | Keras/TensorFlow HDF5, as released by RadImageNet | Downloaded from the source above |
| `radimagenet_resnet50_backbone.pt` | PyTorch state-dict, keyed to match `ct_iqa.models.resnet50.ResNet50Backbone.state_dict()` | `python -m ct_iqa.models.radimagenet_weights` |

## Conversion process

This repo's model loader (`OhashiResNet50.load_radimagenet_weights`, in
`src/ct_iqa/models/ohashi_resnet50.py`) expects a **PyTorch** state-dict
(`.pt`), not the raw Keras `.h5`. Convert the `.h5` first:

```bash
python -m ct_iqa.models.radimagenet_weights \
    --input weights/pretrained/radimagenet/resnet50/RadImageNet-ResNet50_notop.h5 \
    --output weights/pretrained/radimagenet/resnet50/radimagenet_resnet50_backbone.pt
```

The conversion (kernel transpose HWIO->OIHW, conv-bias-into-BatchNorm
folding, and why `ResNet50Backbone` places its stride-2 convs on the 1x1
reduce conv rather than the 3x3 conv to match Keras's layout) is documented
in `ct_iqa.models.radimagenet_weights`'s module docstring. The conversion
mathematics are an exact re-parameterization, not an approximation.

Then point `configs/model.yaml`'s `radimagenet_weights_path` (or
`ExperimentConfig.radimagenet_weights_path` directly, see
`src/ct_iqa/config.py`) at the converted `.pt` file.

## Provenance and verification status

**As of this migration, RadImageNet weights have not been confirmed present
and loaded in any real training run in this repository.** The baseline
notebook and `ExperimentConfig` default `radimagenet_weights_path` to
`null`/`None`, meaning the ResNet50 backbone runs with random initialization
until a real, converted `.pt` file is obtained and explicitly pointed to.
Do not treat any existing result in this repository as evidence of a
successful RadImageNet initialization unless the corresponding experiment's
`config.json`/`config.yaml` records a non-null `radimagenet_weights_path`
AND `OhashiResNet50.load_radimagenet_weights`'s returned `WeightLoadReport`
for that run reports `loaded=True` with a non-zero matched-key count. See
`docs/replication/deviations.md` for how this status is reported in
experiment write-ups.
