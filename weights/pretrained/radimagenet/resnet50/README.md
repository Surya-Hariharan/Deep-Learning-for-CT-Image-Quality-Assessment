# weights/pretrained/radimagenet/resnet50/

Local-only storage for the RadImageNet-pretrained ResNet50 backbone.
Nothing under `weights/` is committed to git (`*.h5`, `*.pt`, `*.pth`, etc.
are all ignored -- see `.gitignore`) because these files are large binaries
best fetched/converted locally rather than duplicated in the repo.
`weights/pretrained/` holds only externally-sourced pretrained weights --
never experiment-generated checkpoints, which live under
`experiments/<NNN_name>/checkpoint/` instead (see
`docs/internal/decisions/README.md`).

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

**UPDATED 2026-09-13, by the scientific pipeline audit.** Both files listed
above are now present locally and have been verified as follows:

- `OhashiResNet50.load_radimagenet_weights(...)` against
  `radimagenet_resnet50_backbone.pt` reports `loaded=True`, **265 matched
  keys**, **0 unexpected keys**, and 53 "missing" keys -- all 53 of which
  are `num_batches_tracked` counters (non-learned inference bookkeeping,
  never present in a converted checkpoint by design, not a sign of an
  incomplete load). 265 + 53 = 318 = the full key count of
  `ResNet50Backbone.state_dict()`.
- A real, preprocessed LDCT-IQAC batch (`[B, 1, 224, 224]`, internally
  replicated to `[B, 3, 224, 224]`) passes through the RadImageNet-loaded
  model and produces a finite, `[0, 1]`-bounded Sigmoid output.
- `tests/integration/test_radimagenet_weights_integration.py` (conversion +
  loading) and `tests/integration/test_ldct_iqac_pipeline_integration.py::test_configured_radimagenet_path_loads_and_matches_backbone`
  exercise this end to end and pass on this machine.

`configs/model.yaml`'s `radimagenet_weights_path` has accordingly been
updated to point at `radimagenet_resnet50_backbone.pt` (previously `null`).
Because `weights/pretrained/` is gitignored, this path will not resolve on
a machine that hasn't obtained/converted the file -- `torch.load` then
raises `FileNotFoundError` (never a silent fallback to ImageNet weights or
to random initialization; see `docs/replication/deviations.md`).

**What is still NOT verified:** this project has not independently
re-verified the RadImageNet authors' own training procedure or the
resulting features' quality (see "Source" above), and **no full training
run using these weights has been executed and persisted in this repository
as of this audit** -- weight *loading* is verified; a trained *result*
using them is not. Do not treat any number currently reachable from this
repository's code as a finished RadImageNet-initialized replication result
until an experiment's `config.json` records this weights path AND a
completed training run's checkpoint/metrics exist under
`experiments/<NNN_name>/`.
