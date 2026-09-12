# weights/

Local-only storage for pretrained backbone weights. Nothing under this
directory is committed to git (`*.h5`, `*.pt`, `*.pth`, etc. are all
ignored — see `.gitignore`) because these files are large binaries best
fetched from their original source rather than duplicated in the repo.

## weights/pretrained/RadImageNet-ResNet50_notop.h5

RadImageNet-pretrained ResNet50 backbone, "notop" (no classification head),
in Keras/TensorFlow `.h5` format as distributed by the RadImageNet authors
(https://github.com/BMEII-AI/RadImageNet, access requested via their form).

This repo's model loader (`ohashi_resnet50.load_radimagenet_weights`)
expects a **PyTorch** state-dict (`.pt`/`.pth`), not the raw Keras `.h5`.
Convert it first, then point `Config.radimagenet_weights_path` (see
`src/ct_iqa/config.py`) at the converted `.pt` file.
