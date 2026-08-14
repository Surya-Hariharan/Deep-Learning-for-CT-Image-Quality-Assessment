# RadImageNet Checkpoint & Isolated Training Environment

Provisioning record for the isolated TensorFlow environment and the
RadImageNet ResNet50 checkpoint, performed 2026-08-14, following
`docs/radimagenet_environment_audit.md` (feasibility audit) and
`docs/research_decisions.md` "Baseline Model Specification Decision"
(model spec resolution). This document records what was provisioned and
verified — **training was not started, the production dataset was not
generated, and the committed model architecture/dropout/VIF/degradation
parameters were not changed.**

---

## 1. Python version

**Isolated environment**: Python **3.10.20** (CPython, via `uv`-managed
toolchain, `C:\Users\surya\AppData\Roaming\uv\python\cpython-3.10.20-windows-x86_64-none\python.exe`).
Chosen because it is the newest Python in the ≤3.10 line already available
on this machine (`py -0` lists it as `Astral/CPython3.10.20`), closest to
Ohashi's reported Python 3.10.13.

**This project's main/test environment remains untouched**: Python 3.13.14
(`py -3.13`, Microsoft Store CPython) — nothing about it was modified by
this provisioning. The two environments are structurally independent (see
§7).

## 2. TensorFlow version

**Installed: `tensorflow==2.10.1`, NOT `2.10.10`.**

`requirements.txt` and multiple docs in this project (inherited from the
project brief) record Ohashi's reported version as "TensorFlow 2.10.10."
**This version does not exist.** Checked directly against PyPI's release
metadata during this session: the entire `2.10.x` line only ever shipped
`2.10.0` and `2.10.1` — no `2.10.10` release exists or ever existed.
`2.10.10` is almost certainly a transcription error somewhere upstream of
this project (in the project brief, or in whatever source it was copied
from) — plausibly a `2.10.1` + stray trailing `0`, but this is a guess, not
a confirmed correction of the original paper.

**This was not silently resolved.** The discrepancy was surfaced to the
user explicitly before installing anything; `tensorflow==2.10.1` (the
closest real release in the same minor line) was selected on the user's
explicit direction, not assumed. `docs/research_decisions.md` should be
updated with this finding as a formal decision the next time methodology
docs are revisited (not done as part of this provisioning task, which is
scoped to environment/checkpoint work only) — flagged here so it is not
lost.

`keras==2.10.0` was installed as TensorFlow 2.10.1's paired Keras
dependency (TF/Keras were still coupled at this release; standalone Keras
did not yet exist as a separate versioned package for this TF line).

## 3. CUDA / cuDNN

**Not installed, anywhere on this machine.** No CUDA Toolkit directory
exists (`C:\Program Files\NVIDIA GPU Computing Toolkit` does not exist), and
none of the specific DLLs TensorFlow 2.10.1 looks for at import/GPU-init
time are present on `PATH`:

```
cudart64_110.dll    (CUDA 11.x runtime)
cublas64_11.dll
cublasLt64_11.dll
cufft64_10.dll
curand64_10.dll
cusolver64_11.dll
cusparse64_11.dll
cudnn64_8.dll        (cuDNN 8.x)
```

TensorFlow 2.10.1 is compiled with CUDA support
(`tf.test.is_built_with_cuda() == True`) and, at runtime, explicitly logs
each of the above DLLs as not found before falling back to CPU-only
execution — this is TensorFlow's own graceful degradation path, not a
crash. **Only the NVIDIA display driver is installed; the separate CUDA
Toolkit (which ships the runtime DLLs above) and cuDNN (a separate
NVIDIA-distributed redistributable, requiring its own account-gated
download) are not.**

## 4. GPU

**Physically present, driver-visible, but NOT visible to this TensorFlow
installation.**

```
$ nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv
name, memory.total [MiB], driver_version, compute_cap
NVIDIA GeForce RTX 4060 Laptop GPU, 8188 MiB, 610.74, 8.9
```

- GPU: NVIDIA GeForce RTX 4060 Laptop GPU, 8,188 MiB (8 GB) VRAM.
- Driver version: 610.74. `nvidia-smi`'s own header additionally reports
  `CUDA UMD Version: 13.3` — the *maximum* CUDA API version this driver
  can service, not a statement that CUDA 13.3 (or any CUDA Toolkit) is
  actually installed; it is not (§3).
- Compute capability: 8.9 (Ada Lovelace) — comfortably new enough for any
  CUDA 11.x-era software; compute capability is not the blocker here.

```
$ python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
GPU devices: []
All physical devices: [PhysicalDevice(name='/physical_device:CPU:0', device_type='CPU')]
```

**Result: `tf.config.list_physical_devices('GPU')` returns an empty list.**
Only `CPU:0` is enumerated. This exactly confirms the prediction in
`docs/radimagenet_environment_audit.md` §6.

## 5. RadImageNet checkpoint

| Field | Value |
| --- | --- |
| Exact filename | `RadImageNet-ResNet50_notop.h5` (as extracted; the official reference training script `thyroid/thyroid_train.py` in the source repo names it `RadImageNet-ResNet50-notop.h5` — hyphen vs. underscore is the only difference, same file) |
| Source | Official `BMEII-AI/RadImageNet` GitHub repo's own linked Google Drive release: `RadImageNet_models-20230414T114049Z-001.zip`, from `https://drive.google.com/file/d/1UgYviv2K6QPM1SCexqqab5-yTgwoAFEc` (one of the two TensorFlow-weights links published directly in that repo's README) |
| Zip size (downloaded) | 1,821,860,458 bytes (≈1.7 GiB, matches Google Drive's own reported "(1.7G)") — bundles all four RadImageNet TensorFlow architectures (ResNet50, DenseNet121, InceptionV3, InceptionResNetV2) plus an unrelated 1,537,082,265-byte GAN generator file (`radimagegan_64x64_generator.pkl`), not needed here |
| Extracted file size | 94,852,768 bytes (≈90.5 MiB) |
| Format | Keras HDF5 (`.h5`); magic bytes verified: `89 48 44 46 0D 0A 1A 0A` (HDF5 signature) |
| Framework | TensorFlow/Keras (loaded via `tensorflow.keras.applications.ResNet50(weights=<path>, ...)`) |
| Download date | 2026-08-14 |
| Checksum | No official checksum is published anywhere by the source (confirmed in `docs/radimagenet_environment_audit.md` §3). **Locally computed SHA-256** (this is the only verification available — it does not independently prove authenticity against a canonical source-published value, since none exists): |

```
SHA256 (RadImageNet-ResNet50_notop.h5) =
1f1d1c9fe6ab6980daa5e528145fe0db249c55415f8b3e1f3e3b1624d184f159
```

**Local path**: `data/external/radimagenet/RadImageNet-ResNet50_notop.h5`
(project-relative). **Confirmed Git-ignored** — matched by the existing
`*.h5` rule in `.gitignore` (`git check-ignore -v` → `.gitignore:63:*.h5`).
Never appears in `git status` or `git ls-files`.

**Acquisition note**: an initial attempt supplied
`C:\Users\surya\OneDrive\Desktop\RadImageNet-main` — this was found to be
the GitHub *source-code* repository (downloaded via "Download ZIP"), not
the weights; it contains no `.h5` file anywhere (verified by exhaustive
search). This was reported back rather than silently treated as the
checkpoint. The actual weights were then fetched from the official Google
Drive zip link above (user-approved automated download), and only the
`RadImageNet-ResNet50_notop.h5` member was extracted and kept; the zip and
all other extracted members (including the unrelated GAN file) were
deleted after extraction, not retained anywhere on disk.

## 6. Environment creation procedure

```bash
# 1. Isolated Python 3.10 interpreter (uv-managed, already present on this machine)
py -0   # confirms Astral/CPython3.10.20 is available

# 2. Isolated venv, OUTSIDE the project repository (never git-trackable by construction)
mkdir -p "C:/Users/surya/.venvs"
uv venv "C:/Users/surya/.venvs/ct-iqa-tf210" --python 3.10.20

# 3. TensorFlow 2.10.1 (see §2 for why not 2.10.10) + h5py
uv pip install --python "C:/Users/surya/.venvs/ct-iqa-tf210/Scripts/python.exe" \
    "tensorflow==2.10.1" "h5py"

# 4. numpy ABI fix -- REQUIRED, not optional (see §8 "compatibility limitations")
uv pip install --python "C:/Users/surya/.venvs/ct-iqa-tf210/Scripts/python.exe" "numpy==1.23.5"
```

**Location**: `C:\Users\surya\.venvs\ct-iqa-tf210` — a machine-local path
outside the repository, analogous to `LDCT_IQA_DATA_ROOT`'s convention of
keeping large/machine-specific artifacts out of the versioned tree. Not
recorded in any config file (would be a machine-specific path, which this
project's own convention forbids — see `src/ct_iqa/utils/paths.py`).
Anyone reproducing this environment on another machine should follow the
procedure above, adjusting only the venv location.

## 7. Isolation guarantee

- The isolated venv lives entirely outside this repository's directory
  tree (`git check-ignore` on its path returns `fatal: ... outside
  repository` — structurally, not just by `.gitignore` pattern, it cannot
  ever be tracked).
- Nothing was installed into, or changed about, the project's main Python
  3.13 environment (`py -3.13`) — verified: `py -3.13 -c "import
  tensorflow"` still raises `ModuleNotFoundError` after this provisioning
  (checked in §10).
- `requirements.txt` was not modified by this task (TensorFlow remains
  commented out there, per existing project convention — the isolated
  environment is provisioned and used entirely outside that file).

## 8. Compatibility limitations (the exact problem, and possible isolated solutions)

**GPU is not usable by this TensorFlow installation without further,
out-of-scope system changes.** Exact problem, precisely stated:

- TensorFlow 2.10.1's GPU support requires the **CUDA 11.x Toolkit runtime
  DLLs** and **cuDNN 8.1** specifically — not whatever CUDA version the
  installed NVIDIA driver happens to report supporting. The driver
  (610.74, reporting `CUDA UMD Version: 13.3`) is new enough to run CUDA
  11.x software (NVIDIA drivers are backward-compatible), but the actual
  CUDA 11.x/cuDNN 8.1 **library files** are a separate, additional install
  — driver presence alone does not provide them.
- TensorFlow 2.10 is also the **last** TensorFlow release with native
  Windows GPU support at all; TensorFlow 2.11+ dropped it, requiring WSL2
  instead. This makes 2.10.1 simultaneously the *correct* choice for
  matching Ohashi's environment and *particularly* fragile to get GPU
  support working on Windows, since there is no newer, better-supported
  path within the same native-Windows constraint.

**Possible isolated solutions (documented for a future decision, NOT
performed by this task — installing a system-wide CUDA Toolkit is exactly
the kind of "modify the machine beyond an isolated venv" action this task's
instructions said to stop short of, not silently do):**

1. **Install CUDA Toolkit 11.2 and cuDNN 8.1 manually** (NVIDIA developer
   account required for cuDNN), add their `bin` directories to `PATH`
   (or to the venv's activation script only, to keep the change
   session-scoped rather than machine-wide). This is the standard,
   best-documented path for TF 2.10 GPU support on Windows.
2. **Investigate pip-installable CUDA runtime wheels**
   (`nvidia-cudnn-cu11`, `nvidia-cublas-cu11`, etc.) as a venv-local
   alternative to a system installer — not attempted or verified in this
   session; Windows wheel availability and TF 2.10's specific DLL-name
   `dlopen` mechanism (rather than a Python-import-based loader) make this
   uncertain to work without testing, not a known-good path.
3. **Accept CPU-only training.** Slower, but requires no further
   environment changes; the smoke tests in this document (§9) confirm CPU
   execution works correctly today, right now, with zero additional setup.

**No TensorFlow version was silently upgraded to work around this** —
2.10.1 was kept throughout, per the task's explicit instruction.

## 9. Model-loading smoke-test result

All performed in the isolated environment (§6); **no training, no
fine-tuning, no compiled/saved model** — see full transcript in this
session's tool output, summarized here:

**GPU/tensor smoke test:**
```
Built with CUDA: True
GPU devices: []
All physical devices: [PhysicalDevice(name='/physical_device:CPU:0', device_type='CPU')]

tensor op result (matmul):
[[19. 22.]
 [43. 50.]]
device placement: /job:localhost/replica:0/task:0/device:CPU:0
```
Correct result, confirming TensorFlow itself works end-to-end on CPU.

**RadImageNet checkpoint loading smoke test:**
```python
base_model = tf.keras.applications.ResNet50(
    weights="data/external/radimagenet/RadImageNet-ResNet50_notop.h5",
    input_shape=(224, 224, 3), include_top=False, pooling="avg",
)
```
```
Model loaded successfully.
Input shape: (None, 224, 224, 3)
Output shape: (None, 2048)
Number of layers: 176
Total params: 23587712
First layer: input_1 InputLayer
Last layer: avg_pool GlobalAveragePooling2D
layer 'conv1_conv' present: True
layer 'conv2_block1_1_conv' present: True
layer 'conv5_block3_out' present: True
```
The weights loaded into the architecture without any shape-mismatch error
(Keras raises immediately on a shape mismatch between stored weights and
the instantiated architecture — silence here is itself a positive
correctness signal, not just an absence of complaint). 23,587,712 params
matches the standard ResNet50-no-top parameter count exactly. Known
backbone layer names (`conv1_conv`, `conv2_block1_1_conv`,
`conv5_block3_out`) are present, confirming this is genuinely a ResNet50
architecture, not a substituted or malformed one.

**Model-specification cross-check (shape-verification only — the resulting
probe model was immediately discarded: not compiled, not trained, not
saved, matching decision A-14):**
```
freeze_backbone=False (fine-tuned) consistency check:
  base_model.trainable (Keras default): True

Input shape : (None, 224, 224, 3)   (expect (None, 224, 224, 3))  -- MATCH
Output shape: (None, 1)             (expect (None, 1) sigmoid regression) -- MATCH
Total params: 23589761
Backbone params: 23587712
Head params added: 2049   (= 2048 weights + 1 bias for Dense(1), exactly right)
Forward pass on dummy zeros -> output: [[0.51301116]]  shape: (1, 1)
```
Confirms: input matches `ModelSpec.input_size=224` with 3 channels;
`base_model.trainable` defaults to `True`, consistent with
`ModelSpec.freeze_backbone=False` (S-01, fine-tuned); attaching
`Dropout(0.5) -> Dense(1, activation="sigmoid")` on top of the GAP-pooled
backbone output produces exactly a `(None, 1)` sigmoid output, matching the
head Ohashi specifies and this project's `ModelSpec.head` (decision A-22
for the 0.5 rate). **This is the full extent of verification performed —
no model was compiled, trained, or persisted.**

**Preprocessing pipeline check** (main project environment, Python 3.13,
no TensorFlow involved — pure numpy, matches
`docs/radimagenet_environment_audit.md` §8's earlier check, re-run here
for completeness):
```
dummy (LDCT-IQAC-shaped) input: (512, 512)
after central_crop (no resize): (224, 224)
after grayscale_to_rgb (channel replication): (224, 224, 3) uint8
matches (224,224,3): True
```

- **Central crop, no resize**: confirmed — `central_crop()` never resizes;
  it raises if the source is smaller than the crop size (never happens for
  512×512 LDCT-IQAC images) and otherwise crops directly. This matches
  Ohashi's explicit statement that images are "cropped to the central
  region according to the input size," with no resize step mentioned.
- **Grayscale/RGB conversion**: channel replication
  (`grayscale_to_rgb_method="channel_replication"`) — PROJECT-ADAPTATION,
  unchanged by this task, since LDCT-IQAC's own PNGs already store 3
  identical channels natively.
- **Normalization**: still genuinely **UNKNOWN / NOT SPECIFIED BY OHASHI**
  (`intensity_normalization: null`,
  `configs/quality/resnet50_vif.yaml:preprocessing`). Not invented by this
  task. One additional, non-authoritative data point surfaced while
  reading the RadImageNet reference training script
  (`thyroid/thyroid_train.py`, quoted in
  `docs/radimagenet_environment_audit.md`): it uses
  `ImageDataGenerator(rescale=1./255, preprocessing_function=preprocess_input)`
  — i.e. a rescale to `[0,1]` combined with `tensorflow.keras.applications.resnet50.preprocess_input`
  (which itself does ImageNet-style BGR reordering and per-channel mean
  subtraction). This is offered as a documented observation about *one*
  downstream RadImageNet application's own code, not evidence about what
  Ohashi did, and is **not** adopted here.

## 10. Confirmation: main environment untouched

```
$ py -3.13 -c "import tensorflow"
ModuleNotFoundError: No module named 'tensorflow'
```
Re-checked after all of the above — the project's Python 3.13 test/dev
environment has nothing installed into it by this provisioning.

---

## Summary

| Item | Status |
| --- | --- |
| RadImageNet checkpoint | Obtained (official source), verified (HDF5-valid, loads into `ResNet50` with zero shape errors, SHA-256 recorded), Git-ignored |
| Isolated training environment | Created (Python 3.10.20, TensorFlow 2.10.1 — not 2.10.10, which does not exist), fully separate from the main Python 3.13 environment |
| GPU | Physically present (RTX 4060, 8GB), driver-visible, **not usable by this TensorFlow install without a separate CUDA 11.2/cuDNN 8.1 install (§8, not performed)** |
| Checkpoint load / architecture cross-check | PASS (§9) |
| Model training | **NOT STARTED** |
| Production dataset generation | **NOT PERFORMED** |
| Committed model architecture / dropout / VIF / degradation parameters | **UNCHANGED** |
