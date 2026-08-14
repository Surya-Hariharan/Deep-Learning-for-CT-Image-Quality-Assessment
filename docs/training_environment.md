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
explicit direction, not assumed.

**Update (2026-08-14, follow-up task):** `requirements.txt`,
`docs/radimagenet_environment_audit.md`, and `docs/project_status.md` have
now all been updated to state the verified `2.10.1` install alongside —
not instead of — the historical "2.10.10" claim they each inherited from
the project brief (that claim is preserved as a record of what was
originally stated, not deleted). This document (`docs/training_environment.md`)
remains the single authoritative source for the actual, verified
environment.

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
- **Normalization**: at the time this section was first written, still
  genuinely UNKNOWN / NOT SPECIFIED BY OHASHI. **Resolved 2026-08-14**
  (decision A-23, `docs/research_decisions.md`) — see the next section for
  the full evidence trace and the resulting specification. `configs/quality/resnet50_vif.yaml:preprocessing.intensity_normalization`
  has been updated from `null` accordingly.

## 9a. Normalization evidence trace and decision (U-M05, decision A-23)

Investigated systematically, per the six-source checklist this task was
given, before adopting anything:

| Source | Finding |
| --- | --- |
| **A. Ohashi's paper** | Never obtained (as with every OHASHI-SPECIFIED item in this project) — no statement available. |
| **B. Ohashi supplementary material / source code** | Searched (2026-08-14); none found publicly. |
| **C. RadImageNet's paper** (Mei et al. 2022) | States only "All images were resized to 224×224 pixels and used as the inputs" (own pretraining) and "downscaled to 256×256" (downstream experiments). **No pixel-scaling, mean-subtraction, or normalization scheme stated anywhere.** |
| **D. Official RadImageNet repository** (`BMEII-AI/RadImageNet`) | No base-pretraining script is published — only downstream fine-tuning examples per medical application (`acl/`, `thyroid/`, etc.). |
| **E. RadImageNet downstream training scripts** | `thyroid/thyroid_train.py` and `acl/acl_train.py` (both checked directly) each use `tensorflow.keras.applications.imagenet_utils.preprocess_input` (default `mode="caffe"`) as a Keras `ImageDataGenerator.preprocessing_function`, **combined with an additional `rescale=1./255`.** This combination is internally inconsistent: `preprocess_input(mode="caffe")`'s mean constants (103.939, 116.779, 123.68) are calibrated for `[0,255]`-scale input; applying them after an independent `/255` rescale does not correspond to any standard, principled normalization scheme — most plausibly a copy-paste artifact in the official repo's own example code, present identically in both files checked. |
| **F./G. The checkpoint file itself** | Inspected directly via `h5py` (not inferred from weight values, per instruction) — `RadImageNet-ResNet50_notop.h5`'s embedded `model_config` shows the saved architecture is `InputLayer -> ZeroPadding2D -> Conv2D -> ...` (175 layers total), with **no** `Rescaling`/`Normalization`/`Lambda` preprocessing layer anywhere. The checkpoint provides zero structural evidence resolving this question either way. |

**Decision (A-23): `tensorflow.keras.applications.resnet50.preprocess_input`
applied to raw `[0,255]`-scale pixel values, with NO additional `/255`
rescale** — i.e. adopting the *corrected* form of the one repeated (if
flawed) signal available from (E), rather than either copying its
inconsistency verbatim or discarding the signal entirely.
**Classification: UNKNOWN → PROJECT-ADAPTATION baseline** — explicitly not
OHASHI-SPECIFIED, and not a confirmed statement of RadImageNet's own base
training either (per this task's item 3: RadImageNet's example code using
a preprocessing operation does not prove Ohashi used it, and this decision
does not conflate the two). Not chosen by any hyperparameter search or
validation comparison, matching decision A-22's own scope constraint.

**Implementation**: `src/ct_iqa/preprocessing/normalization.py::resnet50_preprocess_input`
— a pure-numpy replica (no TensorFlow dependency, so it runs in the main
Python 3.13 environment). **Cross-checked bit-exact against the real
`tensorflow.keras.applications.resnet50.preprocess_input`** in the
isolated environment on a full 224×224×3 random array: `max abs diff =
0.0`, `np.array_equal == True`.

## 9b. Preprocessing specification (authoritative)

The complete, single-source baseline preprocessing pipeline —
`src/ct_iqa/preprocessing/pipeline.py::preprocess_for_model` — composing
three independently-provenanced steps:

| Step | Operation | Provenance |
| --- | --- | --- |
| 1. Input image format | 2-D grayscale or 3-D `(H, W, 3)` RGB array, raw `[0,255]`-scale pixel values (uint8 or float), minimum 224×224 (LDCT-IQAC's 512×512 comfortably satisfies this) | source data property, not a choice |
| 2. Central crop | `central_crop(image, size=224)` — crops the exact center 224×224 window; **raises rather than resizing** if the source is smaller than 224×224 (never triggered for LDCT-IQAC) | OHASHI-SPECIFIED (Section 9/10: "cropped to the central region according to the input size," no resize step mentioned) |
| 3. Crop dimensions | 224 × 224 | OHASHI-SPECIFIED |
| 4. Grayscale/RGB conversion | `grayscale_to_rgb` — channel replication (a no-op for LDCT-IQAC, whose PNGs already store 3 identical channels) | PROJECT-ADAPTATION (unchanged by this task) |
| 5. Channel construction | 3 channels, RGB order until step 6 | matches step 4 |
| 6. Pixel value range | Raw `[0,255]` scale is preserved through steps 2–5; only step 7 transforms it (no separate `/255` rescale — decision A-23) | PROJECT-ADAPTATION (A-23) |
| 7. Normalization | `resnet50_preprocess_input` — RGB→BGR channel reorder, then subtract ImageNet per-channel means `[103.939, 116.779, 123.68]` (BGR order) | PROJECT-ADAPTATION (A-23), resolves U-M05 |
| 8. Output tensor shape | `(224, 224, 3)` | derived from steps 2–3 |
| 9. Output dtype | `float32` | matches the checkpoint's own saved `InputLayer` `dtype: "float32"` (confirmed via the `model_config` inspection in §9a) |

No resizing is performed anywhere in this pipeline — central crop is the
Ohashi-specified operation, and it is never combined with, or substituted
by, a resize step.

## 10. GPU feasibility investigation (RTX 4060 + TensorFlow 2.10.1/CUDA 11.2) — investigation only, nothing installed

Follow-up to §4/§8, specifically asked: can the RTX 4060 realistically be
used with the legacy TF 2.10/CUDA 11.2 stack at all, even after installing
the missing CUDA Toolkit/cuDNN? **No CUDA/cuDNN was installed to answer
this — the question is answered from hardware/software compatibility
evidence, not from a live test.**

**The situation, restated precisely:**

- RTX 4060 Laptop GPU: **Ada Lovelace architecture, compute capability
  8.9** (confirmed via `nvidia-smi --query-gpu=compute_cap`, §4). Ada
  Lovelace launched October 2022.
- TensorFlow 2.10 (and the CUDA 11.2/cuDNN 8.1 combination it targets)
  released September 2022 — essentially simultaneously with Ada Lovelace,
  meaning Google's TF 2.10 build pipeline had no realistic window to add
  `sm_89`/`compute_89` to its official wheel's target compute-capability
  list. TensorFlow's official prebuilt PyPI wheels are compiled once
  against a **fixed list of target compute capabilities** baked in at
  build time (not detected dynamically) — a GPU whose compute capability
  isn't in that list is not necessarily unusable (CUDA's PTX forward-
  compatibility mechanism can sometimes JIT-compile for a newer
  architecture from an older virtual-architecture PTX target), but this is
  markedly less reliable than a wheel built with native support.
- Community-sourced, practically-oriented evidence (checked 2026-08-14):
  guides covering Ada Lovelace (RTX 40-series) + TensorFlow consistently
  recommend **TensorFlow ≥2.13 with CUDA 11.8**, not TensorFlow
  2.10/CUDA 11.2 — one representative guide explicitly states deep
  learning libraries on this hardware should use "CUDA Toolkit up to
  11.8," treating 11.8 as closer to a practical *minimum* than an
  arbitrary upper bound, and separately warns that CUDA 11.8 itself needs
  driver 525+ (this machine's driver, 610.74, comfortably exceeds that —
  driver version is not the limiting factor here, see §4).
- **This project deliberately uses TensorFlow 2.10.1, not 2.13+, to stay
  close to Ohashi's reported environment** (§2) — the same reasoning that
  motivated the isolated Python 3.10 environment in the first place. This
  creates a direct tension: the TensorFlow release that best matches
  Ohashi's reported environment is not the release the wider community has
  found reliable on this specific (newer-than-the-software) GPU
  generation.

**Conclusion: native-Windows GPU use of this specific TF 2.10.1/CUDA
11.2/cuDNN 8.1 stack on an RTX 4060 carries real, non-trivial risk of
failing at runtime even after the CUDA Toolkit/cuDNN are correctly
installed** (the classic symptom would be a "no kernel image available for
this device" / "CUDA error: no binary for GPU architecture" failure at the
first GPU op, not an import-time failure) — this is a evidenced concern
raised by this investigation, not a confirmed failure, since the
toolkit/driver combination was never actually installed and tested (out of
scope for this task).

**Is WSL2 a better option?** Investigated as instructed, with an important
caveat: **WSL2 does not resolve this specific problem.** WSL2's usual
value proposition is restoring GPU support for **TensorFlow 2.11+**, which
dropped *native Windows* GPU builds entirely (§8) — but this project is
deliberately staying on 2.10.1, which still has native Windows GPU support
in principle. Running the *same* TensorFlow 2.10.1 wheel inside WSL2
would not change its baked-in compute-capability list; the Ada Lovelace
compatibility question is a property of the TensorFlow **wheel build**,
not of Windows vs. Linux as a host OS. WSL2 would only plausibly help if
combined with a **newer** TensorFlow release (2.13+, whose wheels do
target Ada-era compute capabilities) — which reopens the same
Ohashi-environment-fidelity trade-off this section already identifies, not
a clean solution.

**Most defensible training environment, given this investigation:**
CPU-only, using the already-provisioned and already-verified
`tensorflow==2.10.1` in the isolated Python 3.10.20 environment (§9). This
is the only combination confirmed working end-to-end today, with zero
additional installation risk, and it does not compromise fidelity to
Ohashi's reported TensorFlow version. GPU training remains a documented,
open possibility — not pursued further by this task, per instruction, and
not without first either (a) accepting the Ada Lovelace compatibility risk
and testing a real CUDA 11.2/cuDNN 8.1 install, or (b) revisiting the
TensorFlow-version-fidelity trade-off explicitly, as its own future
decision.

## 11. Confirmation: main environment untouched

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
| Isolated training environment | Created (Python 3.10.20, TensorFlow 2.10.1 — verified working; project brief's historical "2.10.10" claim does not exist as a release, corrected 2026-08-14), fully separate from the main Python 3.13 environment |
| GPU | Physically present (RTX 4060, 8GB), driver-visible, **not usable by this TensorFlow install without a separate CUDA 11.2/cuDNN 8.1 install (§8/§10, not performed); §10 additionally finds real Ada Lovelace/TF-2.10-wheel compatibility risk even after installing it** |
| Checkpoint load / architecture cross-check | PASS (§9) |
| Input normalization (U-M05) | **RESOLVED 2026-08-14** as PROJECT-ADAPTATION baseline (decision A-23): `resnet50_preprocess_input`, cross-checked bit-exact against real TensorFlow (§9a) |
| Preprocessing specification | Complete and authoritative (§9b): `src/ct_iqa/preprocessing/pipeline.py::preprocess_for_model`, tested in `tests/test_preprocessing_pipeline.py` |
| Model training | **NOT STARTED** |
| Production dataset generation | **NOT PERFORMED** |
| Committed model architecture / dropout / VIF / degradation parameters | **UNCHANGED** |
