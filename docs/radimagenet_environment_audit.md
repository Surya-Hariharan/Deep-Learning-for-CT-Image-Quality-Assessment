# RadImageNet Checkpoint & Model-Environment Feasibility Audit

Audit only — no training, no checkpoint download, no production dataset
generation, no change to the committed research architecture
(`src/ct_iqa/models/resnet50.py`, `src/ct_iqa/training/trainer.py`,
`configs/quality/resnet50_vif.yaml`). Performed 2026-08-14, ahead of any
model-training authorization, as a prerequisite check separate from — and
not implying — that authorization.

Provenance tags follow the taxonomy fixed for this component
(`docs/research_decisions.md`): OHASHI-SPECIFIED / PROJECT-ADAPTATION /
REFERENCE-IMPLEMENTATION-CONVENTION / OBSERVED-PILOT-BEHAVIOR / UNKNOWN /
NOT-SPECIFIED / EXPERIMENTAL-EXTENSION.

---

## 1. Exactly which checkpoint the Ohashi methodology requires

Per `docs/ohashi_methodology.md` (OHASHI-SPECIFIED):

```
Input -> central crop -> 224x224 -> RadImageNet-pretrained ResNet50
       -> Dropout -> Fully Connected(1) -> Sigmoid -> quality score
```

- **Backbone**: ResNet50, pretrained on the **RadImageNet** database (Mei et
  al., 2022, "RadImageNet: An Open Radiologic Deep Learning Research
  Dataset for Effective Transfer Learning," *Radiology: Artificial
  Intelligence* — the RadImageNet project's own paper, not Ohashi's; Ohashi
  cites RadImageNet as the pretraining source, not the other way round).
- **Framework**: Ohashi's own reported environment is TensorFlow 2.10.10 /
  Python 3.10.13 (`requirements.txt` header comment, carried from the
  project brief). The RadImageNet project itself publishes weights for
  **both** TensorFlow and PyTorch (see §3) — Ohashi's specific framework
  choice narrows this to the **TensorFlow/Keras** release for a faithful
  replication.
- **Fine-tuning status**: confirmed **fine-tuned, not frozen** (decision
  S-01, `docs/research_decisions.md`): "By fine-tuning these pre-trained
  models with our IQA dataset..." — explicit paper text, not inferred.

This is the same checkpoint already named as blocker **U-M08** in
`docs/research_decisions.md` ("RadImageNet ResNet50 checkpoint | cannot
build the model at all without it | obtain the RadImageNet release") and in
`configs/quality/resnet50_vif.yaml:model.weights_path: null`. This audit
adds detail to that existing blocker; it does not introduce a new one.

## 2. Local availability

**Not available.** Searched the repository, the full working tree, and
common local download locations (`~/Downloads`) for any file or directory
matching `*radimagenet*`, `*.h5`, or `resnet50*` weight files — none found.
`configs/quality/resnet50_vif.yaml:model.weights_path` is `null`.
`src/ct_iqa/models/resnet50.py:unresolved_prerequisites()` lists "RadImageNet
ResNet50 checkpoint not obtained (weights_path unset)" as its first
blocker, and that function was re-run this pass (§8) — still true.

## 3. Official/research-authoritative source

The RadImageNet project's own GitHub repository:
**https://github.com/BMEII-AI/RadImageNet** (organization: BMEII-AI, the
paper's home lab). Its README states, quoted directly:

- TensorFlow weights: `https://drive.google.com/drive/folders/1Es7cK1hv7zNHJoUW0tI0e6nLFVYTqPqK`
  (folder) or `https://drive.google.com/file/d/1UgYviv2K6QPM1SCexqqab5-yTgwoAFEc`
  (single file) — both Google Drive links published directly in the README,
  not gated behind a request form.
- PyTorch weights: `https://drive.google.com/file/d/1RHt2GnuOYlc_gcoTETtBDSW73mFyRAtR`.
- The underlying **dataset** (raw images, separate from the pretrained
  weights) requires a request via `https://www.radimagenet.com/`, with a
  fallback contact of `twdeyer@radimagenet.com` / `twdeyer@gmail.com` if no
  response. **The pretrained model weights are not documented as requiring
  that same request process** — but see the caveats below before treating
  this as a green light.

**Caveats, stated plainly rather than smoothed over:**

- The README does not publish a checksum, a version tag, or a stable
  release artifact for the weights — they live on Google Drive, which is
  neither versioned nor a guaranteed-permanent host. A downloaded file
  cannot be verified byte-for-byte against a canonical hash from the
  README alone.
- The README states no explicit license or acceptable-use terms for the
  weights themselves (distinct from the dataset's request-gated terms).
  This should be confirmed directly (e.g. by contacting the maintainers)
  before any weights are used in a way that will be published or shared,
  not assumed absent because this audit couldn't find one.
- Secondary/mirror sources exist (a PyTorch "unofficial" port at
  `github.com/Warvito/radimagenet-models`, and a HuggingFace mirror at
  `huggingface.co/Lab-Rasool/RadImageNet`) — **not used as the source of
  record here**, since neither is the paper's own release, and a faithful
  Ohashi replication should trace back to the same weights RadImageNet
  itself published, not a third party's re-export.
- This audit did not download anything from any of the above links —
  per instruction, nothing is fetched without first showing exactly what
  is required (this section) and obtaining explicit approval.

## 4. Expected specification vs. this project's plan

| Property | RadImageNet's own published spec | This project's plan (`configs/quality/resnet50_vif.yaml`, `src/ct_iqa/models/resnet50.py`) | Compatible? |
| --- | --- | --- | --- |
| Architecture | ResNet50 (also DenseNet121, InceptionV3, InceptionResNetV2 available; only ResNet50 in scope here) | ResNet50 only | Yes |
| Framework | TensorFlow/Keras release exists (also PyTorch) | TensorFlow implied by Ohashi's reported environment | Yes, if the TF release is used |
| Input size | 224×224 (RadImageNet's own paper: "All images were resized to 224×224 pixels and used as the inputs of the neural networks") | 224×224 (`input_shape: [224, 224, 3]`) | **Yes — matches exactly** |
| Input channels | 3 (RGB-shaped; medical images likely channel-replicated the same way this project already does) | 3, via `grayscale_to_rgb` channel replication (verified: LDCT-IQAC PNGs already store 3 identical channels, `docs/dataset_audit.md`) | Yes |
| Checkpoint format | Not stated by the README; TensorFlow release is most likely a Keras `.h5` weights file (standard for this era/toolchain) but **not confirmed** — no filename is published | `weights_path: null` (format-agnostic placeholder) | **Unconfirmed, not blocking today** — must be checked when the actual file is obtained (§9) |
| Preprocessing / normalization | **Not documented** by the official README or the paper's methods excerpt available to this audit | `intensity_normalization: null` (already tracked as U-M05, NOT SPECIFIED BY OHASHI) | **Unresolved on both sides** — this is not a new gap this audit found, it confirms the existing one |
| Original classification head | 165-class softmax output (RadImageNet spans CT/MRI/US, 11 anatomic regions, per the RadImageNet paper) — "A global average pooling layer, a dropout layer, and an output layer activated by the softmax function were introduced after the last layer of the pretrained models" | Replace with: GAP (implicit in ResNet50's standard pooled-output mode) -> Dropout -> FC(1) -> Sigmoid | **Structurally compatible** — RadImageNet's own transfer-learning recipe is GAP + dropout + output layer, exactly the shape Ohashi's head reuses; only the final layer (165-way softmax vs 1-unit sigmoid) and the dropout rate differ, both already tracked as open items (dropout: U-M01; head replacement itself is the intended usage pattern, not a mismatch) |
| Backbone fine-tuning | RadImageNet's own ablations found "unfreezing all layers consistently achieved the best performance" — supports fine-tuning as a reasonable choice, though this is RadImageNet's own downstream-task finding, not evidence about what Ohashi specifically did | Fine-tuned (S-01, confirmed from Ohashi's own paper text, independent of RadImageNet's finding) | Yes, and independently corroborated |

**Overall: architecturally compatible.** Input size matches exactly
(224×224, no resize/rescale mismatch to reconcile). The head-replacement
pattern this project plans (drop RadImageNet's 165-way softmax head,
attach GAP -> dropout -> FC(1) -> sigmoid) is the same shape of transformation
RadImageNet's own paper recommends for downstream transfer learning, not an
unusual adaptation. Preprocessing/normalization is the one property neither
side documents — already tracked as U-M05, not a new finding, but this
audit could not close it either.

## 5. Current TensorFlow/Keras environment

```
$ py -3.13 -c "import tensorflow"
ModuleNotFoundError: No module named 'tensorflow'
$ py -3.13 -c "import keras"
ModuleNotFoundError: No module named 'keras'
$ py -3.13 -c "import torch"
ModuleNotFoundError: No module named 'torch'
```

**Neither TensorFlow, Keras, nor PyTorch is installed** in this
environment (`py -3.13`, the interpreter this repository's tests and
scripts currently run under — see `docs/project_status.md` for the
Microsoft Store CPython 3.13.14 vs. Anaconda 3.13.9 environment-drift note).
This matches `requirements.txt`'s own header comment and
`unresolved_prerequisites()`'s second blocker, both of which say the same
thing; this audit re-confirms it rather than assuming it.

`requirements.txt` has `tensorflow==2.10.10` present but **commented out**,
annotated "methodological match to Ohashi et al.; requires Python <=3.10" —
already correctly flagged as a hard version constraint before this audit.

## 6. Can Ohashi's specified training environment actually run here?

**No, not without a separate environment.** Concretely:

- **Python version mismatch.** This repository's active interpreter is
  Python 3.13.14 (`py -3.13`). TensorFlow 2.10.10 supports Python 3.7–3.10
  only (per its published wheel metadata/classifiers — not independently
  re-verified by fetching PyPI in this audit, but this is the same
  constraint already recorded in `requirements.txt`'s own comment, and is
  consistent with TensorFlow's well-documented Python-version support
  windows for that release line). `pip install tensorflow==2.10.10` would
  fail outright in this Python 3.13 environment before any GPU question
  even arises.
- **GPU present, but a further generation gap.** This machine has an NVIDIA
  GeForce RTX 4060 (laptop), 8 GB VRAM, driver reporting `CUDA UMD Version:
  13.3` (`nvidia-smi`, checked this pass). TensorFlow 2.10.x's GPU build
  targets CUDA 11.2 / cuDNN 8.1 and was the **last** TensorFlow release
  with native Windows GPU support before TensorFlow dropped native Windows
  GPU builds in 2.11+ (subsequent Windows GPU use requires WSL2). A CUDA
  11.2-targeted wheel is not guaranteed to run against a driver this far
  ahead without separately installing a matching (older) CUDA
  toolkit/cuDNN alongside it, independent of the Python-version blocker
  above.
- **Conclusion**: reproducing Ohashi's *exact* reported environment
  (Python 3.10.13, TensorFlow 2.10.10) on this machine requires a
  **separate, isolated environment** (e.g. a pinned conda env at Python
  3.10) — it cannot be `pip install`ed into the environment this
  repository's tests currently run under. This is a environment-provisioning
  task, not a code change, and is not performed by this audit.
- CPU-only execution of TensorFlow 2.10 under a correctly-versioned Python
  3.10 environment would work regardless of the GPU/CUDA question, just
  much slower — a fallback worth recording, not a recommendation to use it
  for full training.

## 7. Can the planned architecture do what's required?

Checked against `src/ct_iqa/models/resnet50.py` (code inspection + a
dummy-data run, §8) and `configs/quality/resnet50_vif.yaml`:

| Requirement | Status |
| --- | --- |
| Load RadImageNet weights | **Not yet buildable** — `ModelSpec.pretrained_weights = "radimagenet"` is declared; `build_model()` correctly refuses until `weights_path` is set (§2) and a framework is installed (§5). No code exists yet that actually calls a framework's weight-loading API (by design — decision A-14, this phase is spec-only). |
| Replace the classification head | Declared, not yet implemented: `ModelSpec.head = ("dropout", "fully_connected_1", "sigmoid")`. Matches RadImageNet's own recommended head-replacement shape (§4). No Keras/PyTorch model-graph code exists yet to perform this replacement. |
| Add the required dropout | **Rate unresolved** (`dropout_probability: None` in `ModelSpec`, `dropout_rate: null` in `configs/quality/resnet50_vif.yaml` — tracked as U-M01). The *presence* of a dropout layer is OHASHI-SPECIFIED; the *rate* is not, and per instruction 10 of this audit, this rate is **not** being invented here to make anything run. |
| Produce one sigmoid regression output | Declared (`head` ends in `("fully_connected_1", "sigmoid")`, `target.synthetic.range: [0.0, 1.0]` matching a sigmoid's range) — not yet implemented as executable code, same reason as above. |
| Fine-tune the backbone | **Resolved at the decision level** (S-01: fine-tuned, not frozen) but **not reflected in `ModelSpec`'s default** — `ModelSpec.freeze_backbone` still defaults to `None` (NOT SPECIFIED BY OHASHI) even though `docs/research_decisions.md` decision S-01 already resolved this. **This is a genuine inconsistency this audit found between the resolved research decision and the code's spec dataclass** — flagged here, **not corrected**, per this task's explicit instruction not to modify the committed architecture. |

**Summary**: the plan is architecturally sound and nothing found
contradicts it, but every one of these five capabilities is currently a
**declared specification**, not **executable code** — `build_model()`
raises unconditionally by design (decision A-14) and no framework is
installed to build against even if it didn't.

## 8. Model-construction smoke test (dummy data, no checkpoint required)

A full Keras-graph smoke test (`tf.keras.applications.ResNet50(weights=None)`,
attach the new head, verify output shape) was **not** run — it requires
TensorFlow, which is not installed (§5), and installing it was not
authorized by this task (§6 also notes it cannot target the correct Python
version in this environment regardless). Per instruction 8, only a test
that needs **neither** the checkpoint **nor** an uninstalled framework was
run: the actual framework-agnostic preprocessing path plus the existing
spec-validation code, executed against dummy data this pass:

```
$ py -3.13 -c "... see below ..."
dummy shape: (512, 512)
after central_crop: (224, 224)
after grayscale_to_rgb: (224, 224, 3) uint8
matches expected (224,224,3): True

DEFAULT_SPEC: ModelSpec(backbone='resnet50', pretrained_weights='radimagenet',
  input_size=224, crop_strategy='central_crop',
  head=('dropout', 'fully_connected_1', 'sigmoid'),
  dropout_probability=None, freeze_backbone=None,
  grayscale_to_rgb_method='channel_replication')

unresolved_prerequisites: [
  'RadImageNet ResNet50 checkpoint not obtained (weights_path unset)',
  'No deep-learning framework (tensorflow/torch) installed in this environment',
  'dropout_probability NOT SPECIFIED BY OHASHI -- must become a PROJECT ADAPTATION before training',
  'freeze_backbone NOT SPECIFIED BY OHASHI -- must become a PROJECT ADAPTATION before training',
]

build_model() raised ModelNotBuildable as expected:
Model construction is not authorized at this phase. Unresolved:
  - RadImageNet ResNet50 checkpoint not obtained (weights_path unset)
  - No deep-learning framework (tensorflow/torch) installed in this environment
  - dropout_probability NOT SPECIFIED BY OHASHI -- must become a PROJECT ADAPTATION before training
  - freeze_backbone NOT SPECIFIED BY OHASHI -- must become a PROJECT ADAPTATION before training
```

**Result**: the real preprocessing pipeline
(`ct_iqa.preprocessing.cropping.central_crop` +
`ct_iqa.preprocessing.channels.grayscale_to_rgb`) produces exactly the
`(224, 224, 3)` array shape RadImageNet's ResNet50 expects (§4), from a
dummy array shaped like a real LDCT-IQAC image (512×512). `build_model()`
correctly refuses with every current blocker enumerated, matching this
audit's other findings — no silent success, no accidental construction.
This is the full extent of what can be smoke-tested without either the
checkpoint or a framework install.

## 9. Nothing downloaded

No checkpoint, weight file, or archive was downloaded by this audit. The
exact links this project would need (§3) are stated above for review, not
fetched. If approved, the next action is a **manual, explicit** download of
the TensorFlow RadImageNet ResNet50 weights from the links in §3 into a
location outside version control (this repo's `.gitignore` already covers
`*.h5`/`*.pt`/checkpoint-shaped files generically — see
`docs/project_status.md`'s Git-safety section) — not an automated step in
any script in this repository.

## 10. Unresolved parameters — restated, not invented

Per instruction 10, none of the following were resolved or guessed to make
anything run; they remain exactly as tracked in `docs/research_decisions.md`:

- **U-M01** — dropout probability.
- **U-M05** — RadImageNet preprocessing / intensity normalization (this
  audit looked for it in the official README and paper excerpt available
  and did not find it documented there either — the gap is confirmed to
  exist on both the Ohashi side and the RadImageNet side, not just
  assumed).
- **`freeze_backbone` default in `ModelSpec`** — not changed to `True`
  despite S-01 already resolving the *decision*; flagged as an
  inconsistency (§7) for a future, separate, deliberate code change, not
  patched here.
- Checkpoint file format (`.h5` vs. other) — not confirmed, not guessed.

---

## Report

**Checkpoint status:** NOT AVAILABLE LOCALLY. Not downloaded by this audit.

**Authoritative source:** `https://github.com/BMEII-AI/RadImageNet` (README
links to Google Drive-hosted TensorFlow and PyTorch weight files; no
checksum, version tag, or explicit license published for the weights
themselves — see §3 caveats).

**Framework compatibility:** Ohashi's reported framework is TensorFlow
2.10.10 on Python 3.10.13. RadImageNet publishes a TensorFlow release, so
the framework choice is compatible in principle — but TensorFlow 2.10.10
cannot be installed in this repository's current Python 3.13 environment
(hard version-support mismatch, not merely untested).

**TensorFlow compatibility:** NOT INSTALLED. Cannot be installed as pinned
(`==2.10.10`) in the current Python 3.13 environment; would require a
separate, isolated Python ≤3.10 environment. GPU present (RTX 4060, 8GB)
but generations ahead of TF 2.10's targeted CUDA 11.2/cuDNN 8.1 — further
compatibility work (or CPU fallback) would be needed even after solving the
Python-version constraint.

**Architecture compatibility:** COMPATIBLE IN SPECIFICATION. Input size
matches exactly (224×224). Head-replacement pattern matches RadImageNet's
own recommended transfer-learning recipe (GAP + dropout + output layer).
No executable model-building code exists yet (by design, decision A-14);
the dummy-data preprocessing smoke test (§8) confirms the input pipeline
produces the correct shape.

**GPU/CPU feasibility:** GPU present but environment-incompatible with the
pinned TensorFlow version without additional environment work; CPU
execution under a correctly-versioned Python 3.10 environment is a viable,
slower fallback.

**Unresolved parameters:** U-M01 (dropout probability), U-M05 (RadImageNet
preprocessing/normalization — confirmed undocumented on both sides, not
just on Ohashi's), checkpoint file format (unconfirmed pending actual
download), plus a found-not-fixed inconsistency between `ModelSpec.freeze_backbone`'s
default (`None`) and the already-resolved S-01 decision (fine-tuned).

**Blockers:**
1. RadImageNet checkpoint not obtained (U-M08) — requires a human decision
   to download from the Google Drive links in §3, outside this audit's
   scope.
2. No compatible deep-learning framework installed — requires provisioning
   a separate Python ≤3.10 environment for TensorFlow 2.10.10, a
   machine/environment change, not a code change.
3. Dropout probability (U-M01) unresolved — needs a recorded
   PROJECT-ADAPTATION decision before training, independent of the above.
4. `ModelSpec.freeze_backbone` default not yet updated to reflect S-01 —
   a small, deliberate future code change, out of scope for this audit.

**Exact next action:** none automatic. If and when authorized:
1. A human downloads the TensorFlow RadImageNet ResNet50 weights from the
   links in §3 (manual step, outside version control).
2. Separately, provision an isolated Python 3.10 environment with
   `tensorflow==2.10.10` for training, since it cannot coexist with this
   repository's current Python 3.13 tooling environment.
3. Record a PROJECT-ADAPTATION decision for `dropout_probability` (U-M01)
   in `docs/research_decisions.md` before it is used.
4. Only then does implementing `build_model()` for real become meaningful;
   until all of the above are true, `build_model()` should keep raising, as
   it does today.

**Full 1,000-reference production dataset generation and model training
remain NOT AUTHORIZED and were NOT performed by this audit.**

---

## Resolution update (2026-08-14, same day, follow-up task)

The two blockers §7 and §10 flagged as fixable-without-a-checkpoint were
resolved in a follow-up task, per its own explicit instructions (no weights
downloaded, no TensorFlow installed, no new environment created, no
training, no dataset generation):

- **`ModelSpec.freeze_backbone` inconsistency** (§7, §10): fixed.
  `ModelSpec.freeze_backbone` now defaults to `False`, matching decision
  S-01 exactly — no new interpretation introduced.
  `configs/quality/resnet50_vif.yaml:model.freeze_backbone` updated to
  `false` to match. Regression-guarded by
  `tests/test_models_training_specs.py::test_default_spec_backbone_is_fine_tuned_not_frozen`.
- **Dropout probability** (U-M01, §10): resolved as decision A-22
  (`docs/research_decisions.md`) — a documented PROJECT-ADAPTATION
  baseline of `0.5`, sourced from RadImageNet's own base-model training
  recipe (Mei et al. 2022, the paper this checkpoint comes from), **not**
  from Ohashi's paper. Not chosen by any hyperparameter search or
  validation comparison. `configs/quality/resnet50_vif.yaml:model.dropout_rate`
  updated to `0.5` to match. Full rationale and alternatives considered:
  `docs/research_decisions.md` decision A-22;
  `docs/deviations_from_ohashi.md` DEV-05 states plainly that this remains
  a genuine, acknowledged point of possible divergence from Ohashi's own
  (unknown) rate, not a confirmed match to it.

**What did NOT change:** every blocker in §2–§6 and §9 of this audit
(checkpoint not obtained, no compatible framework installed, no weights
downloaded, TensorFlow 2.10.10/Python-3.13 incompatibility) is exactly as
it was. `unresolved_prerequisites(DEFAULT_SPEC)` now returns exactly 2
items (checkpoint, framework) instead of 4 — verified by
`tests/test_models_training_specs.py::test_resolved_default_spec_no_longer_blocks_on_dropout_or_freeze`.
`build_model()` still raises unconditionally. See
`docs/project_status.md` §12 for the current, consolidated readiness
statement.

---

## Version-string correction note (2026-08-14, second follow-up task)

Every "TensorFlow 2.10.10" reference above (§1, §6, §7's report, and this
addendum's own "What did NOT change" line) documents what this project
historically inherited as Ohashi's reported version — that description of
the *claim* is left as originally written, not rewritten, per this
project's convention of not editing historical records after the fact.

**What has since been established as fact, not claim:** TensorFlow
`2.10.10` does not exist as a PyPI release — the `2.10.x` line only ever
shipped `2.10.0` and `2.10.1` (confirmed directly against PyPI's release
metadata, `docs/training_environment.md` §2). Every conclusion drawn above
using "2.10.10" as a stand-in for "the Ohashi-matching TensorFlow release"
still holds using `2.10.1` instead — the Python-version incompatibility
(§6), the native-Windows-GPU-support argument, and the checkpoint/framework
analysis are all substantively unaffected, since none of them depended on
the exact trailing digit. The actual verified, installed, isolated-environment
combination is **Python 3.10.20 + TensorFlow 2.10.1** — see
`docs/training_environment.md` for the authoritative, current record.
