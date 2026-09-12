# Decisions log

A running log of structural/process decisions made about this repository
that aren't paper facts or replication-methodology decisions (those belong
in `docs/paper/` and `docs/replication/` respectively). Newest first.

---

## 2026-09-13 -- `configs/model.yaml`'s `radimagenet_weights_path` now points at the verified checkpoint

**Decision:** Changed `radimagenet_weights_path` from `null` to
`weights/pretrained/radimagenet/resnet50/radimagenet_resnet50_backbone.pt`.

**Why:** The scientific pipeline audit (2026-09-13) actually exercised
`OhashiResNet50.load_radimagenet_weights` against the real, locally-present
converted checkpoint and confirmed it loads cleanly (265/318 keys matched,
0 unexpected). Leaving the config `null` while a verified-working file sat
unused on disk meant the "real" experiment (`experiments/001_resnet50_baseline`)
would have silently trained with a random backbone despite RadImageNet
weights being available and working -- a genuine configuration gap, not a
scientific-methodology change. See
`weights/pretrained/radimagenet/resnet50/README.md` for the verification
record and `docs/replication/deviations.md` for the corresponding update.

**How it applies:** `weights/pretrained/` is gitignored, so this path will
not resolve on a machine without the file -- `torch.load` raises
`FileNotFoundError` in that case (never a silent fallback to ImageNet
weights or to random init). To deliberately run without RadImageNet
weights, pass `radimagenet_weights_path=None` explicitly when constructing
`ExperimentConfig`, rather than relying on the YAML default.

---

## 2026-09-13 -- Added a real-data smoke test inside the training notebook, not just synthetic unit tests

**Decision:** `notebooks/03_training/01_resnet50_baseline.ipynb` now runs 2
real batches through `train_one_step` (forward, backward, `optimizer.step()`,
gradient-existence and parameter-change checks) immediately after model
construction, THEN restores the model's pre-smoke-test weights, before the
separately-gated `RUN_FULL_TRAINING` cell.

**Why:** The prior smoke-test cell (`RUN_FULL_TRAINING = False`) skipped
training entirely rather than verifying anything -- it never actually
exercised backward()/optimizer.step() on real data before a contributor
would commit to a real 30-epoch run. `tests/unit/test_training_step.py`
only covers this with synthetic random tensors, not the real
dataset+preprocessing+RadImageNet-weights path. The restoration step
afterward is necessary so the smoke test's 2 optimizer steps don't
contaminate the real run's starting point.

**How it applies:** Any future training notebook (e.g. for experiments 002
or 003) should follow the same pattern: a small, restored smoke test before
the real, separately-gated run.

---

## 2026-09-13 -- `notebooks/02_data_preparation/01_dataset_preparation.ipynb` is a validation notebook, not a generation notebook

**Decision:** This notebook validates the existing raw-data ->
preprocessing -> model-input path; it does not precompute or generate any
new data into `data/interim/`/`data/processed/`.

**Why:** LDCT-IQAC needs no synthetic degradation or VIF-label generation
(see `docs/replication/deviations.md`) -- there is nothing this project's
"data preparation" stage needs to *produce*. What was actually missing was
verification that the pipeline is correct and leakage-free, so that's what
the notebook does. `data/interim/` and `data/processed/` remain empty
(with `.gitkeep`) as a result -- this is expected, not an oversight.

**How it applies:** Do not add a degradation/VIF step to this notebook (or
anywhere in `src/ct_iqa/`) unless the project's dataset or scope actually
changes to require reproducing the paper's own data-construction pipeline.

---

## 2026-09-12 -- Experiment checkpoints live under `experiments/`, not `weights/`

**Decision:** Experiment-generated checkpoints (`best.pt`, per-run config,
predictions) are written to `experiments/<NNN_name>/checkpoint/`, never to
`weights/checkpoints/` or anywhere under `weights/`.

**Why:** `weights/pretrained/` holds externally-sourced pretrained weights
(RadImageNet) -- a fundamentally different kind of artifact from a
checkpoint this project's own training loop produced. Keeping them under
one `weights/` tree risked the two being confused (e.g. accidentally
treating a stale experiment checkpoint as a pretrained backbone, or vice
versa). `ExperimentConfig.checkpoint_dir` is now a derived property
(`<experiment_dir>/checkpoint`), not an independently-settable field, so
the two locations cannot drift apart by passing inconsistent config values.

**How it applies:** `ct_iqa.training.checkpointing.save_checkpoint`/
`load_checkpoint` always operate relative to `config.checkpoint_dir`.
`configs/training.yaml`'s `experiment_dir` is the one place this is
configured.

---

## 2026-09-12 -- `ExperimentConfig` stays; YAML configs are additive, not a replacement

**Decision:** `configs/*.yaml` are the on-disk, human-editable source of
truth for hyperparameters; `ExperimentConfig` (a Python dataclass) remains
the runtime interface every notebook/module actually consumes.
`ExperimentConfig.from_yaml_files()` merges the four YAML files into one
`ExperimentConfig` instance.

**Why:** Removing `ExperimentConfig` entirely (in favor of passing raw
dicts/YAML around) would have discarded working, tested, type-checked code
for no benefit. The problem being solved was never "the config isn't in
YAML," it was "notebook paths, Python defaults, and (a since-added)
`configs/*.yaml` could silently disagree." A single merge function closes
that gap without removing the dataclass.

**How it applies:** New notebooks should call
`ExperimentConfig.from_yaml_files()` rather than constructing
`ExperimentConfig(...)` with hardcoded path/hyperparameter literals.

---

## 2026-09-12 -- Notebook generators live under `tools/`, not `scripts/`

**Decision:** The Python scripts that generate `notebooks/*.ipynb` files
(`build_eda_notebook.py`, `build_training_notebook.py`,
`build_evaluation_notebook.py`) live under `tools/`. The old `scripts/`
directory no longer exists.

**Why:** These are dev-time tooling, not reusable ML implementation
(`src/`) and not research-execution records (`notebooks/`). One further
script previously under `scripts/` (`convert_radimagenet_weights.py`) was
NOT moved to `tools/` -- it was moved into `src/ct_iqa/models/` instead,
because it's genuinely reusable, tested library code
(`ct_iqa.models.radimagenet_weights`), not a one-off tool.

**How it applies:** Never hand-edit a generated notebook's `.ipynb` JSON as
the primary way of changing it -- edit the corresponding `tools/build_*.py`
generator and re-run it, or the generator and the notebook will drift apart
(as happened once already with stale dataset paths, and once more subtly
with a markdown-cell-generation bug -- see
`docs/repository_architecture_audit.md`, section 5).

---

## 2026-09-12 -- No LR-search / final-training / replication-results notebooks created yet

**Decision:** `notebooks/03_training/02_learning_rate_search.ipynb`,
`notebooks/03_training/03_final_training.ipynb`,
`notebooks/04_evaluation/02_prediction_analysis.ipynb`,
`notebooks/04_evaluation/03_replication_results.ipynb`, and
`notebooks/02_data_preparation/01_dataset_preparation.ipynb` were **not**
created during this migration.

**Why:** None of that work has actually happened yet -- there is no
learning-rate search, no final training run, no prediction-analysis pass,
and no replication-results writeup to record. LDCT-IQAC also needs no
separate "data preparation" step beyond what `ct_iqa.data.ldct_iqac`
already does on the fly (no precomputed interim/processed artifacts exist).
Creating empty placeholder notebooks for stages that haven't happened would
misrepresent the project's actual experimental status.

**How it applies:** Create each notebook only when the corresponding work
is actually being done, following the pattern established by
`notebooks/03_training/01_resnet50_baseline.ipynb` /
`notebooks/04_evaluation/01_test_set_evaluation.ipynb` (config -> data ->
model -> training/evaluation -> results -> notes, generated from a
`tools/build_*.py` script, never hand-edited directly).

---

## 2026-09-12 -- No `LICENSE` file added

**Decision:** No `LICENSE` file was created during this migration.

**Why:** The correct license for this project cannot be determined from
the existing repository (no license file, no license field, no stated
intent) or from a "standard" default -- and the repository also builds on
a dataset (LDCT-IQAC) and pretrained weights (RadImageNet) with their own
distinct usage terms, which interact with whatever license is eventually
chosen. Inventing a license would risk misrepresenting the project owner's
actual intent.

**How it applies:** This requires an explicit decision from the project
owner. See the root `README.md`'s "Citation & License" section, which
documents this as an open item rather than asserting a license.
