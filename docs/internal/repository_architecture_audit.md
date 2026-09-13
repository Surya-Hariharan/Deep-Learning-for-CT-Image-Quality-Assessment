# Repository Architecture Audit

Date: 2026-09-12
Scope: audit only. No files were moved, renamed, deleted, or rewritten while producing this document.

## 1. Current Repository Structure

```
Deep-Learning-for-CT-Image-Quality-Assessment/
├── .gitignore
├── data/
│   ├── train/
│   │   ├── train.json                  (tracked)
│   │   └── image/*.tif                 (untracked, gitignored, 1000 files)
│   └── test/
│       ├── test.json                   (tracked)
│       └── images/*.tiff               (untracked, gitignored, 300 files)
├── docs/
│   └── git_worktree_audit.md           (created this session, prior task)
├── notebooks/
│   ├── 01_eda_ldct_iqac.ipynb
│   └── 02_ohashi_resnet50_ldct_iqac.ipynb
├── scripts/
│   ├── build_eda_notebook.py
│   ├── build_ohashi_notebook.py
│   └── convert_radimagenet_weights.py
├── src/
│   └── ct_iqa/
│       ├── __init__.py
│       ├── config.py
│       ├── data/
│       │   ├── __init__.py
│       │   └── ldct_iqac.py
│       ├── evaluation/
│       │   ├── __init__.py
│       │   └── metrics.py
│       ├── models/
│       │   ├── __init__.py
│       │   ├── ohashi_resnet50.py
│       │   └── resnet50.py
│       ├── training/
│       │   ├── __init__.py
│       │   └── trainer.py
│       └── utils/
│           ├── __init__.py
│           └── seed.py
├── tests/
│   ├── test_dataset.py
│   ├── test_metrics.py
│   ├── test_model.py
│   ├── test_radimagenet_weights.py
│   └── test_training_step.py
└── weights/
    ├── README.md
    └── pretrained/                     (untracked, gitignored)
        ├── RadImageNet-ResNet50_notop.h5
        └── radimagenet_resnet50_backbone.pt
```

**No `README.md`, `LICENSE`, `pyproject.toml`, or `.python-version` exist anywhere in the repository.** There is no `configs/`, `experiments/`, or `results/` directory. `docs/` currently contains only the Git audit produced in the prior task. This is a very early-stage repository: one dataset, one baseline architecture, one training script, no packaging metadata.

## 2. Target Repository Structure

(As specified in the task — reproduced here for reference; see the task prompt for the full annotated tree.) Key structural shifts from the current state:

- Root gets real packaging/metadata files (`README.md`, `LICENSE`, `pyproject.toml`, `.python-version`).
- `configs/` externalizes hyperparameters currently hardcoded in `ExperimentConfig`.
- `data/` gets `raw/interim/processed/labels` stages instead of flat `train/`/`test/`.
- `docs/` splits paper-facts (`docs/paper/`) from our-implementation facts (`docs/replication/`), plus a `docs/decisions/` log.
- `src/ct_iqa/` gains `preprocessing/`, `degradation/`, `labeling/` packages (currently entirely absent) and splits `training/`/`evaluation/`/`data/` into finer modules than exist today.
- `notebooks/` becomes four numbered stage-directories instead of two flat, ad-hoc-numbered files.
- `weights/` nests pretrained weights by source/architecture instead of a single flat `pretrained/` folder, and adds a `checkpoints/` counterpart.
- `experiments/` and `results/` are new — nothing currently records per-run experiment metadata or persisted output artifacts outside of a checkpoint directory.
- `tests/` splits into `unit/` and `integration/`.

## 3. File-by-File Migration Table

| Current Path | Classification | Target Path | Reason | Changes Required |
|---|---|---|---|---|
| `.gitignore` | KEEP (edit in place) | `.gitignore` | Root config file, correct location already. | Update stale `data/training/`, `data/testing/`, `data/testing/**`, `!data/testing/test.json` rules to `data/train/`, `data/test/` (see [git_worktree_audit.md](git_worktree_audit.md) §4) — cosmetic, no behavior change since extension rules already cover it. |
| `data/train/train.json` | MOVE | `data/labels/ldct_iqac/train.json` | It is a label file (filename→score), not raw imagery — belongs in the `labels` stage, not mixed with a stage named after the split. Namespaced under `ldct_iqac/` per Rule 6 (keep our dataset identity explicit, never implied to be the paper's dataset). | Update `ExperimentConfig.train_json_path` default; update both notebooks/generators; update `.gitignore` allow-list line `!data/testing/test.json` → new path. |
| `data/train/image/*.tif` (1000 files, untracked) | MOVE | `data/raw/ldct_iqac/train/image/*.tif` | These are the original, unmodified TIFFs as received — canonical "raw" stage. No preprocessing happens on disk today (crop/normalize happens in `__getitem__`), so nothing currently belongs in `interim/`/`processed/`. | Update `ExperimentConfig.train_image_dir` default; update `.gitignore` (`data/raw/**` already ignores it, so the specific `data/train/...` path rules become obsolete — simplification, not just a rename). |
| `data/test/test.json` | MOVE | `data/labels/ldct_iqac/test.json` | Same reasoning as train labels. | Same as above, test-side. |
| `data/test/images/*.tiff` (300 files, untracked) | MOVE | `data/raw/ldct_iqac/test/images/*.tiff` | Same reasoning as train images. | Same as above, test-side. |
| `notebooks/01_eda_ldct_iqac.ipynb` | MOVE + RENAME | `notebooks/01_exploration/01_dataset_eda.ipynb` | Pure EDA (file/label consistency, pixel stats, score histograms, sample grids) — no training, no model. Matches target's `01_exploration/` stage exactly. | Update the hardcoded `data/training/`/`data/testing/` path constants inside the notebook (see §10) to the new `data/raw/ldct_iqac/...` paths. Regenerate via `scripts/build_eda_notebook.py` (see below) rather than hand-editing the `.ipynb`. |
| `notebooks/02_ohashi_resnet50_ldct_iqac.ipynb` | SPLIT + MOVE + RENAME | `notebooks/03_training/01_resnet50_baseline.ipynb` (config → model → train → checkpoint) **and** `notebooks/04_evaluation/01_test_set_evaluation.ipynb` (load checkpoint → evaluate → metrics → plots → save predictions) | This single notebook currently does BOTH training (Rule 2 scope: `notebooks/03_training/`) AND held-out test-set evaluation with metrics/plots/persistence (Rule 4 scope: `notebooks/04_evaluation/`). Per Rule 2 and Rule 4 these are two distinct notebook purposes and belong in two directories. | Split cells 1–18 (imports, config, dataset build, model init, optimizer, training loop, loss curve, checkpoint save) into the training notebook; split cells 19–28 (checkpoint load, test-set evaluation, metrics, scatter plot, predictions/metrics/config persistence, summary) into the evaluation notebook. The evaluation notebook must load its model from the checkpoint path written by the training notebook rather than sharing in-memory state. Fix stale dataset paths (§10). |
| `scripts/build_eda_notebook.py` | MOVE + REWRITE | `tools/build_eda_notebook.py` (new `tools/` dir — see note below) | Not application source, not a research record, not reusable ML implementation — a one-off notebook code-generator. The target tree has no directory for this kind of dev tooling; `scripts/` is explicitly *not* preserved by Rule 8, so a new minimal `tools/` (or equivalent) location is proposed rather than silently keeping `scripts/`. | Update the notebook target path (`notebooks/01_eda_ldct_iqac.ipynb` → `notebooks/01_exploration/01_dataset_eda.ipynb`) and the dataset path constants written into the generated notebook. |
| `scripts/build_ohashi_notebook.py` | SPLIT + MOVE + REWRITE | `tools/build_training_notebook.py` + `tools/build_evaluation_notebook.py` | Mirrors the notebook split above — one generator per resulting notebook, so each stays a focused, regeneratable script rather than one script emitting two different notebooks' worth of unrelated cells. Also: its own docstring already refers to the wrong output filename (`notebooks/01_ohashi_resnet50_ldct_iqac.ipynb`) — stale even relative to today's actual output path (`notebooks/02_...`), an existing inconsistency, not something introduced by this migration. | Split cell list at the training/evaluation boundary (see notebook row above); fix dataset paths; fix the internal docstring's stale output-filename reference; also fix the dead `md()` markdown-cell issue (§10) while rewriting, since these files are being touched anyway. |
| `scripts/convert_radimagenet_weights.py` | MOVE | `src/ct_iqa/models/radimagenet_weights.py` | This is reusable, tested, package-relevant implementation (Keras→PyTorch weight conversion tied specifically to `ResNet50Backbone`'s layout), not a one-off script — Rule 8 explicitly says to move such functionality into `src/` rather than leave it in `scripts/`. `tests/test_radimagenet_weights.py` already imports it as `from scripts.convert_radimagenet_weights import convert`, i.e. it is already being treated as importable library code, just from the wrong package. | Update the import in `tests/test_radimagenet_weights.py`; update the `--input`/`--output` CLI defaults' surrounding docstring cross-references in `ohashi_resnet50.py` and `weights/README.md`; keep the `if __name__ == "__main__":` CLI entry point so it remains runnable as `python -m ct_iqa.models.radimagenet_weights`. |
| `src/ct_iqa/__init__.py` | KEEP | `src/ct_iqa/__init__.py` | Correct location; package-level docstring is accurate and not stale. | None. |
| `src/ct_iqa/config.py` | KEEP location; REWRITE contents | `src/ct_iqa/config.py` | Matches the target path exactly. However: (1) its dataset-path defaults (`data/training/image`, `data/training/train.json`, `data/testing/images`, `data/testing/test.json`) are **currently wrong** — they point at directories that no longer exist after the `730bb2c` rename commit (`data/training/`→`data/train/`, `data/testing/`→`data/test/`) — this is a functional bug, not just a naming mismatch; (2) the target architecture introduces `configs/*.yaml` as the source of truth for hyperparameters, but `config.py` currently hardcodes every default as a dataclass literal with no YAML loading path at all. | Fix the four stale path defaults immediately (independent of the broader migration — this is currently broken). As a further step, add a `from_yaml`/loader path so `ExperimentConfig` is constructed from `configs/dataset.yaml` + `configs/training.yaml` rather than only from code-level defaults, once `configs/` exists. |
| `src/ct_iqa/data/__init__.py` | KEEP | `src/ct_iqa/data/__init__.py` | Empty package marker, correct location. | Optional: export `LDCTIQACDataset`, `normalize_score`, `denormalize_score` for a cleaner public API once `data/loader.py`/`data/splits.py` exist alongside it. |
| `src/ct_iqa/data/ldct_iqac.py` | KEEP (do not rename to `dataset.py`) | `src/ct_iqa/data/ldct_iqac.py` | The target tree's generic `dataset.py` name is aspirational for a multi-dataset future; renaming a dataset-specific loader to a generic name would violate Rule 6 (never blur our dataset's identity) more than it would help today, since there is currently exactly one dataset. Recommend keeping the specific name and treating `dataset.py` as a future abstraction layer if/when a second dataset is added. | Fix the stale `data/training/image/`, `data/testing/images/` path references in the module docstring (cosmetic — the code itself takes paths as constructor arguments and has no hardcoded paths). Part of `build_dataloaders`/`build_test_dataloader` (currently in `trainer.py`, see below) logically belongs alongside this file as `data/loader.py` and `data/splits.py`. |
| `src/ct_iqa/evaluation/__init__.py` | KEEP | `src/ct_iqa/evaluation/__init__.py` | Correct location. | None. |
| `src/ct_iqa/evaluation/metrics.py` | KEEP now; SPLIT recommended later | `src/ct_iqa/evaluation/metrics.py` (keep `mse`, `compute_metrics`) + `evaluation/correlation.py` (move `plcc`, `srocc`, `krocc`) + `evaluation/calibration.py` (move `five_parameter_logistic_fit`, `compute_metrics_with_logistic_mapping`) | The file is currently small (98 lines) and coherent, so the split is not urgent — but it is the one file that maps directly onto three separate target modules, so the split should happen when the file grows or when `calibration.py`/`correlation.py` gain their own dataset-specific content. | Update `compute_metrics`'s internal calls to the moved functions' new import paths; update the two notebook/test imports (`from ct_iqa.evaluation.metrics import compute_metrics, compute_metrics_with_logistic_mapping`) if the split happens. |
| `src/ct_iqa/models/__init__.py` | KEEP | `src/ct_iqa/models/__init__.py` | Correct location. | None. |
| `src/ct_iqa/models/ohashi_resnet50.py` | KEEP | `src/ct_iqa/models/ohashi_resnet50.py` | Exact match to target path. Contains the authoritative `OhashiResNet50` class: regression head (`Dropout → Linear(2048,1) → Sigmoid`), `load_radimagenet_weights`, `verify_backbone_loaded`. This is exactly Rule 1's example — architecture lives here, notebooks only instantiate it, and that separation already holds today. | None required by the migration itself. Optional future refactor: extract `WeightLoadReport` + `load_radimagenet_weights` into a `models/weights.py` (the target's `models/heads.py` slot is currently unused — see §4). |
| `src/ct_iqa/models/resnet50.py` | KEEP | `src/ct_iqa/models/resnet50.py` | Exact match to target path. Generic ResNet50 bottleneck backbone, correctly kept separate from the Ohashi-specific head (Rule 1 compliance). | None. |
| `src/ct_iqa/training/__init__.py` | KEEP | `src/ct_iqa/training/__init__.py` | Correct location. | None. |
| `src/ct_iqa/training/trainer.py` | KEEP core; SPLIT out non-training-loop pieces | `src/ct_iqa/training/trainer.py` (keep `train_one_step`, `evaluate_loader`, `train`, `TrainingHistory`) + `data/loader.py` (move `build_dataloaders`, `build_test_dataloader` — these build `DataLoader`s from `LDCTIQACDataset`, which is dataset-domain, not training-loop-domain) + `training/optimizers.py` (move `build_optimizer`) + `training/losses.py` (move `build_loss`) + `training/checkpointing.py` (extract the inline `torch.save(model.state_dict(), best_checkpoint_path)` best-checkpoint logic from `train()`) | This is the file most out of step with the target's finer-grained `training/` package (`losses.py`, `optimizers.py`, `checkpointing.py` are all currently folded into one file, and dataloader construction is training-code that is actually dataset-code). Also contains a stale path reference (`data/testing/` in the module docstring). | Update imports across `trainer.py` itself and both training/evaluation notebooks once split. Fix the stale `data/testing/` docstring reference regardless of whether the split happens. |
| `src/ct_iqa/utils/__init__.py` | KEEP | `src/ct_iqa/utils/__init__.py` | Correct location. | None. |
| `src/ct_iqa/utils/seed.py` | KEEP | `src/ct_iqa/utils/seed.py` | Exact match to target path (`utils/seed.py`). | None. |
| `tests/test_dataset.py` | MOVE | `tests/unit/test_dataset.py` | Pure unit test — uses `tmp_path`, synthetic TIFFs, no real dataset, no model, no I/O beyond the fixture. | Update `pyproject.toml`/pytest config (once it exists) or `pytest.ini`/`conftest.py` `testpaths`/rootdir assumptions so `tests/unit/` is discovered; no import changes needed (imports are already fully qualified `ct_iqa.*`). |
| `tests/test_metrics.py` | MOVE | `tests/unit/test_metrics.py` | Pure unit test — synthetic arrays only, no I/O, no model. | Same test-discovery note as above. |
| `tests/test_model.py` | MOVE | `tests/unit/test_model.py` | Pure unit test — synthetic tensors, no real weights, no dataset. | Same. |
| `tests/test_radimagenet_weights.py` | SPLIT | `tests/unit/test_radimagenet_weights.py` (synthetic-state-dict tests, always run) + `tests/integration/test_radimagenet_weights_integration.py` (the four `@requires_h5`/`@requires_converted_pt`-gated tests that touch real, locally-downloaded weight files) | The file already self-documents this exact split in its own module docstring ("Synthetic-state-dict tests... / Real-file integration tests...") — the `tests/unit/` vs `tests/integration/` distinction the target structure asks for already exists conceptually in this file, just not physically. | Update the `from scripts.convert_radimagenet_weights import convert` import to `from ct_iqa.models.radimagenet_weights import convert` once that module moves (see `scripts/convert_radimagenet_weights.py` row above) — this import will otherwise break silently (only the two `@requires_h5`-gated tests exercise it, and those are skipped when the `.h5` file isn't present locally, so a broken import here could go unnoticed in CI unless the weight file happens to be present). |
| `tests/test_training_step.py` | KEEP (unit) | `tests/unit/test_training_step.py` | Fast, CPU-only, single optimization step on random tensors through the real model — no real dataset or file I/O, appropriately a unit test despite exercising the full model forward/backward path. | Test-discovery note only. |
| `weights/README.md` | MOVE + REWRITE | `weights/pretrained/radimagenet/resnet50/README.md` | Matches the target's deeper nesting (`weights/pretrained/<source>/<architecture>/`), which anticipates more than one pretrained source/architecture eventually sharing `weights/pretrained/`. | Update the two file paths it documents (`weights/pretrained/RadImageNet-ResNet50_notop.h5` → `weights/pretrained/radimagenet/resnet50/RadImageNet-ResNet50_notop.h5`, and the converted `.pt` sibling) and its reference to `scripts/convert_radimagenet_weights.py` once that file moves. |
| `weights/pretrained/RadImageNet-ResNet50_notop.h5` (untracked, gitignored, local-only) | MOVE | `weights/pretrained/radimagenet/resnet50/RadImageNet-ResNet50_notop.h5` | Same nesting rationale as the README. Purely a local-disk relocation — the file is never tracked by git either way. | Update `scripts/convert_radimagenet_weights.py`'s (or its moved successor's) `--input` CLI default. |
| `weights/pretrained/radimagenet_resnet50_backbone.pt` (untracked, gitignored, local-only) | MOVE | `weights/pretrained/radimagenet/resnet50/radimagenet_resnet50_backbone.pt` | Same as above. | Update the `--output` CLI default and `ExperimentConfig.radimagenet_weights_path` usage examples in notebooks/docs. |
| `__pycache__/` directories (7 locations, all gitignored) | REMOVE (leave as-is; not a migration concern) | n/a | Generated Python bytecode cache. Already correctly gitignored and untracked — nothing to move. | None. Will regenerate automatically wherever `.py` files end up after migration. |
| `docs/git_worktree_audit.md` | KEEP | `docs/git_worktree_audit.md` | Output of the prior Git-hygiene audit task; fits at the `docs/` root alongside the target's `docs/README.md`, distinct from the paper/replication/decisions split (it documents repository *tooling* state, not the paper or the replication). | None. |

## 4. Python Architecture Audit

Tracing the Ohashi replication component-by-component, per the task's specific request:

| Component | Currently lives in | Target location | Status |
|---|---|---|---|
| ResNet50 backbone (stem, 4 bottleneck stages, global-avg-pool) | `src/ct_iqa/models/resnet50.py` — `Bottleneck`, `ResNet50Backbone` | `src/ct_iqa/models/resnet50.py` | **Already correct.** Fully in `src/`, generic (no Ohashi-specific code), matches target exactly. |
| RadImageNet backbone weight *loading* (state-dict key matching, BN epsilon correction, missing/unexpected-key reporting) | `src/ct_iqa/models/ohashi_resnet50.py` — `OhashiResNet50.load_radimagenet_weights`, `WeightLoadReport` | `src/ct_iqa/models/ohashi_resnet50.py` (or split to `models/weights.py`) | **Already correct**, fully in `src/`. |
| RadImageNet backbone weight *conversion* (Keras `.h5` → PyTorch state-dict) | `scripts/convert_radimagenet_weights.py` | `src/ct_iqa/models/radimagenet_weights.py` | **Misplaced.** This is reusable, tested implementation currently sitting in `scripts/` and already imported as if it were library code by `tests/test_radimagenet_weights.py`. See migration table. |
| Regression head (`Dropout → Dense(1) → Sigmoid`) | `src/ct_iqa/models/ohashi_resnet50.py` — inline in `OhashiResNet50.__init__`/`forward` | `src/ct_iqa/models/ohashi_resnet50.py` (target also lists an unused `models/heads.py` slot) | **Already correct** location-wise. Not extracted into a separate `heads.py` — acceptable while there is only one head; extraction only pays off with a second head/architecture. |
| Dropout | Same class, `self.dropout = nn.Dropout(p=dropout_p)` | Same | Correct. `dropout_p` has no silent default (raises `TypeError` if omitted) — this is a deliberate, documented, and tested (`test_dropout_p_is_required`) choice because the paper never specifies the probability. |
| Sigmoid | Same class, `self.sigmoid = nn.Sigmoid()` | Same | Correct. |
| Preprocessing (center crop, `[-1,1]` normalization) | `src/ct_iqa/data/ldct_iqac.py` — `_center_crop`, inline `2.0 * array - 1.0` in `__getitem__` | `src/ct_iqa/preprocessing/crop.py`, `preprocessing/normalize.py` | **Gap, not misplacement.** `src/ct_iqa/preprocessing/` does not exist at all yet. The logic is currently correctly implemented but embedded inside the dataset class rather than factored into standalone, independently-testable transform functions. This is the single largest architectural gap relative to the target tree. |
| Degradation (synthetic image-quality degradation used in the *original* Ohashi paper to generate a range of quality levels from clean CT images) | **Nowhere.** No degradation code exists in this repository. | `src/ct_iqa/degradation/noise.py`, `degradation/blur.py`, `degradation/pipeline.py` | **Not applicable to the current replication, and this must be documented, not silently treated as a missing feature.** LDCT-IQAC (this project's dataset) already ships images spanning a range of real quality levels with human radiologist scores — it does not need synthetic degradation to create quality variation, unlike the paper's own dataset construction. Building `degradation/` now would be inventing paper-methodology that this replication does not use. Per Rule 6/Rule 5, this must be recorded explicitly in `docs/replication/deviations.md`, not left as an unexplained empty directory. |
| VIF (Visual Information Fidelity) labeling — the *paper's* regression target | **Nowhere.** No VIF computation exists. | `src/ct_iqa/labeling/vif.py` | **Not applicable, for the same reason as degradation.** This project's regression target is LDCT-IQAC's radiologist-assigned quality score (`SCORE_MIN=0.0`, `SCORE_MAX=4.0`, documented explicitly in `ldct_iqac.py`'s module docstring as "NOT the VIF metric used... in the original Ohashi paper"). The codebase already gets this distinction right in its documentation; the gap is only that `docs/replication/deviations.md` doesn't exist yet to make it discoverable outside of a source-code docstring. |
| Training loop (optimizer step, loss, checkpoint-on-best) | `src/ct_iqa/training/trainer.py` — `train`, `train_one_step`, `evaluate_loader` | `src/ct_iqa/training/trainer.py` (+ split-outs, see §3) | **Already correctly in `src/`,** notebooks only call it. Internal organization needs the split described in §3 (optimizer/loss/checkpoint builders currently inlined). |
| Evaluation (MSE/PLCC/SROCC/KROCC, 5PL mapping) | `src/ct_iqa/evaluation/metrics.py` | `src/ct_iqa/evaluation/metrics.py` (+ split-outs) | **Already correctly in `src/`.** |

**No architecture code was found living inside a notebook.** Both notebooks (via their generator scripts) only ever *import and call* `ct_iqa.*` — `OhashiResNet50(...)`, `build_dataloaders(...)`, `train(...)`, `compute_metrics(...)`, etc. Rule 1 is already satisfied in practice; nothing needs to move *out* of a notebook. This is the one area of the current codebase that already matches the target discipline exactly.

## 5. Notebook Audit

### `notebooks/01_eda_ldct_iqac.ipynb` (generated by `scripts/build_eda_notebook.py`)

- **Purpose:** Exploratory data analysis of the LDCT-IQAC dataset — file/label consistency check, image property inspection (mode/size/dtype/pixel range), duplicate-image check, score-distribution statistics and histograms, sample image grids across the score range.
- **Current contents:** 13 code cells, **zero markdown cells** (confirmed: `grep` for `"cell_type": "markdown"` across both notebooks returns no matches at all).
- **Duplicate code:** None against `src/` — this notebook does not re-implement anything in `ct_iqa` (there is no `ct_iqa` import at all; it works directly against the JSON/TIFF files with local helper functions defined inline, which is appropriate for one-off EDA rather than reusable pipeline code).
- **Does it define a model / perform training?** No.
- **Target notebook:** `notebooks/01_exploration/01_dataset_eda.ipynb`.
- **Required changes:** Update the four hardcoded path constants (`TRAIN_IMAGE_DIR`, `TRAIN_JSON_PATH`, `TEST_IMAGE_DIR`, `TEST_JSON_PATH` — currently `data/training/image`, `data/training/train.json`, `data/testing/images`, `data/testing/test.json`) to match wherever the migration actually lands the dataset (§3). **These paths are already broken today**, independent of any migration — the directories were renamed by commit `730bb2c` and this notebook (and its generator) were never updated, so re-running it right now would fail its own `assert p.exists()` checks. Also consider adding real markdown cells (see §10) since the header comment claims "code-only... per project convention" but the sibling notebook's generator clearly intended rich markdown that never made it in.

### `notebooks/02_ohashi_resnet50_ldct_iqac.ipynb` (generated by `scripts/build_ohashi_notebook.py`)

- **Purpose:** End-to-end Ohashi-ResNet50 baseline run: config → dataset build → model init → shape/parameter verification → forward-pass smoke test → optimizer/loss setup → **gated** full training → checkpoint load → held-out test-set evaluation → metrics (raw and 5PL-mapped) → predicted-vs-ground-truth plot → persistence of predictions/metrics/config.
- **Current contents:** 23 code cells, **zero markdown cells** — despite the generator script (`build_ohashi_notebook.py`) containing 15 separate calls to an `md(text)` helper with substantial, carefully-written markdown content (objective, scope guardrails, per-section explanations, a documented known deviation from the task brief, a final experiment summary). **That markdown is silently discarded**: `def md(text): pass` — the function is a no-op by construction (unlike the EDA notebook's generator, which never calls `md()` at all and openly states the code-only convention in its own module docstring). This looks unintentional rather than a deliberate "code-only" choice for this specific notebook, since so much narrative content was clearly written to be shown.
- **Duplicate code against `src/`:** None found. Every model/data/training/evaluation call is an import from `ct_iqa.*` (`OhashiResNet50`, `LDCTIQACDataset`, `build_dataloaders`, `build_test_dataloader`, `build_optimizer`, `build_loss`, `train`, `evaluate_loader`, `compute_metrics`, `compute_metrics_with_logistic_mapping`, `set_seed`). Rule 1 compliance confirmed.
- **Does it define a model?** No — it instantiates `OhashiResNet50(dropout_p=config.dropout_p, in_channels=1)`, correctly matching Rule 1's example verbatim.
- **Does it perform training?** Yes — cell 16 runs the real `train()` loop, currently with `RUN_FULL_TRAINING = True` hardcoded (the generator's own markdown, discarded per above, says this is "gated... NOT run automatically" and should default to requiring a manual flip to `True` — the code and the (unrendered) documentation of the code already disagree with each other).
- **Should it be split?** **Yes.** It currently covers two of the four target notebook stages (`03_training/` and `04_evaluation/`) in one file. See the migration table (§3) for the exact cell-range split.
- **Target notebook(s):** `notebooks/03_training/01_resnet50_baseline.ipynb` + `notebooks/04_evaluation/01_test_set_evaluation.ipynb`.
- **Required changes:** Fix the four stale dataset path defaults (inherited from `ExperimentConfig`, see §3/§10); split into two notebooks at the checkpoint-save/checkpoint-load boundary; restore the markdown narrative as real markdown cells (or deliberately move that narrative into `docs/replication/` instead, per Rule 5 — either is acceptable, but silently dropping it is not); reconcile the `RUN_FULL_TRAINING` flag's actual default with what the (currently invisible) documentation claims about it.

### Naming (Rule 7)

Both existing notebooks already have descriptive, purpose-specific names (`01_eda_ldct_iqac`, `02_ohashi_resnet50_ldct_iqac`) — no `test.ipynb`/`final.ipynb`/`working.ipynb`-style names exist. The only naming problem is that the current flat numbering (`01`, `02`) does not encode *stage* (exploration vs. training vs. evaluation) the way the target's directory-per-stage + notebook-per-purpose scheme does, and one file's numbering already required renumbering once before (per the git log: "Add EDA notebook as 01, renumber Ohashi baseline notebook to 02" — commit `54cbbb7`), which is exactly the kind of flat-numbering churn the target's stage-directory scheme is designed to avoid.

## 6. Data Pipeline Audit

Tracing `dataset → preprocessing → degradation → labeling → split → loader → training` against what actually exists:

```
data/train/image/*.tif, data/train/train.json   (raw images + human labels)
data/test/images/*.tiff, data/test/test.json    (raw images + human labels)
        │
        ▼
LDCTIQACDataset.__init__   (src/ct_iqa/data/ldct_iqac.py)
    - loads + validates JSON labels
    - lists + validates image files
    - raises loudly on any image/label mismatch, malformed JSON, wrong schema
        │
        ▼
LDCTIQACDataset.__getitem__
    - PIL open (mode 'F' enforced)          <- preprocessing: format validation
    - _center_crop(array, image_size, ...)  <- preprocessing: crop (NOT resize)
    - 2.0 * array - 1.0                     <- preprocessing: RadImageNet [-1,1] normalize
        │
        ▼
build_dataloaders(config)   (src/ct_iqa/training/trainer.py)
    - LDCTIQACDataset(train_image_dir, train_json_path)
    - torch.utils.data.random_split(..., generator=seeded)   <- split
    - DataLoader(train_subset), DataLoader(val_subset)        <- loader
        │
        ▼
train(model, train_loader, val_loader, config)   (src/ct_iqa/training/trainer.py)
```

**Findings:**

- **Degradation and labeling (VIF) stages do not exist** — see §4 for why this is a documented scope decision (LDCT-IQAC already provides real quality-varied images with human scores) rather than an oversight, and why it must be written down in `docs/replication/deviations.md` rather than left implicit.
- **Preprocessing is real and correct, but not modularized** — crop and normalize logic lives inline inside `Dataset.__getitem__` rather than as standalone functions in a `preprocessing/` package. This makes the transform steps untestable in isolation from the `Dataset` class (the existing tests in `tests/test_dataset.py` do test crop/normalize behavior, but only indirectly through `ds[0]`).
- **Split logic lives inside `training/trainer.py`, not `data/`** — `build_dataloaders`'s `random_split` call is dataset-splitting logic (target: `data/splits.py`) that currently sits in the training package because that's where a `DataLoader` is also needed. This is the clearest concrete instance of the `training/trainer.py` decomposition recommended in §3.
- **No `interim/` or `processed/` stage is populated or needed today** — since preprocessing happens on-the-fly in `__getitem__` rather than being precomputed to disk, `data/raw/` is the only populated stage. This is fine as-is; `interim/`/`processed/` should stay empty (with `.gitkeep`, consistent with the existing `.gitignore`'s `!data/**/.gitkeep` allow-list) until/unless a precompute step is introduced.
- **Validation exists but isn't named `validation.py`** — the image/label consistency checks (`_load_labels`, `_list_image_files`, and the mismatch-detection logic in `LDCTIQACDataset.__init__`) are exactly what the target's `data/validation.py` is for; they're currently correct but embedded in `ldct_iqac.py` rather than factored out. Low priority to extract given the dataset-specific validation is tightly coupled to `LDCTIQACDataset`'s own construction.

## 7. Training Pipeline Audit

Tracing `config → dataset → model → optimizer → loss → training → checkpoint → evaluation`:

```
ExperimentConfig                              (src/ct_iqa/config.py)
    - seed, batch_size, epochs, lr, optimizer="adam", loss="mse", image_size
    - dropout_p: REQUIRED, no default (paper doesn't specify it)
    - train/test image_dir + json_path        <- ⚠ stale defaults, see §10
    - val_fraction, radimagenet_weights_path, checkpoint_dir, device
        │
        ▼
build_dataloaders(config) / build_test_dataloader(config)   (trainer.py)
        │
        ▼
OhashiResNet50(dropout_p=config.dropout_p, in_channels=1)   (models/ohashi_resnet50.py)
    optionally: model.load_radimagenet_weights(config.radimagenet_weights_path)
        │
        ▼
build_optimizer(model, config)  -> torch.optim.Adam            (trainer.py)
build_loss(config)              -> nn.MSELoss                  (trainer.py)
        │
        ▼
train(model, train_loader, val_loader, config)                 (trainer.py)
    for epoch in range(config.epochs):
        train_one_step(...) per batch     <- normalized [0,1] target space
        evaluate_loader(..., val_loader)  <- val loss, denormalized preds/targets
        if val_loss improves: torch.save(model.state_dict(), checkpoint_dir/best.pt)
        │
        ▼
(notebook) evaluate_loader(model, test_loader, criterion)       <- held-out test set
compute_metrics(...) / compute_metrics_with_logistic_mapping(...)   (evaluation/metrics.py)
```

**Findings:**

- The pipeline is coherent end-to-end and every stage is exercised by at least one test (`tests/test_training_step.py` for one optimizer step, `tests/test_model.py` for the model, `tests/test_metrics.py` for evaluation) or by the notebook's smoke-test cells.
- `build_optimizer`/`build_loss` both deliberately **reject** anything other than `"adam"`/`"mse"` (raising `ValueError`) rather than silently supporting arbitrary strings — appropriately strict given only the Ohashi-specified configuration is implemented; this is worth preserving behavior-for-behavior when these functions move to `training/optimizers.py`/`training/losses.py`.
- Checkpoint selection is "best-val-loss so far", saved every time it improves — correct and simple, but currently has no `checkpointing.py` home and no eventual per-run `experiments/NNN_.../` directory to save into (checkpoints currently go to a bare `checkpoint_dir` string like `"checkpoints/ohashi_resnet50"`, which is itself a path convention that predates — and conflicts with — the target's `weights/checkpoints/` + `experiments/NNN_name/` structure). This needs a decision during migration: does `config.checkpoint_dir` point into `weights/checkpoints/<run>/` or `experiments/<NNN_name>/checkpoints/`? The target tree implies the latter (each `experiments/` entry is presumably self-contained), but nothing in the current code assumes any particular one.
- `ExperimentConfig.checkpoint_dir` default (`"checkpoints/ohashi_resnet50"`) doesn't match **any** target-tree location — not `weights/checkpoints/`, not `experiments/`. This is a genuine open design question for the migration, not just a path rename, and should be resolved explicitly (see §14/§16) rather than guessed at silently.

## 8. Evaluation Pipeline Audit

Tracing `predictions → metrics → calibration → results`:

```
evaluate_loader(model, test_loader, criterion, device)   (trainer.py)
    -> avg_loss, raw_predictions (denormalized), raw_targets
        │
        ▼
compute_metrics(y_true, y_pred)                (evaluation/metrics.py)
    -> {mse, plcc, srocc, krocc}   (raw score space, no logistic mapping)
        │
        ▼ (optional, explicit opt-in)
five_parameter_logistic_fit(y_pred, y_true)    (evaluation/metrics.py)
compute_metrics_with_logistic_mapping(...)     (evaluation/metrics.py)
    -> {mse, plcc, srocc, krocc}   (after 5PL mapping onto ground-truth scale)
        │
        ▼
(notebook, currently) predictions_record -> checkpoint_dir/test_predictions.json
                       metrics_record     -> checkpoint_dir/test_metrics.json
                       config             -> checkpoint_dir/config.json
```

**Findings:**

- Raw-space and logistic-mapped metrics are correctly kept separate and never silently conflated (`compute_metrics` never internally calls the logistic path) — this matches the module's own stated design intent and Rule 5's spirit (don't blur two methodologically distinct things into one).
- There is **no `results/` directory at all** — predictions, metrics, and config are currently written into the training `checkpoint_dir` (e.g., `checkpoints/ohashi_resnet50_random_init_exp01/test_predictions.json`), which mixes model weights and evaluation outputs in one folder. The target's `results/{figures,tables,predictions,metrics}/` split does not exist yet and nothing currently writes to it.
- No figures are currently saved to disk — the notebook calls `plt.show()` for both the loss curve and the predicted-vs-ground-truth scatter plot, so they only exist inside the notebook's own output cells, not as standalone files under `results/figures/`.
- `calibration.py`'s content (the 5PL fit) is currently in `metrics.py` — see §3/§4.

## 9. Documentation Audit

- **`README.md` does not exist.** There is no top-level entry point describing what this repository is, how to set it up, or how to run anything. This is the single most impactful missing document for "understandable to another developer without reverse-engineering."
- **`LICENSE` does not exist.**
- **`docs/` contains exactly one file** (`git_worktree_audit.md`, from the prior task) — none of `docs/README.md`, `docs/paper/*`, `docs/replication/*`, or `docs/decisions/*` exist.
- **What currently substitutes for documentation is scattered across code docstrings** — and it is good, detailed material: `ldct_iqac.py`'s module docstring records the full dataset audit (file counts, PIL mode, pixel range, score range, TIFF-not-PNG deviation from a task brief); `ohashi_resnet50.py`'s module docstring records exactly what the paper specifies vs. what is an explicit implementation choice (dropout probability, channel replication); `resnet50.py`'s module docstring records a specific, deliberate architectural correction (stride placement on the 1x1 vs 3x3 conv, to match Keras/RadImageNet rather than torchvision) with its rationale; `convert_radimagenet_weights.py`'s module docstring records the exact Keras→PyTorch conversion math (bias-folding, kernel transpose). **All of this is exactly docs/paper/ vs docs/replication/ material, just not yet extracted into `docs/`.** Specifically:
  - Belongs in `docs/paper/architecture.md`: "backbone ResNet50, input 224×224, RadImageNet pretraining, Dropout→Dense→Sigmoid head, Sigmoid replaces Softmax" (what the paper states).
  - Belongs in `docs/replication/deviations.md`: dropout probability not specified by the paper (0.5 chosen here); grayscale→3-channel replication; TIFF-not-PNG; ResNet "v1" stride placement chosen specifically to match RadImageNet's Keras weights (a deviation from the more common torchvision "v1.5" convention, done *for* faithful replication, not despite it); no real RadImageNet weights currently available/verified, so the current baseline runs random-init and is reported as such; LDCT-IQAC's radiologist quality score used as the regression target instead of the paper's VIF metric; no synthetic degradation pipeline, because LDCT-IQAC's images already vary in real quality.
  - Belongs in `docs/replication/dataset_adaptation.md`: LDCT-IQAC vs. the paper's own dataset — different, and this must never be implied to be the same dataset (Rule 6). The current docstrings already get this right in spirit (`ldct_iqac.py` is explicit that its score is "NOT the VIF metric... in the original Ohashi paper") but this fact currently lives only in a Python docstring, invisible to a reader who doesn't open that specific file.
  - Belongs in `docs/replication/reproducibility.md`: seeding (`utils/seed.py`), the seeded `random_split` for the validation carve-out, `ExperimentConfig.save`/`.load` for recording exact run configuration.
- **Nothing currently documents the RadImageNet weight *acquisition* process end-to-end** in one place — `weights/README.md` covers "where the .h5 comes from and that it needs conversion," `convert_radimagenet_weights.py`'s docstring covers "how the conversion math works," and `ohashi_resnet50.py`'s docstring covers "how to use `scripts/convert_radimagenet_weights.py` and why weights are never auto-downloaded" — three different files each holding one third of a single procedure a new contributor needs.
- **Nothing describes the LDCT-IQAC dataset's provenance/license/access terms** outside of a code comment — worth a `docs/replication/dataset_adaptation.md` section, distinct from the `.gitignore`'s prose about the *other* dataset in this repo's ecosystem (`data/external/` — the PNG/NGP pulmonary-nodule dataset, which is **entirely unrelated to the current Ohashi/LDCT-IQAC work** but whose `.gitignore` rules and warning comments are still present and correct; nothing currently explains in `docs/` why two unrelated datasets' safety rules coexist in one `.gitignore`).

## 10. Import/Dependency Problems

1. **Broken dataset-path defaults (currently live, not just a migration risk).** `ExperimentConfig`'s four dataset path defaults (`data/training/image`, `data/training/train.json`, `data/testing/images`, `data/testing/test.json`) reference directories that stopped existing the moment commit `730bb2c` renamed `data/training/`→`data/train/` and `data/testing/`→`data/test/`. Any code that constructs `ExperimentConfig()` with defaults (both notebooks do exactly this) will now fail at the `assert p.exists()` checks or at `LDCTIQACDataset.__init__`'s `FileNotFoundError`. **This needs fixing regardless of whether the broader architecture migration happens at all** — it is an existing regression, not a future one.
2. **Same stale-path problem recurs in 7 files total:** `src/ct_iqa/config.py` (defaults), `src/ct_iqa/data/ldct_iqac.py` (docstring only, not functional), `src/ct_iqa/training/trainer.py` (docstring only), both notebook `.ipynb` files (functional — hardcoded path constants), and both notebook-generator scripts (functional — the source of the notebooks' hardcoded constants). Fixing only `config.py` is not sufficient; the notebooks/generators have their own independent hardcoded copies of the same paths and must each be fixed separately (there is no single source of truth for these paths today — a `configs/dataset.yaml` would fix this class of bug structurally, not just this one instance of it).
3. **Fragile package-style import from `scripts/`.** `tests/test_radimagenet_weights.py` does `from scripts.convert_radimagenet_weights import convert`, which only resolves because `scripts/` happens to be importable as a package from the repository root under pytest's default rootdir insertion behavior — there is no `scripts/__init__.py`, so this relies on Python's implicit namespace package support and pytest's `sys.path` manipulation rather than an explicit, declared package structure. This is exactly the kind of import that "becomes invalid after restructuring" if `scripts/` is renamed/moved/removed (which §3 recommends) — the two tests gated behind `requires_h5` would then fail with an `ImportError` at collection time instead of a clean skip, the first time someone actually has the `.h5` file locally to trigger them.
4. **Repo-root-relative imports depend on notebook working directory.** Both notebook generators insert repo-root-resolution boilerplate at the top (`_cwd = Path.cwd(); ... os.chdir(...)`, `sys.path.insert(0, str(_repo_root / "src"))`) specifically because `ct_iqa` is not installed as a package (no `pyproject.toml`, so no `pip install -e .`) — every notebook must independently guess its way to the repo root and manually splice `src/` onto `sys.path`. This is fragile (two independent copies of nearly-identical boilerplate, each guessing root via `(cwd / "data").exists()`) and will need to be redone for every new notebook created under the new `notebooks/0N_stage/` subdirectories, since one more directory level changes the relative path math. **Introducing `pyproject.toml` with an installable `ct_iqa` package (`pip install -e .`) would eliminate this whole class of fragility** — every notebook could just `import ct_iqa` without any `sys.path` surgery, regardless of which subdirectory it lives in.
5. **No circular dependencies found.** The import graph is a clean DAG: `training.trainer` → `data.ldct_iqac` + `config`; `models.ohashi_resnet50` → `models.resnet50`; `evaluation.metrics` has no internal `ct_iqa` dependencies; `utils.seed` has no internal dependencies. This should be preserved during the `training/trainer.py` split (§3) — e.g. `training/optimizers.py` and `training/losses.py` must not import back from `trainer.py`.
6. **No unused modules found** — every existing `src/ct_iqa/*` file is imported by at least one test or one notebook generator.
7. **`scripts/build_ohashi_notebook.py`'s own internal docstring is stale relative to its own code**, independent of the wider migration: its module docstring says it generates `notebooks/01_ohashi_resnet50_ldct_iqac.ipynb`, but the actual `out_path` in the same file is `notebooks/02_ohashi_resnet50_ldct_iqac.ipynb` (left over from the renumbering in commit `54cbbb7`, which evidently updated the write path but not the docstring above it).
8. **The `md()` no-op in `build_ohashi_notebook.py`** (§5) is not strictly an import/dependency problem, but is exactly the kind of "silent behavior divergence between what the code appears to do and what it actually does" that this audit's import/dependency pass is meant to catch — flagged here since it will resurface as confusing merge/diff noise if someone "fixes" it inside an otherwise-unrelated migration commit rather than as its own deliberate change.

## 11. Naming Problems

- **`data/train/` vs. `data/training/`, `data/test/` vs. `data/testing/`** — the directories were renamed once already (commit `730bb2c`) without updating any consumer, so the *current* on-disk names and the *code's* names for the same thing now disagree (see §10). Any further renaming (e.g., the `data/raw/ldct_iqac/...` scheme proposed in §3) must update every consumer in the same change, not incrementally.
- **`scripts/build_ohashi_notebook.py` outputs `notebooks/02_ohashi_resnet50_ldct_iqac.ipynb`, but its own docstring says `01_...`** — see §10, item 7.
- **`checkpoint_dir="checkpoints/ohashi_resnet50_random_init_exp01"` (set inline in the notebook) vs. `checkpoint_dir: str = "checkpoints/ohashi_resnet50"` (the `ExperimentConfig` default)** — two different naming conventions for the same concept already coexist (one encodes the weight-init strategy and an experiment number, the other doesn't), and neither matches any target-tree location (`weights/checkpoints/` or `experiments/NNN_name/`). This is the same open question flagged in §7.
- **No `test.ipynb`/`final.ipynb`/`working.ipynb`/`experiment_new.ipynb`-style names exist anywhere** — Rule 7 is already being followed for notebook naming; the only issue is stage-encoding (numbering resets per top-level file rather than being scoped to a stage directory), not vague/non-descriptive names.
- **`ldct_iqac.py` vs. a hypothetical generic `dataset.py`** — addressed in §3; recommend keeping the specific name.

## 12. Duplicate Functionality

- **None found between notebooks and `src/`** — confirmed by inspecting both notebook generator scripts cell-by-cell; every model/data/training/evaluation operation is an import and call into `ct_iqa.*`, never a re-implementation. Rule 1 is already satisfied.
- **None found between `scripts/` and `src/`**, with one partial exception: `convert_radimagenet_weights.py` is not *duplicated* anywhere, but it is *misplaced* — it's genuine, non-duplicated, reusable logic that happens to sit outside the package that already imports it as if it were inside (§3, §10 item 3).
- **Markdown content exists in exactly one place and is then discarded, not duplicated** — the `build_ohashi_notebook.py` `md()` calls are the *only* copy of that narrative text anywhere in the repository; the bug is that it never reaches the notebook, not that it's duplicated elsewhere. Worth noting because "duplicate functionality" audits sometimes miss the inverse failure mode (content that should be duplicated — once in the notebook, once in `docs/` — but currently exists nowhere at all after generation).
- **Two independent copies of repo-root-resolution boilerplate** (one per notebook generator) — not exactly "duplicate functionality" in the Rule-8 sense (it's boilerplate, not a reusable unit someone forgot to extract), but is the kind of copy-pasted logic that a `pyproject.toml`-installable package would eliminate entirely (§10 item 4) rather than needing a third, fourth, fifth copy every time a new notebook is added.

## 13. Generated/Temporary Files

- **`__pycache__/` (7 directories: `scripts/`, `src/ct_iqa/`, `src/ct_iqa/data/`, `src/ct_iqa/evaluation/`, `src/ct_iqa/models/`, `src/ct_iqa/training/`, `src/ct_iqa/utils/`, `tests/`)** — Python bytecode cache, correctly gitignored, correctly untracked. No action needed; excluded from the migration table above since there is nothing to "migrate" (they regenerate wherever the corresponding `.py` files end up).
- **`notebooks/*.ipynb` are themselves generated artifacts** (from `scripts/build_*_notebook.py`), not hand-authored — worth calling out explicitly since it changes how "migrating a notebook" should actually be done: the correct migration mechanism is editing and re-running the generator script, not hand-editing the `.ipynb` JSON, or the generator and the notebook will drift out of sync again (as they already have once, via the stale dataset paths).
- **No `.ipynb_checkpoints/`, no stray `.tmp`/`.bak`/`.swp` files, no OS metadata (`.DS_Store`, `Thumbs.db`), no IDE metadata (`.vscode/`, `.idea/`) found anywhere in the tree** — confirmed during the prior Git-hygiene audit and re-confirmed here; nothing in this category needs removal.
- **`checkpoints/ohashi_resnet50*/` referenced by config/notebook defaults does not currently exist on disk** (no training run has been executed and persisted in this checked-out copy) — not a file to remove, just a note that the "generated experiment output" location is currently empty and, per §7, doesn't have an agreed-upon home in the target tree yet (`weights/checkpoints/` vs. `experiments/NNN_.../`).

## 14. Migration Plan

An ordered plan, sequenced so that nothing is left in a broken, half-migrated state at any intermediate step (each phase should leave `pytest` passing before starting the next):

**PHASE 0 — Fix the live regression (independent of everything else)**
Fix `ExperimentConfig`'s four stale dataset-path defaults and the matching hardcoded paths in both notebook generator scripts, so the existing pipeline is at least runnable again against `data/train/`/`data/test/` as they exist today. This should happen *before* any structural migration, so that "does the pipeline still work" is answerable at every subsequent phase.

**PHASE 1 — Create scaffolding**
Add `README.md`, `LICENSE`, `pyproject.toml` (with an installable `ct_iqa` package, `[project.scripts]`/`pip install -e .` so notebooks can drop their `sys.path` boilerplate), `.python-version`, and the new empty directories (`configs/`, `docs/paper/`, `docs/replication/`, `docs/decisions/`, `experiments/`, `results/{figures,tables,predictions,metrics}/`, `weights/pretrained/radimagenet/resnet50/`, `weights/checkpoints/`, `notebooks/01_exploration/` … `04_evaluation/`, `tests/unit/`, `tests/integration/`, `src/ct_iqa/{preprocessing,degradation,labeling}/`). No file moves yet.

**PHASE 2 — Move data**
Relocate `data/train/`→`data/raw/ldct_iqac/train/`, `data/test/`→`data/raw/ldct_iqac/test/`, and the two label JSONs into `data/labels/ldct_iqac/`. Update `.gitignore` accordingly (this can simplify, not just rename, the dataset-safety rules — `data/raw/**` already covers the new location generically). Update `ExperimentConfig` defaults to match.

**PHASE 3 — Move/extract architecture and reusable implementation**
Move `scripts/convert_radimagenet_weights.py` → `src/ct_iqa/models/radimagenet_weights.py`. Split `training/trainer.py` into `trainer.py` (loop only) + `data/loader.py` + `data/splits.py` + `training/optimizers.py` + `training/losses.py` + `training/checkpointing.py`. Optionally split `evaluation/metrics.py` into `metrics.py`/`correlation.py`/`calibration.py`. Extract `preprocessing/crop.py` and `preprocessing/normalize.py` from `data/ldct_iqac.py`'s inline crop/normalize logic, with `ldct_iqac.py` calling into them rather than reimplementing.

**PHASE 4 — Rebuild notebooks via their generators**
Update and split `scripts/build_eda_notebook.py` → `tools/build_eda_notebook.py` (writes `notebooks/01_exploration/01_dataset_eda.ipynb`); split `scripts/build_ohashi_notebook.py` → `tools/build_training_notebook.py` + `tools/build_evaluation_notebook.py` (writing the two notebooks described in §3/§5). Fix the `md()` no-op while doing this rewrite. Regenerate both/all three notebooks from the fixed generators rather than hand-editing `.ipynb` files.

**PHASE 5 — Fix imports**
Update every import touched by Phases 2–4: the moved `convert_radimagenet_weights`/`radimagenet_weights` import in `tests/test_radimagenet_weights.py`; any cross-references inside `ohashi_resnet50.py`'s docstring; `weights/README.md`'s path references; the evaluation notebook's checkpoint-loading path (it must now load from wherever the training notebook saved to, since they're separate notebooks/kernels after the split).

**PHASE 6 — Rebuild documentation**
Populate `docs/paper/` (paper facts only, extracted from existing docstrings — architecture, preprocessing, training, evaluation as the paper states them) and `docs/replication/` (this project's deviations and adaptations — dropout probability, TIFF vs PNG, LDCT-IQAC vs. the paper's dataset, no-degradation/no-VIF rationale, random-init-until-real-weights status, ResNet stride-placement correction). Write `docs/decisions/README.md` as a running decisions log going forward. Write the root `README.md` and `docs/README.md`.

**PHASE 7 — Update tests**
Move `tests/*.py` into `tests/unit/` (all five current files) and split `tests/test_radimagenet_weights.py`'s integration-gated tests into `tests/integration/test_radimagenet_weights_integration.py`. Configure `pyproject.toml`'s `[tool.pytest.ini_options]` `testpaths = ["tests"]` so both subdirectories are discovered.

**PHASE 8 — Run complete validation**
`pytest` (full suite, both unit and integration — integration tests will still skip without local weight files, which is expected and correct), re-run all three (post-split) notebooks end-to-end via `jupyter nbconvert --execute` or equivalent, confirm `git status` shows only the intended changes, confirm no dataset/weight binaries were accidentally staged (`git status` should stay silent on `data/raw/`, `data/labels/` non-JSON files, and `weights/pretrained/`).

## 15. Risk Assessment

- **Highest risk: silently losing the `weights/pretrained/*.h5`/`*.pt` files or the `data/train/image/`/`data/test/images/` TIFFs during a directory move**, since none of them are git-tracked — a move-then-delete-source mistake (rather than a true `git mv`/rename) would be unrecoverable via git. Any actual migration step touching these must be a verified copy-then-delete, or a plain filesystem move confirmed present at the destination before the source is removed — never a delete-and-recreate.
- **Breaking the two `@requires_h5`/`@requires_converted_pt`-gated integration tests silently** — because they're skip-gated on file presence, an import error introduced by moving `convert_radimagenet_weights.py` would only surface on a machine that actually has the `.h5` file locally (Phase 5 must be verified with the weight files present, not just on a clean checkout where the tests would just skip and hide the breakage).
- **Notebook/generator drift recurring** — this has already happened once (dataset paths) and once more subtly (the `md()` no-op). Any migration must treat the generator script as the single source of truth and regenerate, not hand-patch the `.ipynb` — hand-patching would reintroduce exactly the "generator and output disagree" failure mode currently present.
- **`checkpoint_dir`/experiment-output location is genuinely undecided** (§7) — migrating without first deciding whether runs live under `weights/checkpoints/<run>/` or `experiments/<NNN_name>/` risks a second, avoidable renaming churn shortly after this migration completes.
- **`pyproject.toml` introduction changes how `ct_iqa` is imported everywhere** (from manual `sys.path.insert` to an installed package) — this is a high-value fix (§10 item 4) but touches every notebook and potentially CI/test invocation; it should be validated by running the full test suite and both notebooks *before* removing the old `sys.path` boilerplate, not after.
- **Low risk, but non-zero:** the `.gitignore` simplification implied by moving to `data/raw/ldct_iqac/...` (letting the generic `data/raw/**` rule take over from the current `data/train/`/`data/test/`-specific comments) must be double-checked against the existing dataset-safety rules for the *other* dataset in this `.gitignore` (`data/external/`, the PNG/NGP dataset) to make sure a broad edit doesn't accidentally loosen that unrelated dataset's protection.
- **No risk to already-committed history** — every file discussed in this audit that is tracked by git (`.gitignore`, the two label JSONs, both notebooks, all `src/`, all `tests/`, `weights/README.md`) is small and cheaply re-committable; nothing here requires history rewriting, and none of the recommendations in this document require any Git operation beyond ordinary `git mv`/`git add` in a future migration commit.

## 16. Final Recommendation

The proposed target structure **is internally consistent** and is a reasonable, standard research-repository layout for where this project needs to go — the `paper/` vs. `replication/` documentation split in particular directly solves a real, already-visible problem (valuable paper-vs-implementation distinctions currently exist only inside code docstrings, e.g. "NOT the VIF metric... in the original Ohashi paper", where a future contributor is unlikely to find them).

Two things are worth the user's explicit attention before migrating, since they are not simply "move file A to path B" and this audit should not paper over that:

1. **`degradation/` and `labeling/vif.py` will remain empty after migration**, and should stay empty, because this replication's dataset (LDCT-IQAC) doesn't need synthetic degradation or VIF labels — it already has real quality variation and human labels. This is a legitimate, already-well-documented (in code) project decision, not a gap to fill in during the migration. The migration's job here is purely to make that decision *visible* in `docs/replication/deviations.md`, not to build unused code to fill the directories.
2. **The `checkpoint_dir`/`experiments/`/`weights/checkpoints/` relationship is undefined** (§7, §15) and should be decided *before* Phase 3/7 of the migration, since it affects `ExperimentConfig`'s default and every notebook that reads/writes it.

Everything else in this audit — the `training/trainer.py` decomposition, the `scripts/`→`src/`+`tools/` split, the notebook split, the stale-path fix, the docs extraction — is a straightforward, low-risk mechanical migration once those two decisions are made. No blocking architectural contradictions were found between the current codebase and the proposed target structure.

This audit produced no file changes. Awaiting review before any migration work begins.
