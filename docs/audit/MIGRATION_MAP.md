# Migration map and rollback

Actions: KEEP (untouched), MOVE (path changed, history kept with `git mv`), ADD (new), EDIT (per-file edit),
WRAP (old path kept as a thin re-export), LEGACY (kept unchanged for historical reproducibility).
No global search-and-replace was used; every edit below was made file by file.

## Source (`src/ct_iqa/`)

| Original | Action | New / current | Verification |
|---|---|---|---|
| `config.py` | MOVE | `config/legacy.py`; `config/__init__.py` re-exports `ExperimentConfig`, `DEFAULT_IMAGE_SIZE`, `SUPPORTED_ARCHITECTURES`, `_CONFIG_FIELD_NAMES`, `_yaml_default` | existing `test_config.py` and all importers pass |
| - | ADD | `config/schema.py`, `config/loader.py` | `tests/unit/test_study_config.py` |
| `models/resnet50.py` | WRAP | re-exports from `models/backbone.py` | legacy tests pass; `bn_eps` default equals the old PyTorch default |
| - | ADD | `models/backbone.py` (copy of resnet50.py + `bn_eps`), `heads.py`, `registry.py`, `weights.py` | `test_registry_and_bn_eps.py`, regression tests |
| `models/ohashi_resnet50.py` | LEGACY | unchanged | `test_model.py`, `test_radimagenet_weights.py`; equivalence test vs registry model |
| `models/radimagenet_weights.py` | KEEP | re-exported as `models.weights.convert` | existing tests |
| `data/ldct_iqac.py`, `preprocessing/*`, `utils/seed.py` | KEEP | - | existing tests |
| `data/splits.py` | EDIT (append) | adds three-way split and manifest functions; `train_val_split` untouched | `test_three_way_split.py`, `test_data_loader_and_splits.py` |
| `data/loader.py` | EDIT (append) | adds study loaders; historical functions untouched | synthetic end-to-end tests |
| - | ADD | `data/sampling.py` | `test_losses.py` |
| `training/trainer.py`, `training/losses.py`, `training/optimizers.py` (historical function) | LEGACY / KEEP | `optimizers.py` gains `build_study_optimizer` (append) | existing tests |
| `training/checkpointing.py` | EDIT (append) | adds atomic `save_run_checkpoint`, `load_run_checkpoint` | synthetic end-to-end tests |
| - | ADD | `training/engine.py`, `training/selection.py`, `losses/{types,mse,ranking,gaussian_nll,registry}.py` | `test_losses.py`, end-to-end |
| `evaluation/{correlation,calibration}.py` | KEEP | - | existing tests |
| `evaluation/metrics.py` | EDIT (append) | adds `mae`, `rmse`, `compute_point_metrics` | existing + new tests |
| - | ADD | `evaluation/{intervals,bootstrap}.py`, `experiment/{run,metadata}.py`, `cli.py` | `test_intervals.py`, `test_bootstrap.py`, end-to-end |

Not created from the proposed target tree: `utils/reproducibility.py` and `utils/logging.py` (existing `utils/seed.py` and
standard `logging` suffice); `calibration/` as a separate package (conformal calibration is in `evaluation/intervals.py`);
`models/weights.py` does not absorb `radimagenet_weights.py` (re-exported instead).

## Other files

| Path | Action | Note |
|---|---|---|
| `configs/{dataset,model,preprocessing,training}.yaml` | LEGACY | used by `ExperimentConfig.from_yaml_files` for historical tools |
| `configs/base.yaml`, `configs/experiments/*.yaml` | ADD | study v2 |
| `experiments/001_*`, `002_*`, `003_*`, `results/*`, `data/labels/*`, `docs/paper/*`, `docs/replication/*`, notebooks, `tools/build_*`, `tools/evaluate_baseline_test_set.py` | KEEP | unchanged |
| `experiments/001_resnet50_baseline/config.yaml` | EDIT | header note only, values unchanged |
| `.gitignore` | EDIT | header text only; no rule removed or weakened |
| `pyproject.toml` | EDIT | description; console script `ct-iqa` |
| `README.md`, `docs/README.md` | EDIT | study-v2 section, index; dataset-availability wording corrected |
| `tools/create_split_manifest.py`, `tools/inspect_checkpoint_compatibility.py` | ADD | |
| `docs/internal/*` | KEEP | historical snapshots, some paths stale (labeled in docs index) |
| `src/ct_iqa.egg-info/`, `__pycache__/`, `.benchmarks/` | REMOVE | generated/empty, see CLEANUP_MANIFEST |

Planned but deferred: the old notebooks and `tools/build_*` generators are kept because replacing them with equivalents
requires re-running training and evaluation, which was not done. Removal of `OhashiResNet50` waits for their migration.

## Rollback

1. Everything is on branch `study-v2-rebuild`; `main` (commit `821461ee24c7fea4d6c1b787c626cbde076ee998`) is untouched.
   `git switch main` restores the original tracked tree (uncommitted work on the branch stays in the working tree; use
   `git stash -u` first if you want it kept).
2. To discard the rebuild: `git switch main && git branch -D study-v2-rebuild` (only after confirming nothing is wanted).
3. Ignored artifacts were never modified. If one is damaged, restore from `C:\Users\surya\ct_iqa_backup` and verify with
   `sha256sum -c docs/audit/BACKUP_MANIFEST.sha256` run from that directory (hashes listed there are of the originals).
4. The `git mv` of `config.py` is a staged rename; `git mv src/ct_iqa/config/legacy.py src/ct_iqa/config.py` after removing
   the new `config/` files reverses it.
