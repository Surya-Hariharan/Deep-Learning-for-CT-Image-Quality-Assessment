# Initial audit (Stage 0), reconciled with the live repository

Audit date: 2026-10-16. Read-only. Reconciled at the start of Stage 1: the live repository matched the audit
(HEAD `821461ee24c7fea4d6c1b787c626cbde076ee998` on `main`, clean working tree, 137 tracked files, 0 untracked).
No discrepancy between the audit and the live tree was found.

## Baseline

| Item | Value |
|---|---|
| Original branch / commit | `main` / `821461ee24c7fea4d6c1b787c626cbde076ee998` |
| Working branch for the rebuild | `study-v2-rebuild` |
| Stack | Python 3.10, PyTorch >= 2.0, NumPy, SciPy, Pillow, h5py, PyYAML |
| Lint / type-check / CI | none configured at audit time |
| Secrets | no credential-pattern matches in source, configs or docs |

## Ignored artifacts present (not tracked by Git)

| Path | Size | SHA-256 (prefix) |
|---|---|---|
| `experiments/001_resnet50_baseline/checkpoint/best.pt` | 282 MB | `b482931f57b0` |
| `experiments/002_learning_rate_search/lr_1e-2/checkpoint/best.pt` | 282 MB | `2907a0926aa6` |
| `experiments/002_learning_rate_search/lr_1e-3/checkpoint/best.pt` | 282 MB | `068d7f9ba91f` |
| `experiments/002_learning_rate_search/lr_1e-4/checkpoint/best.pt` | 282 MB | `8b20e31fe272` |
| `experiments/002_learning_rate_search/lr_1e-5/checkpoint/best.pt` | 282 MB | `6b04669d221c` |
| `experiments/003_final_training/checkpoint/final.pt` | 282 MB | `07185bae687d` |
| `weights/pretrained/radimagenet/resnet50/RadImageNet-ResNet50_notop.h5` | 95 MB | `1f1d1c9fe6ab` |
| `weights/pretrained/radimagenet/resnet50/radimagenet_resnet50_backbone.pt` | 94 MB | `45cba0d53f1d` |
| `data/raw/ldct_iqac/` (1000 train + 300 test images) | ~1.3 GB | not copied; left in place, never modified |

Full hashes: [BACKUP_MANIFEST.sha256](BACKUP_MANIFEST.sha256) and [BACKUP_MANIFEST.json](BACKUP_MANIFEST.json).

## Findings that drive the rebuild

1. **Runtime coupling.** The only model class is `OhashiResNet50` (`src/ct_iqa/models/ohashi_resnet50.py`).
   `SUPPORTED_ARCHITECTURES` in `src/ct_iqa/config.py` is a validation set, not a factory. About 25 call sites
   (tools, tests, notebooks) construct the class directly.
2. **Hard-coded objective.** `build_loss` accepts only MSE and `build_optimizer` only Adam. `train()` assumes one
   scalar output per sample, which blocks the ranking and Gaussian objectives.
3. **BatchNorm epsilon.** `load_radimagenet_weights` sets every BatchNorm `eps` to 1e-3 as a side effect. `eps`
   is not stored in `state_dict`, so a model rebuilt for evaluation runs at PyTorch's default 1e-5. Training and
   validation used 1e-3; the recorded test evaluation used 1e-5. The impact on historical predictions is
   **unquantified** (no inference was run). Whether 1e-3 matches the source convention is unverified.
4. **Preprocessing evidence.** The `2x - 1` pixel mapping is documented as taken from RadImageNet's reference code,
   but no copy of that evidence is in the repository.
5. **Provenance gaps.** No checksums were recorded for the pretrained weights; checkpoints store config and seed
   but not git SHA, library versions or weights hash.
6. **Split and leakage risk.** The existing 900/100 split is an image-level random split; the labels carry no
   patient identifiers, so patient-level independence cannot be claimed.
7. **Test-set exposure.** The official 300-image test set was used for historical evaluation of experiments 001
   and 003 and for the 001-vs-003 comparison. It is therefore not an untouched confirmatory set.
8. **Overwrite risk.** `save_checkpoint` writes `best.pt` unconditionally; regenerating notebooks from `tools/`
   discards stored outputs.
9. **Stale material.** `.gitignore` header describes a different dataset (PNG/NIfTI); `docs/internal/git_worktree_audit.md`
   is a snapshot of an older state; `experiments/001_resnet50_baseline/config.yaml` records a planned config that
   differs from the run's `config.json`.

## Historical experiments (immutable)

| ID | Purpose | Model | Selection | Test PLCC / SROCC / KROCC / MSE / MAE / RMSE |
|---|---|---|---|---|
| 001 | Baseline, retrained 2026-09-14 | `OhashiResNet50`, RadImageNet init | best val PLCC (epoch index 20) | 0.8667 / 0.8699 / 0.6816 / 0.3291 / 0.4815 / 0.5736 |
| 002 | LR search (validation only) | same | lowest val loss per run | not evaluated on test |
| 003 | Train on all 1000 images | same | final epoch | 0.8490 / 0.8525 / 0.6611 / 0.4474 / 0.5195 / 0.6689 |

These numbers are copied from `results/metrics/` and are not regenerated. The new study uses separate
identifiers (`A_reference_resnet50`, `B_ranking_aware_resnet50`, `C_uncertainty_aware_resnet50`).

The full audit report was delivered in the Stage 0 conversation; this file is its repository-side summary.
