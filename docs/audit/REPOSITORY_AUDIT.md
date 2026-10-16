# Repository audit (initial, 2026-10-16)

Method: read-only inspection of code, configs, docs, run artifacts and notebook *saved outputs*; execution was inferred only from artifacts that exist on disk, never from
the presence of training code. Baseline: branch `main`, HEAD `6c3e3d6`, uncommitted changes limited to notebooks 00-02 (executed outputs added; left untouched).

## 1. What the repository contains
- `src/ct_iqa/`: study-v2 framework (strict YAML config, frozen-split manifest tools, custom ResNet50 backbone with `bn_eps`, regression and Gaussian heads, MSE / MSE+ranking /
  Gaussian-NLL losses, metrics, paired bootstrap, split-conformal intervals, immutable run directories with run manifests, B grid and selection rule, CLI) plus legacy code for 001-003.
- `configs/` (base + experiments A/B/C; legacy YAMLs), `tools/` (split-manifest creator, checkpoint inspector), `tests/` (unit, regression, integration), `docs/` (protocol, ADR-002/003, per-experiment specs,
  limitations), notebooks 00-07, `weights/pretrained/radimagenet/resnet50/` (SHA-256-verified files, gitignored), `data/` (1000 labeled train TIFFs, 300 test TIFFs, label JSONs).
- Environment: project `.venv` is CPU-only torch 2.14; the GPU environment used for this study is the conda env `torch-gpu` (Python 3.10.20, torch 2.5.1+cu121, RTX 4060 Laptop 8 GB).

## 2. What each experiment implements
| Experiment | Implementation | Status before this work |
|---|---|---|
| Historical 001 (baseline) | `OhashiResNet50`, MSE, Adam 1e-3, 900/100 split, 30 epochs, seed 42, CPU, val-PLCC selection, BN eps 1e-3 train / 1e-5 eval | Executed (checkpoint, history, config exist). Test PLCC 0.8667, RMSE 0.5736 (n=300) |
| Historical 002 (LR search) | 4 runs, lr 1e-2..1e-5, validation only | Executed; lr 1e-3 chosen |
| Historical 003 (final training) | all 1000 images, no validation split, final-epoch checkpoint | Executed. Test PLCC 0.8490, RMSE 0.6689 |
| Study v2 A | `resnet50_regression` + MSE, 800/100/100 frozen split, bn_eps 1.001e-5 | Implemented, synthetic-tested; **never run** |
| Study v2 B | A + `MSERankingLoss` (hinge pairwise, in-batch pairs with label gap >= delta), 7-config grid | Implemented, synthetic-tested; **never run** |
| Study v2 C | `resnet50_gaussian` (mean + clamped log-variance) + Gaussian NLL + split conformal | Implemented, synthetic-tested; **never run** |

The study-v2 A/B/C are *new* experiments; the historical 001-003 have different meanings (baseline / LR search / final training) and different protocols. They are not directly comparable.

## 3. What has genuinely been executed (evidence)
- 001-003: `config.json`, `history.json`, checkpoints (SHA-256 recorded in `docs/audit/CHECKPOINT_COMPATIBILITY.json`). Test metrics were read from git history (commit `109f4ba^`); no run manifests or logs exist.
- Notebooks 00-02 have saved outputs (18/18 environment/data checks passed; 02's real-image check skipped). Notebooks 03-07 have **no** executed cells; their run flags are off.
- No study-v2 run existed before this audit (`experiments/study_v2/` held only the manifest).

## 4. Results available and provenance
Only the historical test metrics (above) and validation histories. They come from `README`s and deleted-then-recoverable metric JSONs; the official test set was used by 001 and 003.

## 5. Incomplete, inconsistent, or unverified
1. `normalize.py` and `configs/preprocessing.yaml` claim `2x-1` was "confirmed" from RadImageNet; ADR-002 says only partially verified (PyTorch release); a TF fine-tuning script uses Keras `preprocess_input`. Resolved to the extent possible by [PREPROCESSING.md](../methodology/PREPROCESSING.md) (probe: `2x-1` and caffe-style indistinguishable).
2. Historical READMEs mix 0-based and 1-based epoch numbers; `001/validation_metrics.json` equals final-epoch values, not the selected epoch.
3. Historical BN eps mismatch (train 1e-3, test eval 1e-5); effect unquantified.
4. The test set was examined by 001/003 (and the 001 README says it was untouched during training) - so any new test result is retrospective.
5. Notebook 06 reads the stage-1 B selection record while notebook 04 produces the final stage-2 record. Notebook 06 must be pointed at the final record.
6. The old protocol file listed `config.snapshot.yaml`; the code writes `config.snapshot.json` + `config.source.yaml`. Selection prose said "difference > half-width" while the B doc said "interval excludes zero"; both are now reported ([EXPERIMENT_PROTOCOL](../methodology/EXPERIMENT_PROTOCOL.md) section 5).
7. README's doc map referenced non-existent directories (fixed in the README rewrite).
8. The constant-width conformal comparator for C was pending and unimplemented (added in analysis).
9. **New (this audit): the image-level split leaks anatomy.** All validation/calibration images have a near-identical-anatomy partner in train; test images have none ([DATASET_AND_EDA](../methodology/DATASET_AND_EDA.md)). This was not documented anywhere before and affects the interpretation of every validation metric, bootstrap interval and the conformal exchangeability assumption.

## 6. Authoritative files
Protocol: `docs/methodology/EXPERIMENT_PROTOCOL.md`; split: `experiments/study_v2/protocol/split_manifest.json` (hash verified, never regenerated); configs: `configs/base.yaml` + `configs/experiments/*.yaml`;
run records: `experiments/study_v2/*/runs/*/run_manifest.json` and `run_registry.jsonl`; decisions: ADR-002/003; historical evidence: `experiments/00*_*/`.

## 7. Plan and risks
Execution stages: static tests (done, 176 passed) -> EDA (done) -> preprocessing probe (done) -> protocol freeze (done) -> smoke run (done, 1 epoch, pipeline OK) -> A (seed 42) -> B grid -> selection -> final A/B/C at seeds 1, 2, 3 -> freeze marker -> retrospective test evaluation -> analysis -> README.
Risks: anatomy leakage (optimistic validation, narrow intervals); single image-level split; 3 seeds (low power); B selection reuses the validation set that also picks checkpoints; CUDA nondeterminism; the project working tree is dirty (modified notebooks, new files), which is recorded in every run manifest.
