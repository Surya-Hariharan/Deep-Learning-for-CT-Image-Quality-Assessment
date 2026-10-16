# Reproducibility

## Environment

Python 3.10 (`.python-version`), dependencies pinned in `uv.lock`. Setup: `uv sync --extra dev` creates `.venv`
(ignored). Development of this rebuild used torch 2.14.0 / numpy / scipy as resolved by the lock (CPU build on Windows).

## Data and weights (not in Git)

- Images: `data/raw/ldct_iqac/train/image/*.tif` (1000) and `data/raw/ldct_iqac/test/images/*.tiff` (300); labels are
  tracked in `data/labels/ldct_iqac/`.
- RadImageNet weights: obtain `RadImageNet-ResNet50_notop.h5` from the RadImageNet authors, convert with
  `python -m ct_iqa.models.radimagenet_weights`, and verify against `weights/pretrained/radimagenet/resnet50/CHECKSUMS.sha256`:
  `.h5` `1f1d1c9f...d184f159`, `.pt` `45cba0d5...a0f785`. The loader refuses a file whose hash differs from
  `model.pretrained_sha256`.

## Split

`experiments/study_v2/protocol/split_manifest.json` (seed 2026, SHA-256 `be5831c5...0a16`) lists the exact filenames of
each partition. `ct_iqa.data.splits.load_manifest` re-verifies non-overlap, full coverage of the labeled images, and the
hash on every run. The manifest is never regenerated (`tools/create_split_manifest.py` refuses to overwrite).

## Commands (all exist in `ct_iqa.cli`)

```bash
# config and provenance checks only (no training, no inference)
uv run python -m ct_iqa.cli validate-config --config configs/experiments/a_reference.yaml
uv run python -m ct_iqa.cli validate-config --config configs/experiments/b_ranking_aware.yaml
uv run python -m ct_iqa.cli validate-config --config configs/experiments/c_uncertainty_aware.yaml

# training (EXPENSIVE: 30 epochs; requires approval)
uv run python -m ct_iqa.cli train --config configs/experiments/a_reference.yaml --device cuda

# re-evaluate a finished run on validation or calibration
uv run python -m ct_iqa.cli evaluate --run-dir experiments/study_v2/A_reference_resnet50/runs/<run_id> --split validation

# official test set: only after the protocol is frozen; retrospective
uv run python -m ct_iqa.cli final-test --run-dir <run_dir> --acknowledge-retrospective
```

Tests that do not touch real data or checkpoints: `uv run pytest tests/unit tests/regression tests/integration/test_study_synthetic_end_to_end.py`
(the regression suite skips checkpoint-structure tests if the gitignored checkpoints are absent). The older
`tests/integration/test_ldct_iqac_pipeline_integration.py` and `test_radimagenet_weights_integration.py` read real data and
weights and perform one real optimizer step; run them deliberately.

## Run record

Every run directory contains `config.snapshot.json`, `run_manifest.json` (git commit, branch and dirty flag, Python and
library versions, device, seed, split hash, weights hash, selection rule, best epoch with 1-based indexing, artifact
SHA-256), `history.json`, `checkpoints/best.pt`, validation metrics/predictions (and `calibration.json` for C). Run
directories are created with `exist_ok=False`; a run is never overwritten. Resuming an interrupted run is not implemented.

## Sources of nondeterminism

`set_seed` seeds Python, NumPy, torch and CUDA and sets cuDNN deterministic mode, but does not call
`torch.use_deterministic_algorithms`; some CUDA kernels remain nondeterministic. Batch order depends only on
`train.seed` (dedicated shuffle generator). CPU and GPU runs of the same seed are not expected to match. A synthetic test
confirms identical training losses for two CPU runs with the same seed on this machine; that is not a guarantee across
hardware or library versions.

## Historical experiments

Reproducing 001-003 requires the legacy path (`ct_iqa.models.ohashi_resnet50`, `ct_iqa.config.ExperimentConfig`, the
notebook generators, now archived: [ARCHIVED_ARTIFACTS](archived-artifacts.md)) and reproduces their original BatchNorm behavior (training eps 1e-3 after weight
load, test evaluation eps 1e-5). Their six checkpoints are structurally loadable into the new regression model
(CHECKPOINT_COMPATIBILITY), but evaluating them at a different epsilon is a different
numerical function; the effect is unquantified.

## Reproducing the study-v2 results (executed 2026-10-16)

Environment used: conda env `torch-gpu` (Python 3.10.20, torch 2.5.1+cu121, RTX 4060 Laptop GPU 8 GB, Windows 11). The locked CPU environment (`uv.lock`) runs the tests but is too slow for training.
Run from the repository root with the package importable (`pip install -e .` in that env). Each 30-epoch run takes about 6-8 minutes on that GPU.

```bash
python tools/run_eda.py                                   # dataset audit -> reports/eda/
python tools/preprocessing_probe.py                        # frozen-backbone probe -> reports/preprocessing_probe/
python tools/run_study.py --exp A --seed 42 --purpose selection
for L in 0.03 0.1 0.3 1.0; do python tools/run_study.py --exp B --seed 42 --purpose selection --lam $L --margin 0.05 --delta 0.25; done
python tools/select_b.py --stage 1 --a-run <A seed-42 run dir>
for M in 0.02 0.1; do python tools/run_study.py --exp B --seed 42 --purpose selection --lam 0.3 --margin $M --delta 0.25; done
python tools/run_study.py --exp B --seed 42 --purpose selection --lam 0.3 --margin 0.05 --delta 0.5
python tools/select_b.py --stage 2 --a-run <A seed-42 run dir>
python tools/run_study.py --exp C --seed 42 --purpose selection
for S in 1 2 3; do for E in A C; do python tools/run_study.py --exp $E --seed $S --purpose final; done; done
for S in 1 2 3; do python tools/run_study.py --exp B --seed $S --purpose final --lam 0.3 --margin 0.05 --delta 0.5; done
# create experiments/study_v2/protocol/FROZEN (see EXPERIMENT_PROTOCOL.md section 8), then per final run:
python -m ct_iqa.cli final-test --run-dir <run_dir> --acknowledge-retrospective
python tools/build_reports.py                             # all tables/figures -> reports/experiments/
```

`tools/run_study.py` records the run purpose (smoke / selection / final) in the run manifest and `experiments/study_v2/run_registry.jsonl`.
Expect small numerical differences on other hardware (CUDA nondeterminism). Run directories and checkpoints are not committed.
