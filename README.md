# Quality-Aware Lung CT

A research-grade replication of a no-reference CT image quality assessment
(IQA) baseline, adapted to a dataset with real human-labeled quality scores.

## 1. Project overview

This repository implements and trains a deep-learning model that predicts a
CT image's quality **without needing a clean reference image to compare
against** ("no-reference" IQA). The current, and only, implemented baseline
is a **RadImageNet-pretrained ResNet50** regressor, replicating the
ResNet50 configuration from the reference paper below on this project's own
dataset.

**PYTHON = implementation, NOTEBOOKS = training + experiments.** All
reusable code (model architecture, data loading, preprocessing, training
loop, evaluation metrics) lives in `src/ct_iqa/`. Notebooks under
`notebooks/` only configure and run that code -- they never redefine it.

## 2. Research objective

Reproduce a specific, well-defined no-reference CT-IQA baseline (ResNet50 +
RadImageNet pretraining + a lightweight regression head) as a foundation for
further quality-aware CT work, while being explicit and honest about every
point where this replication's dataset, weight availability, or
unspecified paper details required a deliberate choice.

## 3. Reference paper

> Ohashi et al., *"Development of a No-Reference CT Image Quality
> Assessment Method Using RadImageNet Pre-trained Deep Learning Models."*

Only the paper's **ResNet50** baseline is replicated here; the paper's
InceptionResNetV2 variant and any further methodology are out of scope. See
`docs/paper/` for what the paper specifies, in detail.

## 4. Our dataset

This project trains and evaluates on **LDCT-IQAC**, not the reference
paper's own dataset. LDCT-IQAC provides 1000 training / 300 test CT images
(TIFF, single-channel float, 512x512) each with a **radiologist-assigned
quality score** in `[0, 4]`. See
[`docs/replication/dataset_adaptation.md`](docs/replication/dataset_adaptation.md)
for the full comparison against the paper's own dataset -- **the two are
not the same, and are never treated as interchangeable anywhere in this
codebase or its documentation.**

## 5. Relationship to the Ohashi methodology

| | Paper | This project |
|---|---|---|
| Backbone | ResNet50 or InceptionResNetV2 | ResNet50 only |
| Pretraining | RadImageNet | RadImageNet (weights not yet verified loaded -- see §10) |
| Regression target | VIF (from synthetic degradation) | LDCT-IQAC radiologist quality score |
| Degradation pipeline | Synthetic (noise/blur) | None -- not needed; see §15 |
| Training config | Adam, MSE, batch 64, 30 epochs, lr 1e-3 | Same |
| Evaluation | PLCC/SROCC/KROCC, 5PL-mapped | Same, plus raw (unmapped) metrics reported alongside |

Full detail, including every specific deviation and why, is in
[`docs/replication/deviations.md`](docs/replication/deviations.md).

## 6. Architecture

**PAPER FACT:** ResNet50 backbone, 224x224 input, RadImageNet pretraining,
classification head replaced by `Dropout -> Dense(1) -> Sigmoid`.

**OUR IMPLEMENTATION** (`src/ct_iqa/models/`):

- `resnet50.py` -- a from-scratch standard ResNet50 backbone
  (`torchvision` is not used), with stride-2 downsampling placed on the 1x1
  reduce conv (Keras/RadImageNet convention, not torchvision's), so
  RadImageNet's Keras-trained weights map onto it correctly.
- `ohashi_resnet50.py` -- `OhashiResNet50`: the backbone above + the
  Ohashi regression head, plus RadImageNet weight loading
  (`load_radimagenet_weights`).
- `radimagenet_weights.py` -- Keras `.h5` -> PyTorch state-dict conversion
  for RadImageNet weights (runnable as
  `python -m ct_iqa.models.radimagenet_weights`).

**All architecture lives in `src/ct_iqa/models/` and nowhere else.**
Notebooks only ever do `from ct_iqa.models.ohashi_resnet50 import
OhashiResNet50; model = OhashiResNet50(...)`.

## 7. Repository structure

```
├── README.md, LICENSE, pyproject.toml, .gitignore, .python-version
├── configs/            YAML hyperparameters (dataset/preprocessing/model/training)
├── data/
│   ├── raw/ldct_iqac/  original CT images (gitignored; see §9)
│   ├── labels/ldct_iqac/  filename -> quality-score JSON (tracked)
│   └── interim/, processed/  currently unused (no precompute step exists)
├── docs/
│   ├── paper/          what the paper specifies, ONLY
│   ├── replication/    our adaptation, deviations, reproducibility notes
│   └── decisions/      repository/process decisions log
├── src/ct_iqa/         all reusable implementation (see §6, §11, §12)
├── notebooks/          research execution: 01_exploration -> 02_data_preparation
│                       -> 03_training -> 04_evaluation (see §11, §12)
├── tools/               notebook-generator scripts (dev tooling, not src/ or notebooks/)
├── weights/pretrained/  externally-sourced pretrained weights (gitignored; see §10)
├── experiments/         per-run checkpoints + config (gitignored binaries; see §14)
├── results/             evaluation outputs: figures/tables/predictions/metrics
└── tests/unit/, tests/integration/
```

See `docs/repository_architecture_audit.md` for the full audit this
structure was migrated from, file by file.

## 8. Installation

```bash
git clone <this repository>
cd quality-aware-lung-ct
pip install -e .          # installs `ct_iqa` so notebooks can `import ct_iqa`
                           # with no sys.path manipulation
pip install -e ".[dev]"   # + pytest, nbformat, for running tests/regenerating notebooks
```

Requires Python >= 3.10 (developed against 3.10; see `.python-version`).

## 9. Dataset setup

This repository does **not** include the LDCT-IQAC images (patient-derived
CT data; see the dataset-safety notes in `.gitignore`). To run anything
beyond `pytest`'s synthetic-data unit tests, place the dataset at:

```
data/raw/ldct_iqac/train/image/*.tif     (1000 files)
data/raw/ldct_iqac/test/images/*.tiff    (300 files)
```

`data/labels/ldct_iqac/train.json` and `test.json` (filename -> quality
score) are already included in this repository (they contain no pixel
data). Paths are configured in `configs/dataset.yaml`.

## 10. Pretrained weights setup

RadImageNet weights are not bundled (large, and not this project's to
redistribute). See
[`weights/pretrained/radimagenet/resnet50/README.md`](weights/pretrained/radimagenet/resnet50/README.md)
for the full source/conversion/verification-status writeup. Summary:

1. Obtain `RadImageNet-ResNet50_notop.h5` from
   https://github.com/BMEII-AI/RadImageNet (access requested via their
   form) and place it at
   `weights/pretrained/radimagenet/resnet50/RadImageNet-ResNet50_notop.h5`.
2. Convert it: `python -m ct_iqa.models.radimagenet_weights`.
3. Point `configs/model.yaml`'s `radimagenet_weights_path` at the resulting
   `.pt` file.

**As of this writing, no run in this repository has RadImageNet weights
confirmed loaded** -- the backbone runs randomly initialized until you
complete the steps above. This is reported honestly wherever it matters
(see `docs/replication/deviations.md`), never silently glossed over.

## 11. Training

`notebooks/03_training/01_resnet50_baseline.ipynb` (generated by
`tools/build_training_notebook.py` -- regenerate via that script rather
than hand-editing the `.ipynb`): loads `ExperimentConfig` from
`configs/*.yaml`, builds the dataset/model, and runs the training loop
(`ct_iqa.training.trainer.train`). The full 30-epoch run is gated behind a
`RUN_FULL_TRAINING` flag inside the notebook so re-running it doesn't
silently trigger a long training run. The best-validation-loss checkpoint
is saved to `experiments/001_resnet50_baseline/checkpoint/best.pt`.

## 12. Evaluation

`notebooks/04_evaluation/01_test_set_evaluation.ipynb` (generated by
`tools/build_evaluation_notebook.py`) is **independent** of the training
notebook -- it loads the checkpoint from disk, evaluates on the held-out
LDCT-IQAC test set, and writes predictions/metrics/figures to `results/`.
If no checkpoint exists yet, it still runs against an untrained model as an
explicitly-labeled pipeline sanity check, never presented as a real result.

## 13. Reproducibility

See [`docs/replication/reproducibility.md`](docs/replication/reproducibility.md)
for seeding, the train/validation split, and configuration provenance.
Summary: `ct_iqa.utils.seed.set_seed(config.seed)` seeds Python/NumPy/PyTorch;
the validation split is a seeded `random_split`; every experiment run should
save its resolved config alongside its checkpoint.

## 14. Current experimental status

**No full training run has been executed and persisted in this repository
as of this migration.** `experiments/001_resnet50_baseline/` currently
contains only configuration scaffolding (`README.md`, `config.yaml`), no
checkpoint. `experiments/002_learning_rate_search/` and
`experiments/003_final_replication/` are placeholders for work that has not
started -- see each directory's `README.md` and
`docs/decisions/README.md`. The pipeline itself is implemented and tested
end-to-end (see §16), but no scientific result should be read out of this
repository yet.

## 15. Known deviations

Full list with rationale in
[`docs/replication/deviations.md`](docs/replication/deviations.md). The
two most important:

- **No synthetic-degradation/VIF-labeling pipeline.** LDCT-IQAC already
  provides real quality-varying images with human labels, so this project
  does not need (and does not implement) the paper's own dataset
  construction pipeline.
- **RadImageNet weights not yet verified loaded in any result** (§10).

## 16. Limitations

- Only the ResNet50 baseline is implemented; no InceptionResNetV2, no
  further architectural novelty.
- Results on LDCT-IQAC are **not directly comparable** to the paper's own
  reported numbers (different dataset, different label semantics -- see
  §4/§5).
- The Dropout probability (`configs/model.yaml`: `0.5`) is this project's
  choice, not a value taken from the paper (the paper does not specify
  one).
- No data augmentation, learning-rate scheduling, or early stopping is
  implemented for the baseline configuration (deliberately, to stay close
  to the paper's stated training setup).

## 17. Citation & License

**Cite the reference paper** (Ohashi et al. -- see §3) if you build on the
methodology this project replicates, and cite LDCT-IQAC if you use that
dataset.

**License: undetermined.** No `LICENSE` file exists in this repository.
The correct license depends on a decision only the project owner can make
(and interacts with the usage terms of both LDCT-IQAC and RadImageNet) --
see `docs/decisions/README.md`. Until a `LICENSE` file is added, no license
should be assumed.
