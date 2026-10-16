# Project overview

**Purpose.** Compare three learning objectives for no-reference CT image quality assessment on LDCT-IQAC 2023, all on
one ResNet50-family backbone initialized from RadImageNet: reference regression (A), ranking-aware regression (B),
uncertainty-aware regression with conformal intervals (C). The earlier work in this repository (experiments 001-003)
adapted the Ohashi et al. ResNet50 baseline to LDCT-IQAC; it is preserved unchanged.

**Scope.** ResNet50 family only; image-level analysis; the official test set is a previously examined benchmark and is
used retrospectively. See [RESEARCH_QUESTIONS](RESEARCH_QUESTIONS.md) and [LIMITATIONS_AND_THREATS](LIMITATIONS_AND_THREATS.md).

**Dataset.** 1000 labeled training images and 300 official test images, radiologist scores in [0, 4]
([DATASET_AND_LABELS](DATASET_AND_LABELS.md)).

**Main components.** Package `src/ct_iqa/` ([REPOSITORY_ARCHITECTURE](REPOSITORY_ARCHITECTURE.md)); YAML configs in
`configs/`; CLI `python -m ct_iqa.cli`; notebooks in `notebooks/`; frozen protocol in `experiments/study_v2/protocol/`.

## Experiment summary

| ID | Model | Loss | Extra | Status |
|---|---|---|---|---|
| A | `resnet50_regression` | MSE | - | NOT RUN |
| B | `resnet50_regression` | MSE + lambda * pairwise ranking | validation-only tuning grid | NOT RUN |
| C | `resnet50_gaussian` | Gaussian NLL | split-conformal intervals | NOT RUN |
| 001-003 | `OhashiResNet50` (legacy) | MSE | historical | COMPLETE, immutable |

## Repository map

```text
configs/            base.yaml, experiments/{a,b,c}_*.yaml, legacy dataset/model/preprocessing/training yaml
src/ct_iqa/         package (see architecture doc)
tests/              unit/, integration/ (synthetic + legacy real-data), regression/
experiments/        001-003 (historical, immutable), study_v2/ (protocol + new runs)
results/            historical figures, tables, metrics, predictions
notebooks/          00-07 study notebooks; historical notebooks in numbered subfolders
docs/               this documentation; paper/ and replication/ are historical
data/labels/        tracked label JSON; data/raw/ ignored images
weights/pretrained/ ignored binaries; README and CHECKSUMS.sha256 tracked
tools/              audit scripts, split-manifest creator, legacy notebook generators
```

## Setup and entry points

```bash
uv sync --extra dev
uv run python -m ct_iqa.cli validate-config --config configs/experiments/a_reference.yaml
uv run pytest tests/unit tests/regression
```

Details and the full command list: [REPRODUCIBILITY](REPRODUCIBILITY.md). Reading order: [docs/README](README.md).
