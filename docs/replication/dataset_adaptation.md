# Dataset adaptation: LDCT-IQAC vs. the paper's dataset

**OUR IMPLEMENTATION.** This document exists specifically so that this
project's dataset is never silently conflated with the reference paper's
own dataset (see the migration's non-negotiable rule: "do not treat the
authors' dataset as our dataset").

## Our dataset: LDCT-IQAC

This project trains and evaluates against **LDCT-IQAC**, loaded by
`src/ct_iqa/data/ldct_iqac.py`:

| Split | Images | Format | Labels |
|---|---|---|---|
| train | 1000 `.tif` files, `data/raw/ldct_iqac/train/image/` | PIL mode `'F'` (32-bit float grayscale), 512x512, pixel range `[0, 1]` | `data/labels/ldct_iqac/train.json`: `{filename: score}`, 1000 entries |
| test | 300 `.tiff` files, `data/raw/ldct_iqac/test/images/` | same format | `data/labels/ldct_iqac/test.json`: `{filename: score}`, 300 entries |

The label is a **radiologist-assigned quality score** in `[0.0, 4.0]`
(training scores are multiples of 1/5 -- 5 raters, integer 0-4 votes
averaged; test scores are multiples of 1/6 -- 6 raters). Every image has
exactly one label and vice versa in both splits (verified;
`LDCTIQACDataset` fails loudly if this is ever not true), with no
duplicate images and no non-image files found in either directory.

**Empirically re-verified 2026-09-13** by the scientific pipeline audit
(`notebooks/02_data_preparation/01_dataset_preparation.ipynb`,
`tests/integration/test_ldct_iqac_pipeline_integration.py`) against the
actual on-disk files:

| | train | test |
|---|---|---|
| n | 1000 | 300 |
| min / max | 0.0 / 4.0 | 0.0 / 4.0 |
| mean / median | 2.073 / 2.0 | 2.131 / 2.167 |
| std | 1.066 | 1.087 |
| unique values | 21 (step 1/5, confirming 5 raters) | 25 (step 1/6, confirming 6 raters) |

No NaN/Inf pixels, no unreadable files, no duplicate image content within
or across splits, and **zero filename/content/label-key overlap between
train and test** were found. `SCORE_MIN = 0.0` / `SCORE_MAX = 4.0` (used by
`normalize_score`/`denormalize_score`) exactly bound the observed range in
both splits -- this constant is empirically justified, not assumed.

## The paper's dataset

The reference paper regresses against a **VIF (Visual Information
Fidelity)** score, computed by applying synthetic degradation (e.g. noise,
blur) to clean CT images at controlled severities and measuring the
resulting VIF between the degraded and clean image. This is a fundamentally
different label-generation process from LDCT-IQAC's human-rated scores.

This project has not obtained, and does not use, the paper's own dataset.
**Nothing in this repository should be read as replicating the paper's
dataset construction** -- only its model architecture, training
configuration, and evaluation methodology are replicated, against a
different (but conceptually comparable -- both are CT image quality scores
in a bounded range) dataset.

## Why this substitution

LDCT-IQAC was the dataset actually available to this project. It already
provides real images spanning a range of quality levels with human-expert
labels, which is the property the paper's synthetic-degradation+VIF
pipeline exists to manufacture in the absence of such a dataset. Using
LDCT-IQAC directly means this project does not need to reimplement that
pipeline (see `deviations.md`) -- but it also means any reported
correlation numbers are **not directly comparable** to the paper's own
reported numbers, since the label distributions and generation processes
differ. Any comparison between this project's results and the paper's
reported results must state this explicitly, not present them as
equivalent.
