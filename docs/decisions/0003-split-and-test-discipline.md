# ADR-003: Data split and official-test-set discipline

Status: accepted and frozen. Date: 2026-10-16.

## Context

Historical experiments 001-003 used a 900/100 train/validation split of the 1000 labeled training images and
evaluated experiments 001 and 003 on the official 300-image test set, then compared them there. The test set is
therefore not an untouched confirmatory set for any new study. The labels carry no patient identifiers.

## Decision

1. **Partition.** One image-level split of the 1000 training images into 800 train / 100 validation /
   100 calibration, produced by `numpy.random.default_rng(2026).permutation` over the sorted filenames and stored in
   `experiments/study_v2/protocol/split_manifest.json` (SHA-256 `be5831c56d097a8fdafa7797cda52329ec2dd3aba9dfc88d6e1c92fc694e0a16`).
   It was generated once by `tools/create_split_manifest.py`, which refuses to overwrite an existing manifest.
   It is never re-drawn, including if partition statistics look unbalanced (validation mean 1.954 and calibration
   mean 1.936 versus 2.105 for train are within sampling variation for n = 100 with label std about 1.07).
2. **Roles.**

   | Partition | n | Used for | Never used for |
   |---|---|---|---|
   | train | 800 | gradient updates | selection, calibration, reporting as generalization |
   | validation | 100 | checkpoint selection, hyperparameter tuning (including B's ranking weight) | training, calibration |
   | calibration | 100 | Experiment C conformal calibration only (and reporting its coverage diagnostics) | training or selection in **any** experiment |
   | official test | 300 | one retrospective evaluation after the protocol is frozen | selection, tuning, calibration, seed choice, variant choice |

3. **Same partition for A, B and C.** A and B simply do not use the calibration images. Multiple training seeds, if
   used later, change only initialization of the head and data order; the split never changes. Any different split
   needed for a sensitivity analysis must be labeled as such and stored under its own manifest.
4. **Terminology.** "Image-level split", never "patient-independent split".
5. **Test-set exposure is disclosed.** Results on the official test set are labeled *retrospective evaluation on a
   previously examined benchmark*. The framework enforces this by code: the test loader requires an explicit
   `--final-evaluation` acknowledgement and a frozen protocol marker (`experiments/study_v2/protocol/FROZEN`), and no
   training, selection or calibration code imports it.

## Consequences

- Validation estimates from 100 images are noisy (PLCC standard error roughly 0.01-0.02 at r around 0.95); differences
  between variants smaller than that cannot be called real. Paired bootstrap intervals are required for comparisons.
- The calibration set of 100 gives a finite 90% conformal rank (k = 91 of 100) but only coarse coverage control.
- Because the split is at image level, slices from one patient may sit in several partitions, which can make
  validation and calibration optimistic. This is listed in `docs/limitations-and-threats.md`.
