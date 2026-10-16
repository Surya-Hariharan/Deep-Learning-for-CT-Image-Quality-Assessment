# Dataset and labels

Each statement is tagged with its basis: **[repo]** observed in this repository on 2026-10-16 (labels read; images
checked by earlier tests), **[official]** stated on the challenge website, **[secondary]** stated in a later paper that
uses the data, **[unverified]** not confirmed.

## Source

LDCTIQAC 2023 ("Low-dose Computed Tomography Perceptual Image Quality Assessment" challenge, MICCAI 2023):
[challenge site](https://ldctiqac2023.grand-challenge.org/), data on Zenodo
[10.5281/zenodo.7833096](https://doi.org/10.5281/zenodo.7833096). A benchmark paper is cited by a search result as
Lee et al., *Medical Image Analysis* 99 (2025) 103343 [secondary; full bibliographic record not verified here].
Images derive from the NIH/AAPM Low Dose CT Grand Challenge data [official].

## Sizes and labels

| Fact | Value | Basis |
|---|---|---|
| Training images | 1000 | [repo] `train.json`; [secondary] |
| Official test images | 300 | [repo] `test.json`; [secondary] |
| Score range | 0.0 - 4.0 (0-4 Likert; 0 very poor, 4 excellent) | [repo] min/max in both label files; [secondary] |
| Raters, training scores | five experienced radiologists, scores averaged | [official]; [repo] all 1000 scores are multiples of 1/5 |
| Raters, test scores | six (inferred) | [repo] all 300 test scores are multiples of 1/6; [secondary] says "5-6"; [official] page does not state a test-rater count |
| Viewing condition | abdominal soft-tissue window, width/level 350/40 | [official] |
| Training label mean (std) | 2.073 (1.066) | [repo] |
| Test label mean (std) | 2.131 (1.087) | [repo] |

Training and test raters are not guaranteed to be the same people, and test labels average six rather than five
ratings, so label noise differs slightly between splits [inferred from the above].

## Image representation [repo, from the earlier audit and loader checks]

TIFF, PIL mode `F` (32-bit float), 512x512, pixel values in [0, 1]. Training files `NNNN.tif`, test files
`testNNN.tiff`. Every image has exactly one label and vice versa; no filename, label-key or image-content overlap
between training and test (integration tests `test_no_*_overlap*`). The score scale used for modelling is
`(y - 0) / 4` for training and the inverse for evaluation. Raw images are ignored by Git and are never modified.

## Metadata that is NOT available

No patient identifiers, scanner/protocol fields, acquisition dose, slice position, or per-rater scores are provided in
the repository labels [repo]. The official page does not mention patient grouping [official]. Consequently:

- splits are **image-level**; patient-independence cannot be claimed;
- slices from the same patient may lie in train, validation and calibration partitions;
- inter-rater variability cannot be modelled directly; only averages exist.

## Partitions used in study v2

Frozen 800 / 100 / 100 split of the 1000 training images ([ADR-003](decisions/ADR-003-split-and-test-discipline.md),
`experiments/study_v2/protocol/split_manifest.json`, SHA-256 `be5831c5...0a16`). Label statistics by partition (from the
manifest): train mean 2.105 (std 1.066), validation 1.954 (1.111), calibration 1.936 (1.001).

## Historical split (experiments 001-003)

900 / 100 `random_split` with seed 42; unchanged and documented in `docs/replication/dataset_adaptation.md`
(earlier project notes on the same data).

## Access

Obtain the data from the challenge site or Zenodo; place images under `data/raw/ldct_iqac/{train/image,test/images}`
and keep `data/labels/ldct_iqac/*.json` as tracked. Licensing terms are those of the data providers
[unverified here]; do not redistribute images.
