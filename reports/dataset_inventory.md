# Dataset Inventory

**Generated:** 2026-08-13
**Method:** read-only scan (`scripts/dataset/inspect_datasets.py`), manifest build
(`scripts/dataset/build_manifests.py --hash`), validation
(`scripts/dataset/validate_datasets.py`).
**No dataset file was modified, converted or deleted.** One relocation was
performed and verified; see [Relocation record](#relocation-record).

---

## Headline finding

**Six of the seven expected datasets are not present.** The only data in this
repository is the **LDCT-IQAC 2023** dataset, which was not on the expected
list at all. None of PNG, LIDC-IDRI, LUNA16, LNDb, DeepLesion or CQ500 has been
downloaded.

This blocks both replication components:

- **Component A (NGP-Net)** — the PNG longitudinal dataset is absent, and no
  NGP-Net paper or reference implementation exists anywhere in the repository.
  Nothing about the architecture can be verified.
- **Component B (Ohashi IQA)** — DeepLesion and CQ500 are absent, so the 105
  clean reference images cannot be assembled and the 17,745-image degradation
  dataset cannot be generated.

---

## Inventory table

| Dataset | Location | Format | Patients | Studies | Images | Annotations | Size | Status | Intended role |
| --- | --- | --- | ---: | ---: | ---: | --- | ---: | --- | --- |
| **ldctiqa** | `data/raw/ldctiqa/` | PNG (extracted) | unknown | unknown | 1,000 | 1,000 MOS scores (JSON) | 230.4 MB | **PRESENT — VERIFIED** | subjective IQA validation (deferred) |
| png | `data/raw/png/` | — | 103 exp. | 378 exp. | — | nodule-level longitudinal | 0 | **NOT DOWNLOADED** | longitudinal growth prediction |
| lidc_idri | `data/raw/lidc_idri/` | — | — | — | — | XML, 4 reads/scan | 0 | **NOT DOWNLOADED** | detection / segmentation / auxiliary |
| luna16 | `data/raw/luna16/` | — | — | — | — | CSV candidates + annotations | 0 | **NOT DOWNLOADED** | detection benchmark |
| lndb | `data/raw/lndb/` | — | — | — | — | CSV + masks | 0 | **NOT DOWNLOADED** | external validation |
| deeplesion | `data/raw/deeplesion/` | — | — | — | 90 needed | DL_info.csv | 0 | **NOT DOWNLOADED** | IQA reference images |
| cq500 | `data/raw/cq500/` | — | — | — | 15 needed | reads.csv | 0 | **NOT DOWNLOADED** | IQA reference images |

Counts marked "exp." are expectations from the project brief, not observations.

---

## LDCT-IQAC 2023 — verified properties

Every property below was measured directly from the files, not read from the
source README.

| Property | Verified value |
| --- | --- |
| Image count | 1,000 |
| Format | PNG, all files pass a decode + verify pass |
| Dimensions | 512 × 512 — uniform across all 1,000 |
| Colour mode | RGB, uint8 |
| Channel content | all 3 channels identical in all 1,000 files (grayscale replicated) |
| Pixel range | min 0 and max 255 in every image |
| Corrupt / truncated | 0 |
| Zero-byte files | 0 |
| Duplicate content (SHA-256) | 0 duplicate groups — all 1,000 distinct |
| Duplicate identifiers | 0 |
| Labels | 1,000 — exact 1:1 match with the image files, no orphans either way |
| Label range | 0.0 – 4.0, 21 distinct values on a 0.2 grid, mean 2.073 |
| Label semantics | mean of 5 radiologists, abdominal soft-tissue window 350/40 |
| Already preprocessed? | Yes — converted from the challenge originals to 8-bit PNG |

### Identifiers and geometry — absent

`patient_id`, `study_id`, `series_id`, `slice_index`, `pixel_spacing`,
`slice_thickness` and scanner information are **not distributed** with this
release. They are recorded as `null` in the manifest, not imputed.

**Consequence:** patient-level splitting of this dataset is *impossible*, so
`configs/datasets/ldctiqa.yaml` marks its split strategy as
`BLOCKED - no patient identifiers`. Any future use for training must either
resolve provenance upstream or accept that patient-level leakage cannot be
excluded. The validator fails any run that assigns splits without patient ids.

### Role — reassigned

This dataset **cannot serve as an Ohashi reference-image source**, for two
independent reasons:

1. Its images are *already degraded* (sparse-view streak artifacts + noise).
   Ohashi requires **clean** references to degrade and to compute VIF against.
2. Its labels are *subjective* MOS values, whereas the first replication target
   is *objective* VIF.

It is therefore assigned the role `subjective_iqa_validation` — the natural
dataset for the later subjective-correlation stage, which the project plan
explicitly defers until the objective VIF pipeline reproduces. See
`reports/research_decisions.md`, decision `D-02`.

---

## Relocation record

The dataset was found loose at the repository root and moved into the raw-data
layout. Contents were not altered.

| From | To |
| --- | --- |
| `data/LDCTiqa_png/` | `data/raw/ldctiqa/LDCTiqa_png/` |
| `data/ground_truth_score.json` | `data/raw/ldctiqa/ground_truth_score.json` |
| `data/README.md` | `data/raw/ldctiqa/SOURCE_README.md` |
| `data/gitattributes` | `data/raw/ldctiqa/gitattributes` |

Verified after the move: file count 1,000 → 1,000; SHA-256 of a sampled file
unchanged; all 1,000 labels still resolve to existing files.

---

## Duplicate, corrupt and archive status

- **Duplicate copies of a dataset:** none. Only one dataset exists on disk.
- **Duplicate files within a dataset:** none (SHA-256 over all 1,000 files).
- **Corrupted or incomplete archives:** none — there are no archives on disk;
  the only dataset arrived extracted.
- **Unrecognised paths under `data/`:** none after relocation.

---

## Validation result

`python scripts/dataset/validate_datasets.py` → **23 checks: 15 pass, 8 warn, 0 fail**

All 8 warnings are the honest consequences of missing data, not defects:

| Warning | Meaning |
| --- | --- |
| 6 × "manifest empty — dataset not downloaded" | one per absent dataset |
| "ldctiqa: pixel spacing absent" | spacing genuinely not distributed |
| "lidc/luna16 overlap: patient ids not populated" | cannot check until both exist |

---

## What must be acquired next

| Priority | Dataset | Needed for | Blocking |
| --- | --- | --- | --- |
| 1 | DeepLesion (90 slices) | Ohashi references | IQA dataset generation |
| 1 | CQ500 (15 slices) | Ohashi references | IQA dataset generation |
| 2 | Ohashi et al. (2025) full text | 4 unknown parameters | correctness of the IQA dataset |
| 3 | NGP-Net paper / repository | every architecture detail | all of Component A |
| 4 | PNG longitudinal dataset | growth prediction | all of Component A |
| 5 | LIDC-IDRI, LUNA16, LNDb | detection / segmentation / external validation | pipeline stages 4–7 |
