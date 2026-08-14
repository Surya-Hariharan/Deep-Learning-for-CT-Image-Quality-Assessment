# LDCT-IQAC Dataset Audit

**Audit type:** Read-only, dataset-first. No images generated, no labels generated, no training run, no dataset or config modification.
**Audited on:** 2026-08-14
**Scope:** `data/raw/ldctiqa/` only (the reference-image source). Generated/derivative artifacts elsewhere in `data/` are inventoried for context but not analyzed as part of this audit.

---

## 1. Dataset identity

- Source: **LDCT-IQAC 2023 Grand Challenge**, distributed via the Hugging Face mirror `MikaJesse/LDCTiqa_png`.
- Local copy: `data/raw/ldctiqa/` (222 MB on disk).
- Per `SOURCE_README.md`: 1,000 low-dose CT slices, each with an expert perceptual quality score, published to support No-Reference IQA research. Slices contain **sparse-view streak artifacts + noise** intrinsic to the challenge's low-dose acquisition/reconstruction protocol (i.e. these are not necessarily pristine references).
- Citation: Lee et al., "Low-dose computed tomography perceptual image quality assessment," *Medical Image Analysis* 99 (2025), 103343.
- License: not explicitly stated by the HF mirror; README defers to the original challenge site for usage rights.

## 2. Actual directory structure

```
data/raw/ldctiqa/
├── .gitkeep
├── gitattributes                  # verbatim HF LFS-tracking config (not used by this repo's git)
├── SOURCE_README.md               # 91 lines, HF dataset card
├── ground_truth_score.json        # 1,000 entries: {"0000.png": 2.8, ...}
└── LDCTiqa_png/
    └── 0000.png … 0999.png        # 1,000 flat PNG files, no subdirectories
```

No hidden/system files beyond `.gitkeep`. No unexpected files. No train/test split files, no auxiliary metadata files (no patient/study/series IDs are distributed).

**File counts and sizes:**

| Item | Count | Size |
|---|---|---|
| Reference images (`.png`) | 1,000 | 230,414,992 bytes (≈ 219.7 MiB) total |
| Label file (`ground_truth_score.json`) | 1 | 22,003 bytes |
| Documentation (`SOURCE_README.md`) | 1 | 4,165 bytes |
| `gitattributes` (HF artifact, unrelated to this repo) | 1 | 2,504 bytes |

Largest image: `0021.png` (345,518 B). Smallest: `0101.png` (153,217 B). File-size spread (~2.3×) is consistent with PNG compression varying with local image entropy/noise level — not a structural anomaly.

**Other `data/` contents (context only, not part of this audit's scope):**
`data/manifests/` (committed CSV/Parquet manifests for this and other datasets), `data/interim/quality_iqa_checkpoint/` (an **in-progress, incomplete** production generation run — `progress_manifest.json` reports `status: "in_progress"`, 5/1000 references completed, `updated_at: 2026-08-14T06:21:22Z`), `data/interim/vif_pilot/` and `vif_full_grid_pilot/` (earlier pilot runs), `data/processed/quality_iqa/degraded/` (degraded images already generated from earlier pilot work), `data/external/radimagenet/` (pretrained backbone weights). None of these were modified, resumed, or inspected beyond a directory-level listing.

## 3. File inventory

| Filename | Format | Size | Records | Purpose |
|---|---|---|---|---|
| `LDCTiqa_png/*.png` | PNG, 8-bit RGB | 153 KB – 346 KB each | 1,000 files | Reference CT images |
| `ground_truth_score.json` | JSON object | 22,003 B | 1,000 key→value pairs | Expert quality score per image, keyed by filename |
| `SOURCE_README.md` | Markdown | 4,165 B | — | Original HF dataset card; documents scoring methodology |
| `gitattributes` | Git LFS config | 2,504 B | — | Irrelevant artifact from the HF repo, not consulted by this repo's own `.gitattributes` |

No train/test split file exists. No auxiliary files (segmentation masks, DICOM headers, patient metadata) are present.

## 4. Image characteristics

All 1,000 images share **identical** structural properties — no variation found:

| Property | Value |
|---|---|
| Resolution | 512 × 512 (all 1,000) |
| Channels / mode | RGB (3 channels), and verified **all three channels are pixel-identical** on every sampled image (grayscale data replicated into RGB) |
| dtype / bit depth | `uint8`, 8-bit |
| Pixel value range | min 0, max 255 (full 8-bit range used) |
| Mean intensity (avg across images) | 93.08 |
| Std of intensity (avg across images) | 86.17 |
| File format | PNG (100%) |

No dimension, channel, dtype, or format variation was found — the dataset is fully homogeneous on these axes.

## 5. Image integrity

Read-only scan (PIL `verify()` + full pixel decode) on all 1,000 files:

- **Valid: 1,000 / 1,000**
- **Invalid/corrupted/truncated/zero-byte: 0**
- No NaN/Inf values (not applicable to uint8, and none of the underlying data triggered decode errors)
- No unexpected dimensions, channels, or bit depth

## 6. Label / expert-score audit

Source file: `ground_truth_score.json`, a flat JSON object mapping `"NNNN.png" → float`.

| Statistic | Value |
|---|---|
| Field name | value keyed by filename (no named field — root object is `{filename: score}`) |
| Score type | float, quantized to steps of 0.2 |
| Count with scores | 1,000 |
| Missing scores | 0 |
| Duplicate score *entries* (same filename twice) | 0 (checked: 1,000 unique keys) |
| Orphan scores (score with no matching image) | 0 |
| Min | 0.0 |
| Max | 4.0 |
| Unique values | 21 (0.0, 0.2, 0.4, … 4.0 — exactly the grid produced by averaging 5 integer 0–4 ratings) |
| Non-integer (non-whole) values | 765 / 1000 |
| Mean | 2.0728 |
| Median | 2.0 |
| Std | 1.0662 |

**Percentiles:**

| P1 | P5 | P10 | P25 | P50 | P75 | P90 | P95 | P99 |
|---|---|---|---|---|---|---|---|---|
| 0.0 | 0.4 | 0.6 | 1.2 | 2.0 | 3.0 | 3.6 | 3.8 | 4.0 |

**Full score-frequency distribution:**

| Score | Count | Score | Count | Score | Count |
|---|---|---|---|---|---|
| 0.0 | 25 | 1.4 | 65 | 2.8 | 42 |
| 0.2 | 19 | 1.6 | 65 | 3.0 | 65 |
| 0.4 | 22 | 1.8 | 74 | 3.2 | 35 |
| 0.6 | 42 | 2.0 | 53 | 3.4 | 42 |
| 0.8 | 48 | 2.2 | 56 | 3.6 | 31 |
| 1.0 | 56 | 2.4 | 60 | 3.8 | 51 |
| 1.2 | 51 | 2.6 | 62 | 4.0 | 36 |

The 21-value grid on step 0.2, with no values that aren't multiples of 0.2, is direct empirical confirmation of the README's claim of "5 raters, integer 0–4 scale each, averaged" (5 raters × integer/5 = multiples of 0.2). This is evidence *for* the documented aggregation method, not an assumption.

**Do not convert to classes / do not assume a 1=bad…4=excellent mapping:** no such mapping is defined by the source. See Section 7 for what *is* documented.

## 7. What the expert score actually means

From `SOURCE_README.md` (the only documentation shipped with the dataset — no separate paper-derived rubric or per-radiologist file is present):

- **Who assigned scores:** "five experienced radiologists."
- **Number of observers:** 5.
- **Scoring scale:** integer 0–4 per radiologist (README states "The scores range from 0 (very poor) to 4 (excellent)" for the *final* score, and the empirical 0.2-step grid confirms 5 integer raters averaged).
- **Aggregation:** mean of the 5 radiologists' scores (README: "the average of the scores assigned by the five radiologists").
- **Viewing condition:** abdominal soft-tissue window, width/level 350/40.
- **What the score represents:** README explicitly frames this as **perceptual image quality**, in the context of NR-IQA research and radiation-dose optimization — not diagnostic accuracy, not lesion-detection performance, not a defined clinical-usability rubric.
- **What each anchor (0, 1, 2, 3, 4) individually means:** **UNKNOWN.** The README gives only the two endpoints ("0 = very poor," "4 = excellent") in prose; it does not define 1, 2, or 3, and no per-radiologist scoring rubric or the original challenge's instructions are shipped with this local copy. Do not infer intermediate anchor meanings.
- **Inter-rater information (agreement, individual scores):** UNKNOWN — only the pre-averaged final score is distributed; individual radiologist scores are not present in this file.

## 8. Image ↔ label alignment

| Metric | Value |
|---|---|
| IMAGE COUNT | 1,000 |
| SCORE COUNT | 1,000 |
| MATCHED COUNT | 1,000 |
| MISSING SCORE COUNT | 0 |
| ORPHAN SCORE COUNT | 0 |
| Duplicate image filenames | 0 |
| Duplicate score keys | 0 |
| Filename/case/extension mismatches | none — both sides use lowercase `NNNN.png`, zero-padded to 4 digits, `0000`–`0999` |

Alignment is exact and total.

## 9. Duplicate / near-duplicate audit

- **Exact duplicates (SHA-256 of raw file bytes, all 1,000 files hashed):** 0 duplicate groups, 0 duplicate images.
- **Near-duplicate check:** not performed computationally (would require pairwise perceptual-hash or SSIM comparison across 1,000×1,000 pairs); given zero exact duplicates and the dataset being CT slices from what is presumably a set of distinct patients/studies (per the *challenge's* framing as a diverse benchmark), near-duplication risk was not otherwise flagged by directory structure, filename pattern, or file-size clustering. This is a "not evaluated" gap, not a "confirmed absent" finding — flagged in Section 12.
- **Reference-level split implication:** because **no patient/study/series ID is distributed** with this dataset (confirmed — see Sections 2–3, and `configs/datasets/ldctiqa.yaml` independently marks `patient_id`/`study_id`/`series_id` as "UNKNOWN / REQUIRES VERIFICATION"), it is not possible from the files on disk to confirm that all 1,000 slices come from 1,000 distinct patients. If multiple slices in the set originate from the same patient/study (common in CT slice-level datasets), naive random train/val/test splitting could leak patient identity across splits. This is a real, currently **unresolved** risk that should be checked against the original challenge documentation before any split is finalized — not decided here.

## 10. Quality (score) distribution

- The distribution is **roughly unimodal and centered near the scale midpoint** (mean 2.07, median 2.0, out of a 0–4 scale), with substantial spread (std 1.07) and full range coverage (0.0–4.0, all used).
- It leans **moderately toward the mid-to-upper range**: P25=1.2, P75=3.0, so the interquartile range sits above the scale's own midpoint (2.0) only slightly — the distribution is close to symmetric, not heavily skewed.
- Not heavily concentrated in a narrow band, and not bimodal/extreme (0.0 and 4.0 combined account for only 61/1000 ≈ 6.1% of images).
- Overall characterization: **approximately balanced with moderate spread**, not strongly imbalanced.

**Lowest-score references** (score 0.0, first 10 by filename): `0008, 0075, 0187, 0188, 0208, 0266, 0281, 0294, 0314, 0370`
**Highest-score references** (score 4.0, first 10 by filename): `0780, 0801, 0832, 0837, 0870, 0914, 0934, 0960, 0961, 0981`
**Middle-range references** (score exactly 2.0, sample of 10): `0446, 0464, 0529, 0538, 0553, 0555, 0564, 0569, 0584, 0610`

## 11. Reference-level dataset suitability

| Factor | Assessment |
|---|---|
| Sample size | 1,000 — adequate for the planned degradation/VIF pipeline |
| Quality-score coverage | 100% (1,000/1,000), zero missing |
| Image consistency | Perfectly uniform (512×512, RGB-replicated-grayscale, uint8, PNG) |
| Label completeness | Complete, no gaps, no orphans |
| Corruption | None found (0/1000 invalid) |
| Duplicates | 0 exact; near-duplicate status unevaluated |
| Score distribution | Full-range, roughly balanced, moderate spread |
| Reference independence (patient-level leakage) | **Unknown / unverified** — no patient/study ID shipped |
| Obvious leakage concerns | The unresolved patient-independence question (above) is the only leakage concern surfaced by this audit |

**Conclusion: SUITABLE WITH CONDITIONS.**
The dataset is clean, complete, and internally consistent — no corruption, no missing/orphan labels, no exact duplicates, well-behaved score distribution. The one open condition is patient/study-level independence across the 1,000 slices, which cannot be verified from the files currently on disk and should be resolved (via the original LDCT-IQAC challenge documentation, if it specifies patient counts) before any split strategy is finalized.

## 12. Critical distinction (for downstream work)

- **A. Original reference image** — the actual LDCT-IQAC PNG in `LDCTiqa_png/`. Confirmed to already contain low-dose acquisition artifacts (sparse-view streaks + noise) per the source README — it is *not* a pristine/clean reference in the traditional full-reference-IQA sense.
- **B. Expert score** — the human MOS in `ground_truth_score.json`, representing radiologist-rated *perceptual quality* of image A itself. This is a property of the real image.
- **C. Synthetic degraded image** — does not exist for this dataset yet in the audited raw directory (some exist under `data/processed/quality_iqa/degraded/` from earlier pilot runs, out of scope here); would be created later by applying controlled degradation to A.
- **D. VIF score** — does not exist in the raw directory; would be a full-reference metric comparing C against A.

**B and D are different quantities from different pipelines and must not be conflated.** This audit only characterizes A and B.

## 13. Project assumptions vs. actual findings

| Assumption | Actual finding | Correct? |
|---|---|---|
| 1,000 reference CT images | 1,000 PNG files present, all valid | ✅ Correct |
| One expert score per reference | 1,000 scores, 1:1 with images, 0 missing/orphan | ✅ Correct |
| Expert scores on a 0–4 scale | Confirmed: min 0.0, max 4.0 | ✅ Correct |
| Expert score ≠ synthetic VIF target | Confirmed structurally: only `ground_truth_score.json` (expert MOS) exists in raw data; no VIF values are present anywhere in `data/raw/` | ✅ Correct |
| Image format | Not explicitly stated as an assumption, but implied PNG by dataset name | ✅ Correct — 100% PNG |
| Image dimensions | Not stated as an assumption | Found: 512×512, uniform | N/A — now documented |
| Channels | Not stated as an assumption | Found: RGB with all 3 channels identical (replicated grayscale) | N/A — now documented, and worth flagging since it affects preprocessing (treat as single-channel data) |
| Dataset directory | Not stated as an assumption | `data/raw/ldctiqa/LDCTiqa_png/` + `ground_truth_score.json` at `data/raw/ldctiqa/` | N/A — now documented |
| Label file | Not stated as an assumption | `ground_truth_score.json`, flat `{filename: score}` object | N/A — now documented |
| Duplicate status | Not stated as an assumption | 0 exact duplicates; near-duplicates unevaluated | N/A — now documented |
| Data integrity | Not stated as an assumption | 1000/1000 valid, 0 corrupted | N/A — now documented |

**No project assumption in the prompt was found to be incorrect.** The one meaningful gap is something the prompt did *not* assume either way: patient/study-level independence across the 1,000 slices is unverified (Section 9/11), and the "reference images are pristine" implication some Ohashi-style pipelines carry is explicitly **not** true here — this dataset's images already contain acquisition-level artifacts, a fact this project's own `configs/datasets/ldctiqa.yaml` already documents as a "CAVEAT (documented, not resolved)."

## 14. Directory structure recommendation

The current structure is already clean and does not need reorganization:

```
data/
├── raw/ldctiqa/              # source-of-truth, git-ignored, verified in this audit
├── manifests/                 # small, committed CSV/Parquet descriptions of the raw data
├── interim/                   # in-progress/intermediate generation artifacts
├── processed/                 # finished derived datasets (degraded images, etc.)
└── external/                  # third-party assets (e.g. pretrained backbone weights)
```

This matches the recommended `raw / interim / processed / external` convention already in place. No change recommended.

## 15. Git safety

- `git check-ignore -v` confirms both `data/raw/ldctiqa/LDCTiqa_png/0000.png` and `data/raw/ldctiqa/ground_truth_score.json` are ignored via `.gitignore:14` (`data/raw/**`).
- `git ls-files data/` returns only the small `data/manifests/*.csv|.parquet|.header.json` files (22 tracked files) — no raw pixels, no checkpoints, no large binaries are tracked.
- `data/interim/**` and `data/processed/quality_iqa/degraded/**` are also explicitly ignored (with `.gitkeep` allow-listed so directories persist).
- `checkpoints/` and `experiments/**/checkpoints/` are ignored.
- `git status --short` at audit time shows only unrelated in-progress doc/script changes (`scripts/quality/generate_production_dataset.py`, `tests/test_production_generation.py`, and new `docs/*.md` files) — nothing dataset-related is staged or modified by this audit.
- Nothing was added or committed as part of this audit.

## 16. Known limitations of this audit

- Near-duplicate detection (perceptual hashing / SSIM across all pairs) was not run — only exact byte-level duplicates were checked.
- Patient/study/series-level provenance cannot be determined from the files on disk; this dataset does not ship that metadata.
- Individual-radiologist scores and the original challenge's full scoring rubric are not available locally — only the aggregated mean score and a two-sentence description of the anchors.
- License terms are not resolved locally (README defers to the original challenge site).

## 17. Recommended next actions

1. Before deciding any split strategy: check the original LDCT-IQAC 2023 challenge documentation (external to this repo) for patient/study counts, to resolve the reference-independence question in Section 9/11.
2. Treat the RGB channels as redundant (single effective channel) in any preprocessing design decision — do not assume 3 independent channels of information.
3. Continue treating expert score (MOS) and VIF as separate, non-interchangeable targets, consistent with `configs/datasets/ldctiqa.yaml`'s existing `secondary_role` annotation.
4. If near-duplicate risk matters for the eventual split design, run a perceptual-hash pass as a follow-up — not performed here.
5. No corruption, alignment, or integrity remediation is needed — the dataset is clean as-is.
