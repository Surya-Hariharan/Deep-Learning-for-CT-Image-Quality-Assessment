# Dataset Adaptation: LDCT-IQAC replaces DeepLesion + CQ500

**One sentence description of this project:** an Ohashi-method CT-NR-IQA
replication using the LDCT-IQAC dataset as a project-specific replacement for
the original DeepLesion/CQ500 reference-image dataset.

This is not "an exact reproduction of the Ohashi paper." It is a
methodological replication with an alternative reference dataset. The
difference is documented here in full, and nowhere silently absorbed.

## What changed, and only what changed

| | Ohashi original | This project |
| --- | --- | --- |
| Reference-image source | DeepLesion (90) + CQ500 (15) | LDCT-IQAC (1,000) |
| Reference count | 105 | 1,000 |
| Reference availability | not downloaded in this repo | present, verified (1,000/1,000 images, 0 corrupt) |
| Synthetic dataset total | 17,745 | 169,000 |
| Reference split (60/20/20) | 63 / 21 / 21 references | 600 / 200 / 200 references |
| Synthetic image split | 10,647 / 3,549 / 3,549 | 101,400 / 33,800 / 33,800 |
| Reference identity | synthetic placeholder ids (dataset not present) | real `image_id` (0000–0999) from the manifest |
| Extra per-reference signal | none | `expert_score` — 5-radiologist MOS, 0–4 scale |

Nothing else changes. Degradation grids, condition families, VIF as the
synthetic target, the reference-level split protocol, the model architecture,
training configuration, calibration, and the three-stage evaluation structure
are all preserved from `docs/ohashi_methodology.md`.

## Why LDCT-IQAC

DeepLesion and CQ500 are not present in this repository (see
`reports/dataset_inventory.md`). LDCT-IQAC is present, fully verified (1,000
PNGs, uniform 512×512, 0 corrupt, 0 duplicates, 100% label coverage), and is
CT imagery — the closest available substitute that keeps the rest of the
pipeline meaningful without waiting on an external download.

## What LDCT-IQAC brings that Ohashi's references didn't

LDCT-IQAC images carry a pre-existing radiologist mean-opinion score
(`expert_score`, 0–4). This is a genuine extra asset, not a replacement for
VIF:

- **`vif_score`** — Full-Reference VIF of a *degraded* image against its own
  *clean* reference. This remains the synthetic-stage training target,
  exactly as Ohashi specifies. Computed fresh for all 169,000 records.
- **`expert_score`** — the LDCT-IQAC MOS of the *reference image itself*,
  constant across all 169 degraded variants of that reference. Carried
  through the manifest for the subjective and real-image evaluation stages
  only (`docs/ohashi_methodology.md`, Stages 2–3). Never substituted
  for `vif_score` as a training target. See `src/ct_iqa/data/manifest.py:DegradedRecord`.

## Known caveat (not resolved, documented)

Ohashi's references were treated as pristine/clean CT slices. LDCT-IQAC images
already contain sparse-view streak artifacts and noise from their own
acquisition/reconstruction protocol — that is the entire premise of the
LDCT-IQAC challenge. Every LDCT-IQAC image is used as "clean" here only
*relative to* the additional synthetic degradation layered on top of it; it is
not a claim that the image is pristine in an absolute sense. This is a real
methodological difference from Ohashi's assumed reference quality and is
tracked as deviation **DEV-01** in `docs/deviations_from_ohashi.md`.

## What is explicitly NOT changed by this adaptation

- LDCT-IQAC's `expert_score` does **not** become the synthetic training
  target. VIF does, exactly as Ohashi specifies.
- The reference-level split protocol is unchanged — only the split *sizes*
  scale with the new reference count.
- No new degradation type, model backbone, or evaluation metric is introduced.
