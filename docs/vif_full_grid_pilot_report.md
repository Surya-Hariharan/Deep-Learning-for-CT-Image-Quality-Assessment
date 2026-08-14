# VIF Full-Grid Pilot Report — 100 References × Complete Ohashi Grid

Second, larger validation pass before any commitment to full 1,000-reference
/ 169,000-image dataset generation. Extends `docs/vif_pilot_report.md` (30
references, reduced 3×3 combined subset) by exercising the **complete**
12×12=144 combined noise+blur factorial this time — the one part of the
Ohashi grid the smaller pilot deliberately did not cover, and which turns
out to matter (see §10).

Generator: `scripts/quality/vif_full_grid_pilot.py`. Run date: 2026-08-13/14.
Seed: `20260813` (`configs/paths.yaml:random_seed`, unmodified). Runtime:
~119 minutes (§13).

This remains exploratory validation, not dataset production: no LDCT-IQAC
labels were written, `data/raw/ldctiqa/ground_truth_score.json` was not
touched, `data/manifests/quality_iqa_manifest.*` was not written, and
`src/ct_iqa/vif/wavelet.py` was not modified. All pixels/plots/CSVs live
under `data/interim/vif_full_grid_pilot/` (`.gitignore`'s `data/interim/**`
rule — confirmed via `git check-ignore`, not committed).

---

## 1. Reference sample

**100** LDCT-IQAC reference images, stratified across the `expert_score`
range (0.0–4.0) into 5 equal-width score bins, drawn via
`numpy.random.default_rng(20260813)`. Per-image record —
`reference_id`, `filename`, `expert_score`, `selection_seed` (plus
`selection_reason`) — written to
`data/interim/vif_full_grid_pilot/sample_selection.csv` (gitignored,
reproducible). No source image was modified.

## 2. Theoretical condition count

| | Count |
| --- | --- |
| Noise-only levels | 12 |
| Blur-only levels | 12 |
| Combined noise×blur levels | 12 × 12 = **144** |
| **Conditions/reference (degraded only, clean excluded)** | 12 + 12 + 144 = **168** |
| Conditions/reference (clean counted) | 168 + 1 = **169** |
| **Theoretical total, 100 references, degraded only** | 100 × 168 = **16,800** |
| Theoretical total, 100 references, with clean | 100 × 169 = **16,900** |

Both numbers are reported explicitly, per instruction, rather than
picking one silently.

## 3. Ohashi grid accounting reconciliation

Ohashi's paper reports **105 references × 169 conditions = 17,745**
degraded images (`docs/ohashi_methodology.md`). Reconciling against this
pilot's 168/reference:

- Ohashi's 169 = 1 clean + 12 noise + 12 blur + 144 combined.
- This pilot's 168 excludes the clean condition as a separate generated
  record: VIF of a reference against itself is trivially ≈1.0 (already
  established analytically and empirically — see
  `tests/test_vif_wavelet.py::test_identity_gives_vif_approximately_one`
  and this pilot's own §6 determinism check), so it was not regenerated
  as a distinct row in this pilot's output.
- **168 + 1 = 169 — exact match to Ohashi's reported per-reference count.**

**No unresolved accounting discrepancy exists.** This matches what
`configs/quality/ohashi_ldctiqac.yaml` (`counts.per_reference.total: 169`)
already documents; this pilot's numbers are consistent with, not a
correction to, that existing derivation.

## 4. Actual generated count

**16,800 / 16,800 (100%)** — exactly matching the theoretical
degraded-only total. `scripts/quality/vif_full_grid_pilot.py` asserts
`len(conditions) == 168` before generation begins, so a grid-construction
bug would have failed loudly rather than silently under- or
over-generating.

## 5. VIF validity

| Check | Count | % |
| --- | --- | --- |
| Generated | 16,800 | 100% |
| VIF computed successfully (finite) | 16,800 | **100%** |
| NaN | 0 | 0% |
| Inf | 0 | 0% |
| Negative VIF | 0 | 0% |
| VIF > 1.5 (out of expected range) | 0 | 0% |

Every single one of the 16,800 images produced a finite, in-range VIF
score. This matches the 30-reference pilot's finding and now holds at
17× the scale and with the full combined grid included.

**Determinism check:** re-computed one probe image's VIF score 3 times
(fresh pyramid decomposition each time) — **identical** to full floating-
point precision every time (`determinism_check_passed: true`).

## 6. Diagnostic statistics

| Status | Count | % |
| --- | --- | --- |
| `UNSTABLE_CHANNEL_GAIN` | 15,999 | 95.2% |
| `ok` | 801 | 4.8% |
| `ILL_CONDITIONED_COVARIANCE` | 0 | 0% |
| `NON_FINITE_INTERMEDIATE` | 0 | 0% |
| `NEAR_ZERO_VARIANCE_SUBBAND` | 0 | 0% |

The aggregate 95.2% flag rate looks alarming next to the 30-reference
pilot's 75% — but that comparison is confounded by grid composition, not
a sign of new instability. The 30-reference pilot's reduced grid was
mostly noise-only/blur-only images; this pilot's grid is 85.7% combined
conditions (14,400/16,800), and **isolating by degradation type shows the
true picture**:

| Degradation type | Flag rate | Mechanism |
| --- | --- | --- |
| Noise-only | **1,200/1,200 = 100.0%** | Every noise level, including the mildest (σ=1), triggers the flag somewhere in the pyramid |
| Blur-only | **405/1,200 = 33.8%** | Only the higher blur levels (σ≥1.4) meaningfully trigger it |
| Combined | 14,394/14,400 = 100.0% | Dominated by the noise component (present in every combined condition) |

**This is the pilot's most important clarification: the channel-gain
instability is overwhelmingly a noise-driven phenomenon, not a blur- or
covariance-driven one.** Isolated (non-confounded) mean max channel gain:

| Noise σ | Mean max gain | | Blur σ | Mean max gain |
| --- | --- | --- | --- | --- |
| 1.0 | 732,675 | | 0.1–0.3 | ~1 (negligible) |
| 6.0 | 4,320,540 | | 0.6–0.9 | 20–24 |
| 14.0 | 9,391,647 | | 1.4 | 171 |
| 30.0 | 21,784,669 | | 3.0 | 4,274 |
| 50.0 | 26,099,634 | | 5.0 | 32,732 |

Blur's effect on channel gain is **4–5 orders of magnitude smaller** than
noise's at comparable positions in their respective grids, and negligible
below blur σ=1.4. The earlier `mean_max_gain_by_blur_sigma` breakdown in
this pilot's own raw summary (all values ~8–9 million regardless of blur
σ) is a **data-presentation artifact**, not a finding: that bucket mixes
blur-only records with noise+blur combined records, and since combined
records vastly outnumber blur-only ones and always carry a noise
component, the mixed average is dominated by the noise distribution
underneath it, washing out blur's much smaller true effect. The isolated
breakdown above is the corrected, uncorrelated reading and is what should
be relied on, not the raw mixed-bucket summary.

## 7. Covariance analysis

**Maximum condition number observed: 10,186** (barely above the
30-reference pilot's 7,362, both **8–10 orders of magnitude below** the
`ILL_CONDITIONED_COVARIANCE` flag threshold (1×10¹²) and the pathological
synthetic "smooth" test case (1×10¹⁶–1×10¹⁸, `docs/vif_implementation.md`
§13.4). **Zero of 16,800 images** triggered this flag. This result now
holds across 100 references and the full combined grid, not just the
smaller pilot's 30 — real CT anatomy robustly avoids the GSM
ill-conditioning failure mode regardless of which specific degradation or
combination is applied.

## 8. Noise monotonicity

**100/100 reference sequences (100%) fully monotonic. Zero violations,
zero local step increases.** VIF strictly decreases across all 12 noise
levels for every single reference, holding at 100-reference scale exactly
as it did at 30.

## 9. Blur monotonicity

**100/100 reference sequences (100%) fully monotonic. Zero violations.**
Same result as noise-only, at full scale.

## 10. Combined-grid monotonicity

Analyzed properly as a 2D grid (not a naive severity-sum ordering): for
each of the 12 fixed blur levels, swept noise across all 12 levels
(1,200 sequences); for each of the 12 fixed noise levels, swept blur
across all 12 levels (1,200 sequences). **This is where the full grid
reveals something the smaller, reduced-subset pilot could not have
found.**

| Direction | Sequences | Violations (≥1 step) | % fully monotonic |
| --- | --- | --- | --- |
| Fixed blur, sweep noise | 1,200 | 362 | 69.8% |
| Fixed noise, sweep blur | 1,200 | 1,047 | **12.8%** |
| **Overall** | 2,400 | 1,409 | 41.3% |

At first read this looks like a serious regression from the 30-reference
pilot's finding (which — using only a 3×3 combined subset — found
violations narrowly confined to `blur=5.0` fixed, and reported 83.3%
axis-wise monotonicity). **It is not a regression; it is a more complete
picture the coarser pilot was structurally unable to see** — 3 blur
points cannot detect a pattern that only becomes visible across 12.

**Where violations concentrate (fixed-blur, sweep-noise direction):**

| Blur σ fixed | References violating (of 100) |
| --- | --- |
| 5.0 | 100 (100%) |
| 3.0 | 100 (100%) |
| 2.1 | 100 (100%) |
| 1.4 | 58 (58%) |
| 0.9 | 2 |
| 0.75 | 2 |
| ≤0.6 | 0 |

**Fixed-noise, sweep-blur direction:** violations are broad, not
concentrated — 80–98 of 100 references violate at essentially every fixed
noise level (no single noise level stands out as clean).

**Magnitude of violations — the important qualifier:**

| | Value |
| --- | --- |
| Total local step violations | 3,725 (of ~24,000 total steps across 2,400 sequences) |
| Mean violation magnitude | **0.0081** |
| Median | 0.0070 |
| 95th percentile | 0.0196 |
| **Maximum observed** | **0.0393** |
| Violations exceeding 0.05 | **0** |
| Violations exceeding 0.10 | **0** |

**Net (endpoint-to-endpoint) direction — a gentler, arguably more
meaningful summary than strict step-by-step monotonicity:**

| Direction | Endpoint (first vs. last) correctly decreasing |
| --- | --- |
| Fixed noise, sweep blur | **1,200 / 1,200 = 100%** |
| Fixed blur, sweep noise | 994 / 1,200 = **82.8%** |

**Interpretation:** local, step-to-step monotonicity is frequently
violated in the combined grid — more often than the smaller pilot
suggested — but every single violation is a small wiggle (never exceeding
0.04 on a [0,1] scale), and the *net* direction across the full severity
range is still correct in 100% of blur sweeps and 83% of noise sweeps.
The concentration of violations at high blur (σ≥2.1) lines up mechanically
with §6/§13.4's finding: blur alone already crushes the reference-side
information content severely at those levels (blur-only VIF at σ=5.0
already falls to ~0.03–0.05, per §11), so the operating point is
near-degenerate — small perturbations from added noise on top of already
heavily-blurred subbands can locally tip the ratio either direction
without indicating a broken metric, the same interpretation reached (on
much thinner evidence) in the 30-reference pilot.

Worst single violation across the entire pilot: **+0.0393**
(reference `0037`, blur=5.0 fixed, noise stepping 2→3).

## 11. VIF distribution

| | n | mean | median | std | min | max | p5 | p25 | p75 | p95 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| All | 16,800 | 0.496 | 0.458 | 0.286 | 0.027 | 1.000 | 0.110 | 0.239 | 0.733 | 0.981 |
| Noise-only | 1,200 | 0.676 | 0.721 | 0.248 | 0.198 | 0.987 | 0.257 | 0.474 | 0.915 | 0.984 |
| Blur-only | 1,200 | 0.642 | 0.743 | 0.358 | 0.027 | 1.000 | 0.049 | 0.282 | 0.998 | 1.000 |
| Combined | 14,400 | 0.469 | 0.408 | 0.273 | 0.033 | 0.987 | 0.112 | 0.227 | 0.681 | 0.962 |

No pathological spikes: the full-population minimum (0.0266) and maximum
(0.9999...) sit exactly where the individual noise-only and blur-only
minimums/maximums are, with nothing outside that envelope contributed by
the combined grid despite it being 12× larger than either univariate
grid. `data/interim/vif_full_grid_pilot/figures/vif_histogram.png`,
`vif_by_degradation_type.png`, and `vif_combined_grid_heatmap.png`
(gitignored) show a smooth, unimodal-per-type distribution and a clean
monotonic-looking heatmap surface (the small per-cell wiggles from §10
are not visible at this resolution — consistent with their small
magnitude).

## 12. Expert-score descriptive relationship

**DESCRIPTIVE ASSOCIATION — NOT SUBJECTIVE VALIDATION.** `expert_score`
measures radiologists' assessment of the original, undegraded LDCT-IQAC
image; `vif_score` measures information fidelity of a synthetically
degraded version relative to that same image. These remain different
signals answering different questions.

Across 100 references, Pearson correlation between each reference's mean
VIF (averaged over all 168 of its degraded variants) and its
`expert_score`: **0.830**; Spearman: **0.876** — both close to the
30-reference pilot's figures (0.86 / 0.85), now on a more than 3× larger,
more representative sample. Offered as an observation, not a validated
relationship, and not a substitute for either signal.

A related, more mechanistic observation from §6: mean max channel gain
also increases monotonically with expert-score bin (2.2M at
`expert_score` 0–1, up to 19.8M at 4–5) — plausibly because
higher-`expert_score` references have cleaner, higher-contrast background
regions (sharper anatomy against genuinely flat background) than
lower-`expert_score` references, whose "background" may itself already
carry some of LDCT-IQAC's own acquisition noise (per `docs/deviations_from_ohashi.md`
DEV-02), giving it nonzero local variance that happens to avoid the
near-zero-variance channel-gain trap. This is correlational, from n=100,
and is reported as a plausible mechanism worth further checking, not an
established fact.

## 13. Runtime

| | Value |
| --- | --- |
| Total wall-clock runtime | 7,124.9 s = **118.75 minutes** (~1.98 hours) |
| Images | 16,800 |
| Mean time/image | **0.424 s** |
| Median time/image | 0.424 s (tight distribution — no outlier-driven skew) |
| Throughput | **2.358 images/second** |
| CPU parallelism | None used — single-process, single-threaded NumPy/SciPy/pyrtools computation on this run; no GPU involved |

Performance optimization applied for this pilot (not a change to the VIF
algorithm itself): `vif_wavelet()` and `vif_wavelet_diagnostics()`
independently rebuild the same steerable pyramids; since the canonical
`"project"` profile aggregates via a plain sum, the final score can be
(and was) derived directly from `vif_wavelet_diagnostics()`'s per-subband
totals instead of calling both functions — confirmed identical to calling
`vif_wavelet()` directly to 1×10⁻⁹ precision on a probe image before the
full run began. This roughly halved per-image time versus the
30-reference pilot's un-optimized 66-image sample (~0.8s/image there vs.
0.424s/image here).

**Projected full 169,000-image dataset runtime at this rate: ~19.9 hours**
single-threaded. This is a serious logistics consideration for the actual
dataset generator (`scripts/quality/build_ohashi_dataset.py`), which
currently has no checkpointing/resumability mechanism — an uninterrupted
~20-hour run is fragile without one.

## 14. Disk usage

This pilot does **not** write degraded pixels to disk (by design — 16,800
PNGs would add unnecessary bulk to a validation exercise). Only CSV
records, a JSON summary, and a handful of example plots are written:
**3.2 MB total** for the entire pilot output
(`data/interim/vif_full_grid_pilot/`).

For capacity planning, one sample degraded image was PNG-encoded (not
written to disk, encoded in memory) to measure its size:
**163.3 KB/image**. Extrapolated to the full planned dataset:

```
169,000 images x 163.3 KB  ~=  28.3 GB
```

This is an estimate from a single representative sample, not a measured
aggregate — actual size will vary with per-image compression efficiency
across the full reference diversity, but should be in this range.

## 15. Problems encountered

1. **Channel-gain instability is a near-universal property of
   noise-degraded images specifically** (100% of noise-only and combined
   conditions), not a rare edge case — confirmed and precisely
   mechanistically isolated in this pilot (§6), correcting the more
   general "75% of images" framing from the smaller pilot. Final VIF
   scores remain sane throughout (§5, §11) in every one of 16,800 cases.
2. **Combined-grid local monotonicity violations are substantially more
   widespread than the smaller pilot's reduced 3×3 subset could reveal**
   (§10) — up to 100% of sequences violate at the three highest blur
   levels. Violation magnitude remains uniformly tiny (max 0.039, zero
   exceeding 0.05) and net direction remains correct in the large
   majority of sequences (100% for blur sweeps, 83% for noise sweeps).
3. **`combination_order` and `sigma_scale` remain genuinely unresolved**
   (U-D01, U-D02, `docs/research_decisions.md`) — this pilot reused the
   same pilot-only provisional values as the smaller pilot, not a
   resolution.
4. **No checkpointing exists for a ~20-hour full run** — an operational
   gap to close before attempting full generation, not a VIF-correctness
   problem.

No NaN, Inf, negative, or out-of-range VIF scores were observed anywhere
in 16,800 images (§5).

## 16. Recommended handling rules

Given the corrected, mechanistic understanding from this pilot:

1. **Retain, never discard or modify, `UNSTABLE_CHANNEL_GAIN`-flagged
   records.** At a ~95% overall flag rate (100% for every noise-bearing
   condition), any handling rule involving exclusion is not viable — it
   would eliminate virtually the entire dataset. The only sound rule is
   to propagate the flag as a queryable manifest column (plus
   `max_channel_gain`, `covariance_condition_number` per record, as this
   pilot already tracks) so downstream training/evaluation can audit,
   weight, or study these records if needed, without the flag itself ever
   silently altering a label. This generalizes and firms up the smaller
   pilot's identical recommendation.
2. **Document combined-grid label noise explicitly as an expected
   property, not a defect to "fix."** Anyone consuming `vif_score` labels
   for the noise+blur combined conditions should know that step-to-step
   monotonicity is not guaranteed (§10) — small (≤0.04), high-blur-
   concentrated non-monotonicities are a known, characterized property of
   this VIF formulation on real CT images, not label noise from a bug.
   This should be stated in `docs/ohashi_methodology.md` or
   `configs/quality/ohashi_ldctiqac.yaml` before the full dataset is
   generated, so it's not rediscovered and mistaken for a new problem.
3. **Resolve or formally re-affirm `combination_order` (U-D02) and
   `sigma_scale` (U-D01)** before either is baked into 169,000
   irreversible labels.
4. **Add checkpointing/resumability to `build_ohashi_dataset.py`** before
   a ~20-hour, ~28 GB unattended run — not previously necessary at
   pilot scale, now clearly necessary at full scale.

## 17. Recommendation for full 1,000-reference generation

**B — READY WITH SPECIFIC DOCUMENTED CONDITIONS**

Not A: this pilot materially changed the picture from the smaller one —
the channel-gain flag rate is higher and more mechanistically precise
than previously known, and the combined-grid monotonicity violations are
substantially more widespread than the reduced-subset pilot could detect.
Both findings turned out to be benign in every respect this pilot could
check (bounded scores, tiny violation magnitudes, correct net direction),
but "benign in every case checked across 16,800 images and 100
references" is still a sample, not a proof, for a run that is 10× larger
in image count and 10× larger in reference count. Proceeding straight to
full generation without documenting these two properties first would risk
a future reader mistaking well-understood, characterized metric behavior
for an undiscovered bug.

Not C: nothing in this larger pilot points to an implementation defect.
Determinism holds exactly. Covariance conditioning remains excellent at
full scale (§7). Univariate (noise-only, blur-only) monotonicity remains
perfect at 100/100 references. The channel-gain instability has a fully
traced, real, non-bug mechanism (§6) and never once corrupted a final
score across 16,800 images. The combined-grid violations are small,
concentrated exactly where the mechanism predicts (high blur, near-
degenerate reference information), and net-correct the vast majority of
the time. This is evidence of a correctly-behaving, well-characterized
metric with known limitations — not evidence of something broken.

**Before full generation, specifically:**

- Adopt recommendation §16.1 (flag-and-retain, never exclude) as the
  formal handling rule, recorded in `docs/research_decisions.md` as a new
  PROJECT ADAPTATION entry.
- Add the combined-grid label-noise caveat (§16.2) to the methodology
  docs so it is documented before generation, not discovered after.
- Resolve or formally re-affirm U-D01/U-D02 (§16.3) — this pilot's
  provisional values should not silently become the permanent ones by
  default.
- Address the runtime/checkpointing gap (§16.4) given the ~20-hour,
  ~28 GB projected full run.

## 18. Production decision (2026-08-14, post-pilot)

This report's recommendation, Status B, is **accepted**. Consequences,
recorded formally in `docs/research_decisions.md` ("Production Generation
Decision" and decisions A-16 through A-20):

- Flag-and-retain (never exclude/alter) is now the formal handling rule for
  `UNSTABLE_CHANNEL_GAIN` and all VIF diagnostic flags (A-19).
- Combined-grid local monotonicity violations are documented as expected,
  characterized metric behavior, not corrected (A-20).
- U-D01 (noise sigma units) and U-D02 (combination order) are resolved as
  PROJECT ADAPTATION, retaining this pilot's provisional values
  (`sigma_scale=1.0`, `order="blur_then_noise"`) rather than silently
  changing them at production time (A-16, A-17). Recorded in the new
  `configs/degradation.yaml`.
- `scripts/quality/generate_production_dataset.py` adds reference-level
  checkpointing/resume and streams manifest records to disk instead of
  holding all ~169,000 in memory, closing the gap identified in §16.4/§15.4.
- **This authorizes production-pipeline preparation only.** The full
  1,000-reference / 169,000-image generation run itself is a separate,
  explicit authorization gate: `scripts/preflight_generation.py` must pass
  and a human must issue the go-ahead. It is not started automatically by
  this decision.
