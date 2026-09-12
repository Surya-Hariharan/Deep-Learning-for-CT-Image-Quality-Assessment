# Evaluation (paper)

**PAPER FACT**, as replicated in `src/ct_iqa/evaluation/`:

- Correlation is reported via **PLCC** (Pearson linear correlation
  coefficient), **SROCC** (Spearman rank-order correlation coefficient),
  and **KROCC** (Kendall rank-order correlation coefficient) --
  `src/ct_iqa/evaluation/correlation.py`.
- Before computing PLCC/SROCC, predicted scores are mapped onto the
  ground-truth scale via a **five-parameter logistic (5PL) fit** --
  standard VQEG/ITU-T-style IQA evaluation methodology --
  `src/ct_iqa/evaluation/calibration.py`.

## Implementation note

This project reports **both** raw-space metrics (`compute_metrics`, no
mapping) and 5PL-mapped metrics (`compute_metrics_with_logistic_mapping`),
labeled separately, rather than only the paper's 5PL-mapped numbers -- see
`ct_iqa.evaluation.metrics`'s module docstring. This is an
**implementation decision** (more transparent reporting), not a deviation
from what the paper measures.
