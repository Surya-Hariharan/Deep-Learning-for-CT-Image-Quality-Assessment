# Documentation index

Start with the root [`README.md`](../README.md). Suggested reading order for the study-v2 rebuild:

1. [PROJECT_OVERVIEW](PROJECT_OVERVIEW.md) - purpose, scope, map, entry points
2. [RESEARCH_QUESTIONS](RESEARCH_QUESTIONS.md) - questions, hypotheses, criteria
3. [LITERATURE_REVIEW](LITERATURE_REVIEW.md) - Ohashi et al., RadImageNet, LDCT-IQAC
4. [DATASET_AND_LABELS](DATASET_AND_LABELS.md) - verified facts, missing metadata
5. [REPOSITORY_ARCHITECTURE](REPOSITORY_ARCHITECTURE.md) and [MODEL_ARCHITECTURE](MODEL_ARCHITECTURE.md) - modules and diagrams
6. [PREPROCESSING_AND_WEIGHTS](PREPROCESSING_AND_WEIGHTS.md) - input pipeline, checksums, BatchNorm epsilon
7. [EXPERIMENT_PROTOCOL](EXPERIMENT_PROTOCOL.md) and [EVALUATION_METRICS](EVALUATION_METRICS.md)
8. Experiment specifications: [A](experiments/EXPERIMENT_A_REFERENCE.md), [B](experiments/EXPERIMENT_B_RANKING_AWARE.md), [C](experiments/EXPERIMENT_C_UNCERTAINTY_AWARE.md)
9. [REPRODUCIBILITY](REPRODUCIBILITY.md), [RESULTS_AND_COMPARISON](RESULTS_AND_COMPARISON.md), [LIMITATIONS_AND_THREATS](LIMITATIONS_AND_THREATS.md), [CHANGELOG](CHANGELOG.md)
10. [ARCHIVED_ARTIFACTS](ARCHIVED_ARTIFACTS.md) - historical notebooks, scripts and results removed from the tree, and how to recover them

Decision records: [ADR-001 model registry](decisions/ADR-001-model-registry.md),
[ADR-002 BatchNorm and preprocessing](decisions/ADR-002-batchnorm-and-preprocessing.md),
[ADR-003 split and test discipline](decisions/ADR-003-split-and-test-discipline.md).

Audit trail: [INITIAL_AUDIT](audit/INITIAL_AUDIT.md), [CLEANUP_MANIFEST](audit/CLEANUP_MANIFEST.md),
[MIGRATION_MAP](audit/MIGRATION_MAP.md), [CHECKPOINT_COMPATIBILITY](audit/CHECKPOINT_COMPATIBILITY.md),
[VALIDATION_REPORT](audit/VALIDATION_REPORT.md), backup hashes in `audit/BACKUP_MANIFEST.sha256`.

Historical material (kept unchanged as the record of experiments 001-003):

- **[`paper/`](paper/)** - what the reference paper (Ohashi et al.) specifies, and nothing else.
- **[`replication/`](replication/)** - how the earlier project adapted that methodology: dataset differences,
  deviations, reproducibility notes, results writeup.
- **[`internal/`](internal/)** - engineering history (structure audits, decisions log). Some paths in these files refer to
  the pre-rebuild layout.
