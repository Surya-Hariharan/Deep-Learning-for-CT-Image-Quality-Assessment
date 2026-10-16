# Documentation index

Start with the root [`README.md`](../README.md). Suggested reading order for the study-v2 rebuild:

1. [Project Overview](project-overview.md) - purpose, scope, map, entry points
2. [Research Questions](research-questions.md) - questions, hypotheses, criteria
3. [Literature Review](literature-review.md) - Ohashi et al., RadImageNet, LDCT-IQAC
4. [Dataset and Labels](dataset-and-labels.md) - verified facts, missing metadata
5. [Repository Architecture](repository-architecture.md) and [Model Architecture](model-architecture.md) - modules and diagrams
6. [Preprocessing and Weights](preprocessing-and-weights.md) - input pipeline, checksums, BatchNorm epsilon
7. [Experiment Protocol](experiment-protocol.md) and [Evaluation Metrics](evaluation-metrics.md)
8. Experiment specifications: [A](experiments/a-reference.md), [B](experiments/b-ranking-aware.md), [C](experiments/c-uncertainty-aware.md)
9. [Reproducibility](reproducibility.md), [Results and Comparison](results-and-comparison.md), [Limitations and Threats](limitations-and-threats.md), [Changelog](CHANGELOG.md)
10. [Archived Artifacts](archived-artifacts.md) - historical notebooks, scripts and results removed from the tree, and how to recover them

Decision records: [ADR 0001: model registry](decisions/0001-model-registry.md),
[ADR 0002: BatchNorm and preprocessing](decisions/0002-batchnorm-and-preprocessing.md),
[ADR 0003: split and test discipline](decisions/0003-split-and-test-discipline.md).

Audit trail: [Initial Audit](audit/initial-audit.md), [Cleanup Manifest](audit/CLEANUP_MANIFEST.md),
[Migration Map](audit/MIGRATION_MAP.md), [Checkpoint Compatibility](audit/checkpoint-compatibility.md),
[Validation Report](audit/validation-report.md), backup hashes in `audit/BACKUP_MANIFEST.sha256`.

Historical material (kept unchanged as the record of experiments 001-003):

- **[`paper/`](paper/)** - what the reference paper (Ohashi et al.) specifies, and nothing else.
- **[`replication/`](replication/)** - how the earlier project adapted that methodology: dataset differences,
  deviations, reproducibility notes, results writeup.
- **[`internal/`](internal/)** - engineering history (structure audits, decisions log). Some paths in these files refer to
  the pre-rebuild layout.
