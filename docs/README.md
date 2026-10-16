# Documentation index

Start with the root [`README.md`](../README.md).

| Document | Contents |
|---|---|
| [project-overview.md](project-overview.md) | purpose, scope, main components, entry points |
| [methodology/EXPERIMENT_PROTOCOL.md](methodology/EXPERIMENT_PROTOCOL.md) | frozen protocol: split, roles, seeds, B grid and selection rule, C calibration, test boundary (`experiment-protocol.md` is a pointer) |
| [methodology/DATASET_AND_EDA.md](methodology/DATASET_AND_EDA.md) | dataset audit, label distributions, anatomy-overlap finding |
| [methodology/PREPROCESSING.md](methodology/PREPROCESSING.md) | pre-registered preprocessing probe, results, checkpoint compatibility |
| [audit/REPOSITORY_AUDIT.md](audit/REPOSITORY_AUDIT.md) | initial repository audit |
| [experiments/COMPARATIVE_ANALYSIS.md](experiments/COMPARATIVE_ANALYSIS.md) | A vs B vs C results, paired comparisons, claim-to-evidence table |
| [experiments/ERROR_ANALYSIS.md](experiments/ERROR_ANALYSIS.md) | error patterns by score bin, worst cases, uncertainty vs error |
| [PROVENANCE.md](PROVENANCE.md) | methodological sources and attribution |
| [experiments/a-reference.md](experiments/a-reference.md) | Experiment A: reference MSE regression |
| [experiments/b-ranking-aware.md](experiments/b-ranking-aware.md) | Experiment B: MSE + pairwise ranking loss |
| [experiments/c-uncertainty-aware.md](experiments/c-uncertainty-aware.md) | Experiment C: Gaussian NLL + split-conformal intervals |
| [dataset-and-labels.md](dataset-and-labels.md) | LDCT-IQAC facts, label scale, known gaps |
| [preprocessing-and-weights.md](preprocessing-and-weights.md) | crop, `2x-1`, weight checksums, BatchNorm epsilon |
| [model-architecture.md](model-architecture.md) | backbone, heads, output spaces |
| [evaluation-metrics.md](evaluation-metrics.md) | PLCC/SROCC/KROCC, errors, interval metrics |
| [limitations-and-threats.md](limitations-and-threats.md) | threats to validity |
| [reproducibility.md](reproducibility.md) | seeds, environments, provenance |
| [decisions/0002-batchnorm-and-preprocessing.md](decisions/0002-batchnorm-and-preprocessing.md) | decision record: BatchNorm epsilon and preprocessing |
| [decisions/0003-split-and-test-discipline.md](decisions/0003-split-and-test-discipline.md) | decision record: split and test-set discipline |
| [archived-artifacts.md](archived-artifacts.md) | removed historical material and how to recover it |

`audit/` holds only the backup hash manifests (`BACKUP_MANIFEST.*`) and the checkpoint-structure report
(`CHECKPOINT_COMPATIBILITY.json`).
