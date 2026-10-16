# Documentation index

Start with the root [`README.md`](../README.md).

| Document | Contents |
|---|---|
| [project-overview.md](project-overview.md) | purpose, scope, main components, entry points |
| [experiment-protocol.md](experiment-protocol.md) | frozen split, roles of each partition, comparison and test-set rules |
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
