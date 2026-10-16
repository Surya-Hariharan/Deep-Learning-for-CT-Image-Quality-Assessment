# Archived artifacts (removed from the working tree)

To keep the repository focused on the study-v2 experiments (A, B, C), historical material was removed from the working
tree. Everything is still in git history, and a copy exists outside this repository.

| What | Paths | Recover from |
|---|---|---|
| Historical result files of experiments 001-003 | `results/figures/`, `results/tables/`, `results/metrics/`, `results/predictions/` | `109f4ba^` |
| Historical notebooks and their generators | `notebooks/01_exploration/`, `02_data_preparation/`, `03_training/`, `04_evaluation/`; `tools/build_*_notebook.py`, `tools/evaluate_baseline_test_set.py` | `9f15916^` |
| Reference-paper summaries, replication notes, engineering history, rebuild audits and superseded docs | `docs/paper/`, `docs/replication/`, `docs/internal/`, `docs/audit/*.md`, `docs/decisions/0001-model-registry.md`, `docs/{research-questions,literature-review,repository-architecture,results-and-comparison}.md`, `docs/CHANGELOG.md` | `c4e3ad7` |

Older documents that mention these paths (for example `experiments/00x_*/README.md` and comments in `src/ct_iqa/config/legacy.py`)
remain accurate records of what existed then.

## Recovering a file

```bash
# show a file without touching the working tree
git show c4e3ad7:docs/replication/deviations.md

# restore a whole folder into the working tree
git checkout 9f15916^ -- notebooks/03_training
```

## What stays

Source (`src/ct_iqa/`), tests, configs, the frozen split manifest (`experiments/study_v2/protocol/`), the historical
experiment records (`experiments/001_*`, `002_*`, `003_*`), `data/labels/`, the study-v2 notebooks `notebooks/00-07`, and
`tools/create_split_manifest.py` / `tools/inspect_checkpoint_compatibility.py`. The six historical checkpoints and the
pretrained RadImageNet weights are not tracked by git (ignored binaries) and were not removed.

## Caveat

Running restored historical notebooks or generators can overwrite checkpoints and evaluate the official test set; read
[`experiment-protocol.md`](experiment-protocol.md) first.
