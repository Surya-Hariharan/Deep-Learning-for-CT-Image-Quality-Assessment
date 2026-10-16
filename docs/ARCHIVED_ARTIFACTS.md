# Archived artifacts (removed from the working tree)

To keep the repository focused on the study-v2 experiments (A, B, C), the historical artifacts below were removed from the
working tree. **Nothing was rewritten:** every file is still in git history, so older documents that mention these paths
(`docs/replication/`, `docs/internal/`, `experiments/00x_*/README.md`, parts of `docs/audit/`) remain accurate records of
what existed at the time. A copy also exists outside this repository.

| What | Paths | Removed in | Recover from |
|---|---|---|---|
| Historical result files of experiments 001-003 (19 figures, 7 tables, 2 metrics JSON, 2 prediction CSVs) | `results/figures/`, `results/tables/`, `results/metrics/`, `results/predictions/` | `109f4ba` | `109f4ba^` |
| Historical notebooks (001-003 exploration, preparation, training, LR search, final training, evaluation, analysis) | `notebooks/01_exploration/`, `02_data_preparation/`, `03_training/`, `04_evaluation/` | `9f15916` | `9f15916^` |
| Their generator and evaluation scripts | `tools/build_*_notebook.py` (7 files), `tools/evaluate_baseline_test_set.py` | `9f15916` | `9f15916^` |

## Recovering a file

```bash
# show a file without touching the working tree
git show 109f4ba^:results/metrics/001_resnet50_baseline_test_metrics.json

# restore a whole folder into the working tree
git checkout 9f15916^ -- notebooks/03_training
```

## What stays

Source (`src/ct_iqa/`), tests, configs, the frozen split manifest (`experiments/study_v2/protocol/`), the historical
experiment records (`experiments/001_*`, `002_*`, `003_*`: configs, histories, READMEs), `data/labels/`, the study-v2
notebooks `notebooks/00-07`, and `tools/create_split_manifest.py` / `tools/inspect_checkpoint_compatibility.py`.

The six historical checkpoints and the pretrained RadImageNet weights are not tracked by git (ignored binaries) and were
not removed.

## Caveat

Running the historical notebooks or generators after restoring them can overwrite checkpoints and evaluate the official
test set; see [`audit/VALIDATION_REPORT.md`](audit/VALIDATION_REPORT.md) and the study-v2
[`EXPERIMENT_PROTOCOL.md`](EXPERIMENT_PROTOCOL.md) before doing so.
