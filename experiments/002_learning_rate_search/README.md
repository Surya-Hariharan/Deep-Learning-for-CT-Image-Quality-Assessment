# Experiment 002: Learning Rate Search

**Status: complete.**

## Objective

Determine the best learning rate for the Ohashi-style RadImageNet ResNet50
model on LDCT-IQAC, empirically -- using ONLY the validation split. The
300-image test set was never loaded, referenced, or used for any decision
in this experiment.

## Fixed configuration (identical across all four runs)

Adam optimizer, MSE loss, batch size 64, 30 epochs,
seed 42, dropout 0.5, RadImageNet-initialized ResNet50
backbone (freshly reloaded from disk for each run, never carried over from
another run), identical 900/100 train/validation split for all four runs.

## Comparison table (validation only)

| LR | Best Epoch | Best Val Loss | Final Train Loss | Final Val Loss | PLCC | SROCC | KROCC | Status |
|---|---|---|---|---|---|---|---|---|
| 0.01 | 25 | 0.009523 | 0.010098 | 0.092842 | 0.9464 | 0.9551 | 0.8248 | stable |
| 0.001 | 26 | 0.003540 | 0.003809 | 0.004658 | 0.9750 | 0.9730 | 0.8738 | stable |
| 0.0001 | 19 | 0.003980 | 0.003584 | 0.004656 | 0.9722 | 0.9737 | 0.8780 | stable |
| 1e-05 | 29 | 0.008117 | 0.011510 | 0.008117 | 0.9434 | 0.9465 | 0.8174 | stable |

## Selection

**SELECTED LEARNING RATE: 0.001**

Selected by lowest best-validation-loss (0.003540) among stable runs, per the
predefined criterion in docs/replication/reproducibility.md. No test-set metric of any
kind was used to make or influence this selection.

## Ohashi paper context (historical context only -- not a claim of superiority either way)

The Ohashi paper's own reported configuration uses learning rate 1e-3 for its
ResNet50 baseline (see docs/paper/training.md) -- for the paper's own dataset and
VIF-based regression target, not LDCT-IQAC. 
This search's empirically-selected learning rate for LDCT-IQAC (1e-3) happens to match the paper's reported value. This is reported as an observation, not evidence that either dataset's choice is more 'correct' -- the two experiments differ in dataset and label semantics (see docs/replication/dataset_adaptation.md).

## Artifacts

- Per-LR runs: `experiments/002_learning_rate_search/lr_*/` (`checkpoint/best.pt`, `history.json`, `config.json`, `README.md`)
- Comparison table: `results/tables/002_learning_rate_search.csv`
- Figures: `results/figures/002_learning_rate_search_validation_loss.png`, `..._validation_correlation.png`

## Test set status

**TEST SET WAS NOT USED DURING LR SEARCH.** No test image, label, DataLoader, or
metric was constructed or computed anywhere in `notebooks/03_training/02_learning_rate_search.ipynb`.

## Next step

Experiment 003 -- Final Training, using the selected learning rate above. Not started
by this experiment.
