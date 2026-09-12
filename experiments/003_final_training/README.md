# Experiment 003: Final Training

**Status: complete.**

## Objective

Final Ohashi-style RadImageNet ResNet50 CT-IQA model for this LDCT-IQAC
adaptation, using the learning rate selected by experiment 002
(**LR = 1e-3**, see `experiments/002_learning_rate_search/README.md`), trained
on ALL 1000 labeled LDCT-IQAC training images. This is an Ohashi-style model
adapted to LDCT-IQAC -- not an exact reproduction of the original paper's
experiment (see docs/replication/dataset_adaptation.md).

## Dataset

- Training samples used: 1000 (ALL labeled LDCT-IQAC training images)
- Validation samples used: 0 -- **IMPLEMENTATION DECISION**: experiment 002 already used the
  900/100 split to select the learning rate; holding out validation data again here would only
  reduce the final model's training data with no remaining selection to justify it. Not a paper fact.
- Test samples used: 0 -- the 300-image test set was never loaded in this experiment.

## Model

`OhashiResNet50` (`src/ct_iqa/models/`): RadImageNet-pretrained ResNet50 backbone
(PAPER FACT: ResNet50, RadImageNet pretraining) -> Global Average Pooling -> Dropout(0.5)
(IMPLEMENTATION DECISION: exact probability not specified by the paper) -> Linear(2048,1) ->
Sigmoid. No architectural change from experiments 001/002.

**RadImageNet initialization: verified.** RadImageNet weights loaded from weights/pretrained/radimagenet/resnet50/radimagenet_resnet50_backbone.pt: 265 matched, 53 missing, 0 unexpected. 53 num_batches_tracked counters left at 0 (expected -- not learned parameters).

## Preprocessing / target

OUR IMPLEMENTATION, unchanged from experiments 001/002: `LDCT-IQAC image [0,1] -> center crop
224x224 (PAPER FACT) -> [-1,1] normalization (IMPLEMENTATION DECISION) -> grayscale replicated to
3 channels inside the model (IMPLEMENTATION DECISION) -> ResNet50`. Target:
raw score [0,4] -> normalize_score -> [0,1] (Sigmoid-compatible).

## Configuration

| | |
|---|---|
| optimizer | Adam |
| loss | MSE (normalized [0,1] target space) |
| learning rate | 0.001 (selected by experiment 002) |
| batch size | 64 |
| epochs | 30 (completed, no early stopping, no validation-based stopping) |
| dropout | 0.5 |
| seed | 42 |

## Environment

Python 3.10.20, PyTorch 2.5.1+cu121, CUDA available: True, GPU: NVIDIA GeForce RTX 4060 Laptop GPU, device=cuda.

## Actual measured results

**These are real, measured numbers from this run -- no validation metric exists for this run
(no validation split), and no test metric was computed (test set untouched).**

- Epochs completed: **30/30**, no NaN/Inf loss at any epoch
  (checked every epoch during training; the run would have stopped immediately and raised if one occurred).
- Final (epoch 30) training loss: **0.001783**
- Training loss trajectory: started at 0.023430, ended at 0.001783.

## Checkpoint

`checkpoint/final.pt` (gitignored) -- **the epoch-30 (final) model, not a
validation-selected best checkpoint** (none exists for this run). Contains `model_state_dict`,
`optimizer_state_dict`, `epoch` (29, 0-indexed), the resolved `config`, `seed`
(42), and `checkpoint_type="final_epoch"`. **Independently reloaded in a fresh model
instance/process** and verified to produce finite, [0,1]-bounded predictions on a real training batch.

## Unexpected events

None. No NaN/Inf loss, no collapsed predictions, gradients present throughout.

## Test set status

**TEST SET NOT USED DURING FINAL TRAINING.** No test image, label, DataLoader, or metric was
constructed or computed anywhere in this experiment.

## What this experiment does NOT show

Test-set performance was not evaluated here. The next step is an independent evaluation of
`checkpoint/final.pt` against the untouched 300-image LDCT-IQAC test set. This experiment also
does not reproduce the original Ohashi paper's numerical results -- different dataset, different
label semantics (see docs/replication/dataset_adaptation.md) -- and no such claim is made here.

## Next step

FINAL INDEPENDENT TEST-SET EVALUATION, using `experiments/003_final_training/checkpoint/final.pt`
against the untouched 300-image LDCT-IQAC test set. Not performed by this experiment.
