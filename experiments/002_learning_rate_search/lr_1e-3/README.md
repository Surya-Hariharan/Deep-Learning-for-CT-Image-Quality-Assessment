# LR search run: lr_1e-3

**Learning rate: 0.001**. Part of experiment 002 (learning rate search) --
see `experiments/002_learning_rate_search/README.md` for the cross-LR
comparison and final selection.

Fresh RadImageNet-initialized `OhashiResNet50`, same 900/100 train/validation
split and seed (42) as every other LR in this search. Trained for
30 epochs (Adam, MSE, batch size 64).

- Status: stable
- Best epoch (0-indexed): 26
- Best validation loss (normalized MSE): 0.003540
- Final training loss: 0.003809
- Final validation loss: 0.004658
- Validation correlation on the BEST checkpoint (raw [0,4] scale, validation split only):
  PLCC=0.9750, SROCC=0.9730, KROCC=0.8738, MSE=0.0563

Checkpoint: `checkpoint/best.pt` (model + optimizer state + epoch + val_loss + config + seed).
TEST SET NOT USED -- this run never touched test data.
