# LR search run: lr_1e-5

**Learning rate: 1e-05**. Part of experiment 002 (learning rate search) --
see `experiments/002_learning_rate_search/README.md` for the cross-LR
comparison and final selection.

Fresh RadImageNet-initialized `OhashiResNet50`, same 900/100 train/validation
split and seed (42) as every other LR in this search. Trained for
30 epochs (Adam, MSE, batch size 64).

- Status: stable
- Best epoch (0-indexed): 29
- Best validation loss (normalized MSE): 0.008117
- Final training loss: 0.011510
- Final validation loss: 0.008117
- Validation correlation on the BEST checkpoint (raw [0,4] scale, validation split only):
  PLCC=0.9434, SROCC=0.9465, KROCC=0.8174, MSE=0.1304

Checkpoint: `checkpoint/best.pt` (model + optimizer state + epoch + val_loss + config + seed).
TEST SET NOT USED -- this run never touched test data.
