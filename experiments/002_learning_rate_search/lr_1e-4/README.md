# LR search run: lr_1e-4

**Learning rate: 0.0001**. Part of experiment 002 (learning rate search) --
see `experiments/002_learning_rate_search/README.md` for the cross-LR
comparison and final selection.

Fresh RadImageNet-initialized `OhashiResNet50`, same 900/100 train/validation
split and seed (42) as every other LR in this search. Trained for
30 epochs (Adam, MSE, batch size 64).

- Status: stable
- Best epoch (0-indexed): 19
- Best validation loss (normalized MSE): 0.003980
- Final training loss: 0.003584
- Final validation loss: 0.004656
- Validation correlation on the BEST checkpoint (raw [0,4] scale, validation split only):
  PLCC=0.9722, SROCC=0.9737, KROCC=0.8780, MSE=0.0637

Checkpoint: `checkpoint/best.pt` (model + optimizer state + epoch + val_loss + config + seed).
TEST SET NOT USED -- this run never touched test data.
