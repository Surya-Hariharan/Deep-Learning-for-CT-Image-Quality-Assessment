# Paper summary

**Reference:** Ohashi et al., *"Development of a No-Reference CT Image
Quality Assessment Method Using RadImageNet Pre-trained Deep Learning
Models."*

Everything in `docs/paper/` describes what this paper states. It does not
describe this project's implementation, adaptations, or results -- those
live in `docs/replication/`. Where this project's source code or
documentation is uncertain about an exact paper detail (e.g. an exact
hyperparameter value not confirmed against the primary source), that
uncertainty is called out explicitly rather than presented as fact.

## What the paper does (as understood from this project's working notes)

The paper proposes a **no-reference** (no clean reference image required)
CT image quality assessment method: a deep convolutional network,
pretrained on **RadImageNet** (a radiology-specific pretraining corpus,
distinct from ImageNet), fine-tuned as a **regressor** to predict an image
quality score directly from a CT image.

Two backbone architectures are evaluated in the paper: **ResNet50** and
**InceptionResNetV2**. This project implements and replicates **only the
ResNet50 baseline** -- see `docs/replication/replication_scope.md`.

## Files in this section

- `architecture.md` -- backbone, head, input size, activation functions.
- `preprocessing.md` -- image preprocessing as the paper describes it.
- `training.md` -- optimizer, loss, batch size, epochs, learning rate.
- `evaluation.md` -- correlation metrics and the 5-parameter-logistic
  mapping used before computing them.

Each of those files should be read alongside its counterpart in
`docs/replication/` (particularly `docs/replication/deviations.md`), which
records anywhere this project's implementation could not, or deliberately
did not, match the paper exactly.
