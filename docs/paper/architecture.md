# Architecture (paper)

**PAPER FACT**, as replicated by this project's ResNet50 baseline
(`src/ct_iqa/models/`):

- Backbone: **ResNet50**.
- Pretraining: **RadImageNet** (not ImageNet).
- Input size: **224x224**.
- The original classification head is **replaced** with a regression head:
  `features -> Dropout -> Dense(1) -> Sigmoid`.
- **Sigmoid** replaces the original network's Softmax output activation,
  since the task is regression (a single continuous quality score), not
  classification.

The paper also evaluates an **InceptionResNetV2** backbone; that is
explicitly **out of scope** for this project (see
`docs/replication/replication_scope.md`) and is not implemented anywhere in
`src/ct_iqa/`.

## What the paper does not specify

- The exact **Dropout probability** before the final Dense layer. The paper
  states a Dropout layer is present but does not give its probability.
  This project does not invent one on the paper's behalf -- see
  `docs/replication/deviations.md` for the explicit, documented choice made
  here instead.
- Whether the ResNet50 implementation follows the "v1" stride placement
  (stride on the 1x1 reduce conv, as in `keras.applications.ResNet50`) or
  the "v1.5" placement (stride on the 3x3 conv, as in `torchvision`'s
  ResNet50) -- these produce numerically different features from the same
  pretrained weights. See `docs/replication/deviations.md` for why this
  project's backbone specifically matches the Keras/RadImageNet convention.
