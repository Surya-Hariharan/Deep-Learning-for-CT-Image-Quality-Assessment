# Preprocessing (paper)

**PAPER FACT:** "each image was cropped to the central region according to
the input size required by the model" -- i.e. a **center crop** to
224x224, explicitly **not** a resize, in order to preserve the original CT
images' spatial resolution rather than compressing/interpolating it away.

## What the paper does not specify

- **Pixel value normalization formula.** The paper does not state the
  exact normalization applied to pixel values before feeding them to the
  RadImageNet-pretrained backbone. See `docs/replication/deviations.md` for
  the normalization this project uses instead (RadImageNet's own
  convention, confirmed from RadImageNet's official reference code rather
  than assumed).
- **Source image format.** The paper's own dataset construction/format is
  not something this project has independently verified beyond what is
  needed to replicate the preprocessing steps above. See
  `docs/replication/dataset_adaptation.md` for this project's dataset
  (LDCT-IQAC) and its actual on-disk format (TIFF).
- **Synthetic degradation.** The paper's own methodology (as referenced by
  this project's task brief) constructs training data by applying
  synthetic degradation (e.g. noise, blur) to clean CT images and computing
  a **VIF** (Visual Information Fidelity) score as the regression target.
  **This project's current implementation does not perform this step at
  all** -- see `docs/replication/deviations.md` for why.
