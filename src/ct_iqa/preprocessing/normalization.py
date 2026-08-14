"""Input normalization for the RadImageNet-pretrained ResNet50 backbone.

Resolves U-M05 (docs/research_decisions.md, decision A-23): neither
Ohashi's paper nor the RadImageNet paper (Mei et al. 2022) specifies an
input normalization scheme. This project's PROJECT-ADAPTATION baseline
replicates ``tensorflow.keras.applications.resnet50.preprocess_input``'s
default "caffe" mode -- RGB->BGR channel reorder, then per-channel
ImageNet mean subtraction, applied to raw ``[0, 255]``-scale pixel values
-- in pure numpy, so it works in this project's main (TensorFlow-free)
Python environment. See decision A-23 for the full evidence trace,
including the official RadImageNet repository's own downstream example
code, which was found to combine this transform with an inconsistent
extra ``/255`` rescale -- deliberately NOT reproduced here.

NOT a claim that this is what Ohashi's own pipeline did, and NOT a claim
that this is confirmed to be how RadImageNet's own checkpoint was itself
trained.
"""

from __future__ import annotations

import numpy as np

# ImageNet channel means, BGR order -- Keras's own published constants
# (`keras.applications.imagenet_utils.preprocess_input`, mode="caffe", the
# default for `resnet50.preprocess_input`). REFERENCE-IMPLEMENTATION-CONVENTION:
# these are Keras/ImageNet constants, not derived from RadImageNet's or
# LDCT-IQAC's own pixel statistics -- decision A-23 records this explicitly.
BGR_MEAN = np.array([103.939, 116.779, 123.68], dtype=np.float64)


def resnet50_preprocess_input(image: np.ndarray) -> np.ndarray:
    """Replicate ``tf.keras.applications.resnet50.preprocess_input``'s
    default ("caffe") behaviour: RGB -> BGR, then per-channel mean
    subtraction. No additional rescale -- decision A-23 explains why
    combining this with ``/255`` (as RadImageNet's own downstream example
    scripts do) is deliberately not reproduced here.

    Args:
        image: array shaped ``(..., 3)``, RGB channel order, values on the
            raw ``[0, 255]`` scale (uint8 or float). Passing an
            already-``[0, 1]``-rescaled array will silently produce the
            same internally-inconsistent result this decision avoids --
            the caller is responsible for keeping the scale raw.

    Returns:
        ``float64`` array, same shape, BGR channel order, mean-subtracted.
    """
    image = np.asarray(image, dtype=np.float64)
    if image.ndim < 1 or image.shape[-1] != 3:
        raise ValueError(f"expected an array with a trailing 3-channel (RGB) axis, got shape {image.shape}")
    bgr = image[..., ::-1]
    return bgr - BGR_MEAN
