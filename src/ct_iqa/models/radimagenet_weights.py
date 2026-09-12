"""Convert the Keras RadImageNet-ResNet50 `.h5` weights into a PyTorch
state-dict compatible with `ct_iqa.models.resnet50.ResNet50Backbone`.

Moved from `scripts/convert_radimagenet_weights.py` into the `ct_iqa`
package (see docs/repository_architecture_audit.md): this is reusable,
tested implementation tied specifically to `ResNet50Backbone`'s layout, not
a one-off script, and is already imported as library code by
`tests/integration/test_radimagenet_weights_integration.py`. The conversion
mathematics are unchanged from the pre-move implementation.

Source: weights/pretrained/radimagenet/resnet50/RadImageNet-ResNet50_notop.h5
    - `keras.applications.ResNet50`-architecture weights (confirmed via the
      file's embedded `model_config`: 53 Conv2D + 53 BatchNormalization
      layers, standard `convX_blockY_Z_{conv,bn}` naming, no classification
      head).
    - Conv2D kernels are stored HWIO (kernel_h, kernel_w, in, out); PyTorch
      `nn.Conv2d` expects OIHW -- transposed here via `.transpose(3, 2, 0, 1)`.
    - Conv2D layers carry a bias; `ResNet50Backbone`'s convs do not
      (`bias=False`, standard when immediately followed by BatchNorm). This
      is losslessly foldable into the following BatchNorm's running_mean:
      for y = conv(x) + b  ->  BN((y - moving_mean) / sqrt(var + eps)),
      using a bias-free conv with `moving_mean' = moving_mean - b` produces
      an IDENTICAL output, since the bias only ever appears as a constant
      shift immediately absorbed by the mean-subtraction. This is an exact
      re-parameterization, not an approximation.
    - Keras's default BatchNormalization epsilon is 1e-3; PyTorch's default
      `nn.BatchNorm2d` epsilon is 1e-5. This module does not change that
      here (eps is set at load time in `ohashi_resnet50.load_radimagenet_weights`,
      not baked into this checkpoint) -- see that function for why.
    - Stride-2 downsampling in Keras's ResNet50 is applied on the block's
      FIRST 1x1 (reduce) conv (`*_block1_1_conv`) and the shortcut conv
      (`*_block1_0_conv`), never on the 3x3 conv. `ResNet50Backbone` was
      updated (see `ct_iqa/models/resnet50.py`) to match this placement
      exactly, so no stride remapping is needed here -- key-for-key,
      shape-for-shape correspondence holds.

Usage:
    python -m ct_iqa.models.radimagenet_weights \
        --input weights/pretrained/radimagenet/resnet50/RadImageNet-ResNet50_notop.h5 \
        --output weights/pretrained/radimagenet/resnet50/radimagenet_resnet50_backbone.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
import torch

# Keras stage prefix -> our stage name, and block count per stage (must match
# ResNet50Backbone.BLOCKS_PER_STAGE = (3, 4, 6, 3)).
_STAGE_PREFIXES = {"conv2": "stage1", "conv3": "stage2", "conv4": "stage3", "conv5": "stage4"}
_BLOCKS_PER_STAGE = {"conv2": 3, "conv3": 4, "conv4": 6, "conv5": 3}

# Keras sub-layer index -> our conv/bn name.
_SUBLAYER_MAP = {1: "conv1", 2: "conv2", 3: "conv3"}  # "0" (shortcut) handled separately

_DEFAULT_H5_PATH = "weights/pretrained/radimagenet/resnet50/RadImageNet-ResNet50_notop.h5"
_DEFAULT_PT_PATH = "weights/pretrained/radimagenet/resnet50/radimagenet_resnet50_backbone.pt"


def _get_array(h5file: h5py.File, layer_name: str, dataset_name: str) -> np.ndarray:
    group = h5file["model_weights"][layer_name][layer_name]
    return np.array(group[dataset_name])


def _convert_conv_bn_pair(
    h5file: h5py.File, conv_name: str, bn_name: str
) -> dict[str, torch.Tensor]:
    """Convert one (Conv2D, BatchNorm) pair, folding the conv's bias into the BN's running_mean."""
    kernel = _get_array(h5file, conv_name, "kernel:0")  # (H, W, in, out)
    bias = _get_array(h5file, conv_name, "bias:0")  # (out,)
    weight = np.transpose(kernel, (3, 2, 0, 1))  # -> (out, in, H, W), PyTorch OIHW

    gamma = _get_array(h5file, bn_name, "gamma:0")
    beta = _get_array(h5file, bn_name, "beta:0")
    moving_mean = _get_array(h5file, bn_name, "moving_mean:0")
    moving_variance = _get_array(h5file, bn_name, "moving_variance:0")

    # Exact re-parameterization: fold conv bias into BN's running_mean so a
    # bias-free conv (as used by ResNet50Backbone) produces identical output.
    adjusted_running_mean = moving_mean - bias

    return {
        "conv.weight": torch.from_numpy(weight.copy()),
        "bn.weight": torch.from_numpy(gamma.copy()),
        "bn.bias": torch.from_numpy(beta.copy()),
        "bn.running_mean": torch.from_numpy(adjusted_running_mean.copy()),
        "bn.running_var": torch.from_numpy(moving_variance.copy()),
    }


def convert(h5_path: str | Path) -> dict[str, torch.Tensor]:
    """Convert the full RadImageNet ResNet50 `.h5` file into a
    `ResNet50Backbone`-compatible state-dict (missing only
    `num_batches_tracked`, which `load_state_dict(..., strict=False)`
    leaves at its freshly-initialized value of 0 -- an inference-time
    bookkeeping counter, not a learned parameter).
    """
    state_dict: dict[str, torch.Tensor] = {}

    with h5py.File(h5_path, "r") as f:
        # --- stem: conv1_conv + conv1_bn -> stem.0 (conv) + stem.1 (bn) ---
        stem = _convert_conv_bn_pair(f, "conv1_conv", "conv1_bn")
        state_dict["stem.0.weight"] = stem["conv.weight"]
        state_dict["stem.1.weight"] = stem["bn.weight"]
        state_dict["stem.1.bias"] = stem["bn.bias"]
        state_dict["stem.1.running_mean"] = stem["bn.running_mean"]
        state_dict["stem.1.running_var"] = stem["bn.running_var"]

        # --- 4 stages, each with N bottleneck blocks ---
        for keras_prefix, our_stage in _STAGE_PREFIXES.items():
            n_blocks = _BLOCKS_PER_STAGE[keras_prefix]
            for block_idx in range(1, n_blocks + 1):  # Keras blocks are 1-indexed
                our_block = block_idx - 1  # our nn.Sequential blocks are 0-indexed
                block_prefix = f"{keras_prefix}_block{block_idx}"
                our_prefix = f"{our_stage}.{our_block}"

                for keras_sub, our_sub in _SUBLAYER_MAP.items():
                    conv_name = f"{block_prefix}_{keras_sub}_conv"
                    bn_name = f"{block_prefix}_{keras_sub}_bn"
                    converted = _convert_conv_bn_pair(f, conv_name, bn_name)
                    state_dict[f"{our_prefix}.{our_sub}.weight"] = converted["conv.weight"]
                    state_dict[f"{our_prefix}.bn{keras_sub}.weight"] = converted["bn.weight"]
                    state_dict[f"{our_prefix}.bn{keras_sub}.bias"] = converted["bn.bias"]
                    state_dict[f"{our_prefix}.bn{keras_sub}.running_mean"] = converted["bn.running_mean"]
                    state_dict[f"{our_prefix}.bn{keras_sub}.running_var"] = converted["bn.running_var"]

                # Shortcut/downsample conv+bn -- present only on the first block of each stage.
                shortcut_conv = f"{block_prefix}_0_conv"
                if block_idx == 1:
                    shortcut_bn = f"{block_prefix}_0_bn"
                    converted = _convert_conv_bn_pair(f, shortcut_conv, shortcut_bn)
                    state_dict[f"{our_prefix}.downsample.0.weight"] = converted["conv.weight"]
                    state_dict[f"{our_prefix}.downsample.1.weight"] = converted["bn.weight"]
                    state_dict[f"{our_prefix}.downsample.1.bias"] = converted["bn.bias"]
                    state_dict[f"{our_prefix}.downsample.1.running_mean"] = converted["bn.running_mean"]
                    state_dict[f"{our_prefix}.downsample.1.running_var"] = converted["bn.running_var"]

    return state_dict


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=_DEFAULT_H5_PATH)
    parser.add_argument("--output", default=_DEFAULT_PT_PATH)
    args = parser.parse_args()

    state_dict = convert(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state_dict, output_path)
    print(f"converted {len(state_dict)} tensors -> {output_path}")


if __name__ == "__main__":
    main()
