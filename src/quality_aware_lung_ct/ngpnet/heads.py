"""Output heads for NGP-Net.

Migrated unmodified (math-wise) from the author's ``nn/ngpnet.py``.
"""

import torch
from torch import nn, Tensor
from torch.nn import functional as F


class SpatialTransformer(nn.Module):
    ''' Spatial Transformer Network (STN) '''

    def __init__(self,
                 in_size: int = 64,
                 mode_img: str = "bilinear",
                 mode_msk: str = "nearest",
                 spatial_dim: int = 3):
        ''' Args:
        * `in_size`: spatial size of input images.
        * `mode_img`: sample mode for images.
        * `mode_msk`: sample mode for masks.
        * `spatial_dim`: number of spatial dimensions.
        '''
        super(SpatialTransformer, self).__init__()
        self.size = [in_size,] * spatial_dim
        self.mode_img = mode_img
        self.mode_msk = mode_msk

        grid = torch.stack(torch.meshgrid([
            torch.arange(0, s) for s in self.size
        ], indexing="ij")).float().unsqueeze(0)
        self.register_buffer("grid", grid)

    def forward(self, src: Tensor, flow: Tensor, is_msk: bool = False):
        # normalize grid values to [-1,1] for resampler
        loc = self.grid + flow
        for i, s in enumerate(self.size):
            loc[:, i] = 2 * (loc[:, i] / (s - 1) - 0.5)

        # move channels dim to last position
        if len(self.size) == 2:
            loc = loc.permute(0, 2, 3, 1)[..., [1, 0]]
        else:
            loc = loc.permute(0, 2, 3, 4, 1)[..., [2, 1, 0]]

        mode = self.mode_msk if is_msk else self.mode_img
        dst = F.grid_sample(src, loc, mode=mode, align_corners=True)
        return dst
