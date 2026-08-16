"""NGP-Net U-Net decoder (shared by the shape and texture branches).

Migrated unmodified (math-wise) from the author's ``nn/ngpnet.py``.
"""

from typing import Sequence

from torch import nn, Tensor

from .layers import UpBlock
from .temporal import BlockGroup


class UNetDecoder(nn.Module):
    ''' Decoder for U-Net '''

    def __init__(self,
                 out_channels: int = 3,
                 depths: Sequence[int] = (2, 2, 2, 2),
                 img_size: int = 64,
                 feat_dim: int = 8,
                 drop_path_rate: float = 0.0,
                 spatial_dim: int = 3):
        ''' Args:
        * `out_channels`: dimension of output channels.
        * `depths`: number of blocks in each stage.
        * `img_size`: spatial size of input images.
        * `feat_dim: output channels of the tokenizer.
        * `drop_path_rate`: stochastic depth rate.
        * `spatial_dim`: number of spatial dimensions.
        '''
        super(UNetDecoder, self).__init__()
        if spatial_dim in [2, 3]:
            Conv = nn.Conv3d if spatial_dim == 3 else nn.Conv2d
        else:
            raise ValueError("`spatial_dim` should be 2 or 3.")

        self.center = BlockGroup(feat_dim=feat_dim*8,
                                 img_size=img_size//16,
                                 num_blocks=depths[3],
                                 drop_path_rate=drop_path_rate,
                                 spatial_dim=spatial_dim)

        self.upsampler3 = UpBlock(in_channels=feat_dim*8,
                                  out_channels=feat_dim*4,
                                  spatial_dim=spatial_dim)
        self.decoder3 = BlockGroup(feat_dim=feat_dim*4,
                                   img_size=img_size//8,
                                   num_blocks=depths[2],
                                   drop_path_rate=drop_path_rate,
                                   spatial_dim=spatial_dim)

        self.upsampler2 = UpBlock(in_channels=feat_dim*4,
                                  out_channels=feat_dim*2,
                                  spatial_dim=spatial_dim)
        self.decoder2 = BlockGroup(feat_dim=feat_dim*2,
                                   img_size=img_size//4,
                                   num_blocks=depths[1],
                                   drop_path_rate=drop_path_rate,
                                   spatial_dim=spatial_dim)

        self.upsampler1 = UpBlock(in_channels=feat_dim*2,
                                  out_channels=feat_dim,
                                  spatial_dim=spatial_dim)
        self.decoder1 = BlockGroup(feat_dim=feat_dim,
                                   img_size=img_size//2,
                                   num_blocks=depths[0],
                                   drop_path_rate=drop_path_rate,
                                   spatial_dim=spatial_dim)

        self.out_head = nn.Sequential(
            UpBlock(feat_dim, feat_dim, spatial_dim=spatial_dim),
            nn.GroupNorm(num_groups=feat_dim//4, num_channels=feat_dim),
            Conv(feat_dim, out_channels, 3, stride=1, padding=1)
        )

    def forward(self, xs: Tensor, t1: Tensor):
        (e0, e1, e2, e3) = xs
        d3 = self.center(e3, t1)
        d2 = self.decoder3(self.upsampler3(e3 + d3), t1)
        d1 = self.decoder2(self.upsampler2(e2 + d2), t1)
        d0 = self.decoder1(self.upsampler1(e1 + d1), t1)
        out = self.out_head(e0 + d0)
        return out
