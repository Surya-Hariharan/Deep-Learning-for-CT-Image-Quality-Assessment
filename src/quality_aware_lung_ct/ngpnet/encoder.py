"""NGP-Net U-Net encoder.

Migrated unmodified (math-wise) from the author's ``nn/ngpnet.py``.
"""

from typing import Sequence

from torch import nn, Tensor

from .layers import DownBlock
from .temporal import BlockGroup


class UNetEncoder(nn.Module):
    ''' Encoder for U-Net '''

    def __init__(self,
                 in_channels: int = 2,
                 depths: Sequence[int] = (2, 2, 2, 2),
                 img_size: int = 64,
                 feat_dim: int = 8,
                 drop_path_rate: float = 0.0,
                 spatial_dim: int = 3):
        ''' Args:
        * `in_channels`: dimension of input channels.
        * `depths`: number of blocks in each stage.
        * `img_size`: spatial size of input images.
        * `feat_dim: output channels of the tokenizer.
        * `drop_path_rate`: stochastic depth rate.
        * `spatial_dim`: number of spatial dimensions.
        '''
        super(UNetEncoder, self).__init__()
        if spatial_dim not in [2, 3]:
            raise ValueError("`spatial_dim` should be 2 or 3.")

        self.tokenizer = nn.Sequential(
            DownBlock(in_channels, feat_dim, spatial_dim=spatial_dim),
        )

        self.encoder1 = BlockGroup(feat_dim=feat_dim,
                                   img_size=img_size//2,
                                   num_blocks=depths[0],
                                   drop_path_rate=drop_path_rate,
                                   spatial_dim=spatial_dim)
        self.downsampler1 = DownBlock(in_channels=feat_dim,
                                      out_channels=feat_dim*2,
                                      spatial_dim=spatial_dim)

        self.encoder2 = BlockGroup(feat_dim=feat_dim*2,
                                   img_size=img_size//4,
                                   num_blocks=depths[1],
                                   drop_path_rate=drop_path_rate,
                                   spatial_dim=spatial_dim)
        self.downsampler2 = DownBlock(in_channels=feat_dim*2,
                                      out_channels=feat_dim*4,
                                      spatial_dim=spatial_dim)

        self.encoder3 = BlockGroup(feat_dim=feat_dim*4,
                                   img_size=img_size//8,
                                   num_blocks=depths[2],
                                   drop_path_rate=drop_path_rate,
                                   spatial_dim=spatial_dim)
        self.downsampler3 = DownBlock(in_channels=feat_dim*4,
                                      out_channels=feat_dim*8,
                                      spatial_dim=spatial_dim)

    def forward(self, xs: Tensor, t0: Tensor):
        e0 = self.tokenizer(xs)
        e1 = self.downsampler1(self.encoder1(e0, t0))
        e2 = self.downsampler2(self.encoder2(e1, t0))
        e3 = self.downsampler3(self.encoder3(e2, t0))
        return (e0, e1, e2, e3)
