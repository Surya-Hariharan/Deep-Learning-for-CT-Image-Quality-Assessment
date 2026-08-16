"""Time-conditioning modules for NGP-Net.

Migrated unmodified (math-wise) from the author's ``nn/net_utils.py``.
"""

import math

import torch
from torch import nn, Tensor

from .layers import LayerNorm, GlobalRespNorm, DropPath


class TEM(nn.Module):
    ''' Time Embedding Module (TEM) '''

    def __init__(self,
                 time_dim: int = 8,
                 img_size: int = 32,
                 spatial_dim: int = 3):
        ''' Args:
        * `time_dim`: dimension of time embedding.
        * `img_size`: size of feature map.
        * `spatial_dim`: number of spatial dimensions.
        '''
        super(TEM, self).__init__()
        self.shape = [time_dim] + [1] * spatial_dim

        half = (img_size ** spatial_dim) // 2
        scale = -math.log(1e4) / (half - 1)
        self._2i = (scale * torch.arange(half)).exp()

        self.linear = nn.Sequential(
            nn.Linear(2 * half, time_dim),
            nn.SiLU(inplace=True),
            nn.Linear(time_dim, time_dim),
        )

    def forward(self, t: Tensor):
        if self._2i.device != t.device:
            self._2i = self._2i.to(t.device)
        itv = t[:, None] * self._2i[None, :]
        emb = torch.cat((itv.sin(), itv.cos()), dim=-1)
        emb = self.linear(emb).view(t.shape[0], *self.shape)
        return emb


class STEM(nn.Module):
    ''' Spatial-Temporal Encoding Module (STEM) '''

    def __init__(self,
                 in_dim: int,
                 drop_path_rate: float = 0.0,
                 spatial_dim: int = 3):
        ''' Args:
        * `in_dim`: dimension of input channels.
        * `drop_path_rate`: stochastic depth rate.
        * `spatial_dim`: number of spatial dimensions.
        '''
        super(STEM, self).__init__()
        if spatial_dim in [2, 3]:
            Conv = nn.Conv3d if spatial_dim == 3 else nn.Conv2d
        else:
            raise ValueError("`spatial_dim` should be 2 or 3.")

        ### spatial embedding:
        self.norm = LayerNorm(in_dim, eps=1e-6, channels_last=False)
        self.qk_sp = Conv(in_dim, in_dim, 1, groups=in_dim)
        self.v_sp = Conv(in_dim, in_dim, 1, groups=in_dim)
        ### temporal embedding:
        self.qk_tm = Conv(in_dim, in_dim, 1, groups=in_dim)
        self.v_tm = Conv(in_dim, in_dim, 1, groups=in_dim)
        ### attention:
        self.swish = nn.SiLU(inplace=True)
        self.attn = Conv(in_dim, in_dim, 4, padding=3, groups=in_dim, dilation=2)
        ### projection:
        self.proj = nn.Sequential(
            GlobalRespNorm(in_dim, spatial_dim=spatial_dim,
                           eps=1e-6, channels_last=False),
            Conv(in_dim, in_dim, 1, groups=in_dim),
            DropPath(drop_path_rate, scale_by_keep=True),
        )

    def forward(self, xs: Tensor, xt: Tensor):
        x0, xs = xs, self.norm(xs)
        ### query,key,value:
        qk = self.qk_sp(xs) + self.qk_tm(xt)
        v = self.v_sp(xs) + self.v_tm(xt)
        ### attention:
        x = self.attn(self.swish(qk)) * v
        ### projection:
        y = x0 + self.proj(x)
        return y


class ReSTEBlock(nn.Module):
    ''' Residual Spatial-Temporal Encoding Block (ReSTE block) '''

    def __init__(self,
                 in_dim: int,
                 drop_path_rate: float = 0.0,
                 spatial_dim: int = 3):
        ''' Args:
        * `in_dim`: dimension of input channels.
        * `drop_path_rate`: stochastic depth rate.
        * `spatial_dim`: number of spatial dimensions.
        '''
        super(ReSTEBlock, self).__init__()
        Conv = nn.Conv2d if spatial_dim == 2 else nn.Conv3d

        self.conv = Conv(in_dim, in_dim, 3, padding=1)
        self.swish = nn.SiLU(inplace=True)
        self.norm = nn.GroupNorm(num_groups=in_dim//4,
                                 num_channels=in_dim)
        self.block = STEM(in_dim=in_dim,
                          spatial_dim=spatial_dim,
                          drop_path_rate=drop_path_rate)

    def forward(self, x: Tensor, t: Tensor):
        y = self.conv(self.swish(self.norm(x)))
        y = x + self.block(y, t)
        return y


class BlockGroup(nn.Module):
    ''' ReSTE Block Group '''

    def __init__(self,
                 feat_dim: int = 16,
                 img_size: int = 64,
                 num_blocks: int = 2,
                 drop_path_rate: float = 0.0,
                 spatial_dim: int = 3):
        ''' Args:
        * `feat_dim`: channels of image features.
        * `img_size`: size of image features.
        * `num_blocks`: number of blocks in each stage.
        * `drop_path_rate`: stochastic depth rate.
        * `spatial_dim`: number of spatial dimensions.
        '''
        super(BlockGroup, self).__init__()

        self.time_emb = TEM(time_dim=feat_dim,
                            img_size=img_size,
                            spatial_dim=spatial_dim)
        self.blocks = nn.ModuleList([
            ReSTEBlock(in_dim=feat_dim,
                       drop_path_rate=drop_path_rate,
                       spatial_dim=spatial_dim)
            for _ in range(num_blocks)
        ])

    def forward(self, xs: Tensor, t: Tensor):
        xt = self.time_emb(t)
        for block in self.blocks:
            xs = block(xs, xt)
        return xs
