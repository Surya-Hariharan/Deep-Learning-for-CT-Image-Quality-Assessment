"""Generic building blocks for NGP-Net.

Migrated unmodified (math-wise) from the author's ``nn/net_utils.py``.
"""

from typing import Sequence, Union

import torch
from torch import nn, Tensor
from torch.nn import functional as F


def layer_norm(x: Tensor,
               norm_shape: Sequence[int],
               eps: float = 1e-6,
               channels_last: bool = True):
    if channels_last:   # [B, ..., C]
        y = F.layer_norm(x, norm_shape, eps=eps)
    else:               # [B, C, ...]
        mean = x.mean(1, keepdim=True)
        var = (x - mean).pow(2).mean(1, keepdim=True)
        y = (x - mean) / torch.sqrt(var + eps)
    return y


class DropPath(nn.Module):
    ''' Stochastic drop paths per sample for residual blocks.

        Reference: https://github.com/rwightman/pytorch-image-models
    '''

    def __init__(self,
                 drop_prob: float = 0.0,
                 scale_by_keep: bool = True):
        ''' Args:
            * `drop_prob`: drop paths probability.
            * `scale_by_keep`: whether scaling by non-dropped probaility.
        '''
        super(DropPath, self).__init__()
        if drop_prob < 0 or drop_prob > 1:
            raise ValueError("drop_path_prob should be between 0 and 1.")
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x: Tensor):
        if self.drop_prob == 0.0:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        rand_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        if self.scale_by_keep and keep_prob > 0.0:
            rand_tensor.div_(keep_prob)
        return x * rand_tensor


class LayerNorm(nn.Module):
    ''' Layer Normalization (LN) '''

    def __init__(self,
                 norm_shape: Union[Sequence[int], int],
                 eps: float = 1e-6,
                 channels_last: bool = True):
        ''' Args:
        * `norm_shape`: dimension of the input feature.
        * `eps`: epsilon of layer normalization.
        * `channels_last`: whether the channel is the last dim.
        '''
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(norm_shape), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(norm_shape), requires_grad=True)
        self.channels_last = channels_last
        self.norm_shape = (norm_shape,)
        self.eps = eps

    def forward(self, x: Tensor):
        if self.channels_last:  # [B, ..., C]
            y = F.layer_norm(x, self.norm_shape,
                             self.weight, self.bias, self.eps)
        else:                   # [B, C, ...]
            y = layer_norm(x, self.norm_shape, self.eps, False)
            if x.ndim == 4:
                y = self.weight[:, None, None] * y
                y += self.bias[:, None, None]
            else:
                y = self.weight[:, None, None, None] * y
                y += self.bias[:, None, None, None]
        return y


class GlobalRespNorm(nn.Module):
    ''' Global Response Normalization (GRN) '''

    def __init__(self,
                 dim: int,
                 eps: float = 1e-6,
                 channels_last: bool = False,
                 spatial_dim: int = 3):
        ''' Args:
        * `dim`: dimension of input channels.
        * `eps`: epsilon of the normalization.
        * `channels_last`: whether the channel is the last dim.
        * `spatial_dim`: number of spatial dimensions.
        '''
        super(GlobalRespNorm, self).__init__()
        if spatial_dim == 2:
            if channels_last:
                size, self.dims = [1, 1, 1, dim], (1, 2)
            else:
                size, self.dims = [1, dim, 1, 1], (2, 3)
        else:
            if channels_last:
                size, self.dims = [1, 1, 1, 1, dim], (1, 2, 3)
            else:
                size, self.dims = [1, dim, 1, 1, 1], (2, 3, 4)

        self.eps = eps
        self.gamma = nn.Parameter(torch.zeros(*size))
        self.beta = nn.Parameter(torch.zeros(*size))

    def forward(self, x: Tensor):
        ''' Args:
        * `x`: a input tensor in [B, C, (D,) W, H] shape.
        '''
        gx = torch.norm(x, p=2, dim=self.dims, keepdim=True)
        nx = gx / (gx.mean(dim=1, keepdim=True) + self.eps)
        x = self.gamma * (x * nx) + self.beta + x
        return x


class DownBlock(nn.Module):
    ''' Downsampling Block '''

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 3,
                 stride: int = 2,
                 padding: int = 1,
                 spatial_dim: int = 3):
        super(DownBlock, self).__init__()
        if spatial_dim in [2, 3]:
            Conv = nn.Conv3d if spatial_dim == 3 else nn.Conv2d
        else:
            raise ValueError("`spatial_dim` should be 2 or 3.")

        self.conv = Conv(in_channels, out_channels, kernel_size, stride, padding)

    def forward(self, x: Tensor):
        return self.conv(x)


class UpBlock(nn.Module):
    ''' Upsampling Block '''

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 3,
                 stride: int = 2,
                 padding: int = 1,
                 output_padding: int = 1,
                 spatial_dim: int = 3):
        super(UpBlock, self).__init__()
        if spatial_dim in [2, 3]:
            ConvT = nn.ConvTranspose3d if spatial_dim == 3 else nn.ConvTranspose2d
        else:
            raise ValueError("`spatial_dim` should be 2 or 3.")

        self.conv = ConvT(in_channels, out_channels, kernel_size,
                          stride, padding, output_padding)

    def forward(self, x: Tensor):
        return self.conv(x)
