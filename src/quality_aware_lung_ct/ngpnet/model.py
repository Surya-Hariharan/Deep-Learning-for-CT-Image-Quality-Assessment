"""NGP-Net: Nodule Growth Prediction network.

Migrated unmodified (math-wise) from the author's ``nn/ngpnet.py``
(class ``NGPnet``, renamed ``NGPNet`` for naming consistency; see
docs/architecture.md). Measured parameter count at the paper's defaults
(``img_size=64, feat_dim=8``) is 1,984,821, matching the published figure.
"""

from typing import Sequence

import torch
from torch import nn, Tensor

from .encoder import UNetEncoder
from .decoder import UNetDecoder
from .heads import SpatialTransformer


class NGPNet(nn.Module):
    ''' Nodule Growth Prediction network (NGP-Net) '''

    def __init__(self,
                 img_size: int = 64,
                 feat_dim: int = 8,
                 num_classes: int = 2,
                 depths: Sequence[int] = (2, 2, 2, 2),
                 drop_path_rate: float = 0.0,
                 spatial_dim: int = 3):
        ''' Args:
        * `img_size`: spatial size of input images.
        * `feat_dim: output channels of the tokenizer.
        * `num_classes: number of classes.
        * `depths`: number of blocks in each stage.
        * `drop_path_rate`: stochastic depth rate.
        * `spatial_dim`: number of spatial dimensions.
        '''
        super(NGPNet, self).__init__()

        self.encoder = UNetEncoder(in_channels=2,
                                   depths=depths,
                                   img_size=img_size,
                                   feat_dim=feat_dim,
                                   drop_path_rate=drop_path_rate,
                                   spatial_dim=spatial_dim)

        self.shape_net = UNetDecoder(out_channels=num_classes,
                                     depths=depths,
                                     img_size=img_size,
                                     feat_dim=feat_dim,
                                     drop_path_rate=drop_path_rate,
                                     spatial_dim=spatial_dim)

        self.texture_net = UNetDecoder(out_channels=spatial_dim,
                                       depths=depths,
                                       img_size=img_size,
                                       feat_dim=feat_dim,
                                       drop_path_rate=drop_path_rate,
                                       spatial_dim=spatial_dim)

        self.sp_trans = SpatialTransformer(in_size=img_size,
                                           spatial_dim=spatial_dim)

    def ngpnet(self, im0: Tensor, im1: Tensor, tm0: Tensor, tm1: Tensor):
        ''' Full forward pass returning the raw (soft) heads.

        Returns `(im2, mk2_logits, field)`:
        * `im2`: predicted image at `t1 + tm1`, warped from `im1`.
        * `mk2_logits`: (B, num_classes, ...) segmentation logits (soft).
        * `field`: (B, spatial_dim, ...) deformation field.
        '''
        im = torch.cat([im0, im1], dim=1)
        enc = self.encoder(im, tm0)

        mk2 = self.shape_net(enc, tm1)
        field = self.texture_net(enc, tm1)
        im2 = self.sp_trans(im1, field)
        return im2, mk2, field

    def forward(self, im0: Tensor, im1: Tensor, tm0: Tensor, tm1: Tensor):
        ''' Inference-oriented forward pass: hard mask via argmax. '''
        im2, mk2, _ = self.ngpnet(im0, im1, tm0, tm1)
        mk2 = mk2.argmax(dim=1, keepdim=True)
        return im2, mk2
