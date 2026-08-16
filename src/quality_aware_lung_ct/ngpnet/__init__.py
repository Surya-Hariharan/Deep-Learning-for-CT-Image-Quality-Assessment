"""NGP-Net: A Lightweight Growth Prediction Network for Pulmonary Nodules.

Reimplementation of Tang, Luo, Liu, Huang & Zou (IEEE TMI, 2026),
official code: https://github.com/XinKai-Tang/NGP-Net. The architecture and
its mathematical behavior are preserved unmodified from the authors' code
(originally ``nn/ngpnet.py`` and ``nn/net_utils.py``); only module
organization and packaging changed. See docs/architecture.md.
"""

from .model import NGPNet
from .config import NGPNetConfig

__all__ = ["NGPNet", "NGPNetConfig"]
