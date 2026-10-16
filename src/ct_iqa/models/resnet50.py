"""Compatibility re-export of the ResNet50 backbone.

The implementation moved to `ct_iqa.models.backbone` (identical architecture, plus an
explicit `bn_eps` constructor argument). This module is kept so that existing imports
(`ct_iqa.models.ohashi_resnet50`, tests, notebooks) keep working unchanged.
See docs/audit/MIGRATION_MAP.md.
"""

from ct_iqa.models.backbone import Bottleneck, DEFAULT_BN_EPS, ResNet50Backbone, _make_stage

__all__ = ["Bottleneck", "DEFAULT_BN_EPS", "ResNet50Backbone", "_make_stage"]
