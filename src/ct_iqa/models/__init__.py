"""Model architecture specifications. NOT IMPLEMENTED as a trainable model yet
(no framework weights loaded, no training run) -- see ModelSpec in
resnet50.py for what is fixed vs undecided.
"""

from ct_iqa.models.resnet50 import DEFAULT_SPEC, ModelNotBuildable, ModelSpec, build_model, unresolved_prerequisites

__all__ = ["ModelSpec", "DEFAULT_SPEC", "ModelNotBuildable", "build_model", "unresolved_prerequisites"]
