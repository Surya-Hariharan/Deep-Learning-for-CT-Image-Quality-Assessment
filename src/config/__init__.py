"""Configuration package.

`ct_iqa.config.legacy` holds the original flat `ExperimentConfig` used by the historical
experiments 001-003, their notebooks and tools. Its names are re-exported here so every
existing `from ct_iqa.config import ExperimentConfig` keeps working unchanged.

New code uses `ct_iqa.config.schema.StudyConfig` and `ct_iqa.config.loader.load_study_config`.
"""

from ct_iqa.config.legacy import (  # noqa: F401
    DEFAULT_IMAGE_SIZE,
    SUPPORTED_ARCHITECTURES,
    ExperimentConfig,
    _CONFIG_FIELD_NAMES,
    _yaml_default,
)
