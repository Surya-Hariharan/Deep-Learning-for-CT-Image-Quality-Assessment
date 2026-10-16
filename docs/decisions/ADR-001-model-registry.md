# ADR-001: Model registry and compatibility with the historical class

Status: accepted. Date: 2026-10-16.

## Context
The only model was `OhashiResNet50`; the architecture name lived in `SUPPORTED_ARCHITECTURES`, a validation set with no
factory, and about 25 call sites constructed the class directly. Experiments B and C need different outputs and losses.

## Decision
1. A registry (`ct_iqa.models.registry`) maps descriptive names to a head kind on one shared backbone:
   `resnet50_regression`, `resnet50_gaussian`. Only the ResNet50 family is registered. `build_model(ModelConfig)` is the
   only construction path used by study v2.
2. Models return a `ModelOutput(mean, log_var=None)`, in normalized [0, 1] space, so one training loop serves all
   experiments and losses state explicitly which outputs they need.
3. Head layer attribute names (`dropout`, `fc`, `fc_logvar`) are kept at module top level so the regression model's
   `state_dict` keys equal the historical class's (`backbone.*`, `fc.*`). All six historical checkpoints load strictly
   into it ([CHECKPOINT_COMPATIBILITY](../audit/CHECKPOINT_COMPATIBILITY.md)); on synthetic weights the outputs equal the
   legacy class's at equal epsilon (`tests/regression/test_legacy_compatibility.py`).
4. `OhashiResNet50` and its module are **kept unchanged** (not turned into an alias). Rationale: it carries the historical
   weight-load side effect on epsilon that the old runs depended on; rewriting it as a wrapper would risk silently changing
   that behavior. The class is documented as legacy. Removal is deferred until the notebooks and tools that import it are
   migrated and verified, which has not happened.
5. `ct_iqa/config.py` became a package: `config/legacy.py` holds the original code (moved with `git mv`), and
   `config/__init__.py` re-exports its names, so `from ct_iqa.config import ExperimentConfig` still works.

## Consequences
Study-v2 code has no dependency on the Ohashi-named identifiers. Two code paths coexist (legacy and study v2) until a later
migration; the duplication is intentional and recorded in [MIGRATION_MAP](../audit/MIGRATION_MAP.md). Scientific attribution
remains in documentation and docstrings.
