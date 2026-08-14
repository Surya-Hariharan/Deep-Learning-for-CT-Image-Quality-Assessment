"""Tests for the model/training specification stubs.

These assert that NO model gets built and NO training runs -- the modules must
fail loudly (not silently substitute a default) while prerequisites are
unresolved, per the project's "do not train yet" rule.
"""

from __future__ import annotations

import pytest

from ct_iqa.models.resnet50 import (
    DEFAULT_SPEC,
    ModelNotBuildable,
    ModelSpec,
    build_model,
    unresolved_prerequisites,
)
from ct_iqa.training.checkpointing import CheckpointingNotAuthorized, save_checkpoint
from ct_iqa.training.trainer import (
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE_GRID,
    Trainer,
    TrainingNotAuthorized,
    plan_lr_search,
    select_best,
)


def test_build_model_refuses_unconditionally():
    with pytest.raises(ModelNotBuildable):
        build_model()


def test_build_model_lists_every_remaining_blocker():
    """Only the checkpoint and the framework install remain unresolved for
    the DEFAULT_SPEC -- dropout_probability and freeze_backbone are now
    resolved (decisions A-22 and S-01) and must NOT appear as blockers for
    the default spec."""
    with pytest.raises(ModelNotBuildable) as exc_info:
        build_model()
    message = str(exc_info.value)
    assert "RadImageNet" in message
    assert "framework" in message
    assert "dropout_probability" not in message
    assert "freeze_backbone" not in message


def test_default_spec_matches_ohashi_architecture():
    assert DEFAULT_SPEC.backbone == "resnet50"
    assert DEFAULT_SPEC.pretrained_weights == "radimagenet"
    assert DEFAULT_SPEC.input_size == 224
    assert DEFAULT_SPEC.head == (
        "global_average_pooling", "dropout", "fully_connected_1", "sigmoid",
    )


def test_default_spec_backbone_is_fine_tuned_not_frozen():
    """Regression guard for decision S-01: the Ohashi paper confirms the
    RadImageNet backbone is fine-tuned, not frozen ("By fine-tuning these
    pre-trained models with our IQA dataset..."). DEFAULT_SPEC must never
    silently drift back to `None` (unspecified) or `True` (frozen)."""
    assert DEFAULT_SPEC.freeze_backbone is False


def test_default_spec_dropout_probability_is_resolved_baseline():
    """Regression guard for decision A-22: dropout_probability is a
    PROJECT-ADAPTATION baseline (0.5, sourced from RadImageNet's own
    transfer-learning recipe), not left as an unresolved `None`. This is
    NOT a claim that Ohashi's paper specifies this value."""
    assert DEFAULT_SPEC.dropout_probability == 0.5


def test_resolved_default_spec_no_longer_blocks_on_dropout_or_freeze():
    default_blockers = unresolved_prerequisites(DEFAULT_SPEC)
    assert not any("dropout_probability" in b for b in default_blockers)
    assert not any("freeze_backbone" in b for b in default_blockers)
    assert len(default_blockers) == 2  # checkpoint + framework only


def test_explicitly_unresolved_spec_still_blocks():
    """The defensive `is None` checks in unresolved_prerequisites() must
    still function for a caller who explicitly constructs an unresolved
    spec, even though DEFAULT_SPEC itself is now fully resolved."""
    unresolved = ModelSpec(dropout_probability=None, freeze_backbone=None)  # type: ignore[arg-type]
    blockers = unresolved_prerequisites(unresolved)
    assert any("dropout_probability" in b for b in blockers)
    assert any("freeze_backbone" in b for b in blockers)


def test_lr_search_grid_matches_ohashi_spec():
    assert LEARNING_RATE_GRID == (1e-2, 1e-3, 1e-4, 1e-5)
    assert BATCH_SIZE == 64
    assert EPOCHS == 30


def test_plan_lr_search_enumerates_all_four_runs():
    runs = plan_lr_search()
    assert [r.learning_rate for r in runs] == list(LEARNING_RATE_GRID)
    assert all(r.batch_size == 64 and r.epochs == 30 for r in runs)


def test_select_best_picks_lowest_validation_mse():
    results = {1e-2: 0.05, 1e-3: 0.01, 1e-4: 0.02, 1e-5: 0.03}
    assert select_best(results) == 1e-3


def test_select_best_requires_a_complete_search():
    with pytest.raises(ValueError):
        select_best({1e-2: 0.05, 1e-3: 0.01})


def test_select_best_rejects_empty_results():
    with pytest.raises(ValueError):
        select_best({})


def test_trainer_fit_is_not_authorized():
    trainer = Trainer(model_spec=DEFAULT_SPEC)
    assert len(trainer.planned_runs) == 4
    with pytest.raises(TrainingNotAuthorized):
        trainer.fit()


def test_checkpoint_saving_is_not_authorized():
    with pytest.raises(CheckpointingNotAuthorized):
        save_checkpoint()
