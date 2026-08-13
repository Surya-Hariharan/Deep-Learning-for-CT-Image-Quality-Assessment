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


def test_build_model_lists_every_blocker():
    with pytest.raises(ModelNotBuildable) as exc_info:
        build_model()
    message = str(exc_info.value)
    assert "RadImageNet" in message
    assert "dropout_probability" in message
    assert "freeze_backbone" in message


def test_default_spec_matches_ohashi_architecture():
    assert DEFAULT_SPEC.backbone == "resnet50"
    assert DEFAULT_SPEC.pretrained_weights == "radimagenet"
    assert DEFAULT_SPEC.input_size == 224
    assert DEFAULT_SPEC.head == ("dropout", "fully_connected_1", "sigmoid")


def test_default_spec_leaves_unspecified_items_none():
    assert DEFAULT_SPEC.dropout_probability is None
    assert DEFAULT_SPEC.freeze_backbone is None


def test_resolving_unspecified_items_shrinks_blocker_list():
    resolved = ModelSpec(dropout_probability=0.3, freeze_backbone=False)
    default_blockers = unresolved_prerequisites(DEFAULT_SPEC)
    resolved_blockers = unresolved_prerequisites(resolved)
    assert len(resolved_blockers) < len(default_blockers)
    assert not any("dropout_probability" in b for b in resolved_blockers)


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
