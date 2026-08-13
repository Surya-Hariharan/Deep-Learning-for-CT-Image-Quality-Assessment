"""Training procedure specification and the learning-rate search plan.

OHASHI-SPECIFIED: search {1e-2, 1e-3, 1e-4, 1e-5} with Adam, batch size 64,
30 epochs, MSE loss, and select by validation MSE. The paper reports 1e-3 as
best for ResNet50, but this project must still *run* the selection rather
than assume the reported result.

``Trainer.fit`` raises unconditionally: this module plans training (what runs
are needed) without executing any of them -- training is not authorized at
this phase.
"""

from __future__ import annotations

from dataclasses import dataclass, field

LEARNING_RATE_GRID: tuple[float, ...] = (1e-2, 1e-3, 1e-4, 1e-5)  # OHASHI-SPECIFIED
BATCH_SIZE = 64          # OHASHI-SPECIFIED
EPOCHS = 30              # OHASHI-SPECIFIED
OPTIMIZER = "adam"       # OHASHI-SPECIFIED
SELECTION_METRIC = "validation_mse"  # OHASHI-SPECIFIED
REPORTED_BEST_LR = 1e-3  # OHASHI-SPECIFIED (must still be confirmed by running the search)


@dataclass(frozen=True)
class PlannedRun:
    learning_rate: float
    batch_size: int = BATCH_SIZE
    epochs: int = EPOCHS
    optimizer: str = OPTIMIZER
    loss: str = "mse"


def plan_lr_search() -> list[PlannedRun]:
    """The set of runs the LR search requires. Does not execute anything."""
    return [PlannedRun(learning_rate=lr) for lr in LEARNING_RATE_GRID]


def select_best(results: dict[float, float]) -> float:
    """Given ``{learning_rate: validation_mse}`` for a completed search, return
    the winning learning rate. Pure selection logic -- takes results, does not
    produce them.
    """
    if not results:
        raise ValueError("no results to select from")
    missing = set(LEARNING_RATE_GRID) - set(results)
    if missing:
        raise ValueError(f"incomplete search: missing results for {sorted(missing)}")
    return min(results, key=results.get)


class TrainingNotAuthorized(RuntimeError):
    pass


@dataclass
class Trainer:
    """Specification-only trainer. ``fit`` is not implemented at this phase."""

    model_spec: object
    planned_runs: list[PlannedRun] = field(default_factory=plan_lr_search)

    def fit(self, *args, **kwargs):
        raise TrainingNotAuthorized(
            "Training is not authorized at this phase (repository initialization only). "
            f"{len(self.planned_runs)} runs are planned via plan_lr_search(); "
            "none have been executed."
        )
