"""Enumerates the full degradation grid for a reference image.

``build_grid`` produces the 169 conditions per reference (1 clean + 12 noise +
12 blur + 144 combined) that this project uses. Dataset-agnostic: with 105
references (Ohashi's original DeepLesion+CQ500 set) it reproduces the paper's
17,745-image total; with the 1,000-reference LDCT-IQAC adaptation used in this
project it gives 169,000 (see configs/quality/ohashi_ldctiqac.yaml).
"""

from __future__ import annotations

from typing import Iterator, Sequence

from ct_iqa.degradation.combined import Condition
from ct_iqa.degradation.gaussian_blur import BLUR_SIGMAS
from ct_iqa.degradation.gaussian_noise import NOISE_SIGMAS


def build_grid(
    noise_sigmas: Sequence[float] = NOISE_SIGMAS,
    blur_sigmas: Sequence[float] = BLUR_SIGMAS,
    include_clean: bool = True,
) -> list[Condition]:
    """Enumerate all conditions for a single reference image.

    With the default grids and ``include_clean=True`` this yields 169
    conditions per reference (DERIVED, see docs/deviations_from_ohashi.md for
    the arithmetic that forces this).

    ``include_clean`` is exposed because that component is DERIVED from the
    published total rather than read from the paper; flipping it off reproduces
    the 168-per-reference variant for comparison.
    """
    conditions: list[Condition] = []
    if include_clean:
        conditions.append(Condition("clean"))
    conditions += [Condition("noise", noise_sigma=float(s)) for s in noise_sigmas]
    conditions += [Condition("blur", blur_sigma=float(s)) for s in blur_sigmas]
    conditions += [
        Condition("noise_blur", noise_sigma=float(n), blur_sigma=float(b))
        for n in noise_sigmas
        for b in blur_sigmas
    ]
    return conditions


def iter_grid(**kwargs) -> Iterator[Condition]:
    yield from build_grid(**kwargs)
