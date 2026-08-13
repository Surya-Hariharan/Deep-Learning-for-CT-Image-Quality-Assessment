"""Combined noise+blur degradation and the ``Condition`` abstraction shared
across the whole degradation package."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from ct_iqa.degradation.gaussian_blur import apply_gaussian_blur
from ct_iqa.degradation.gaussian_noise import apply_gaussian_noise

DegradationType = Literal["clean", "noise", "blur", "noise_blur"]
Order = Literal["blur_then_noise", "noise_then_blur"]


@dataclass(frozen=True)
class Condition:
    """One degradation condition applied to one reference image."""

    degradation_type: DegradationType
    noise_sigma: float | None = None
    blur_sigma: float | None = None

    @property
    def combined(self) -> bool:
        return self.degradation_type == "noise_blur"

    def suffix(self) -> str:
        """Deterministic id suffix, e.g. ``noise_3_blur_0.45``."""
        if self.degradation_type == "clean":
            return "clean"
        parts = []
        if self.noise_sigma is not None:
            parts.append(f"noise_{_fmt(self.noise_sigma)}")
        if self.blur_sigma is not None:
            parts.append(f"blur_{_fmt(self.blur_sigma)}")
        return "_".join(parts)


def _fmt(value: float) -> str:
    """Format a sigma without trailing zeros, so ids stay stable and readable."""
    return f"{value:g}"


def apply_condition(
    image: np.ndarray,
    condition: Condition,
    *,
    order: Order,
    sigma_scale: float,
    rng: np.random.Generator,
    clip_range: tuple[float, float] | None = None,
    blur_truncate: float = 4.0,
    blur_mode: str = "nearest",
) -> np.ndarray:
    """Apply one condition to one image.

    ``order`` is a required keyword: for the combined condition the result
    genuinely differs between blur-then-noise (models a physical acquisition
    chain) and noise-then-blur (partially smooths the noise away). NOT
    SPECIFIED BY OHASHI which was used, so the caller must state its choice.
    """
    out = np.asarray(image, dtype=np.float64)

    if condition.degradation_type == "clean":
        return out

    steps: list[str] = []
    if condition.degradation_type == "noise":
        steps = ["noise"]
    elif condition.degradation_type == "blur":
        steps = ["blur"]
    elif order == "blur_then_noise":
        steps = ["blur", "noise"]
    else:
        steps = ["noise", "blur"]

    for step in steps:
        if step == "noise":
            out = apply_gaussian_noise(
                out, condition.noise_sigma, sigma_scale, rng, clip_range
            )
        else:
            out = apply_gaussian_blur(out, condition.blur_sigma, blur_truncate, blur_mode)
    return out


def condition_rng(seed: int, reference_id: str, condition: Condition) -> np.random.Generator:
    """Per-(reference, condition) generator.

    Deriving the stream from a stable hash of the ids means regenerating one
    image reproduces exactly the same noise realisation as the full run, and
    the dataset does not depend on iteration order or worker count. PROJECT
    ADAPTATION -- not part of Ohashi's methodology, purely an engineering
    reproducibility measure.
    """
    import hashlib

    key = f"{seed}|{reference_id}|{condition.suffix()}".encode("utf-8")
    digest = hashlib.sha256(key).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "big"))
