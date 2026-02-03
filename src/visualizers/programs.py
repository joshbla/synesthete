from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import random
import torch


@dataclass(frozen=True)
class ProgramSpec:
    """
    Minimal, reproducible description of how a training clip was rendered.

    This is intentionally lightweight: for now, we store a seed and the chosen
    visualizer "family" + class names. Re-rendering is achieved by reseeding
    RNGs before constructing/rendering visualizers.
    """

    seed: int
    family: Literal["geometric", "organic"]
    kind: Literal["primitive", "augmented", "composite"]
    visualizers: tuple[str, ...]
    meta: dict[str, Any] | None = None


def seed_everything(seed: int) -> None:
    """
    Seed Python + torch RNGs for deterministic visual program sampling.

    Note: determinism still depends on implementation details; this is meant for
    debug reproducibility, not cryptographic stability across refactors.
    """

    random.seed(int(seed))
    torch.manual_seed(int(seed))

