import random
import torch
from .base import Visualizer
from .pulse import PulseVisualizer
from .spectrum import SpectrumVisualizer
from .waveform import WaveformVisualizer
from .tiles import TilesVisualizer
from .particles import ParticlesVisualizer
from .contours import ContoursVisualizer
from .trails import TrailsVisualizer
from .kaleidoscope import KaleidoscopeVisualizer
from .augment import AugmentedVisualizer
from .programs import ProgramSpec, seed_everything
from .compositor import CompositorVisualizer

# Keep an explicit balance between "clean geometric" and "organic/painterly/abstract".
_GEOMETRIC_REGISTRY = [
    PulseVisualizer,
    SpectrumVisualizer,
    WaveformVisualizer,
    TilesVisualizer,
]

_ORGANIC_REGISTRY = [
    ParticlesVisualizer,
    ContoursVisualizer,
    TrailsVisualizer,
    KaleidoscopeVisualizer,
]

_BASE_REGISTRY = _GEOMETRIC_REGISTRY + _ORGANIC_REGISTRY

def _make_augmented_from(pool):
    return AugmentedVisualizer(lambda: random.choice(pool)())

def get_random_visualizer(config: dict | None = None, seed: int | None = None, return_spec: bool = False):
    """
    Sample a visualizer instance.

    Args:
      config: optional config dict; may include `visualizers.*` knobs.
      seed: if provided, makes sampling deterministic (for debugging).
      return_spec: if True, return (visualizer, ProgramSpec).
    """
    if seed is None:
        seed = int(torch.randint(0, 2**31 - 1, (1,)).item())
    seed_everything(seed)

    cfg = config or {}
    vcfg = (cfg.get("visualizers", {}) or {}) if isinstance(cfg, dict) else {}
    aug_prob = float(vcfg.get("aug_prob", 0.35))
    organic_prob = float(vcfg.get("organic_prob", 0.5))
    composite_prob = float(vcfg.get("composite_prob", 0.25))

    # 50/50 family balance
    family = "geometric" if random.random() >= organic_prob else "organic"
    pool = _GEOMETRIC_REGISTRY if family == "geometric" else _ORGANIC_REGISTRY

    # Multi-primitive compositor sometimes (Phase 3 combinatorics)
    if random.random() < composite_prob:
        comp = CompositorVisualizer(pool)
        spec = ProgramSpec(seed=seed, family=family, kind="composite", visualizers=tuple(v.__name__ for v in pool))
        return (comp, spec) if return_spec else comp

    # Add an augmented wrapper sometimes for combinatorial lift.
    if random.random() < aug_prob:
        viz = _make_augmented_from(pool)
        spec = ProgramSpec(seed=seed, family=family, kind="augmented", visualizers=(viz.base_factory().__class__.__name__,))
        return (viz, spec) if return_spec else viz

    viz_cls = random.choice(pool)
    viz = viz_cls()
    spec = ProgramSpec(seed=seed, family=family, kind="primitive", visualizers=(viz_cls.__name__,))
    return (viz, spec) if return_spec else viz
