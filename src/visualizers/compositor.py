import random
from typing import Literal

import torch

from .base import Visualizer


BlendMode = Literal["add", "max", "alpha", "multiply"]


def _blend(a: torch.Tensor, b: torch.Tensor, mode: BlendMode, alpha: float) -> torch.Tensor:
    # a,b: (T,3,H,W) or (3,H,W)
    if mode == "add":
        return torch.clamp(a + b * alpha, 0, 1)
    if mode == "max":
        return torch.maximum(a, b * alpha)
    if mode == "multiply":
        return torch.clamp(a * (0.6 + 0.4 * alpha) + b * (0.4 * alpha), 0, 1)
    # alpha blend
    return torch.clamp(a * (1 - alpha) + b * alpha, 0, 1)


class CompositorVisualizer(Visualizer):
    """
    Compose multiple primitive visualizers into a single clip with blend modes.
    """

    def __init__(self, visualizer_factories, num_layers: int | None = None):
        super().__init__()
        self.factories = list(visualizer_factories)
        self.num_layers = num_layers

    def render(self, waveform, fps=30, height=128, width=128, sample_rate=16000, audio_feats=None):
        if not self.factories:
            raise ValueError("CompositorVisualizer requires at least one visualizer factory")

        k = self.num_layers if self.num_layers is not None else random.choice([2, 2, 3])
        k = max(1, min(k, 4))

        # Pick layers
        layers = [random.choice(self.factories)() for _ in range(k)]
        modes: list[BlendMode] = [random.choice(["alpha", "add", "max", "multiply"]) for _ in range(k - 1)]

        # Render first layer
        out = layers[0].render(
            waveform,
            fps=fps,
            height=height,
            width=width,
            sample_rate=sample_rate,
            audio_feats=audio_feats,
        )

        # Blend in subsequent layers
        for layer, mode in zip(layers[1:], modes):
            b = layer.render(
                waveform,
                fps=fps,
                height=height,
                width=width,
                sample_rate=sample_rate,
                audio_feats=audio_feats,
            )
            # Match length if any visualizer differs by 1 frame due to rounding
            T = min(out.shape[0], b.shape[0])
            out = out[:T]
            b = b[:T]
            alpha = random.uniform(0.35, 0.85)
            out = _blend(out, b, mode=mode, alpha=alpha)

        return out

