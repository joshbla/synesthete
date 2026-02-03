import random
import math
import torch

from .base import Visualizer
from .utils import cached_meshgrid, rotate_coords, sample_bilinear


def _postprocess_mirror(frame: torch.Tensor) -> torch.Tensor:
    # frame: (3,H,W)
    if random.random() < 0.5:
        frame = torch.flip(frame, dims=[2])
    if random.random() < 0.5:
        frame = torch.flip(frame, dims=[1])
    return frame


def _postprocess_posterize(frame: torch.Tensor, levels: int) -> torch.Tensor:
    levels = max(2, int(levels))
    return torch.round(frame * (levels - 1)) / (levels - 1)


def _postprocess_blur(frame: torch.Tensor, k: int) -> torch.Tensor:
    # Simple spatial blur via average pooling (kept smooth; avoids "glitchy" artifacts).
    k = max(1, int(k))
    if k == 1:
        return frame
    x = frame.unsqueeze(0)  # (1,C,H,W)
    x = torch.nn.functional.avg_pool2d(x, kernel_size=k, stride=1, padding=k // 2)
    return x.squeeze(0)


def _postprocess_bloom(frame: torch.Tensor, thresh: float, k: int, gain: float) -> torch.Tensor:
    # Add a soft glow from bright areas.
    x = frame
    bright = torch.clamp(x - float(thresh), 0.0, 1.0)
    glow = _postprocess_blur(bright, k=k)
    return torch.clamp(x + glow * float(gain), 0.0, 1.0)


def _postprocess_vignette(frame: torch.Tensor, strength: float) -> torch.Tensor:
    C, H, W = frame.shape
    y, x = cached_meshgrid(H, W, device_str=str(frame.device))
    yy = (y.to(torch.float32) / max(1.0, float(H - 1))) * 2 - 1
    xx = (x.to(torch.float32) / max(1.0, float(W - 1))) * 2 - 1
    r = torch.sqrt(xx * xx + yy * yy).clamp(0, 1)
    mask = (1.0 - float(strength) * (r ** 1.8)).clamp(0.0, 1.0)
    return torch.clamp(frame * mask[None, :, :], 0.0, 1.0)


def _postprocess_colorgrade(frame: torch.Tensor, contrast: float, saturation: float, gamma: float) -> torch.Tensor:
    x = torch.clamp(frame, 0.0, 1.0)
    # gamma
    x = torch.pow(x + 1e-6, float(gamma))
    # contrast around 0.5
    x = torch.clamp((x - 0.5) * float(contrast) + 0.5, 0.0, 1.0)
    # saturation via lerp with grayscale
    gray = x.mean(dim=0, keepdim=True)
    x = torch.clamp(gray + (x - gray) * float(saturation), 0.0, 1.0)
    return x


def _postprocess_warp(frame: torch.Tensor, theta: float, scale: float) -> torch.Tensor:
    C, H, W = frame.shape
    y, x = cached_meshgrid(H, W, device_str=str(frame.device))
    y = y.to(frame.device)
    x = x.to(frame.device)
    cy = (H - 1) / 2
    cx = (W - 1) / 2
    yy, xx = rotate_coords(y, x, cy, cx, theta=theta)
    # slight zoom
    yy = (yy - cy) * scale + cy
    xx = (xx - cx) * scale + cx
    return sample_bilinear(frame, yy, xx).clamp(0, 1)


class AugmentedVisualizer(Visualizer):
    """
    Wrap another visualizer and apply a randomized stack of lightweight postprocesses.

    This is a Phase 3 accelerator: it turns N base primitives into N * K visual
    families without making the data generator brittle.
    """

    def __init__(self, base_visualizer_factory):
        super().__init__()
        self.base_factory = base_visualizer_factory

    def render(self, waveform, fps=30, height=128, width=128, sample_rate=16000, audio_feats=None):
        base = self.base_factory()
        frames = base.render(waveform, fps=fps, height=height, width=width, sample_rate=sample_rate, audio_feats=audio_feats)

        # Choose a postprocess stack once per clip
        ops = []
        # Prefer smooth, non-glitchy transforms. (We intentionally avoid pixelation/edge overlays by default.)
        if random.random() < 0.35:
            ops.append(("mirror", None))
        if random.random() < 0.55:
            theta = random.uniform(-0.35, 0.35)
            scale = random.uniform(0.92, 1.08)
            ops.append(("warp", (theta, scale)))
        if random.random() < 0.55:
            ops.append(("blur", random.choice([1, 3, 5])))
        if random.random() < 0.35:
            ops.append(("bloom", (random.uniform(0.45, 0.7), random.choice([3, 5]), random.uniform(0.2, 0.8))))
        if random.random() < 0.35:
            ops.append(("vignette", random.uniform(0.15, 0.55)))
        if random.random() < 0.75:
            ops.append(("colorgrade", (random.uniform(0.85, 1.35), random.uniform(0.8, 1.5), random.uniform(0.85, 1.2))))
        # Rarely allow posterize (stylized, but can read as "glitchy" if overused)
        if random.random() < 0.08:
            ops.append(("posterize", random.choice([5, 6, 7, 8])))

        if not ops:
            return frames

        out = torch.zeros_like(frames)
        for i in range(frames.shape[0]):
            fr = frames[i]
            for name, arg in ops:
                if name == "mirror":
                    fr = _postprocess_mirror(fr)
                elif name == "posterize":
                    fr = _postprocess_posterize(fr, arg)
                elif name == "warp":
                    th, sc = arg
                    # add a tiny time-varying twist so it doesn't look static
                    fr = _postprocess_warp(fr, theta=th + 0.03 * math.sin(i * 0.1), scale=sc)
                elif name == "blur":
                    fr = _postprocess_blur(fr, k=arg)
                elif name == "bloom":
                    th, k, gain = arg
                    fr = _postprocess_bloom(fr, thresh=th, k=k, gain=gain)
                elif name == "vignette":
                    fr = _postprocess_vignette(fr, strength=arg)
                elif name == "colorgrade":
                    c, s, g = arg
                    fr = _postprocess_colorgrade(fr, contrast=c, saturation=s, gamma=g)
            out[i] = fr
        return out

