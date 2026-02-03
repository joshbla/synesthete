import math
from functools import lru_cache

import torch


def random_saturated_color() -> torch.Tensor:
    """
    Return a high-saturation RGB color tensor in [0, 1], shape (3,).
    """
    rgb = [torch.rand(1).item(), torch.rand(1).item(), torch.rand(1).item()]
    min_idx = rgb.index(min(rgb))
    rgb[min_idx] = 0.0
    max_idx = rgb.index(max(rgb))
    rgb[max_idx] = 1.0
    return torch.tensor(rgb, dtype=torch.float32)


@lru_cache(maxsize=32)
def cached_meshgrid(height: int, width: int, device_str: str = "cpu"):
    """
    Cache a coordinate meshgrid for given size/device.
    Returns (y, x) int64 tensors.
    """
    device = torch.device(device_str)
    y, x = torch.meshgrid(
        torch.arange(height, device=device),
        torch.arange(width, device=device),
        indexing="ij",
    )
    return y, x


def safe_normalize(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    m = x.abs().max()
    return x / (m + eps)


def smoothstep(x: torch.Tensor) -> torch.Tensor:
    # clamp to [0, 1]
    x = torch.clamp(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def rotate_coords(y: torch.Tensor, x: torch.Tensor, center_y: float, center_x: float, theta: float):
    """
    Rotate integer coordinate grids around (center_y, center_x).
    Returns rotated float coords (yy, xx).
    """
    dy = (y.to(torch.float32) - center_y)
    dx = (x.to(torch.float32) - center_x)
    c = math.cos(theta)
    s = math.sin(theta)
    yy = dy * c - dx * s + center_y
    xx = dy * s + dx * c + center_x
    return yy, xx


def sample_bilinear(img: torch.Tensor, yy: torch.Tensor, xx: torch.Tensor) -> torch.Tensor:
    """
    Bilinear sample a (C,H,W) image at float coords.
    Returns (C,H,W) sampled image.
    """
    if img.ndim != 3:
        raise ValueError(f"Expected img (C,H,W), got {tuple(img.shape)}")
    C, H, W = img.shape
    yy = torch.clamp(yy, 0.0, H - 1.001)
    xx = torch.clamp(xx, 0.0, W - 1.001)

    y0 = yy.floor().to(torch.long)
    x0 = xx.floor().to(torch.long)
    y1 = torch.clamp(y0 + 1, max=H - 1)
    x1 = torch.clamp(x0 + 1, max=W - 1)

    wy = (yy - y0.to(yy.dtype)).unsqueeze(0)
    wx = (xx - x0.to(xx.dtype)).unsqueeze(0)

    Ia = img[:, y0, x0]
    Ib = img[:, y0, x1]
    Ic = img[:, y1, x0]
    Id = img[:, y1, x1]

    wa = (1 - wy) * (1 - wx)
    wb = (1 - wy) * wx
    wc = wy * (1 - wx)
    wd = wy * wx

    return Ia * wa + Ib * wb + Ic * wc + Id * wd

