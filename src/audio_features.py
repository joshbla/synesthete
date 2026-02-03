import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class AudioFeatureConfig:
    """
    Frame-aligned audio features for conditioning.

    We intentionally keep this simple and dependency-light (pure torch),
    because the goal of Phase 1 is *identifiable conditioning*, not
    maximizing audio semantics.
    """

    sample_rate: int = 16000
    fps: int = 30
    num_frames: int = 90
    n_fft: int = 512
    num_bands: int = 8
    rolloff: float = 0.85
    eps: float = 1e-8


def audio_feature_dim(num_bands: int) -> int:
    # rms, zcr, centroid, rolloff, flatness + band energies
    return 5 + int(num_bands)


def _frame_chunks(waveform: torch.Tensor, num_frames: int) -> list[torch.Tensor]:
    """
    Split mono waveform into `num_frames` contiguous chunks.

    waveform: (1, N)
    returns: list length num_frames, each (S,)
    """
    if waveform.ndim != 2 or waveform.shape[0] != 1:
        raise ValueError(f"Expected waveform shape (1, N), got {tuple(waveform.shape)}")
    if num_frames <= 0:
        raise ValueError(f"num_frames must be > 0, got {num_frames}")

    N = waveform.shape[1]
    samples_per_frame = max(1, N // num_frames)
    chunks: list[torch.Tensor] = []
    for i in range(num_frames):
        start = i * samples_per_frame
        end = min((i + 1) * samples_per_frame, N)
        chunk = waveform[0, start:end]
        if chunk.numel() < 4:
            # ensure non-empty / minimally valid
            chunk = torch.zeros(4, device=waveform.device, dtype=waveform.dtype)
        chunks.append(chunk)
    return chunks


def compute_audio_timeline(
    waveform: torch.Tensor,
    *,
    sample_rate: int,
    fps: int,
    num_frames: int,
    n_fft: int = 512,
    num_bands: int = 8,
    rolloff: float = 0.85,
    normalize_per_clip: bool = True,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Compute a frame-aligned audio feature timeline.

    Args:
      waveform: (1, N) mono float tensor
      num_frames: number of video frames / feature steps (T)

    Returns:
      feats: (T, F) float tensor

    Notes:
      - Chunks are computed to match the visualizers' "frame chunk" logic:
        contiguous slices of the waveform per frame.
      - Features are designed to be stable and cheap:
        RMS, ZCR, spectral centroid, spectral rolloff, spectral flatness,
        and coarse band energies.
    """
    device = waveform.device
    dtype = waveform.dtype

    chunks = _frame_chunks(waveform, num_frames=num_frames)
    F = audio_feature_dim(num_bands)
    feats = torch.zeros((num_frames, F), device=device, dtype=dtype)

    # Frequency bin centers for centroid calculations
    freqs = torch.linspace(0, sample_rate / 2, n_fft // 2 + 1, device=device, dtype=dtype)
    hann = torch.hann_window(n_fft, device=device, dtype=dtype)

    for i, chunk in enumerate(chunks):
        x = chunk

        # Time-domain
        rms = torch.sqrt(torch.mean(x * x) + eps)

        # ZCR: fraction of sign changes
        s = torch.sign(x)
        s[s == 0] = 1
        zcr = torch.mean((s[1:] != s[:-1]).to(dtype))

        # Spectral
        if x.numel() < n_fft:
            xw = torch.nn.functional.pad(x, (0, n_fft - x.numel()))
        else:
            xw = x[:n_fft]
        xw = xw * hann
        spec = torch.fft.rfft(xw)
        mag = spec.abs() + eps

        mag_sum = mag.sum()
        centroid = (freqs * mag).sum() / mag_sum

        # rolloff frequency: smallest f s.t. cumulative energy >= rolloff * total
        cum = torch.cumsum(mag, dim=0)
        target = rolloff * cum[-1]
        idx = torch.searchsorted(cum, target).clamp(0, mag.shape[0] - 1)
        roll_f = freqs[idx]

        # spectral flatness
        flatness = torch.exp(torch.mean(torch.log(mag))) / (torch.mean(mag) + eps)

        # coarse band energies (linear bins)
        band_energies = torch.zeros(num_bands, device=device, dtype=dtype)
        bins = mag.shape[0]
        band_size = max(1, bins // num_bands)
        for b in range(num_bands):
            start = b * band_size
            end = bins if b == num_bands - 1 else min((b + 1) * band_size, bins)
            band_energies[b] = mag[start:end].mean()

        # Pack
        feats[i, 0] = torch.log1p(rms)
        feats[i, 1] = zcr
        feats[i, 2] = centroid / (sample_rate / 2 + eps)  # normalize 0..1
        feats[i, 3] = roll_f / (sample_rate / 2 + eps)    # normalize 0..1
        feats[i, 4] = flatness
        feats[i, 5:] = torch.log1p(band_energies)

    if normalize_per_clip:
        mean = feats.mean(dim=0, keepdim=True)
        std = feats.std(dim=0, keepdim=True).clamp_min(1e-4)
        feats = (feats - mean) / std

    return feats

