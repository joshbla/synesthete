import torch
import random
import math
from .base import Visualizer
from .utils import cached_meshgrid, random_saturated_color, smoothstep, safe_normalize


class ContoursVisualizer(Visualizer):
    """
    Contour/isolines on a synthetic scalar field, modulated by audio.
    """

    def render(self, waveform, fps=30, height=128, width=128, sample_rate=16000, audio_feats=None):
        num_frames, samples_per_frame = self.get_frame_audio_chunks(waveform, fps, sample_rate=sample_rate)
        device = waveform.device
        video_frames = torch.zeros(num_frames, 3, height, width, device=device)

        y, x = cached_meshgrid(height, width, device_str=str(device))
        y = y.to(device).to(torch.float32) / max(1.0, float(height - 1))
        x = x.to(device).to(torch.float32) / max(1.0, float(width - 1))

        # Random field parameters
        k1 = random.uniform(1.5, 6.0)
        k2 = random.uniform(1.5, 6.0)
        k3 = random.uniform(0.5, 3.0)
        phase = random.uniform(0, 2 * math.pi)
        n_levels = random.choice([4, 6, 8, 10])
        thickness = random.choice([0.01, 0.015, 0.02])
        color_a = random_saturated_color().to(device)
        color_b = random_saturated_color().to(device)

        for i in range(num_frames):
            start = i * samples_per_frame
            end = min((i + 1) * samples_per_frame, waveform.shape[1])
            chunk = waveform[0, start:end]
            if chunk.numel() < 32:
                continue

            if audio_feats is not None and i < audio_feats.shape[0] and audio_feats.shape[1] >= 4:
                amp = torch.sigmoid(audio_feats[i, 0]).clamp(0, 1)
                # use centroid/rolloff-ish features as brightness proxy
                br = float(torch.sigmoid(audio_feats[i, 2]).item())
            else:
                amp = chunk.abs().mean().clamp(0, 1)
                # crude spectral brightness proxy
                mag = torch.fft.rfft(chunk).abs()
                mag = torch.log1p(mag)
                mag = safe_normalize(mag)
                hi = mag[int(0.6 * mag.shape[0]) :].mean()
                br = float(hi.item())

            t = i / max(1, num_frames - 1)
            # Scalar field: mix of sinusoids + radial component
            r = torch.sqrt((x - 0.5) ** 2 + (y - 0.5) ** 2)
            field = (
                torch.sin((x * k1 + t * (0.5 + 2.0 * amp)) * 2 * math.pi + phase)
                + 0.7 * torch.cos((y * k2 - t * (0.3 + 1.5 * amp)) * 2 * math.pi)
                + 0.5 * torch.sin((r * k3 + t * (0.2 + 1.0 * br)) * 2 * math.pi)
            )
            field = (field + 2.2) / 4.4  # roughly into [0,1]

            # Contour lines: where field is near discrete levels
            levels = torch.linspace(0.1, 0.9, n_levels, device=device)
            d = torch.min(torch.abs(field[None, :, :] - levels[:, None, None]), dim=0).values
            lines = 1.0 - smoothstep(d / thickness)

            # Background wash
            c = color_a * (1 - br) + color_b * br
            bg = (0.05 + 0.25 * amp) * (c[:, None, None])

            frame = bg + lines[None, :, :] * c[:, None, None] * (0.4 + 0.8 * amp)
            video_frames[i] = torch.clamp(frame, 0, 1)

        return video_frames

