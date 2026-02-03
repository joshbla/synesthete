import torch
import random
import math
from .base import Visualizer
from .utils import cached_meshgrid, sample_bilinear


class KaleidoscopeVisualizer(Visualizer):
    """
    Kaleidoscope-style symmetry visualizer driven by a simple base field from audio.
    """

    def render(self, waveform, fps=30, height=128, width=128, sample_rate=16000, audio_feats=None):
        num_frames, samples_per_frame = self.get_frame_audio_chunks(waveform, fps, sample_rate=sample_rate)
        device = waveform.device
        video_frames = torch.zeros(num_frames, 3, height, width, device=device)

        # symmetry
        n_sectors = random.choice([4, 6, 8, 10, 12])
        twist = random.uniform(0.0, 1.5)

        y, x = cached_meshgrid(height, width, device_str=str(device))
        y = y.to(device).to(torch.float32)
        x = x.to(device).to(torch.float32)
        cy = (height - 1) / 2
        cx = (width - 1) / 2

        dy = y - cy
        dx = x - cx
        r = torch.sqrt(dy * dy + dx * dx) / (min(height, width) / 2 + 1e-6)
        theta = torch.atan2(dy, dx)  # [-pi, pi]

        for i in range(num_frames):
            start = i * samples_per_frame
            end = min((i + 1) * samples_per_frame, waveform.shape[1])
            chunk = waveform[0, start:end]
            if chunk.numel() < 32:
                continue

            if audio_feats is not None and i < audio_feats.shape[0] and audio_feats.shape[1] >= 4:
                amp = torch.sigmoid(audio_feats[i, 0]).clamp(0, 1)
                br = torch.sigmoid(audio_feats[i, 2]).clamp(0, 1) * 2.0
            else:
                amp = chunk.abs().mean().clamp(0, 1)
                # crude spectral brightness
                mag = torch.fft.rfft(chunk).abs()
                mag = torch.log1p(mag)
                br = mag[int(0.6 * mag.shape[0]) :].mean() / (mag.mean() + 1e-6)
                br = br.clamp(0, 2.0)

            # Build a simple base image in polar coords: a few rotating sine patterns
            t = i / max(1, num_frames - 1)
            base = torch.sin((theta * (n_sectors / math.pi) + t * (1.0 + 3.0 * amp) * twist) * math.pi)
            base = base + 0.7 * torch.cos((r * (4.0 + 8.0 * float(br.item())) - t * (0.5 + 2.0 * amp)) * math.pi)
            base = (base + 2.0) / 4.0
            base = torch.clamp(base, 0, 1)

            # RGB from phase-shifted components
            img = torch.stack(
                [
                    base,
                    torch.clamp(1.0 - base * (0.7 + 0.6 * amp), 0, 1),
                    torch.clamp(base * (0.6 + 0.9 * float(br.item())), 0, 1),
                ],
                dim=0,
            )  # (3,H,W)

            # Kaleidoscope mapping: fold theta into sector and reflect alternating sectors
            sector_angle = 2 * math.pi / n_sectors
            theta2 = (theta + math.pi)  # [0, 2pi]
            sector_idx = torch.floor(theta2 / sector_angle)
            local = theta2 - sector_idx * sector_angle  # [0, sector_angle)
            reflect = (sector_idx % 2) == 1
            local = torch.where(reflect, sector_angle - local, local)
            theta_folded = local - sector_angle / 2  # center around 0

            yy = cy + r * (min(height, width) / 2) * torch.sin(theta_folded)
            xx = cx + r * (min(height, width) / 2) * torch.cos(theta_folded)
            out = sample_bilinear(img, yy, xx)
            video_frames[i] = torch.clamp(out, 0, 1)

        return video_frames

