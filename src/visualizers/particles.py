import torch
import random
import math
from .base import Visualizer
from .utils import cached_meshgrid, random_saturated_color, smoothstep


class ParticlesVisualizer(Visualizer):
    """
    Simple particle field where spawn/brightness/velocity respond to amplitude.
    Implemented with tensor ops (no per-pixel Python loops).
    """

    def render(self, waveform, fps=30, height=128, width=128, sample_rate=16000, audio_feats=None):
        num_frames, samples_per_frame = self.get_frame_audio_chunks(waveform, fps, sample_rate=sample_rate)
        device = waveform.device
        video_frames = torch.zeros(num_frames, 3, height, width, device=device)

        # Grid
        y, x = cached_meshgrid(height, width, device_str=str(device))
        y = y.to(device)
        x = x.to(device)

        # Particle state
        n_particles = random.choice([64, 96, 128, 160])
        pos = torch.stack(
            [
                torch.rand(n_particles, device=device) * (height - 1),
                torch.rand(n_particles, device=device) * (width - 1),
            ],
            dim=1,
        )  # (N,2) in (y,x)
        vel = torch.randn(n_particles, 2, device=device) * 0.5

        color_a = random_saturated_color().to(device)
        color_b = random_saturated_color().to(device)
        radius_base = random.uniform(1.5, 4.5)
        drift = random.uniform(0.1, 0.6)

        for i in range(num_frames):
            start = i * samples_per_frame
            end = min((i + 1) * samples_per_frame, waveform.shape[1])
            chunk = waveform[0, start:end]
            if chunk.numel() < 8:
                continue

            if audio_feats is not None and i < audio_feats.shape[0]:
                amp = torch.sigmoid(audio_feats[i, 0]).clamp(0, 1)
            else:
                amp = chunk.abs().mean().clamp(0, 1)
            # small random acceleration
            vel = vel + torch.randn_like(vel) * (0.1 + drift * amp)
            # damp
            vel = vel * (0.85 + 0.1 * (1 - amp))
            pos = pos + vel

            # respawn out-of-bounds
            oob = (pos[:, 0] < 0) | (pos[:, 0] > height - 1) | (pos[:, 1] < 0) | (pos[:, 1] > width - 1)
            if oob.any():
                pos[oob, 0] = torch.rand(oob.sum(), device=device) * (height - 1)
                pos[oob, 1] = torch.rand(oob.sum(), device=device) * (width - 1)
                vel[oob] = torch.randn(oob.sum(), 2, device=device) * 0.5

            # Render as sum of Gaussian-ish blobs
            # Compute squared distance to each particle center:
            # (H,W) -> (1,H,W) for broadcasting; particle coords -> (N,1,1)
            dy = (y[None, :, :] - pos[:, 0][:, None, None])
            dx = (x[None, :, :] - pos[:, 1][:, None, None])
            d2 = dy * dy + dx * dx

            radius = radius_base * (0.7 + 2.5 * amp)
            # Soft blob kernel: exp(-d2 / (2*sigma^2))
            sigma2 = (radius * radius) * 0.6
            w = torch.exp(-d2 / (2.0 * sigma2 + 1e-6))  # (N,H,W)
            # Normalize by N so brightness doesn't explode
            field = w.mean(dim=0)  # (H,W)

            # Contrast curve
            field = smoothstep(field * (0.8 + 2.0 * amp))

            # Color drift over time based on amp
            mix = float(amp.item())
            c = color_a * (1 - mix) + color_b * mix

            frame = torch.zeros(3, height, width, device=device)
            frame[0] = field * c[0]
            frame[1] = field * c[1]
            frame[2] = field * c[2]
            video_frames[i] = torch.clamp(frame, 0, 1)

        return video_frames

