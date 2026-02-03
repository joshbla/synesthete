import torch
import random
from .base import Visualizer
from .utils import random_saturated_color, cached_meshgrid, rotate_coords, sample_bilinear


class TrailsVisualizer(Visualizer):
    """
    Feedback/trails visualizer: a simple recurrent frame buffer with audio-driven gain/warp.
    """

    def render(self, waveform, fps=30, height=128, width=128, sample_rate=16000, audio_feats=None):
        num_frames, samples_per_frame = self.get_frame_audio_chunks(waveform, fps, sample_rate=sample_rate)
        device = waveform.device
        video_frames = torch.zeros(num_frames, 3, height, width, device=device)

        # initial buffer
        buf = torch.zeros(3, height, width, device=device)
        base_color = random_saturated_color().to(device)
        accent = random_saturated_color().to(device)

        y, x = cached_meshgrid(height, width, device_str=str(device))
        y = y.to(device)
        x = x.to(device)
        cy = (height - 1) / 2
        cx = (width - 1) / 2

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
            # feedback amount (higher amp -> brighter, slightly lower persistence)
            feedback = (0.92 - 0.25 * amp).clamp(0.6, 0.95)

            # small rotation warp tied to audio
            theta = float((amp * 0.25).item())
            yy, xx = rotate_coords(y, x, cy, cx, theta=theta)
            warped = sample_bilinear(buf, yy, xx)

            # inject new energy: a pulsing ring + some noise
            dy = (y.to(torch.float32) - cy)
            dx = (x.to(torch.float32) - cx)
            r = torch.sqrt(dy * dy + dx * dx)
            ring_r = (min(height, width) * (0.15 + 0.25 * float(amp.item())))
            ring = torch.exp(-((r - ring_r) ** 2) / (2.0 * (2.0 + 8.0 * float(amp.item())) ** 2))
            inj_color = base_color * (1 - float(amp.item())) + accent * float(amp.item())
            inject = inj_color[:, None, None] * ring[None, :, :] * (0.4 + 1.2 * amp)

            # update buffer
            buf = torch.clamp(warped * feedback + inject, 0, 1)
            video_frames[i] = buf

        return video_frames

