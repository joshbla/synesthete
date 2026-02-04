import torch

from .base import Visualizer


def _amp01(audio_feats: torch.Tensor, i: int) -> float:
    # audio_feats[:,0] is log1p(rms) (normalized in current pipeline).
    # Map to 0..1 in a way that preserves contrast without saturating too hard.
    x = float(audio_feats[i, 0].item())
    y = 0.5 + 0.25 * x
    if y < 0.0:
        y = 0.0
    if y > 1.0:
        y = 1.0
    return float(y)


class DebugPulseGlobal(Visualizer):
    """
    Deterministic, strongly audio-reactive target:
    amplitude -> global brightness + radius pulse.
    """

    def render(self, waveform, fps=30, height=128, width=128, sample_rate=16000, audio_feats=None):
        num_frames, _spf = self.get_frame_audio_chunks(waveform, fps, sample_rate=sample_rate)
        frames = torch.zeros((num_frames, 3, height, width), dtype=torch.float32)
        if audio_feats is None:
            return frames

        y, x = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
        cy, cx = height // 2, width // 2
        rr = torch.sqrt((y - cy) ** 2 + (x - cx) ** 2)
        base = float(min(height, width)) * 0.18
        span = float(min(height, width)) * 0.22

        for i in range(num_frames):
            if i >= audio_feats.shape[0]:
                break
            a = _amp01(audio_feats, i)
            r = base + span * a
            mask = rr <= r
            # Strong signal: big mass of pixels changes with amp
            brightness = 0.08 + 0.92 * a
            frames[i, 0, mask] = brightness
            frames[i, 1, mask] = 0.25 * brightness
            frames[i, 2, mask] = 0.65 * brightness
            # Subtle background flicker also tied to amp
            frames[i] += (0.02 * a)
        return frames.clamp(0, 1)


class DebugBandsBars(Visualizer):
    """
    Deterministic, strongly audio-reactive target:
    band energies -> bar heights (very obvious spectrum-like mapping).
    """

    def render(self, waveform, fps=30, height=128, width=128, sample_rate=16000, audio_feats=None):
        num_frames, _spf = self.get_frame_audio_chunks(waveform, fps, sample_rate=sample_rate)
        frames = torch.zeros((num_frames, 3, height, width), dtype=torch.float32)
        if audio_feats is None:
            return frames

        # bands are at columns 5: in current feature layout
        if audio_feats.shape[1] < 6:
            return frames

        num_bands = int(audio_feats.shape[1] - 5)
        num_bars = min(16, max(8, num_bands))
        bar_w = max(1, width // num_bars)

        for i in range(num_frames):
            if i >= audio_feats.shape[0]:
                break
            bands = audio_feats[i, 5:].to(torch.float32)
            # linear-ish mapping (avoid sigmoid saturation); clamp for stability
            bands = torch.clamp(0.5 + 0.35 * bands, 0.0, 1.0)
            # interpolate to num_bars
            bars = torch.nn.functional.interpolate(bands.view(1, 1, -1), size=num_bars, mode="linear", align_corners=False)
            bar_vals = bars.view(-1)

            for b in range(num_bars):
                val = float(bar_vals[b].item())
                h = int(val * (height - 1))
                if h <= 0:
                    continue
                x0 = b * bar_w
                x1 = min(width, (b + 1) * bar_w)
                # vivid per-band color
                c = b / max(1, num_bars - 1)
                frames[i, 0, height - h :, x0:x1] = 0.2 + 0.8 * c
                frames[i, 1, height - h :, x0:x1] = 0.9 - 0.7 * c
                frames[i, 2, height - h :, x0:x1] = 0.15 + 0.6 * (1.0 - c)

        return frames.clamp(0, 1)

