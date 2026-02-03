import torch
import random
from .base import Visualizer
from .utils import random_saturated_color, safe_normalize


class TilesVisualizer(Visualizer):
    """
    Grid/tile visualizer driven by coarse FFT band energies.
    """

    def render(self, waveform, fps=30, height=128, width=128, sample_rate=16000, audio_feats=None):
        num_frames, samples_per_frame = self.get_frame_audio_chunks(waveform, fps, sample_rate=sample_rate)
        video_frames = torch.zeros(num_frames, 3, height, width)

        # Randomize tile resolution
        tiles_h = random.choice([8, 10, 12, 16])
        tiles_w = random.choice([8, 10, 12, 16])
        tile_h = max(1, height // tiles_h)
        tile_w = max(1, width // tiles_w)

        base_color = random_saturated_color()
        alt_color = random_saturated_color()
        mode = random.choice(["bars", "checker", "heat"])

        for i in range(num_frames):
            # Prefer band energies from Phase 1 features if provided.
            if audio_feats is not None and i < audio_feats.shape[0] and audio_feats.shape[1] >= 6:
                bands = torch.sigmoid(audio_feats[i, 5:]).to(torch.float32)
                # Expand/tiling into a grid
                n_bins = tiles_h * tiles_w
                bands = bands.view(1, 1, -1)
                vals = torch.nn.functional.interpolate(bands, size=n_bins, mode="linear", align_corners=False).view(-1)
                grid = vals.view(tiles_h, tiles_w)
            else:
                start = i * samples_per_frame
                end = min((i + 1) * samples_per_frame, waveform.shape[1])
                chunk = waveform[0, start:end]
                if chunk.numel() < 32:
                    continue

                # FFT magnitudes
                fft = torch.fft.rfft(chunk)
                mag = torch.log1p(fft.abs())
                mag = safe_normalize(mag)

                # Bin into tiles_h * tiles_w bands
                n_bins = tiles_h * tiles_w
                bin_size = max(1, mag.shape[0] // n_bins)
                vals = []
                for b in range(n_bins):
                    s = b * bin_size
                    e = mag.shape[0] if b == n_bins - 1 else min((b + 1) * bin_size, mag.shape[0])
                    vals.append(mag[s:e].mean())
                grid = torch.stack(vals).view(tiles_h, tiles_w)

            # Render tiles
            frame = torch.zeros(3, height, width)
            for th in range(tiles_h):
                for tw in range(tiles_w):
                    v = grid[th, tw].item()
                    y0 = th * tile_h
                    x0 = tw * tile_w
                    y1 = height if th == tiles_h - 1 else min((th + 1) * tile_h, height)
                    x1 = width if tw == tiles_w - 1 else min((tw + 1) * tile_w, width)

                    if mode == "checker":
                        c = base_color if (th + tw) % 2 == 0 else alt_color
                        frame[:, y0:y1, x0:x1] = c[:, None, None] * v
                    elif mode == "bars":
                        # Fill from bottom within each tile
                        h_fill = int(v * (y1 - y0))
                        if h_fill > 0:
                            frame[:, y1 - h_fill : y1, x0:x1] = base_color[:, None, None]
                    else:  # heat
                        c = base_color * (1 - v) + alt_color * v
                        frame[:, y0:y1, x0:x1] = c[:, None, None]

            video_frames[i] = torch.clamp(frame, 0, 1)

        return video_frames

