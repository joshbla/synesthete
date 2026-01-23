import torch
import random
import math
from .base import Visualizer

class SpectrumVisualizer(Visualizer):
    def render(self, waveform, fps=30, height=128, width=128):
        num_frames, samples_per_frame = self.get_frame_audio_chunks(waveform, fps)
        video_frames = torch.zeros(num_frames, 3, height, width)
        
        # Randomize mode
        mode = random.choice(['vertical', 'horizontal', 'radial'])
        num_bars = random.choice([8, 16, 32])
        
        # Color gradient
        start_color = torch.tensor([random.random(), random.random(), random.random()])
        end_color = torch.tensor([random.random(), random.random(), random.random()])
        
        for i in range(num_frames):
            start = i * samples_per_frame
            end = min((i + 1) * samples_per_frame, waveform.shape[1])
            chunk = waveform[0, start:end]
            
            if chunk.shape[0] < 10:
                continue

            fft = torch.fft.rfft(chunk)
            fft_mag = fft.abs()
            fft_mag = torch.log1p(fft_mag)
            if fft_mag.max() > 0:
                fft_mag = fft_mag / fft_mag.max()
            
            # Binning
            bin_size = fft_mag.shape[0] // num_bars
            bar_vals = []
            for b in range(num_bars):
                if bin_size > 0:
                    val = fft_mag[b*bin_size : (b+1)*bin_size].mean().item()
                else:
                    val = 0
                bar_vals.append(val)
            
            if mode == 'vertical':
                bar_width = width // num_bars
                for b, val in enumerate(bar_vals):
                    bar_h = int(val * height)
                    if bar_h > 0:
                        # Interpolate color
                        t = b / num_bars
                        color = start_color * (1-t) + end_color * t
                        
                        video_frames[i, 0, height-bar_h:, b*bar_width:(b+1)*bar_width] = color[0]
                        video_frames[i, 1, height-bar_h:, b*bar_width:(b+1)*bar_width] = color[1]
                        video_frames[i, 2, height-bar_h:, b*bar_width:(b+1)*bar_width] = color[2]

            elif mode == 'horizontal':
                bar_height = height // num_bars
                for b, val in enumerate(bar_vals):
                    bar_w = int(val * width)
                    if bar_w > 0:
                        t = b / num_bars
                        color = start_color * (1-t) + end_color * t
                        video_frames[i, 0, b*bar_height:(b+1)*bar_height, :bar_w] = color[0]
                        video_frames[i, 1, b*bar_height:(b+1)*bar_height, :bar_w] = color[1]
                        video_frames[i, 2, b*bar_height:(b+1)*bar_height, :bar_w] = color[2]

            elif mode == 'radial':
                # Draw concentric circles or pie slices
                center_h, center_w = height // 2, width // 2
                max_radius = min(height, width) // 2
                
                # We need a polar grid
                y_grid, x_grid = torch.meshgrid(torch.arange(height), torch.arange(width), indexing='ij')
                dy = y_grid - center_h
                dx = x_grid - center_w
                r_grid = torch.sqrt(dy**2 + dx**2)
                theta_grid = torch.atan2(dy, dx) # -pi to pi
                
                # Normalize theta to 0 to 1
                theta_norm = (theta_grid + math.pi) / (2 * math.pi)
                
                # Map theta to bar index
                bar_idx = (theta_norm * num_bars).long().clamp(0, num_bars-1)
                
                # Construct a tensor of values for the whole grid
                vals_tensor = torch.tensor(bar_vals)
                grid_vals = vals_tensor[bar_idx]
                
                mask = r_grid <= (grid_vals * max_radius)
                
                # Color based on angle
                c_r = 0.5 + 0.5 * torch.sin(theta_grid)
                c_g = 0.5 + 0.5 * torch.sin(theta_grid + 2*math.pi/3)
                c_b = 0.5 + 0.5 * torch.sin(theta_grid + 4*math.pi/3)
                
                video_frames[i, 0, mask] = c_r[mask]
                video_frames[i, 1, mask] = c_g[mask]
                video_frames[i, 2, mask] = c_b[mask]

        return video_frames
