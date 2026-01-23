import torch
import random
import math
from .base import Visualizer

class WaveformVisualizer(Visualizer):
    def render(self, waveform, fps=30, height=128, width=128):
        num_frames, samples_per_frame = self.get_frame_audio_chunks(waveform, fps)
        video_frames = torch.zeros(num_frames, 3, height, width)
        
        mode = random.choice(['line', 'circle'])
        thickness = random.randint(1, 3)
        
        # High saturation color
        rgb = [random.random(), random.random(), random.random()]
        min_idx = rgb.index(min(rgb))
        rgb[min_idx] = 0.0
        max_idx = rgb.index(max(rgb))
        rgb[max_idx] = 1.0
        color = torch.tensor(rgb)
        
        y_grid, x_grid = torch.meshgrid(torch.arange(height), torch.arange(width), indexing='ij')
        
        for i in range(num_frames):
            start = i * samples_per_frame
            end = min((i + 1) * samples_per_frame, waveform.shape[1])
            chunk = waveform[0, start:end]
            
            # Downsample chunk to width
            if chunk.shape[0] > width:
                k = chunk.shape[0] // width
                chunk_ds = chunk[:k*width].view(width, k).mean(dim=1)
            else:
                chunk_ds = torch.nn.functional.pad(chunk, (0, width - chunk.shape[0]))
                
            # Normalize to -1..1 then scale to height
            chunk_ds = chunk_ds / (chunk_ds.abs().max() + 1e-6)
            
            if mode == 'line':
                # Draw line across middle
                center_y = height // 2
                offset = (chunk_ds * (height // 3)).long()
                ys = torch.clamp(center_y + offset, 0, height-1)
                xs = torch.arange(width)
                
                for t in range(-thickness//2, thickness//2 + 1):
                    valid_y = torch.clamp(ys + t, 0, height-1)
                    video_frames[i, 0, valid_y, xs] = color[0]
                    video_frames[i, 1, valid_y, xs] = color[1]
                    video_frames[i, 2, valid_y, xs] = color[2]
                    
            elif mode == 'circle':
                center_h, center_w = height // 2, width // 2
                base_radius = height // 4
                
                dy = y_grid - center_h
                dx = x_grid - center_w
                r_grid = torch.sqrt(dy**2 + dx**2)
                theta_grid = torch.atan2(dy, dx) # -pi to pi
                theta_norm = (theta_grid + math.pi) / (2 * math.pi) # 0 to 1
                
                idx_grid = (theta_norm * (width - 1)).long()
                deformation = chunk_ds[idx_grid]
                
                target_radius = base_radius + deformation * (base_radius * 0.5)
                
                mask = (r_grid >= target_radius - thickness) & (r_grid <= target_radius + thickness)
                
                video_frames[i, 0, mask] = color[0]
                video_frames[i, 1, mask] = color[1]
                video_frames[i, 2, mask] = color[2]

        return video_frames
