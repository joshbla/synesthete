import torch
import numpy as np
import random

class Visualizer:
    def render(self, waveform, fps=30, height=128, width=128):
        raise NotImplementedError

class PulseVisualizer(Visualizer):
    def render(self, waveform, fps=30, height=128, width=128):
        duration = waveform.shape[1] / 16000
        num_frames = int(duration * fps)
        video_frames = torch.zeros(num_frames, 3, height, width)
        
        samples_per_frame = waveform.shape[1] // num_frames
        
        # Grid for circle
        y_grid, x_grid = torch.meshgrid(torch.arange(height), torch.arange(width), indexing='ij')
        center_h, center_w = height // 2, width // 2
        dist_sq = (y_grid - center_h)**2 + (x_grid - center_w)**2
        
        # Color base
        r_base = random.random()
        g_base = random.random()
        b_base = random.random()
        
        for i in range(num_frames):
            start = i * samples_per_frame
            end = min((i + 1) * samples_per_frame, waveform.shape[1])
            chunk = waveform[0, start:end]
            amp = chunk.abs().mean().item()
            
            radius = (height // 2) * min(1.0, amp * 5.0)
            if radius > 1:
                mask = dist_sq <= radius**2
                video_frames[i, 0, mask] = r_base * min(1.0, amp * 3.0)
                video_frames[i, 1, mask] = g_base * min(1.0, amp * 3.0)
                video_frames[i, 2, mask] = b_base * min(1.0, amp * 3.0)
                
        return video_frames

class SpectrumVisualizer(Visualizer):
    def render(self, waveform, fps=30, height=128, width=128):
        duration = waveform.shape[1] / 16000
        num_frames = int(duration * fps)
        video_frames = torch.zeros(num_frames, 3, height, width)
        
        samples_per_frame = waveform.shape[1] // num_frames
        
        for i in range(num_frames):
            start = i * samples_per_frame
            end = min((i + 1) * samples_per_frame, waveform.shape[1])
            chunk = waveform[0, start:end]
            
            # Simple FFT
            if chunk.shape[0] > 10:
                fft = torch.fft.rfft(chunk)
                fft_mag = fft.abs()
                # Log scale roughly
                fft_mag = torch.log1p(fft_mag)
                if fft_mag.max() > 0:
                    fft_mag = fft_mag / fft_mag.max()
                
                # Draw bars
                num_bars = 16
                bar_width = width // num_bars
                
                # Resample fft to num_bars
                # Simple binning
                bin_size = fft_mag.shape[0] // num_bars
                
                for b in range(num_bars):
                    if bin_size > 0:
                        val = fft_mag[b*bin_size : (b+1)*bin_size].mean().item()
                    else:
                        val = 0
                        
                    bar_height = int(val * height)
                    if bar_height > 0:
                        # Draw vertical bar
                        # Color gradient based on bar index
                        r = b / num_bars
                        g = 1.0 - (b / num_bars)
                        b_col = 0.5
                        
                        video_frames[i, 0, height-bar_height:, b*bar_width:(b+1)*bar_width] = r
                        video_frames[i, 1, height-bar_height:, b*bar_width:(b+1)*bar_width] = g
                        video_frames[i, 2, height-bar_height:, b*bar_width:(b+1)*bar_width] = b_col
                        
        return video_frames

class NoiseVisualizer(Visualizer):
    def render(self, waveform, fps=30, height=128, width=128):
        # Perlin-ish noise that reacts to amplitude
        duration = waveform.shape[1] / 16000
        num_frames = int(duration * fps)
        video_frames = torch.zeros(num_frames, 3, height, width)
        
        samples_per_frame = waveform.shape[1] // num_frames
        
        # Seed noise
        noise_base = torch.rand(height, width)
        
        for i in range(num_frames):
            start = i * samples_per_frame
            end = min((i + 1) * samples_per_frame, waveform.shape[1])
            chunk = waveform[0, start:end]
            amp = chunk.abs().mean().item()
            
            # Shift noise based on amp
            shift = int(amp * 20)
            current_noise = torch.roll(noise_base, shifts=shift, dims=0)
            
            # Colorize
            video_frames[i, 0] = current_noise * amp * 4.0 # Red channel sensitive
            video_frames[i, 1] = current_noise * 0.2
            video_frames[i, 2] = torch.roll(current_noise, shifts=-shift, dims=1) * amp * 2.0
            
        return video_frames

REGISTRY = [PulseVisualizer, SpectrumVisualizer, NoiseVisualizer]

def get_random_visualizer():
    return random.choice(REGISTRY)()
