import torch
import random
from .base import Visualizer

class PulseVisualizer(Visualizer):
    def render(self, waveform, fps=30, height=128, width=128):
        num_frames, samples_per_frame = self.get_frame_audio_chunks(waveform, fps)
        video_frames = torch.zeros(num_frames, 3, height, width)
        
        # Randomize parameters for this clip
        num_shapes = random.randint(1, 3)
        shapes = []
        for _ in range(num_shapes):
            # Ensure high saturation color
            rgb = [random.random(), random.random(), random.random()]
            min_idx = rgb.index(min(rgb))
            rgb[min_idx] = 0.0
            max_idx = rgb.index(max(rgb))
            rgb[max_idx] = 1.0
            
            shapes.append({
                'center_h': random.randint(int(height*0.2), int(height*0.8)),
                'center_w': random.randint(int(width*0.2), int(width*0.8)),
                'shape_type': random.choice(['circle', 'square', 'diamond']),
                'color': torch.tensor(rgb),
                'scale': random.uniform(0.5, 2.0)
            })

        # Precompute grid
        y_grid, x_grid = torch.meshgrid(torch.arange(height), torch.arange(width), indexing='ij')
        
        for i in range(num_frames):
            start = i * samples_per_frame
            end = min((i + 1) * samples_per_frame, waveform.shape[1])
            chunk = waveform[0, start:end]
            amp = chunk.abs().mean().item()
            
            for shape in shapes:
                # Calculate size based on amp
                base_radius = (height // 4) * shape['scale']
                current_radius = base_radius * min(1.5, amp * 5.0)
                
                if current_radius < 1:
                    continue

                if shape['shape_type'] == 'circle':
                    dist_sq = (y_grid - shape['center_h'])**2 + (x_grid - shape['center_w'])**2
                    mask = dist_sq <= current_radius**2
                elif shape['shape_type'] == 'square':
                    mask = (torch.abs(y_grid - shape['center_h']) <= current_radius) & \
                           (torch.abs(x_grid - shape['center_w']) <= current_radius)
                elif shape['shape_type'] == 'diamond':
                    mask = (torch.abs(y_grid - shape['center_h']) + \
                            torch.abs(x_grid - shape['center_w'])) <= current_radius * 1.4

                # Add color to frame (additive blending)
                for c in range(3):
                    # Boosted gain for visibility
                    val = shape['color'][c] * min(1.0, amp * 6.0)
                    video_frames[i, c, mask] = torch.clamp(
                        video_frames[i, c, mask] + val, 
                        0, 1
                    )
                
        return video_frames
