import torch
import random
from torch.utils.data import IterableDataset
from .audio_gen import AudioGenerator
from .visualizers import get_random_visualizer
from .audio_features import compute_audio_timeline

# Keep the old function for creating validation sets or debug samples
def create_synthetic_dataset(output_dir, num_samples=100, duration=3.0, fps=30, height=128, width=128):
    import soundfile as sf
    from pathlib import Path
    from torchcodec.encoders import VideoEncoder
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating {num_samples} static samples in {output_dir}...")
    
    audio_gen = AudioGenerator()
    
    for i in range(num_samples):
        waveform = audio_gen.generate_sequence(duration=duration, bpm=random.randint(80, 140))
        viz = get_random_visualizer()
        num_frames = max(1, int((waveform.shape[1] / audio_gen.sr) * fps))
        audio_feats = compute_audio_timeline(waveform, sample_rate=audio_gen.sr, fps=fps, num_frames=num_frames)
        video = viz.render(waveform, fps=fps, height=height, width=width, sample_rate=audio_gen.sr, audio_feats=audio_feats)
        
        audio_path = output_dir / f"sample_{i}.wav"
        video_path = output_dir / f"sample_{i}.mp4"
        
        sf.write(audio_path, waveform.squeeze(0).numpy(), 16000)
        
        video_uint8 = video.mul(255).byte()
        encoder = VideoEncoder(video_uint8, frame_rate=fps)
        encoder.to_file(str(video_path))
        
    print("Generation complete.")

import time

class InfiniteAVDataset(IterableDataset):
    """
    A PyTorch IterableDataset that generates audio-video pairs on-the-fly.
    This allows for infinite training without storage bottlenecks.
    """
    def __init__(self, samples_per_epoch=1000, duration=3.0, fps=30, height=128, width=128):
        self.samples_per_epoch = samples_per_epoch
        self.duration = duration
        self.fps = fps
        self.height = height
        self.width = width
        self.audio_gen = AudioGenerator()
        self.sample_rate = self.audio_gen.sr

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            # Single-process data loading, return the full iterator
            iter_start = 0
            iter_end = self.samples_per_epoch
        else:
            # Multi-process data loading, split workload
            per_worker = int(self.samples_per_epoch / worker_info.num_workers)
            worker_id = worker_info.id
            iter_start = worker_id * per_worker
            iter_end = iter_start + per_worker
            
            # Seed random based on worker ID to ensure different data
            random.seed(worker_info.seed)
            torch.manual_seed(worker_info.seed)
            
        for _ in range(iter_start, iter_end):
            start_time = time.time()
            
            # 1. Generate Audio
            waveform = self.audio_gen.generate_sequence(
                duration=self.duration, 
                bpm=random.randint(80, 140)
            )
            
            # 2. Generate Video
            viz = get_random_visualizer()
            num_frames = max(1, int((waveform.shape[1] / self.sample_rate) * self.fps))
            audio_feats = compute_audio_timeline(
                waveform,
                sample_rate=self.sample_rate,
                fps=self.fps,
                num_frames=num_frames,
            )
            video = viz.render(
                waveform, 
                fps=self.fps, 
                height=self.height, 
                width=self.width,
                sample_rate=self.sample_rate,
                audio_feats=audio_feats,
            )
            
            gen_time = time.time() - start_time
            if random.random() < 0.01: # Log 1% of the time to avoid spam
                print(f"[DataGen] Generated sample in {gen_time*1000:.2f}ms")
            
            yield waveform, video

    def __len__(self):
        return self.samples_per_epoch
