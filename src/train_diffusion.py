import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, IterableDataset
import time
from pathlib import Path
import random
import sys
import os

# Ensure we can import from current directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from vae import VAE
from diffusion import DiffusionTransformer, NoiseScheduler
from audio_gen import AudioGenerator
from audio_features import compute_audio_timeline
from audio_features import audio_feature_dim
from visualizers import get_random_visualizer
from tracker import ExperimentTracker
from utils import load_config, get_device

# Dataset that returns (Audio, Latents)
# We generate Video -> Encode with VAE -> Return Latents
class LatentDiffusionDataset(IterableDataset):
    def __init__(self, vae, config, device='cpu'):
        self.config = config
        self.samples_per_epoch = config.get('diffusion', {}).get('samples_per_epoch', 2000)
        self.vae = vae
        self.device = device
        self.sample_rate = config.get('data', {}).get('sample_rate', 16000)
        self.audio_gen = AudioGenerator(sample_rate=self.sample_rate)
        self.fps = self.config.get('data', {}).get('fps', 30)
        self.audio_feature_n_fft = int(self.config.get('diffusion', {}).get('audio_feature_n_fft', 512))
        self.audio_feature_num_bands = int(self.config.get('diffusion', {}).get('audio_feature_num_bands', 8))
        # Use local temporal context in feature space (neighboring frames).
        # 0 means only the current frame's feature vector is used.
        self.audio_feature_context = int(self.config.get('diffusion', {}).get('audio_feature_context', 0))
        self.audio_feature_T = 2 * self.audio_feature_context + 1
        self.style_dim = int(self.config.get('diffusion', {}).get('style_dim', 64))
        
    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            iter_start = 0
            iter_end = self.samples_per_epoch
        else:
            per_worker = int(self.samples_per_epoch / worker_info.num_workers)
            iter_start = worker_info.id * per_worker
            iter_end = iter_start + per_worker
            
        for _ in range(iter_start, iter_end):
            # 1. Generate Data
            # Use procedural audio instead of random noise
            duration = self.config.get('data', {}).get('duration', 3.0)
            waveform = self.audio_gen.generate_sequence(duration=duration) # (1, 48000)
            
            # Ideally we want real audio-viz pairs, but our current viz is random.
            # To make the model learn "Audio -> Viz", we need the viz to be reactive.
            # Our visualizers ARE reactive to the waveform passed in.
            
            viz = get_random_visualizer()
            height = self.config.get('data', {}).get('height', 128)
            width = self.config.get('data', {}).get('width', 128)
            frames = viz.render(waveform, fps=self.fps, height=height, width=width, sample_rate=self.sample_rate)
            num_frames = frames.size(0)
            # Sample one clip-level style latent and reuse for all frames from this clip
            style_z = torch.randn(self.style_dim)
            
            # 2. Encode to Latents
            # We process in batches to save memory if needed, but 90 frames is small
            with torch.no_grad():
                frames = frames.to(self.device)
                mu, log_var = self.vae.encode(frames)
                # We use the mean (mu) as the ground truth latent for diffusion
                # We could sample, but mean is cleaner for training targets
                latents = mu # (90, 256, 8, 8)
                # 2b. Compute audio features aligned to frames
                audio_feats = compute_audio_timeline(
                    waveform,
                    sample_rate=self.sample_rate,
                    fps=self.fps,
                    num_frames=num_frames,
                    n_fft=self.audio_feature_n_fft,
                    num_bands=self.audio_feature_num_bands,
                )  # (T, F)
            
            # 3. Yield pairs
            # We yield (Audio_Features_for_Frame_i, Latent_Frame_i).
            # Conditioning is identifiable: frame i is paired with its aligned audio feature vector (optionally with neighbor context).
            indices = torch.randperm(frames.size(0))
            frames_per_clip = self.config.get('diffusion', {}).get('frames_per_clip', 10)
            for i in indices[:frames_per_clip]:
                # Build a fixed-length context window in feature space: (T_ctx, F)
                # Pad by edge-replication so batching works cleanly.
                ctx = []
                for j in range(i - self.audio_feature_context, i + self.audio_feature_context + 1):
                    jj = min(max(j, 0), num_frames - 1)
                    ctx.append(audio_feats[jj])
                feat_ctx = torch.stack(ctx, dim=0)  # (T_ctx, F)
                yield feat_ctx, latents[i], style_z

class DiffusionTrainer:
    def __init__(self):
        self.device = get_device()
        print(f"Training on {self.device}")
        
        # Load config
        try:
            self.config = load_config()
        except:
            self.config = {}
        
        # 1. Load VAE (Frozen)
        latent_dim = self.config.get('model', {}).get('latent_dim', 256)
        self.vae = VAE(latent_dim=latent_dim).to(self.device)
        try:
            self.vae.load_state_dict(torch.load("vae_checkpoints/vae_latest.pth", map_location=self.device))
            print("Loaded VAE checkpoint.")
        except:
            print("WARNING: No VAE checkpoint found! Training from scratch (garbage in/out).")
            
        self.vae.eval()
        for param in self.vae.parameters():
            param.requires_grad = False
            
        # 2. Init Diffusion Model
        d_model = self.config.get('model', {}).get('d_model', 512)
        height = self.config.get('data', {}).get('height', 128)
        # VAE downsamples by 16 (4 layers of stride 2)
        latent_spatial_size = height // 16 
        num_bands = int(self.config.get('diffusion', {}).get('audio_feature_num_bands', 8))
        style_dim = int(self.config.get('diffusion', {}).get('style_dim', 64))
        self.model = DiffusionTransformer(
            latent_dim=latent_dim,
            d_model=d_model,
            latent_spatial_size=latent_spatial_size,
            audio_feature_dim=audio_feature_dim(num_bands),
            style_dim=style_dim,
        ).to(self.device)
        lr = self.config.get('diffusion', {}).get('learning_rate', 1e-4)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr)
        
        # 3. Tracker
        self.tracker = ExperimentTracker(self.config)
        
        # 4. Noise Schedule
        timesteps = self.config.get('diffusion', {}).get('timesteps', 50)
        self.scheduler = NoiseScheduler(num_timesteps=timesteps, device=self.device)

    def train(self, epochs=None):
        if epochs is None:
            epochs = self.config.get('diffusion', {}).get('epochs', 20)
            
        samples_per_epoch = self.config.get('diffusion', {}).get('samples_per_epoch', 100)
        batch_size = self.config.get('diffusion', {}).get('batch_size', 32)
        
        dataset = LatentDiffusionDataset(self.vae, config=self.config, device=self.device)
        dataloader = DataLoader(dataset, batch_size=batch_size)
        p_style_drop = float(self.config.get('diffusion', {}).get('p_style_drop', 0.2))
        
        self.model.train()
        
        for epoch in range(epochs):
            total_loss = 0
            start_time = time.time()
            batch_count = 0
            
            for batch_idx, (audio, latents, style) in enumerate(dataloader):
                audio = audio.to(self.device)
                latents = latents.to(self.device) # (B, 256, 8, 8)
                style = style.to(self.device)     # (B, style_dim)
                B = latents.shape[0]

                # Style dropout: sometimes remove style conditioning so the model
                # can still operate when style is unspecified at inference.
                if p_style_drop > 0:
                    drop_mask = (torch.rand((B,), device=self.device) < p_style_drop).to(style.dtype)[:, None]
                    style = style * (1.0 - drop_mask)
                
                # Sample random timesteps
                t = torch.randint(0, self.scheduler.num_timesteps, (B,), device=self.device).long()
                
                # Add noise
                noisy_latents, noise = self.scheduler.add_noise(latents, t)
                
                # Predict noise
                self.optimizer.zero_grad()
                noise_pred = self.model(noisy_latents, t, audio, style=style)
                
                # Loss
                loss = nn.functional.mse_loss(noise_pred, noise)
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                batch_count += 1
                
                if batch_idx % 10 == 0:
                    print(f"Epoch {epoch+1} | Batch {batch_idx} | Loss: {loss.item():.4f}")
                    self.tracker.log_metrics({"diff_batch_loss": loss.item()})
            
            if batch_count > 0:
                avg_loss = total_loss / batch_count
                epoch_time = time.time() - start_time
                print(f"Epoch {epoch+1} Complete | Avg Loss: {avg_loss:.4f} | Time: {epoch_time:.2f}s")
                self.tracker.log_metrics({
                    "diff_epoch": epoch + 1,
                    "diff_loss": avg_loss,
                    "diff_time": epoch_time
                })
                
            # Save checkpoint
            Path("diffusion_checkpoints").mkdir(exist_ok=True)
            torch.save(self.model.state_dict(), "diffusion_checkpoints/diff_latest.pth")
            
        self.tracker.finish()

def train_diffusion(epochs=None):
    trainer = DiffusionTrainer()
    trainer.train(epochs=epochs)

if __name__ == "__main__":
    train_diffusion()
