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
from diffusion import DiffusionTransformer
from visualizers import get_random_visualizer
from tracker import ExperimentTracker
from utils import load_config

# Dataset that returns (Audio, Latents)
# We generate Video -> Encode with VAE -> Return Latents
class LatentDiffusionDataset(IterableDataset):
    def __init__(self, vae, samples_per_epoch=2000, device='cpu'):
        self.samples_per_epoch = samples_per_epoch
        self.vae = vae
        self.device = device
        
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
            waveform = torch.randn(1, 48000) # Random audio for now (visualizers are random anyway)
            # Ideally we want real audio-viz pairs, but our current viz is random.
            # To make the model learn "Audio -> Viz", we need the viz to be reactive.
            # Our visualizers ARE reactive to the waveform passed in.
            
            viz = get_random_visualizer()
            frames = viz.render(waveform, fps=30) # (90, 3, 128, 128)
            
            # 2. Encode to Latents
            # We process in batches to save memory if needed, but 90 frames is small
            with torch.no_grad():
                frames = frames.to(self.device)
                mu, log_var = self.vae.encode(frames)
                # We use the mean (mu) as the ground truth latent for diffusion
                # We could sample, but mean is cleaner for training targets
                latents = mu # (90, 256)
                
                # Reshape back to spatial for the diffusion model
                # VAE encode flattens, but we know it came from 8x8
                latents = latents.view(90, 256, 8, 8)
            
            # 3. Yield pairs
            # We yield (Audio, Latent_Frame)
            # We repeat audio for each frame? Or train on sequence?
            # For simplicity, let's train Frame-by-Frame conditioned on the WHOLE audio clip.
            # This allows the model to look at past/future audio context.
            
            # Yield: (Audio_Clip, Latent_Frame_t, t_index)
            # Actually, let's just pick random frames from the clip to avoid correlation
            indices = torch.randperm(frames.size(0))
            for i in indices[:10]: # Just take 10 frames per clip to mix it up
                yield waveform, latents[i]

class DiffusionTrainer:
    def __init__(self):
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        print(f"Training on {self.device}")
        
        # 1. Load VAE (Frozen)
        self.vae = VAE(latent_dim=256).to(self.device)
        try:
            self.vae.load_state_dict(torch.load("vae_checkpoints/vae_latest.pth", map_location=self.device))
            print("Loaded VAE checkpoint.")
        except:
            print("WARNING: No VAE checkpoint found! Training from scratch (garbage in/out).")
            
        self.vae.eval()
        for param in self.vae.parameters():
            param.requires_grad = False
            
        # 2. Init Diffusion Model
        self.model = DiffusionTransformer(latent_dim=256, d_model=512).to(self.device)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=1e-4)
        
        # 3. Tracker
        try:
            config = load_config()
        except:
            config = {}
        self.tracker = ExperimentTracker(config)
        
        # 4. Noise Schedule
        self.num_timesteps = 1000
        self.betas = torch.linspace(0.0001, 0.02, self.num_timesteps).to(self.device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

    def add_noise(self, x_0, t):
        """
        x_0: Clean latents
        t: Timesteps
        Returns: x_t (Noisy), noise (The added noise)
        """
        sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod[t])[:, None, None, None]
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod[t])[:, None, None, None]
        
        noise = torch.randn_like(x_0)
        x_t = sqrt_alphas_cumprod * x_0 + sqrt_one_minus_alphas_cumprod * noise
        return x_t, noise

    def train(self, epochs=10):
        dataset = LatentDiffusionDataset(self.vae, samples_per_epoch=100, device=self.device)
        dataloader = DataLoader(dataset, batch_size=32)
        
        self.model.train()
        
        for epoch in range(epochs):
            total_loss = 0
            start_time = time.time()
            batch_count = 0
            
            for batch_idx, (audio, latents) in enumerate(dataloader):
                audio = audio.to(self.device)
                latents = latents.to(self.device) # (B, 256, 8, 8)
                B = latents.shape[0]
                
                # Sample random timesteps
                t = torch.randint(0, self.num_timesteps, (B,), device=self.device).long()
                
                # Add noise
                noisy_latents, noise = self.add_noise(latents, t)
                
                # Predict noise
                self.optimizer.zero_grad()
                noise_pred = self.model(noisy_latents, t, audio)
                
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

if __name__ == "__main__":
    trainer = DiffusionTrainer()
    trainer.train()
