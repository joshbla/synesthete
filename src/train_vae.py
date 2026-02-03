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
from visualizers import get_random_visualizer
from tracker import ExperimentTracker
from utils import load_config, get_device

# Simple dataset that just generates frames (ignoring audio)
class VisualizerFrameDataset(IterableDataset):
    def __init__(self, config):
        self.config = config
        self.samples_per_epoch = config.get('vae', {}).get('samples_per_epoch', 2000)
        self.batch_size = config.get('vae', {}).get('batch_size', 32)
        
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
            # Generate a clip
            # We don't need real audio, just random waveform to trigger viz
            sample_rate = self.config.get('data', {}).get('sample_rate', 16000)
            duration = self.config.get('data', {}).get('duration', 3.0)
            audio_len = int(sample_rate * duration)
            waveform = torch.randn(1, audio_len) 
            viz = get_random_visualizer()
            fps = self.config.get('data', {}).get('fps', 30)
            height = self.config.get('data', {}).get('height', 128)
            width = self.config.get('data', {}).get('width', 128)
            frames = viz.render(waveform, fps=fps, height=height, width=width, sample_rate=sample_rate) # (T, 3, H, W)
            
            # Yield individual frames to train the VAE on images
            # Shuffle frames to break temporal correlation in batch
            indices = torch.randperm(frames.size(0))
            for i in indices:
                yield frames[i]

def loss_function(recon_x, x, mu, log_var, kld_weight=0.1):
    """VAE Loss = MSE + KLD"""
    MSE = nn.functional.mse_loss(recon_x, x, reduction='sum')
    # KLD = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    # Weight KLD to prevent posterior collapse or explosion
    return MSE + kld_weight * KLD

def train_vae():
    device = get_device()
    print(f"Training VAE on {device}...")
    
    # Load config for tracker
    try:
        config = load_config()
    except:
        config = {}
        
    tracker = ExperimentTracker(config)
    
    # Config
    BATCH_SIZE = config.get('vae', {}).get('batch_size', 64)
    EPOCHS = config.get('vae', {}).get('epochs', 10)
    LR = config.get('vae', {}).get('learning_rate', 1e-4)
    LATENT_DIM = config.get('model', {}).get('latent_dim', 256)
    KLD_WEIGHT = config.get('vae', {}).get('kld_weight', 0.1)
    
    model = VAE(latent_dim=LATENT_DIM).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    dataset = VisualizerFrameDataset(config=config)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE)
    
    output_dir = Path("vae_checkpoints")
    output_dir.mkdir(exist_ok=True)
    
    model.train()
    
    for epoch in range(EPOCHS):
        total_loss = 0
        start_time = time.time()
        batch_count = 0
        
        for batch_idx, frames in enumerate(dataloader):
            frames = frames.to(device)
            
            optimizer.zero_grad()
            recon_batch, mu, log_var = model(frames)
            loss = loss_function(recon_batch, frames, mu, log_var, kld_weight=KLD_WEIGHT)
            
            if torch.isnan(loss):
                print("Warning: Loss is NaN, skipping batch")
                continue
                
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            batch_count += 1
            
            if batch_idx % 20 == 0:
                print(f"Epoch {epoch+1} | Batch {batch_idx} | Loss: {loss.item() / len(frames):.2f}")
                tracker.log_metrics({"vae_batch_loss": loss.item() / len(frames)})
                
        if batch_count > 0:
            avg_loss = total_loss / (batch_count * BATCH_SIZE)
            epoch_time = time.time() - start_time
            print(f"Epoch {epoch+1} Complete | Avg Loss: {avg_loss:.2f} | Time: {epoch_time:.2f}s")
            tracker.log_metrics({
                "vae_epoch": epoch + 1,
                "vae_loss": avg_loss,
                "vae_time": epoch_time
            })
            
            # Log reconstruction sample
            if (epoch + 1) % 1 == 0: # Log every epoch
                with torch.no_grad():
                    # Take first 8 frames from last batch
                    sample = frames[:8]
                    recon, _, _ = model(sample)
                    # Concat original and recon
                    comparison = torch.cat([sample, recon], dim=0)
                    tracker.log_video(comparison.unsqueeze(1), caption=f"VAE Recon Epoch {epoch+1}")
        
        # Save checkpoint
        torch.save(model.state_dict(), output_dir / "vae_latest.pth")

    print("VAE Training Complete.")
    tracker.finish()

if __name__ == "__main__":
    train_vae()
