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
from utils import load_config

# Simple dataset that just generates frames (ignoring audio)
class VisualizerFrameDataset(IterableDataset):
    def __init__(self, samples_per_epoch=2000, batch_size=32):
        self.samples_per_epoch = samples_per_epoch
        self.batch_size = batch_size
        
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
            waveform = torch.randn(1, 48000) 
            viz = get_random_visualizer()
            frames = viz.render(waveform, fps=30) # (T, 3, H, W)
            
            # Yield individual frames to train the VAE on images
            # Shuffle frames to break temporal correlation in batch
            indices = torch.randperm(frames.size(0))
            for i in indices:
                yield frames[i]

def loss_function(recon_x, x, mu, log_var):
    """VAE Loss = MSE + KLD"""
    MSE = nn.functional.mse_loss(recon_x, x, reduction='sum')
    # KLD = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    # Weight KLD to prevent posterior collapse or explosion
    return MSE + 0.1 * KLD

def train_vae():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Training VAE on {device}...")
    
    # Load config for tracker
    try:
        config = load_config()
    except:
        config = {}
        
    tracker = ExperimentTracker(config)
    
    # Config
    BATCH_SIZE = 64
    EPOCHS = 5
    LR = 1e-4 # Reduced LR
    LATENT_DIM = 256
    SAMPLES_PER_EPOCH = 50 # Reduced for speed
    
    model = VAE(latent_dim=LATENT_DIM).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    dataset = VisualizerFrameDataset(samples_per_epoch=SAMPLES_PER_EPOCH)
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
            loss = loss_function(recon_batch, frames, mu, log_var)
            
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
