import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchaudio
import torchvision
from pathlib import Path
import time
import soundfile as sf
from .model import AudioToVideoNet
from .tracker import ExperimentTracker
from .data import InfiniteAVDataset, create_synthetic_dataset

# We keep AVDataset for backward compatibility or validation on static files
class AVDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        self.samples = list(self.data_dir.glob("*.wav"))
        self.data_cache = []
        
        print(f"Loading {len(self.samples)} samples into memory for speed...")
        for wav_path in self.samples:
            # Load Audio
            wav_data, sr = sf.read(wav_path)
            waveform = torch.from_numpy(wav_data).float()
            if waveform.ndim == 1:
                waveform = waveform.unsqueeze(0)
            else:
                waveform = waveform.t()
            
            # Ensure fixed length (pad or trim) - simple approach: trim to 3s
            target_len = 48000 # 3s @ 16kHz
            if waveform.shape[1] > target_len:
                waveform = waveform[:, :target_len]
            elif waveform.shape[1] < target_len:
                padding = torch.zeros(1, target_len - waveform.shape[1])
                waveform = torch.cat([waveform, padding], dim=1)
                
            # Load Video
            mp4_path = wav_path.with_suffix(".mp4")
            
            # Use torchcodec for video decoding
            from torchcodec.decoders import VideoDecoder
            decoder = VideoDecoder(str(mp4_path))
            # decoder[:] returns (T, C, H, W) in uint8 [0, 255]
            video = decoder[:].float() / 255.0
            
            # read_video returns (T, H, W, C)
            # video, _, _ = torchvision.io.read_video(str(mp4_path), pts_unit='sec')
            # video = video.permute(0, 3, 1, 2).float() / 255.0
            
            # Ensure 90 frames (3s * 30fps)
            target_frames = 90
            if video.shape[0] > target_frames:
                video = video[:target_frames]
            elif video.shape[0] < target_frames:
                if video.shape[0] > 0:
                    last_frame = video[-1:]
                    padding = last_frame.repeat(target_frames - video.shape[0], 1, 1, 1)
                    video = torch.cat([video, padding], dim=0)
                else:
                    # Handle empty video case
                    video = torch.zeros(target_frames, 3, 128, 128)
                
            self.data_cache.append((waveform, video))
            
    def __len__(self):
        return len(self.data_cache)
    
    def __getitem__(self, idx):
        return self.data_cache[idx]

def train_model(data_dir, output_dir="models", epochs=20, batch_size=8, learning_rate=1e-3, frame_size=(128, 128), num_frames=90, config=None):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Training on {device}")
    
    # Initialize Tracker
    tracker = ExperimentTracker(config if config else {})
    
    # Use Infinite Dataset
    # We use num_samples from config as "samples per epoch"
    samples_per_epoch = config.get('data', {}).get('num_samples', 1000)
    dataset = InfiniteAVDataset(
        samples_per_epoch=samples_per_epoch,
        duration=config.get('data', {}).get('duration', 3.0),
        fps=config.get('data', {}).get('fps', 30),
        height=frame_size[0],
        width=frame_size[1]
    )
    
    # num_workers=0 for now to avoid complexity with MPS/multiprocessing, 
    # but can be increased for CPU generation speedup.
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=0)
    
    model = AudioToVideoNet(num_frames=num_frames, frame_size=frame_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    
    Path(output_dir).mkdir(exist_ok=True)
    
    model.train()
    total_start = time.time()
    
    print(f"Starting training with Infinite Data Engine ({samples_per_epoch} samples/epoch)...")
    
    for epoch in range(epochs):
        total_loss = 0
        epoch_start = time.time()
        
        # The dataloader will yield exactly samples_per_epoch items because of how we wrote __iter__
        for i, (audio, video) in enumerate(dataloader):
            audio, video = audio.to(device), video.to(device)
            
            optimizer.zero_grad()
            output = model(audio)
            
            loss = criterion(output, video)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # Log batch loss occasionally
            if i % 10 == 0:
                tracker.log_metrics({"batch_loss": loss.item()})
            
        avg_loss = total_loss / len(dataloader)
        epoch_time = time.time() - epoch_start
        print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | Time: {epoch_time:.2f}s")
        
        # Log epoch metrics
        tracker.log_metrics({
            "epoch": epoch + 1,
            "loss": avg_loss,
            "epoch_time": epoch_time
        })
        
        # Log a sample video every 5 epochs (if enabled)
        if (epoch + 1) % 5 == 0 and config.get('tracker', {}).get('log_videos', False):
            with torch.no_grad():
                # Pick first sample from last batch
                sample_out = output[0] # (T, C, H, W)
                tracker.log_video(sample_out, caption=f"Epoch {epoch+1} Sample")
        
    total_time = time.time() - total_start
    print(f"Training complete in {total_time:.2f}s. Model saved.")
    torch.save(model.state_dict(), f"{output_dir}/model_latest.pth")
    
    tracker.finish()
    return model

if __name__ == "__main__":
    train_model("data/train")
