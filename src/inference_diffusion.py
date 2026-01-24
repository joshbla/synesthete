import torch
import torch.nn as nn
from pathlib import Path
import sys
import os

# Ensure we can import from current directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from vae import VAE
from diffusion import DiffusionTransformer, NoiseScheduler
from audio_gen import AudioGenerator
from visualizers import get_random_visualizer

def run_diffusion_inference(model_path="diffusion_checkpoints/diff_latest.pth", output_path="output_diffusion.mp4", input_audio_path=None, num_frames=90):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Running diffusion inference on {device}")
    
    # 1. Load VAE (Frozen)
    vae = VAE(latent_dim=256).to(device)
    try:
        vae.load_state_dict(torch.load("vae_checkpoints/vae_latest.pth", map_location=device))
        print("Loaded VAE checkpoint.")
    except:
        print("WARNING: No VAE checkpoint found! Results will be garbage.")
    vae.eval()
    
    # 2. Load Diffusion Model
    model = DiffusionTransformer(latent_dim=256, d_model=512).to(device)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded Diffusion checkpoint from {model_path}")
    except:
        print(f"WARNING: No Diffusion checkpoint found at {model_path}. Using random weights.")
    model.eval()
    
    # 3. Get Audio
    if input_audio_path:
        import soundfile as sf
        wav_data, sr = sf.read(input_audio_path)
        waveform = torch.from_numpy(wav_data).float()
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        else:
            waveform = waveform.t()
        
        # Resample if needed
        if sr != 16000:
            import torchaudio.transforms as T
            resampler = T.Resample(sr, 16000)
            waveform = resampler(waveform)
    else:
        print("No input audio provided, generating synthetic complex test sound...")
        gen = AudioGenerator(sample_rate=16000)
        waveform = gen.generate_sequence(duration=3.0, bpm=120) # (1, 48000)
        
    # Ensure correct length
    target_len = 48000
    if waveform.shape[1] > target_len:
        waveform = waveform[:, :target_len]
    elif waveform.shape[1] < target_len:
         padding = torch.zeros(1, target_len - waveform.shape[1])
         waveform = torch.cat([waveform, padding], dim=1)
         
    waveform = waveform.unsqueeze(0).to(device) # (1, 1, 48000)
    
    # 4. Sampling Loop
    print("Sampling...")
    scheduler = NoiseScheduler(num_timesteps=50, device=device)
    
    # Batch generation for speed
    batch_size = num_frames
    audio_batch = waveform.repeat(batch_size, 1, 1) # (90, 1, 48000)
    
    # Start with pure noise
    shape = (batch_size, 256, 8, 8)
    
    # Sample
    latents = scheduler.sample(model, audio_batch, shape) # (90, 256, 8, 8)
    
    # 5. Decode Latents
    print("Decoding latents...")
    with torch.no_grad():
        output_frames = vae.decode(latents) # (90, 3, 128, 128)
        
    # 6. Save Video
    print("Saving video...")
    
    # Prepare video tensor: (T, C, H, W) -> (T, H, W, C) for ffmpeg or similar if needed,
    # but torchcodec VideoEncoder expects (T, C, H, W) or (T, H, W, C)?
    # Let's check src/inference.py again.
    # It used: video_uint8 = (video_tensor * 255).byte() -> (T, C, H, W)
    # encoder = VideoEncoder(video_uint8, ...)
    
    video_tensor = output_frames.cpu()
    video_uint8 = (video_tensor * 255).byte() # (T, C, H, W)
    
    from torchcodec.encoders import VideoEncoder
    encoder = VideoEncoder(video_uint8, frame_rate=30)
    temp_video_path = output_path.replace(".mp4", "_temp.mp4")
    encoder.to_file(temp_video_path)
    
    # Save audio temporarily
    temp_audio_path = output_path.replace(".mp4", "_temp.wav")
    audio_to_save = waveform.squeeze(0).cpu() # (1, T)
    
    import torchaudio
    torchaudio.save(temp_audio_path, audio_to_save, 16000)
    
    # Combine with ffmpeg
    import subprocess
    cmd = [
        "ffmpeg", "-y",
        "-i", temp_video_path,
        "-i", temp_audio_path,
        "-c:v", "copy",
        "-c:a", "aac",
        "-strict", "experimental",
        output_path
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    # Clean up temp files
    if os.path.exists(temp_video_path):
        os.remove(temp_video_path)
    if os.path.exists(temp_audio_path):
        os.remove(temp_audio_path)
        
    print(f"Saved generated video to {output_path}")

if __name__ == "__main__":
    run_diffusion_inference()
