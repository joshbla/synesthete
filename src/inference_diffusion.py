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
from audio_features import compute_audio_timeline, audio_feature_dim
from visualizers import get_random_visualizer
from utils import load_config, get_device

def run_diffusion_inference(model_path="diffusion_checkpoints/diff_latest.pth", output_path="output_diffusion.mp4", input_audio_path=None, num_frames=None):
    device = get_device()
    print(f"Running diffusion inference on {device}")
    
    config = load_config()
    if num_frames is None:
        num_frames = config.get('model', {}).get('num_frames', 90)
    
    # 1. Load VAE (Frozen)
    latent_dim = config.get('model', {}).get('latent_dim', 256)
    vae = VAE(latent_dim=latent_dim).to(device)
    try:
        vae.load_state_dict(torch.load("vae_checkpoints/vae_latest.pth", map_location=device))
        print("Loaded VAE checkpoint.")
    except:
        print("WARNING: No VAE checkpoint found! Results will be garbage.")
    vae.eval()
    
    # 2. Load Diffusion Model
    d_model = config.get('model', {}).get('d_model', 512)
    height = config.get('data', {}).get('height', 128)
    latent_spatial_size = height // 16
    num_bands = int(config.get('diffusion', {}).get('audio_feature_num_bands', 8))
    model = DiffusionTransformer(
        latent_dim=latent_dim,
        d_model=d_model,
        latent_spatial_size=latent_spatial_size,
        audio_feature_dim=audio_feature_dim(num_bands),
    ).to(device)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded Diffusion checkpoint from {model_path}")
    except:
        print(f"WARNING: No Diffusion checkpoint found at {model_path}. Using random weights.")
    model.eval()
    
    # 3. Get Audio
    sample_rate = config.get('data', {}).get('sample_rate', 16000)
    if input_audio_path:
        import soundfile as sf
        wav_data, sr = sf.read(input_audio_path)
        waveform = torch.from_numpy(wav_data).float()
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        else:
            waveform = waveform.t()
        
        # Resample if needed
        if sr != sample_rate:
            import torchaudio.transforms as T
            resampler = T.Resample(sr, sample_rate)
            waveform = resampler(waveform)
    else:
        print("No input audio provided, generating synthetic complex test sound...")
        gen = AudioGenerator(sample_rate=sample_rate)
        duration = config.get('data', {}).get('duration', 3.0)
        waveform = gen.generate_sequence(duration=duration, bpm=120) # (1, 48000)
        
    # Ensure correct length
    target_len = int(sample_rate * config.get('data', {}).get('duration', 3.0))
    if waveform.shape[1] > target_len:
        waveform = waveform[:, :target_len]
    elif waveform.shape[1] < target_len:
         padding = torch.zeros(1, target_len - waveform.shape[1])
         waveform = torch.cat([waveform, padding], dim=1)
         
    waveform = waveform.unsqueeze(0).to(device) # (1, 1, 48000)
    
    # 4. Sampling Loop
    print("Sampling...")
    timesteps = config.get('diffusion', {}).get('timesteps', 50)
    scheduler = NoiseScheduler(num_timesteps=timesteps, device=device)

    # Compute a per-frame audio feature timeline and condition each frame on
    # a fixed-length local context window in feature space.
    fps = config.get('data', {}).get('fps', 30)
    n_fft = int(config.get('diffusion', {}).get('audio_feature_n_fft', 512))
    context = int(config.get('diffusion', {}).get('audio_feature_context', 0))
    audio_feats = compute_audio_timeline(
        waveform.squeeze(0).cpu(),  # (1, N) on CPU is fine; we'll move to device after
        sample_rate=sample_rate,
        fps=fps,
        num_frames=num_frames,
        n_fft=n_fft,
        num_bands=num_bands,
    )  # (T, F)

    # Build (T, T_ctx, F) then treat each frame as a batch item: (T, T_ctx, F)
    ctx_feats = []
    for i in range(num_frames):
        ctx = []
        for j in range(i - context, i + context + 1):
            jj = min(max(j, 0), num_frames - 1)
            ctx.append(audio_feats[jj])
        ctx_feats.append(torch.stack(ctx, dim=0))
    audio_batch = torch.stack(ctx_feats, dim=0).to(device)  # (T, T_ctx, F)
    batch_size = audio_batch.shape[0]
    
    # Start with pure noise
    shape = (batch_size, latent_dim, latent_spatial_size, latent_spatial_size)
    
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
    fps = config.get('data', {}).get('fps', 30)
    encoder = VideoEncoder(video_uint8, frame_rate=fps)
    temp_video_path = output_path.replace(".mp4", "_temp.mp4")
    encoder.to_file(temp_video_path)
    
    # Save audio temporarily
    temp_audio_path = output_path.replace(".mp4", "_temp.wav")
    audio_to_save = waveform.squeeze(0).cpu() # (1, T)
    
    import torchaudio
    torchaudio.save(temp_audio_path, audio_to_save, sample_rate)
    
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
