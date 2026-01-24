import torch
import torchaudio
import torchvision
from pathlib import Path
from .model import AudioToVideoNet
from .utils import get_device
# from .data import generate_musical_clip

def run_inference(model_path, output_path="output.mp4", input_audio_path=None, num_frames=90, frame_size=(128, 128)):
    device = get_device()
    print(f"Running inference on {device}")
    
    # Load Model
    model = AudioToVideoNet(num_frames=num_frames, frame_size=frame_size).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Get Audio
    if input_audio_path:
        import soundfile as sf
        wav_data, sr = sf.read(input_audio_path)
        waveform = torch.from_numpy(wav_data).float()
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        else:
            waveform = waveform.t()
    else:
        print("No input audio provided, generating synthetic complex test sound...")
        from .audio_gen import AudioGenerator
        gen = AudioGenerator()
        waveform = gen.generate_sequence(duration=3.0, bpm=120)
        
    # Prepare input
    # Ensure (1, 1, 48000) - wait, we used 32000 before (2s), now 3s = 48000
    target_len = 48000
    if waveform.shape[1] > target_len:
        waveform = waveform[:, :target_len]
    elif waveform.shape[1] < target_len:
         padding = torch.zeros(1, target_len - waveform.shape[1])
         waveform = torch.cat([waveform, padding], dim=1)
         
    waveform = waveform.unsqueeze(0).to(device) # Add batch dim
    
    # Predict
    with torch.no_grad():
        output_frames = model(waveform) # (1, 60, 3, 128, 128)
        
    # Save Video
    # Output is (B, T, C, H, W) in [0, 1]
    # Need (T, H, W, C) in [0, 255] uint8
    # video = output_frames.squeeze(0)
    # video = video.permute(0, 2, 3, 1) # T, H, W, C
    # video = (video * 255).byte().cpu()
    
    # Prepare audio for saving (Time, Channels)
    audio_to_save = waveform.squeeze(0).cpu() # (C, T)
    # audio_to_save = audio_to_save.t() # (T, C) if needed, but 1D array works for mono often
    
    # Use torchcodec to write video, then ffmpeg to add audio
    # video is (H, W, C) uint8
    # torchcodec expects (T, C, H, W)
    # video was permuted to (T, H, W, C) earlier
    
    # Let's revert the permute for torchcodec
    # output_frames is (1, T, C, H, W) -> squeeze -> (T, C, H, W)
    video_tensor = output_frames.squeeze(0).cpu()
    video_uint8 = (video_tensor * 255).byte()
    
    from torchcodec.encoders import VideoEncoder
    encoder = VideoEncoder(video_uint8, frame_rate=30)
    temp_video_path = output_path.replace(".mp4", "_temp.mp4")
    encoder.to_file(temp_video_path)
    
    # Save audio temporarily
    temp_audio_path = output_path.replace(".mp4", "_temp.wav")
    
    # Ensure (C, T) for torchaudio
    if audio_to_save.ndim == 1:
        audio_to_save = audio_to_save.unsqueeze(0)
        
    import torchaudio
    # Use torchaudio to save, it handles formats well
    torchaudio.save(temp_audio_path, audio_to_save, 16000)
    
    # import soundfile as sf
    # sf.write(temp_audio_path, audio_to_save.squeeze().numpy(), 16000)
    
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
    import os
    if os.path.exists(temp_video_path):
        os.remove(temp_video_path)
    if os.path.exists(temp_audio_path):
        os.remove(temp_audio_path)
        
    # torchvision.io.write_video(output_path, video, fps=30, audio_array=audio_to_save, audio_fps=16000, audio_codec='aac')
    print(f"Saved generated video to {output_path}")

if __name__ == "__main__":
    run_inference("models/model_latest.pth")
