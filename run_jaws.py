import torch
import soundfile as sf
import sys
import os
from pathlib import Path

# Ensure we can import from src
sys.path.append(str(Path(__file__).parent))

from src.audio_gen import AudioGenerator
from src.inference_diffusion import run_diffusion_inference
from src.utils import load_config

def main():
    print("🦈 Generating JAWS theme...")
    
    # Load config to get duration/sr
    config = load_config()
    duration = config.get('data', {}).get('duration', 3.0)
    sr = config.get('data', {}).get('sample_rate', 16000)
    
    # Generate Audio
    gen = AudioGenerator(sample_rate=sr)
    waveform = gen.generate_jaws_theme(duration=duration)
    
    # Save to file (keep outputs out of repo root)
    out_dir = Path("outputs/audio")
    out_dir.mkdir(parents=True, exist_ok=True)
    audio_path = str(out_dir / "jaws_theme.wav")
    sf.write(audio_path, waveform.squeeze().numpy(), sr)
    print(f"Saved audio to {audio_path}")
    
    # Run Inference
    print("🦈 Running inference on JAWS theme...")
    run_diffusion_inference(
        model_path="diffusion_checkpoints/diff_latest.pth",
        output_path="outputs/output_jaws.mp4",
        input_audio_path=audio_path
    )
    
    print("\n✅ Done! Check outputs/output_jaws.mp4")

if __name__ == "__main__":
    main()
