import sys
import argparse
from pathlib import Path

# Ensure we can import from src
sys.path.append(str(Path(__file__).parent))

from src.train_vae import train_vae
from src.train_diffusion import train_diffusion
from src.inference_diffusion import run_diffusion_inference
from src.utils import load_config

def parse_args():
    parser = argparse.ArgumentParser(description="Synesthete Latent Diffusion Pipeline")
    parser.add_argument("--force-vae", action="store_true", help="Force retraining of the VAE")
    parser.add_argument("--force-diffusion", action="store_true", help="Force retraining of the Diffusion model")
    parser.add_argument("--force-all", action="store_true", help="Force retraining of everything")
    parser.add_argument("--skip-inference", action="store_true", help="Skip the inference step")
    return parser.parse_args()

def main():
    args = parse_args()
    
    print("=== Starting Synesthete Latent Diffusion Pipeline ===", flush=True)
    
    # Load Config
    cfg = load_config()
    print(f"Loaded config: {cfg['project']['name']} v{cfg['project']['version']}", flush=True)
    
    # 1. Train VAE
    # This is the "Eye" of the model. It learns to see and reconstruct shapes.
    # It needs to be trained first so the Diffusion model has something to predict.
    vae_path = Path("vae_checkpoints/vae_latest.pth")
    if vae_path.exists() and not (args.force_vae or args.force_all):
        print(f"\n[Step 1] Found existing VAE checkpoint at {vae_path}. Skipping training.", flush=True)
    else:
        print("\n[Step 1] Training VAE (The Eye)...", flush=True)
        train_vae()
    
    # 2. Train Diffusion
    # This is the "Brain" of the model. It learns to imagine latents from audio.
    diff_path = Path("diffusion_checkpoints/diff_latest.pth")
    if diff_path.exists() and not (args.force_diffusion or args.force_all):
        print(f"\n[Step 2] Found existing Diffusion checkpoint at {diff_path}. Skipping training.", flush=True)
    else:
        print("\n[Step 2] Training Diffusion Transformer (The Brain)...", flush=True)
        train_diffusion() # Config controls epochs now
    
    # 3. Inference
    # This generates the final video using the trained Brain and Eye.
    if not args.skip_inference:
        print("\n[Step 3] Running Inference...", flush=True)
        run_diffusion_inference(
            model_path="diffusion_checkpoints/diff_latest.pth",
            output_path="output_test.mp4",
            num_frames=90
        )
        print("\n=== Pipeline Complete ===", flush=True)
        print(f"Check output_test.mp4 to see the result!", flush=True)
    else:
        print("\n=== Pipeline Complete (Inference Skipped) ===", flush=True)

if __name__ == "__main__":
    main()
