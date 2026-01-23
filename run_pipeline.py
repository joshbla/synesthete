import sys
from pathlib import Path

# Ensure we can import from src
sys.path.append(str(Path(__file__).parent))

from src.data import create_synthetic_dataset
from src.train import train_model
from src.inference import run_inference
from src.utils import load_config

def main():
    print("=== Starting Synesthete Pipeline ===")
    
    # Load Config
    cfg = load_config()
    print(f"Loaded config: {cfg['project']['name']} v{cfg['project']['version']}")
    
    # 1. Data Generation
    # With Infinite Data Engine, we don't need to pre-generate training data.
    # But we can generate a small validation set if needed, or just skip.
    print("\n[Step 1] Data Generation (Skipped for Infinite Data Engine)")
    # create_synthetic_dataset(...) 
    
    # 2. Training
    print("\n[Step 2] Training Model...")
    train_model(
        data_dir=cfg['data']['train_dir'],
        output_dir=cfg['train']['output_dir'],
        epochs=cfg['train']['epochs'],
        batch_size=cfg['train']['batch_size'],
        learning_rate=cfg['train']['learning_rate'],
        frame_size=tuple(cfg['model']['frame_size']),
        num_frames=cfg['model']['num_frames'],
        config=cfg  # Pass full config for logging
    )
    
    # 3. Inference
    print("\n[Step 3] Running Inference...")
    model_path = f"{cfg['train']['output_dir']}/{cfg['train']['save_name']}"
    run_inference(
        model_path=model_path,
        output_path=cfg['inference']['output_video'],
        num_frames=cfg['model']['num_frames'],
        frame_size=tuple(cfg['model']['frame_size'])
    )
    
    print("\n=== Pipeline Complete ===")
    print(f"Check {cfg['inference']['output_video']} to see the result!")

if __name__ == "__main__":
    main()
