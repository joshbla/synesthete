import yaml
from pathlib import Path
import torch

def load_config(config_path="config/default.yaml"):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def get_device():
    """
    Returns the best available device.
    Prioritizes MPS (Apple Silicon) > CUDA > CPU.
    """
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")
