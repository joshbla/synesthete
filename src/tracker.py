import wandb
import os
from pathlib import Path

class ExperimentTracker:
    """
    Wrapper around Weights & Biases to keep it isolated.
    If wandb fails to initialize or is disabled, this class handles it gracefully.
    """
    def __init__(self, config):
        self.enabled = config.get('tracker', {}).get('enabled', False)
        self.project_name = config.get('project', {}).get('name', 'synesthete')
        self.run = None
        
        if self.enabled:
            try:
                # Initialize wandb
                # We pass the whole config dict so hyperparameters are logged
                self.run = wandb.init(
                    project=self.project_name,
                    config=config,
                    # Set anonymous mode if needed, though user has account now
                    # anonymous="allow" 
                )
                print(f"[Tracker] W&B initialized: {self.run.name}")
            except Exception as e:
                print(f"[Tracker] Failed to initialize W&B: {e}")
                self.enabled = False

    def log_metrics(self, metrics, step=None):
        """
        Log a dictionary of metrics (e.g. {'loss': 0.5, 'epoch': 1})
        """
        if self.enabled and self.run:
            self.run.log(metrics, step=step)

    def log_video(self, video_tensor, caption="Generated Video", fps=30):
        """
        Log a video tensor to W&B.
        Expected tensor: (T, C, H, W) or (T, H, W, C)
        """
        if self.enabled and self.run:
            # wandb expects (T, C, H, W) for video
            if video_tensor.ndim == 4:
                # Check if channels are last (T, H, W, C) -> permute to (T, C, H, W)
                if video_tensor.shape[-1] == 3:
                    video_tensor = video_tensor.permute(0, 3, 1, 2)
                
                # Ensure it's 0-255 uint8 or 0-1 float
                # wandb handles numpy or tensor
                video_data = video_tensor.cpu().numpy()
                
                self.run.log({
                    caption: wandb.Video(video_data, fps=fps, format="mp4")
                })

    def finish(self):
        if self.enabled and self.run:
            self.run.finish()
