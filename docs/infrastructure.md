# Infrastructure & Tooling

## Video Backend (TorchCodec + FFmpeg)

We use a modern, high-performance stack for video I/O to avoid legacy deprecation warnings in PyTorch.

*   **TorchCodec**: The Python library for decoding/encoding video tensors.
*   **FFmpeg**: The system-level library that powers TorchCodec.

### Requirements
You must have FFmpeg installed on your system for the pipeline to work:
```bash
brew install ffmpeg
```

## Experiment Tracking (Weights & Biases)

We use [Weights & Biases (W&B)](https://wandb.ai) to track experiments, visualize loss curves, and log generated video samples during training.

### Free Tier Strategy
To ensure we stay within the **Free Tier** limits and avoid accidental charges or feature bloat:

1.  **Storage Limit**: The free tier includes **5GB** of storage.
    *   *Strategy*: We only log metrics (tiny). Video logging is **disabled by default** in `config/default.yaml` (`log_videos: false`). We do **not** log model checkpoints (large) to W&B; those are saved locally in `models/`.
2.  **Feature Isolation**:
    *   We do **not** use W&B Artifacts (heavy storage).
    *   We do **not** use W&B Weave (LLM app tracing).
    *   We do **not** use W&B Launch/Inference (compute costs).
3.  **Code Isolation**:
    *   All W&B logic is wrapped in `src/tracker.py`.
    *   If you want to disable it entirely, set `tracker.enabled: false` in `config/default.yaml`. The code will gracefully skip all logging.

### Setup
1.  **Login**: Run `uv run wandb login` and paste your API key.
2.  **Run**: Just run the pipeline as normal. The tracker will automatically initialize.
