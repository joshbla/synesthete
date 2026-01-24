# Synesthete

**Synesthete** is a research and development project aimed at building the foundations for a high-level audio-to-video AI model.

## Vision

The ultimate goal is to create a model capable of generating video content that is "representative" of input audio in a semantic and aesthetic sense, going beyond simple frequency-based visualizers. This involves training on pairs of audio and visual data to learn mappings between soundscapes and visual imagery.

## Documentation

*   [**Model Architecture**](docs/model_architecture.md): Details on the hybrid CNN + Transformer architecture.
*   [**Data Engine**](docs/data_engine.md): Details on the Infinite Data Engine and `IterableDataset`.
*   [**Audio Inputs**](docs/audio_inputs.md): Details on the procedural music generation engine.
*   [**Visualizers**](docs/visualizers.md): Descriptions of the synthetic visualizers used for training data.
*   [**Infrastructure**](docs/infrastructure.md): Details on experiment tracking, video backend, and tooling.
*   [**Standards**](docs/standards.md): Guidelines for documentation and project maintenance.

## Current Status: Latent Diffusion Architecture

This repository implements a **Latent Diffusion** architecture to solve the "regression to the mean" problem (gray sludge) inherent in direct regression models.

**Success**: The model now successfully generates distinct, sharp, and novel visuals, breaking free from the blurry averaging of previous iterations. It generalizes well across the latent space, producing coherent shapes and colors.

**Limitations**:
- **Simple Audio**: The current training data uses basic procedural audio.
- **Rhythm**: The model does not yet exhibit strong beat synchronization or rhythmic reactivity.

- **Environment**: Managed by `uv`, optimized for Apple Silicon (MPS acceleration).
- **Core Libraries**: PyTorch, Torchaudio, TorchCodec.
- **Data Engine**: An "Infinite" `IterableDataset` that generates procedural audio-video pairs on-the-fly.

### Model Architecture

The system consists of two trained models:

1.  **Spatial VAE ("The Eye")**:
    -   Compresses 128x128 video frames into **8x8x256 spatial latents**.
    -   Ensures generated images are sharp and valid.

2.  **Diffusion Transformer ("The Brain")**:
    -   Takes random noise and "denoises" it into valid latents, conditioned on audio.
    -   Uses a Transformer backbone with Cross-Attention to audio embeddings.

```
Audio Waveform (48k samples)
         ↓
┌─────────────────────────────────────┐
│  Audio Encoder (CNN)                │
└─────────────────────────────────────┘
         ↓
┌─────────────────────────────────────┐      ┌──────────────┐
│  Diffusion Transformer              │  ←   │ Random Noise │
│  (Denoises Latents)                 │      └──────────────┘
└─────────────────────────────────────┘
         ↓
    Latent Code (8x8x256)
         ↓
┌─────────────────────────────────────┐
│  VAE Decoder (CNN)                  │
└─────────────────────────────────────┘
         ↓
Video Frames (128x128 RGB)
```

**Why Latent Diffusion?**
- **Creativity**: Instead of averaging all possible outputs (sludge), diffusion probabilistically picks *one* valid, sharp output.
- **Efficiency**: The transformer operates on small 8x8 latents rather than full 128x128 pixels.

## Getting Started

1. **Install Prerequisites**:
   ```bash
   brew install ffmpeg
   ```

2. **Sync the environment**:
   ```bash
   uv sync
   ```

3. **Run the Pipeline**:
   The entire process (Data Gen -> Training -> Inference) is orchestrated by a single script:
   ```bash
   uv run python run_pipeline.py
   ```
   This will:
   - Train the model for 20 epochs on your MPS chip using the Infinite Data Engine.
   - Generate `output_test.mp4` demonstrating the result.

## Roadmap

We are following a "crawl, walk, run" strategy. Current progress and next steps:

- [x] **Model Modularity**: `AudioToVideoNet` now has distinct components (Audio Encoder, Temporal Transformer, Frame Decoder).
- [x] **Temporal Awareness**: Transformer encoder with positional encoding provides long-range temporal modeling.
- [ ] **Pre-trained Embeddings**: Transition from raw audio waveforms to using embeddings from industry-standard models (like Wav2Vec2, CLAP, or Whisper) to leverage semantic understanding of sound.
- [ ] **Evaluation Metrics**: Implement automated metrics (e.g., Audio-Visual Sync scores) to objectively measure model performance beyond visual inspection.
- [x] **Latent Diffusion**: Explore diffusion-based decoders for higher quality frame generation.

## Project Structure

- `config/`: Configuration files (YAML).
- `src/`: Source code modules.
    - `audio_gen.py`: Procedural audio generation.
    - `visualizers/`: Package containing algorithmic video generators.
    - `data.py`: Dataset creation and management.
    - `model.py`: PyTorch neural network architecture (Hybrid CNN + Transformer).
    - `diffusion.py`: Diffusion Transformer and Noise Scheduler.
    - `vae.py`: Variational Autoencoder (Spatial).
    - `train.py`: Training loop for main model.
    - `train_vae.py`: Training loop for VAE.
    - `train_diffusion.py`: Training loop for Diffusion.
    - `inference.py`: Video generation from audio.
    - `inference_diffusion.py`: Inference for Diffusion model.
    - `tracker.py`: W&B experiment tracking wrapper.
- `run_pipeline.py`: Main orchestrator script.
- `favorites/`: Curated output samples.
- `scripts/`: Utility and inspection scripts.
