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

## Current Status: Hybrid CNN + Transformer Architecture

This repository implements a modern hybrid architecture combining the strengths of CNNs and Transformers:

- **Environment**: Managed by `uv`, optimized for Apple Silicon (MPS acceleration).
- **Core Libraries**: PyTorch, Torchaudio, TorchCodec.
- **Data Engine**: An "Infinite" `IterableDataset` that generates procedural audio-video pairs on-the-fly during training, eliminating storage bottlenecks and overfitting.

### Model Architecture

The `AudioToVideoNet` uses a three-stage hybrid architecture:

```
Audio Waveform (48k samples, 3s @ 16kHz)
         ↓
┌─────────────────────────────────────┐
│  CNN Audio Encoder                  │  ← Efficient local feature extraction
│  (4 conv layers with GELU + BN)     │    Captures beats, attacks, harmonics
└─────────────────────────────────────┘
         ↓
┌─────────────────────────────────────┐
│  Transformer Temporal Module        │  ← Long-range temporal modeling
│  (4 layers, 8 heads, positional enc)│    Connects musical phrases across time
└─────────────────────────────────────┘
         ↓
┌─────────────────────────────────────┐
│  CNN Frame Decoder                  │  ← Spatial generation with coherence
│  (Transposed convolutions 8→128px)  │    Upsamples to full resolution
└─────────────────────────────────────┘
         ↓
Video Frames (90 frames, 128×128 RGB)
```

**Why Hybrid?** This architecture follows modern best practices from projects like Stable Audio, MusicGen, and AudioLDM:
- CNNs excel at local pattern extraction (translation equivariant, computationally efficient)
- Transformers excel at long-range dependencies (connecting a beat at t=0.5s to visuals at t=2.5s)
- The combination provides both efficiency and expressivity

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
- [ ] **Latent Diffusion**: Explore diffusion-based decoders for higher quality frame generation.

## Project Structure

- `config/`: Configuration files (YAML).
- `src/`: Source code modules.
    - `audio_gen.py`: Procedural audio generation.
    - `visualizers/`: Package containing algorithmic video generators.
    - `data.py`: Dataset creation and management.
    - `model.py`: PyTorch neural network architecture (Hybrid CNN + Transformer).
    - `train.py`: Training loop.
    - `inference.py`: Video generation from audio.
    - `tracker.py`: W&B experiment tracking wrapper.
- `run_pipeline.py`: Main orchestrator script.
- `favorites/`: Curated output samples.
- `scripts/`: Utility and inspection scripts.
