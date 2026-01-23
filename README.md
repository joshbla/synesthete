# Synesthete

**Synesthete** is a research and development project aimed at building the foundations for a high-level audio-to-video AI model.

## Vision

The ultimate goal is to create a model capable of generating video content that is "representative" of input audio in a semantic and aesthetic sense, going beyond simple frequency-based visualizers. This involves training on pairs of audio and visual data to learn mappings between soundscapes and visual imagery.

## Documentation

*   [**Data Engine**](docs/data_engine.md): Details on the Infinite Data Engine and `IterableDataset`.
*   [**Audio Inputs**](docs/audio_inputs.md): Details on the procedural music generation engine.
*   [**Visualizers**](docs/visualizers.md): Descriptions of the synthetic visualizers used for training data.
*   [**Infrastructure**](docs/infrastructure.md): Details on experiment tracking, video backend, and tooling.
*   [**Standards**](docs/standards.md): Guidelines for documentation and project maintenance.

## Current Status: Foundation

This repository currently serves as the foundational setup for:
- **Environment**: Managed by `uv`, optimized for Apple Silicon (MPS acceleration).
- **Core Libraries**: PyTorch, Torchaudio, TorchCodec.
- **Data Engine**: An "Infinite" `IterableDataset` that generates procedural audio-video pairs on-the-fly during training, eliminating storage bottlenecks and overfitting.

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

We are following a "crawl, walk, run" strategy. Having established the foundational pipeline, the next steps for research are:

1.  **Model Modularity**: Refactor `AudioToVideoNet` into swappable components (Audio Encoder, Latent Bridge, Video Decoder) to allow easy experimentation with different architectures (e.g., Transformers vs CNNs).
2.  **Temporal Awareness**: Introduce Recurrent Neural Networks (RNNs) or Transformers to give the model "memory," allowing it to understand rhythm and musical context over time rather than just instantaneous amplitude.
3.  **Pre-trained Embeddings**: Transition from raw audio waveforms to using embeddings from industry-standard models (like CLAP or Jukebox) to leverage semantic understanding of sound.
4.  **Evaluation Metrics**: Implement automated metrics (e.g., Audio-Visual Sync scores) to objectively measure model performance beyond visual inspection.

## Project Structure

- `config/`: Configuration files (YAML).
- `src/`: Source code modules.
    - `audio_gen.py`: Procedural audio generation.
    - `visualizers.py`: Suite of algorithmic video generators.
    - `data.py`: Dataset creation and management.
    - `model.py`: PyTorch neural network architecture.
    - `train.py`: Training loop.
    - `inference.py`: Video generation from audio.
    - `tracker.py`: W&B experiment tracking wrapper.
- `run_pipeline.py`: Main orchestrator script.
- `favorites/`: Curated output samples.
- `scripts/`: Utility and inspection scripts.
