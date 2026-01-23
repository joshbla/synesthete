# Data Engine

Synesthete uses a procedural, on-the-fly data generation engine to train models. This approach replaces traditional static datasets (files on disk) with an infinite stream of unique samples generated in memory during training.

## The Infinite Dataset (`src/data.py`)

The core of this system is the `InfiniteAVDataset` class, which inherits from PyTorch's `IterableDataset`.

### How It Works
1.  **On-Demand Generation**: When the training loop requests a batch, the dataset calls the `AudioGenerator` and a random `Visualizer` immediately.
2.  **No Storage**: Data is generated in RAM, fed to the GPU, and then discarded. No `.wav` or `.mp4` files are written to disk during training.
3.  **Infinite Variety**: Since generation is procedural and randomized, the model effectively never sees the exact same sample twice. This prevents overfitting and forces the model to learn generalizable rules.

### Configuration
The behavior of the engine is controlled via `config/default.yaml`:

```yaml
data:
  num_samples: 1000  # Defines the "length" of one epoch (how many samples to generate before calling the epoch done)
  duration: 3.0      # Length of each clip in seconds
  fps: 30            # Video frame rate
  height: 128        # Video height
  width: 128         # Video width
```

## Static Data Generation
While training uses the infinite engine, we still retain the ability to generate static files for debugging or validation. The `create_synthetic_dataset` function can be called manually to save a batch of `.wav` and `.mp4` files to `data/train` for inspection.
