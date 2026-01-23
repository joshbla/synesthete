# Model Architecture

## Overview

The `AudioToVideoNet` is a hybrid CNN + Transformer architecture for audio-to-video synthesis. It generates 90 video frames (3 seconds @ 30fps) at 128×128 resolution from 48,000 audio samples (3 seconds @ 16kHz).

## Architecture Diagram

```
Audio Waveform (B, 1, 48000)
         │
         ▼
┌─────────────────────────────────────┐
│       CNN Audio Encoder             │
│  Conv1d(1→32→64→128→256)           │
│  GELU activation + BatchNorm        │
│  Stride-4 downsampling              │
└─────────────────────────────────────┘
         │
         ▼ (B, 256, T_compressed)
         │
    Linear Interpolation → (B, 256, 90)
         │
         ▼ permute to (B, 90, 256)
         │
    ┌────┴──────────────────────────────┐
    │                                   │ (Skip Connection)
    ▼                                   ▼
┌─────────────────────────────────────┐ │
│     Positional Encoding             │ │
│  Sinusoidal (max_len=256)           │ │
└─────────────────────────────────────┘ │
         │                              │
         ▼                              │
┌─────────────────────────────────────┐ │
│   Transformer Encoder               │ │
│  4 layers, 8 heads                  │ │
│  d_model=256, d_ff=1024             │ │
│  GELU activation, dropout=0.1       │ │
│  + LayerNorm                        │ │
└─────────────────────────────────────┘ │
         │                              │
         ▼ (B, 90, 256)                 │
         │                              │
    Add Residual ◄──────────────────────┘
         │
         ▼ (B, 90, 256)
         │
    Reshape to (B*90, 256)
         │
         ▼
┌─────────────────────────────────────┐
│   Spatial Seed Projection           │
│  Linear(256 → 256*8*8)              │
│  Reshape to (B*90, 256, 8, 8)       │
└─────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│     CNN Frame Decoder               │
│  ConvTranspose2d upsampling:        │
│  8×8 → 16×16 → 32×32 → 64×64 → 128×128  │
│  GELU + BatchNorm (except final)    │
│  Sigmoid output                     │
└─────────────────────────────────────┘
         │
         ▼
    Reshape to (B, 90, 3, 128, 128)
         │
         ▼
Video Frames (B, 90, 3, 128, 128)
```

## Component Details

### 1. CNN Audio Encoder

Extracts local audio features efficiently using 1D convolutions.

| Layer | In Channels | Out Channels | Kernel | Stride | Output Size (approx) |
|-------|-------------|--------------|--------|--------|---------------------|
| Conv1 | 1 | 32 | 80 | 4 | 12000 |
| Conv2 | 32 | 64 | 4 | 4 | 3000 |
| Conv3 | 64 | 128 | 4 | 4 | 750 |
| Conv4 | 128 | 256 | 4 | 4 | 187 |

**Why CNN for audio encoding?**
- **Translation equivariance**: A beat at t=1s should be encoded the same as at t=2s
- **Local feature extraction**: Beats, attacks, harmonics are local patterns
- **Computational efficiency**: O(n) vs O(n²) for self-attention on 48k samples

### 2. Temporal Transformer

Models long-range dependencies across the audio timeline. Includes a **residual skip connection** around the transformer block to preserve local rhythmic features (beats) while adding long-range context (flow).

| Parameter | Value |
|-----------|-------|
| d_model | 256 |
| nhead | 8 |
| num_layers | 4 |
| dim_feedforward | 1024 |
| activation | GELU |
| dropout | 0.1 |

**Why Transformer for temporal modeling?**
- **Long-range dependencies**: Connects musical phrases across the full 3-second window
- **Self-attention**: Each time step can attend to all other time steps
- **Positional awareness**: Sinusoidal encoding preserves temporal ordering

### 3. CNN Frame Decoder

Generates video frames from temporally-aware features.

| Layer | In Channels | Out Channels | Output Size |
|-------|-------------|--------------|-------------|
| Spatial Seed | 256 | 256 | 8×8 |
| ConvT1 | 256 | 128 | 16×16 |
| ConvT2 | 128 | 64 | 32×32 |
| ConvT3 | 64 | 32 | 64×64 |
| ConvT4 | 32 | 3 | 128×128 |

**Why CNN for frame decoding?**
- **Spatial coherence**: Convolutions maintain local spatial relationships
- **Parameter efficiency**: Shared weights across spatial dimensions
- **Proven effectiveness**: Standard in image generation (GANs, VAEs, diffusion)

## Design Rationale

This hybrid architecture follows the pattern established by modern audio-visual models:

| Model | Audio Encoder | Temporal | Visual Decoder |
|-------|---------------|----------|----------------|
| **Ours** | CNN | Transformer | CNN |
| MusicGen | EnCodec (CNN) | Transformer | - |
| Stable Audio | CNN | Transformer | Diffusion U-Net |
| AudioLDM 2 | CLAP | Transformer | Diffusion U-Net |

### Trade-offs

| Aspect | Pure CNN | Hybrid (Ours) | Pure Transformer |
|--------|----------|---------------|------------------|
| Local features | ✅ Excellent | ✅ Excellent | ⚠️ Needs many layers |
| Long-range deps | ⚠️ Limited | ✅ Good | ✅ Excellent |
| Compute efficiency | ✅ Fast | ✅ Fast | ⚠️ O(n²) attention |
| Parameter count | ✅ Small | ✅ Moderate | ⚠️ Large |

## Parameter Count

Approximate parameter breakdown:
- **Audio Encoder**: ~200K parameters
- **Transformer**: ~2.7M parameters (4 layers × 256 dim)
- **Frame Decoder**: ~1.5M parameters
- **Total**: ~4.4M parameters

## Legacy Model

The original `AudioToVideoNetLegacy` class is preserved for comparison. It uses:
- Same CNN audio encoder (with ReLU instead of GELU)
- MLP frame decoder (no transformer, no CNN decoder)

This can be useful for A/B testing the impact of the transformer on temporal coherence.

## Future Directions

1. **Pretrained Audio Encoders**: Replace CNN with Wav2Vec2, Whisper, or CLAP embeddings
2. **Cross-Attention**: Add audio-to-visual cross-attention for finer conditioning
3. **Latent Diffusion**: Replace CNN decoder with diffusion for higher quality
4. **Larger Transformers**: Scale up for more complex temporal relationships
