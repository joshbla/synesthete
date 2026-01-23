# Audio Inputs

The project currently uses a synthetic audio generation engine to create diverse musical training data. This ensures the model learns to map various acoustic features (pitch, rhythm, timbre) to visual elements.

## Generation Engine (`src/audio_gen.py`)

The `AudioGenerator` class is responsible for creating procedural audio clips.

### Musical Structure
Each generated sample is a 3-second musical sequence (at 16kHz) containing two concurrent parts:

1.  **Bass Line**:
    *   **Waveform**: Sine wave.
    *   **Rhythm**: Long, sustained notes (half notes).
    *   **Role**: Provides the harmonic foundation and low-frequency energy.

2.  **Melody**:
    *   **Waveform**: Sawtooth wave (richer harmonics).
    *   **Rhythm**: Faster, varied notes (quarter, eighth, or whole notes).
    *   **Role**: Provides high-frequency content and rhythmic complexity.

### Composition Logic
*   **Scale**: Notes are chosen from a Major scale relative to a random root frequency (e.g., A3, C4, E4).
*   **Envelope**: An ADSR (Attack, Decay, Sustain, Release) envelope is applied to every note to prevent clicking and ensure a natural, musical amplitude contour.
*   **Mixing**: The bass and melody are mixed together, with the bass slightly louder (0.6) than the melody (0.3).
