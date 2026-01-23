# Visualizers

To train a robust audio-to-video model, we generate synthetic ground-truth videos using a suite of programmatic visualizers. Each visualizer algorithmically maps audio features to visual patterns.

## Visualizer Suite (`src/visualizers.py`)

The system currently employs three distinct visualizers. During data generation, one is randomly selected for each audio sample.

### 1. PulseVisualizer
*   **Concept**: A classic "thumping speaker" effect.
*   **Logic**:
    *   Draws a circle in the center of the frame.
    *   **Radius**: Controlled by the instantaneous amplitude of the audio. Louder = Larger.
    *   **Color**: Randomly assigned base color (RGB) for each clip.
    *   **Brightness**: Scales with amplitude.

### 2. SpectrumVisualizer
*   **Concept**: A frequency analyzer / graphic equalizer.
*   **Logic**:
    *   Performs a Fast Fourier Transform (FFT) on audio chunks.
    *   Divides the frequency spectrum into 16 vertical bars.
    *   **Height**: Represents the magnitude of that frequency band.
    *   **Color**: A gradient from red to green based on the frequency bin index.

### 3. NoiseVisualizer
*   **Concept**: Abstract, textural reaction.
*   **Logic**:
    *   Starts with a static noise field (random static).
    *   **Motion**: The noise texture "rolls" (shifts) vertically based on audio amplitude. Louder sounds cause larger shifts, creating a glitchy, vibrating effect.
    *   **Color**: The RGB channels are separated and shifted by different amounts, creating chromatic aberration effects during loud moments.
