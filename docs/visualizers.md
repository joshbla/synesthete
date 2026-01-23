# Visualizers

To train a robust audio-to-video model, we generate synthetic ground-truth videos using a suite of programmatic visualizers. Each visualizer algorithmically maps audio features to visual patterns.

## Design Philosophy: Abstraction & Generalization

To prevent the model from overfitting to specific, localized visual cues (e.g., "bass always equals a red circle in the center"), the visualizers are designed to be highly stochastic and abstract. This forces the model to learn general representations of musicality—attacks, rhythm, harmonic content—rather than memorizing fixed mappings. This approach encourages "reward hacking" where the model must find the underlying signal amidst varied visual presentations.

## Visualizer Suite (`src/visualizers/`)

The system employs multiple visualizer classes. During data generation, one is randomly selected, and its internal parameters (color, shape, position, motion) are randomized for that specific sample.

### 1. PulseVisualizer
*   **Concept**: Geometric shapes that pulse with audio amplitude.
*   **Randomization**:
    *   **Shapes**: 1 to 3 shapes per clip.
    *   **Type**: Circle, Square, or Diamond.
    *   **Position**: Randomly placed on the screen (not just centered).
    *   **Color**: Random additive colors.
    *   **Size**: Base size varies; pulses with amplitude.

### 2. SpectrumVisualizer
*   **Concept**: Frequency analysis visualization.
*   **Randomization**:
    *   **Mode**:
        *   `vertical`: Classic bottom-up bars.
        *   `horizontal`: Left-to-right bars.
        *   `radial`: Pie-slice segments radiating from the center.
    *   **Resolution**: 8, 16, or 32 frequency bands.
    *   **Color**: Random start and end colors for the gradient.

### 3. WaveformVisualizer
*   **Concept**: Direct visualization of the audio signal time-domain data.
*   **Randomization**:
    *   **Mode**:
        *   `line`: Oscilloscope-style line across the screen.
        *   `circle`: A ring that deforms based on the waveform.
    *   **Style**: Random line thickness and color.
