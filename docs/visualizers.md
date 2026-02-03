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

### 4. TilesVisualizer
*   **Concept**: A tiled/grid “mosaic” where cell intensity/color is driven by coarse frequency energy.
*   **Randomization**:
    *   **Tile resolution**: e.g. 8×8 up to 16×16.
    *   **Mode**: bar-fill / checker / heatmap.
    *   **Palette**: randomized saturated base/alt colors.

### 5. ParticlesVisualizer
*   **Concept**: A particle field rendered as soft blobs; motion/brightness responds to amplitude.
*   **Randomization**:
    *   **Particle count**: dozens to hundreds.
    *   **Radius / drift**: controls density and motion feel.
    *   **Color drift**: shifts between two saturated colors based on audio.

### 6. ContoursVisualizer
*   **Concept**: Contour/isolines on a synthetic scalar field, modulated by audio amplitude and spectral brightness.
*   **Randomization**:
    *   **Field frequencies/phases**: changes the topology of lines.
    *   **Contour count/thickness**: controls visual density.
    *   **Palette**: two-color blend driven by brightness.

### 7. TrailsVisualizer
*   **Concept**: Feedback/trails effect with audio-driven persistence and subtle warping.
*   **Randomization**:
    *   **Feedback amount**: responds to amplitude.
    *   **Warp**: small rotation/zoom drift.
    *   **Colors**: base/accent palette.

### 8. KaleidoscopeVisualizer
*   **Concept**: Symmetry-based kaleidoscope patterns driven by audio-modulated polar fields.
*   **Randomization**:
    *   **Sector count**: e.g. 4–12.
    *   **Twist**: rotation strength over time.
    *   **Color channels**: phase-shifted mapping for variation.

### 9. AugmentedVisualizer (Postprocess Combinatorics)
*   **Concept**: Wraps any base visualizer and applies a randomized stack of lightweight post-process operations.
*   **Why it exists**: Turns \(N\) primitives into \(N \times K\) visual families cheaply (combinatorial diversity) without writing dozens of bespoke renderers.
*   **Operations (examples)**:
    *   mirror flips
    *   mild warp (rotate/zoom)
    *   blur / bloom (soft glow)
    *   vignette
    *   gentle color grading (contrast/saturation/gamma)
    *   (rarely) posterization
