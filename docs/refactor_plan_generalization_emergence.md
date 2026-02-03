## Refactor Plan: Generalization + Emergent Audio Visualizations

This plan is intentionally **fundamentals-first**. The goal is not “add features,” it’s to change the incentives so the easiest thing for the model to learn is:

- **Audio-grounded structure and motion** (it must pay attention to audio)
- **A wide, continuous space of styles** (it can explore and invent)
- **Temporal coherence** (it makes “a video,” not a bag of frames)

If we do that, “emergent” visualizations become a natural result of training on a sufficiently rich distribution rather than a fragile, lucky artifact.

---

### The fundamental problem you’re trying to solve

You want a model that can:

- learn **general rules** about how audio relates to visuals (beats → pulses, band energy → shapes/texture motion, etc.)
- not collapse into approximating a small number of pre-existing visualizer templates
- generate **new visualizations** that feel valid, coherent, and audio-reactive

In practice, models fail this goal in two predictable ways:

- **Audio-ignoring shortcut**: the model learns a strong “visualizer-looking” prior and only weakly uses audio because it can still minimize loss that way.
- **Mode/style collapse**: the model discovers that producing a small family of “safe” outputs is rewarded, especially when the conditional distribution is very multimodal (many valid visuals per audio).

Your refactor should make these shortcuts *expensive* and make real audio-conditioning *cheap*.

---

### Principles (the “physics” of this problem)

- **1) Identifiability (the model must know what part of audio explains what frame)**  
  If you train on (full audio clip) → (random frame), you’ve created an ambiguous task: there is no learnable mapping from “which part of audio” to “this frame.”  
  Fixing this is not an optimization detail—it’s the difference between a learnable problem and a degenerate one.

- **2) Factorization (separate “content from audio” vs “style/aesthetic”)**  
  Many valid visual styles can represent the same audio. If you don’t give the model a clean place to put “style randomness,” it will either:
  - average it out (regression-to-mean), or
  - collapse to a few modes it can reliably reproduce.
  
  The right shape is:  
  \[
  \text{video} \sim p(\text{video} \mid \text{audio}, \text{style})
  \]
  where **audio controls what happens** and **style controls how it looks**.

- **3) Emergence is mostly a data property**  
  A model cannot invent a truly new visual grammar from a tiny set of monolithic generators. What *does* work is training on **compositional structure** (primitives + recombination) so “new” outputs are new *combinations* and interpolations that are still in-distribution.

- **4) Video is not a set of frames**  
  Temporal coherence is part of the concept. If each frame is sampled independently, you’ll always get flicker/jitter and style instability.  
  Coherence is also an “emergence amplifier”: stable style + stable rules across time make outputs feel like a coherent new visualization system.

- **5) You can’t steer what you can’t measure**  
  “Looks cool” isn’t enough when you scale diversity. You need at least a few objective checks that detect:
  - “the model ignores audio”
  - “the model collapses to a few styles”
  - “the model flickers even when style is fixed”

---

### Refactor outcomes (what will be true if this works)

- **Audio sensitivity**: keeping the random seed/style fixed, changing audio changes the visual dynamics in a meaningful way.
- **Style diversity**: keeping audio fixed, sampling style produces many distinct aesthetics without breaking audio-reactivity.
- **Emergent hybrids**: interpolating style produces coherent “in-between” aesthetics; composing primitives yields novel but valid visual systems.
- **Temporal stability**: within a clip, the output looks like one coherent visualization system evolving over time.

---

## Phased plan (high-level, implementation-ready without being code-level)

### Phase 1 — Make conditioning identifiable (time alignment)

**Goal**: remove the audio-ignoring shortcut by making “what audio explains this frame” explicit.

**Why this matters**: if the training target frame is not paired with its corresponding audio slice/timestep, the model can’t learn synchronization even in principle.

**What changes**:
- Introduce a per-frame **audio timeline**: a sequence of features with length \(T = \text{num_frames}\).
  - Think: “one feature vector per video frame,” aligned to the same notion of time the visualizers use.
- Train diffusion on **aligned pairs**:
  - frame \(i\) (or a short subsequence of frames) is conditioned on audio-timeline \(i\) (or the aligned subsequence).
- Stop sampling targets as “random independent frames from a clip” without also providing frame index/time.

**Result**: the model is now solving a well-posed problem: “given what the audio is doing at time \(i\), generate what the visual system should be doing at time \(i\).”

---

### Phase 2 — Factorize variation: add a continuous “style channel” (anti-collapse)

**Goal**: give the model a clean place to put aesthetic randomness so it doesn’t average or collapse.

**Why this matters**: with “extraordinary visualizer variation,” \(p(\text{video} \mid \text{audio})\) is extremely multimodal. Without a style variable, the model is rewarded for finding a few stable modes.

**What changes**:
- Add a **clip-level continuous style latent** (sample once per clip, reused for all frames).
  - This is not “visualizer_id.” It’s a continuous vector you can sample and interpolate.
- During training, sometimes **drop style conditioning** (and sometimes drop audio conditioning) so you can control “faithfulness vs novelty” at inference (the same core idea as classifier-free guidance).

**Result**: the model can produce many valid visual systems for the same audio without collapsing into 2–3 templates.

---

### Phase 3 — Expand diversity the right way: compositional visualizer programs

**Goal**: get genuine novelty by making the training distribution combinatorial and recombinable.

**Why this matters**: emergence is limited by what the data generator can express. A few hand-authored visualizers will mostly yield mixtures of those.

**What changes**:
- Replace “a small registry of monolithic visualizers” with a **program generator**:
  - programs are built from **primitives** (bars, rings, particle fields, contours, trails, grids, distortions, palettes, blending ops, etc.)
  - each program has continuous parameters and composition structure
- Keep named “classic visualizers” as presets for debugging, but train primarily on random programs.

**Result**: “new visualizations” becomes “new programs / new parameter regions,” which are still in-distribution and therefore stable.

---

### Phase 4 — Make it a real video model (temporal coherence)

**Goal**: eliminate flicker and turn “style” into something consistent across time.

**Why this matters**: if frames are independent, outputs never feel like one coherent invented visualization system.

**What changes**:
- Train and sample on **short frame subsequences** (not just single frames).
- Add a minimal temporal mechanism:
  - clip-level style latent persists across frames
  - the model is aware of frame time/index
  - optionally: weak recurrence / conditioning on previous latent(s)

**Result**: stable aesthetics and motion rules across time; novelty reads as “a coherent new visualizer,” not noise.

---

### Phase 5 — Make “ignoring audio” detectable and punishable (evaluation + training pressure)

**Goal**: prevent silent failure when you scale diversity.

**Why this matters**: with strong generative priors, the system can look “cool” while ignoring audio; you need automated signals.

**What changes**:
- Add an “audio shuffle test” to eval:
  - same seed/style, swap audio → output should change meaningfully
- Add a lightweight mismatch penalty:
  - encourage matched (audio, video/latent) pairs to score higher than mismatched ones
- Track diversity:
  - for a fixed audio clip, sample many styles; ensure outputs occupy a broad space (not clustering into a few archetypes)

**Result**: you can scale visual variety without losing audio grounding or collapsing styles.

---

## Practical guidance on “emergence vs memorization”

If you do the above, you’ll get two knobs that matter:

- **Audio conditioning strength**: higher = more faithful/reactive, lower = more abstract/novel.
- **Style sampling temperature/diversity**: higher = more exploration, lower = more consistent/recognizable aesthetics.

To avoid the “memorize a few visualizers” trap:

- Prefer **continuous style** over discrete IDs as the primary driver of variation.
- Keep the generator **compositional** so novelty is recombination, not extrapolation from 3 templates.
- Evaluate routinely with **audio shuffle** and **style diversity** probes.

---

### Minimal “start here” if you want the shortest path to improvement

If you only do two things first:

- **Time-align conditioning** (Phase 1)
- **Add clip-level continuous style + dropout** (Phase 2)

…you will already see a big step in “generalizes + doesn’t collapse,” even before expanding the visualizer family.

---

## Progress / status updates

This section tracks what has been implemented so far, so the plan stays a “living” doc.

### Update (2026-02-03)

- **Phase 0 — Fix contracts**
  - **Completed**: Removed the hardcoded 16kHz assumption in visualizer chunking by threading `sample_rate` through the visualizer API and callers.
  - **Notes**: This makes timing/chunking correct across config changes and prevents subtle alignment bugs.

- **Phase 1 — Make conditioning identifiable (time alignment)**
  - **Completed**: Diffusion conditioning is now an explicit **frame-aligned audio feature timeline** (one feature vector per frame), with optional neighbor context.
  - **What changed**:
    - Training/inference compute per-frame audio features aligned to the same frame chunking concept the visualizers use.
    - Diffusion is conditioned on a fixed-shape tensor `(T_ctx, F)` per frame, so batching is stable as you scale.
    - Audio conditioning includes positional encoding so ordering within the context window matters.
  - **Config added**:
    - `diffusion.audio_feature_n_fft`, `diffusion.audio_feature_num_bands`, `diffusion.audio_feature_context`

- **Phase 2 — Factorize variation (style)**
  - **Completed (minimal)**: Added a clip-level continuous style latent `style_z` and training-time style dropout.
  - **What changed**:
    - A single `style_z` is sampled per clip and reused across that clip’s frames.
    - Diffusion cross-attends to an explicit **style token** alongside audio features (so “style” has a dedicated channel).
    - Training uses **style dropout** (`p_style_drop`) to prevent over-reliance on style and enable “unspecified style” behavior.
    - Inference samples a `style_z` so the same audio can yield multiple aesthetics by resampling style.
    - Inference supports `style_mode` (`random` vs `zero`) and `style_seed` to make audio-reactivity checks easier and repeatable.
  - **Config added**:
    - `diffusion.style_dim`, `diffusion.p_style_drop`

- **Phase 3 — Compositional visualizer programs**
  - **Not started**: Visualizer variety is still driven by a small set of monolithic renderers.

- **Phase 4 — Temporal coherence**
  - **Not started**: Frames are still sampled i.i.d. (even though conditioning is now time-aligned per frame).

- **Phase 5 — Detect/punish “ignoring audio”**
  - **Not started**: No automated “audio shuffle test,” mismatch pressure, or diversity probes yet.

### Practical implication for existing checkpoints

These Phase 1 changes alter the **conditioning distribution** for diffusion (raw full-clip conditioning → frame-aligned feature conditioning). Existing diffusion weights trained under the old scheme are expected to degrade under the new conditioning; retraining diffusion is the intended next step.

