## Audio responsiveness postmortem (Synesthete)

This document explains **what we tried**, **why it mostly didn’t work**, **what finally worked**, and the **general principles** behind getting clear, human-visible audio responsiveness in an audio→video latent diffusion system.

It is written to stand alone outside of chat context.

---

### Problem statement

We want generated videos whose motion/structure is **clearly driven by audio**, in a way a human can notice:

- silence → visuals settle / become minimal
- louder segments → stronger motion/brightness/size changes
- changes in spectral energy (bands) → correspondingly different geometry/colors
- rhythmic structure → periodic visual events

The system uses:

- **procedural audio** to create training waveforms
- **procedural visualizers** to render frames from that waveform
- a **VAE** to compress frames into latents
- a **diffusion model** to predict latents conditioned on audio features (and optional style/temporal signals)

---

### Mental model: why “audio responsiveness” fails so often

In conditional generation, the model will use the conditioning signal only if:

1. The conditioning is **identifiable** (it is clear which part of audio explains which frame), and
2. Using conditioning is **necessary to minimize loss** (ignoring conditioning is expensive).

You can satisfy (1) and still fail (2). In that case the model often learns a strong unconditional prior (“looks like a plausible visualization”) and only weakly uses audio.

This is the *central failure mode we ran into*.

---

## What we tried (and why it wasn’t enough)

### Phase 1 — Identifiable conditioning (time-aligned audio features)

**Change**: we replaced raw waveform conditioning with a per-frame, frame-aligned feature timeline (`compute_audio_timeline`), and fed each frame a local context window.

**Why it helped conceptually**: it makes the task well-posed. The model can now, in principle, learn synchronization.

**Why it didn’t yield clear responsiveness by itself**: even with identifiable conditioning, a diffusion model can still minimize denoising loss by learning an unconditional prior if the conditional signal is not *required* by the objective.

### Phase 2 — Factorize “style” from “audio”

**Change**: introduce a clip-level continuous style latent with dropout.

**Why it helps**: it gives a place to put aesthetic randomness, which reduces collapse and allows variety.

**Why it didn’t fix responsiveness**: style factorization prevents some failure modes, but it still doesn’t force the model to be worse when audio is wrong.

### Phase 3 — More diverse procedural visualizers

**Change**: added more primitives + augmentation/compositing for diversity.

**Why it helps**: novelty and emergent variety come from data diversity and recombination.

**Why it made audio responsiveness harder to see**: diversity increases multimodality. If the objective doesn’t force audio dependence, the model is even more tempted to learn a broad unconditional prior and ignore audio.

### Phase 4 — Minimal temporal coherence

**Change**: contiguous frame snippet sampling, frame index embedding, previous-latent conditioning.

**Why it helps**: reduces flicker, makes outputs more “video-like.”

**Why it didn’t fix responsiveness**: temporal coupling mostly smooths artifacts; it doesn’t guarantee audio is used.

### Phase 5 (early) — Evaluation tools

**Change**: audio shuffle eval and style diversity eval.

**Why it helps**: you can detect regressions and measure whether audio affects outputs at all.

**Why it didn’t itself change behavior**: evaluation doesn’t change training incentives.

---

## What finally worked (the turning point)

### 1) “Silence + track dropout” audio for human-auditable supervision

We introduced a synthetic audio variant with:

- segments where **some tracks drop out**
- a segment of **full silence**

This makes “what the visual should do” extremely obvious to a human: settle during silence, react when a track comes back.

Config knob used in training/inference: `data.audio_silence_mix_prob`.

### 2) Direct shuffle-pressure in diffusion training (the key principle)

We added a **direct training penalty** that makes the model explicitly worse when audio conditioning is shuffled:

- compute the normal diffusion loss \(L_{good}\) with correct audio
- compute the diffusion loss \(L_{bad}\) with shuffled audio
- add a hinge penalty encouraging \(L_{bad} \\ge L_{good} + m\)

Conceptually:

> If the model ignores audio, then shuffled and unshuffled losses will be similar.  
> The penalty makes that solution expensive.

Knobs:

- `diffusion.audio_shuffle_loss_weight`
- `diffusion.audio_shuffle_loss_margin`
- `diffusion.audio_shuffle_loss_prob`

This was the first change that directly attacked the “audio-ignoring shortcut.”

### 3) Preserve absolute loudness cues for silence tests

For “settle during silence,” per-clip normalization can hide absolute loudness information.

We added:

- `diffusion.audio_feature_normalize_per_clip` (can be disabled)
- optional richer channels like absolute RMS / onset proxies

With silence tests, disabling per-clip z-scoring makes the silence cue easier to learn.

### 4) Confirming whether the VAE was the bottleneck (it wasn’t primary)

We ran a diagnostic:

- render a strongly audio-reactive “ground-truth” clip via debug visualizers
- encode→decode through the VAE
- compare “normal audio features” vs “shuffled audio features” at the output

Result: the VAE reduces the strength of the signal but **does not eliminate it**, so the primary failure was **diffusion not learning/using audio**, not the VAE destroying information.

---

## General principles (portable lessons)

### Principle A — Conditioning must be necessary, not just present

If the objective doesn’t punish the model for being audio-invariant, it will often learn to be audio-invariant.

The most reliable fixes are those that make **wrong conditioning measurably worse**:

- loss-gap / hinge penalties between correct vs shuffled conditioning
- contrastive objectives
- discriminator/matcher objectives (with careful tuning)

### Principle B — Human-auditable “probes” are invaluable

Silence and dropouts are a high-signal probe:

- they create unambiguous segments where visuals should settle
- humans can reliably detect whether the model responds

Use these both in training curriculum and eval.

### Principle C — Diversity makes priors stronger (so you need stronger pressure)

When you increase visual diversity without increasing “must-use-audio” pressure, the model can lean on priors more.

Recommended approach:

- bootstrap on a small, deterministic set of strongly reactive targets
- then reintroduce diversity gradually while keeping audio pressure on

### Principle D — Separate diagnosis from generalization

You need two modes:

- **debug/bootstrap mode**: prove the system can learn audio dependence at all
- **full mode**: scale up diversity/generalization

If you skip the bootstrap, you can waste a lot of time chasing architecture changes that don’t address incentives.

---

## How to improve further (next steps)

### 1) Strengthen the direct shuffle pressure

- increase `audio_shuffle_loss_weight` until artifacts appear, then back off
- increase `audio_shuffle_loss_margin` to demand a larger separation
- use harder negatives than random permutation (e.g., maximize mismatch in feature space)

### 2) Apply the pressure at more meaningful levels

Instead of (or in addition to) noise-pred MSE:

- compute and compare reconstruction in **x0-pred** space
- add a lightweight perceptual proxy on decoded frames (cheap statistics) for correct vs shuffled audio

### 3) Curriculum back to diversity

Use:

- `data.audio_debug_mode` for bootstrapping
- `data.audio_debug_mix_prob` to gradually mix real diversity back in
- keep shuffle pressure active so the model can’t regress to the audio-ignoring shortcut

### 4) Audio CFG guidance schedule

Once trained with `p_audio_drop > 0`, audio classifier-free guidance can make effects more obvious:

- tune `inference.audio_guidance_scale` upward until artifacts appear, then back off

### 5) Add explicit “silence” channels

For the specific “settle in silence” behavior:

- add a binary or smoothed “is_silent” feature channel derived from RMS
- optionally train a small auxiliary loss on a “motion energy” proxy during silence segments

---

## Summary

The main reason early attempts didn’t yield human-visible responsiveness was not that audio conditioning was missing; it was that **the training objective allowed a good solution that ignored audio**.

The turning point was adding **explicit training pressure** that makes shuffled/wrong audio strictly worse, and using silence/dropouts as a human-auditable probe.

