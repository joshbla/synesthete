import torch
import numpy as np
import random

class AudioGenerator:
    def __init__(self, sample_rate=16000):
        self.sr = sample_rate

    def normalize(self, waveform: torch.Tensor, target_rms: float = 0.16, peak_limit: float = 0.9) -> torch.Tensor:
        """
        Normalize waveform to a comfortable loudness.

        - target_rms is in linear scale (0..1), not dBFS. 0.16 is moderate and more audible.
        - peak_limit prevents accidental clipping.
        """
        if waveform.numel() == 0:
            return waveform
        x = waveform - waveform.mean()
        rms = torch.sqrt(torch.mean(x * x) + 1e-8)
        if rms > 0:
            x = x * (target_rms / rms)
        peak = x.abs().max()
        if peak > peak_limit:
            x = x * (peak_limit / peak)
        return torch.clamp(x, -1.0, 1.0)

    def smooth(self, waveform: torch.Tensor, kernel_size: int = 9) -> torch.Tensor:
        """
        Simple low-pass smoothing using a moving average FIR.
        Helps reduce grating high-frequency content in saw/square tones.
        """
        k = int(kernel_size)
        if k <= 1:
            return waveform
        x = waveform.view(1, 1, -1)
        kernel = torch.ones(1, 1, k, dtype=x.dtype, device=x.device) / k
        x = torch.nn.functional.pad(x, (k // 2, k // 2), mode="reflect")
        y = torch.nn.functional.conv1d(x, kernel)
        return y.view(-1)
        
    def get_envelope(self, t, attack=0.05, release=0.1):
        envelope = torch.ones_like(t)
        attack_len = int(attack * self.sr)
        release_len = int(release * self.sr)
        
        if attack_len > 0:
            envelope[:attack_len] = torch.linspace(0, 1, attack_len)
        if release_len > 0:
            envelope[-release_len:] = torch.linspace(1, 0, release_len)
        return envelope

    def generate_tone(self, freq, duration, wave_type='sine'):
        t = torch.linspace(0, duration, int(duration * self.sr))
        if wave_type == 'sine':
            wave = torch.sin(2 * np.pi * freq * t)
        elif wave_type == 'square':
            wave = torch.sign(torch.sin(2 * np.pi * freq * t))
        elif wave_type == 'saw':
            wave = 2 * (freq * t - torch.floor(freq * t + 0.5))
        elif wave_type == 'triangle':
            # triangle wave from saw
            saw = 2 * (freq * t - torch.floor(freq * t + 0.5))
            wave = 2.0 * torch.abs(saw) - 1.0
        else:
            wave = torch.sin(2 * np.pi * freq * t)
            
        return wave * self.get_envelope(t)

    def generate_jaws_theme(self, duration=3.0):
        total_samples = int(duration * self.sr)
        full_waveform = torch.zeros(total_samples)
        
        # Jaws is E and F alternating (semitone)
        # E2 = 82.41 Hz, F2 = 87.31 Hz
        note1 = 82.41
        note2 = 87.31
        
        current_sample = 0
        note_idx = 0
        
        # Start slow with pauses, get faster and remove pauses
        # Initial duration in seconds
        current_note_dur = 0.5
        current_pause = 0.4
        min_note_dur = 0.1
        decay = 0.8 # Speed up factor
        
        while current_sample < total_samples:
            freq = note1 if note_idx % 2 == 0 else note2
            
            # Generate tone
            # Use sine for a cleaner, deeper sound (like a cello/bass)
            wave = self.generate_tone(freq, current_note_dur, wave_type='sine')
            
            # Add to waveform
            end_sample = min(current_sample + wave.shape[0], total_samples)
            full_waveform[current_sample:end_sample] += wave[:end_sample-current_sample] * 0.5
            
            # Advance by note duration + pause
            current_sample += int((current_note_dur + current_pause) * self.sr)
            
            # Speed up logic
            # Every pair of notes (E-F), reduce duration and pause
            if note_idx % 2 == 1:
                current_note_dur = max(min_note_dur, current_note_dur * decay)
                current_pause = max(0, current_pause * decay - 0.05) # Pause shrinks faster
                
            note_idx += 1
            
        return self.normalize(full_waveform).unsqueeze(0)

    def generate_sequence(self, duration=3.0, bpm=120):
        total_samples = int(duration * self.sr)
        full_waveform = torch.zeros(total_samples)
        
        # Musical parameters
        root_freq = random.choice([220, 261.63, 329.63, 392.00, 440.00]) # A3, C4, E4, G4, A4
        scale_ratios = [1.0, 1.12, 1.25, 1.33, 1.5, 1.66, 1.875, 2.0] # Major scale
        
        # 1. Bass line (Long notes)
        beat_dur = 60 / bpm
        current_sample = 0
        while current_sample < total_samples:
            note_dur = beat_dur * 2
            freq = root_freq * 0.5 * random.choice([1.0, 1.5, 0.75])
            
            wave = self.generate_tone(freq, note_dur, wave_type='sine')
            
            end_sample = min(current_sample + wave.shape[0], total_samples)
            full_waveform[current_sample:end_sample] += wave[:end_sample-current_sample] * 0.35
            current_sample += int(note_dur * self.sr)

        # 2. Melody (Faster notes)
        current_sample = 0
        while current_sample < total_samples:
            note_dur = beat_dur * random.choice([0.5, 0.25, 1.0])
            freq = root_freq * random.choice(scale_ratios) * random.choice([1, 2])
            
            # Use a softer waveform than raw saw to reduce harshness.
            wave = self.generate_tone(freq, note_dur, wave_type='triangle')
            wave = self.smooth(wave, kernel_size=9)
            
            end_sample = min(current_sample + wave.shape[0], total_samples)
            # Add with less volume
            full_waveform[current_sample:end_sample] += wave[:end_sample-current_sample] * 0.18
            current_sample += int(note_dur * self.sr)
            
        full_waveform = self.normalize(full_waveform)
        return full_waveform.unsqueeze(0) # (1, T)

    def _apply_fade_mask(self, mask: torch.Tensor, fade_s: float = 0.01) -> torch.Tensor:
        """
        Apply short fades at mask edges to avoid clicks.
        mask: (N,) in {0,1} or [0,1]
        """
        N = int(mask.numel())
        fade = max(1, int(fade_s * self.sr))
        if fade <= 1 or N < 4:
            return mask
        m = mask.clone()
        # find rising/falling edges
        dm = m[1:] - m[:-1]
        rises = (dm > 0.5).nonzero(as_tuple=False).view(-1) + 1
        falls = (dm < -0.5).nonzero(as_tuple=False).view(-1) + 1
        for idx in rises.tolist():
            a = max(0, idx - fade)
            b = min(N, idx + fade)
            ramp = torch.linspace(0.0, 1.0, b - a, device=m.device, dtype=m.dtype)
            m[a:b] = torch.maximum(m[a:b], ramp)
        for idx in falls.tolist():
            a = max(0, idx - fade)
            b = min(N, idx + fade)
            ramp = torch.linspace(1.0, 0.0, b - a, device=m.device, dtype=m.dtype)
            m[a:b] = torch.minimum(m[a:b], ramp)
        return torch.clamp(m, 0.0, 1.0)

    def generate_sequence_dropouts(
        self,
        duration: float = 3.0,
        bpm: float = 120,
        *,
        full_silence: bool = True,
        fade_s: float = 0.01,
    ) -> torch.Tensor:
        """
        Generate a 3-track synthetic clip with explicit track dropouts and optional full silence.

        Purpose (Phase 5 human evaluation + training): create segments where audio energy is
        unambiguously low/zero so visuals should settle, and segments where only some tracks
        are active so changes are easy to attribute.
        """
        total_samples = int(duration * self.sr)
        if total_samples <= 1:
            return torch.zeros((1, max(1, total_samples)), dtype=torch.float32)

        # Musical parameters (keep similar palette to generate_sequence)
        root_freq = random.choice([220, 261.63, 329.63, 392.00, 440.00])
        scale_ratios = [1.0, 1.12, 1.25, 1.33, 1.5, 1.66, 1.875, 2.0]
        beat_dur = 60 / float(bpm)

        # Track 1: bass (long notes)
        bass = torch.zeros(total_samples)
        cur = 0
        while cur < total_samples:
            note_dur = beat_dur * 2
            freq = root_freq * 0.5 * random.choice([1.0, 1.5, 0.75])
            wave = self.generate_tone(freq, note_dur, wave_type="sine")
            end = min(cur + wave.shape[0], total_samples)
            bass[cur:end] += wave[: end - cur] * 0.35
            cur += int(note_dur * self.sr)

        # Track 2: melody (faster notes)
        melody = torch.zeros(total_samples)
        cur = 0
        while cur < total_samples:
            note_dur = beat_dur * random.choice([0.5, 0.25, 1.0])
            freq = root_freq * random.choice(scale_ratios) * random.choice([1, 2])
            wave = self.generate_tone(freq, note_dur, wave_type="triangle")
            wave = self.smooth(wave, kernel_size=9)
            end = min(cur + wave.shape[0], total_samples)
            melody[cur:end] += wave[: end - cur] * 0.18
            cur += int(note_dur * self.sr)

        # Track 3: percussion-ish clicks (onsets)
        t = torch.linspace(0, duration, total_samples)
        clicks = torch.zeros_like(t)
        step = max(1, int(self.sr * 0.18))
        clicks[::step] = 1.0
        kernel = torch.hann_window(63)
        perc = torch.nn.functional.conv1d(clicks.view(1, 1, -1), kernel.view(1, 1, -1), padding=31).view(-1)
        perc = perc / (perc.abs().max() + 1e-8)
        perc = perc * 0.12

        # Segment schedule (relative time)
        # [bass, melody, perc] activity
        segs = []
        if full_silence:
            segs = [
                (0.00, 0.22, (1, 1, 1)),
                (0.22, 0.34, (1, 0, 1)),  # melody dropout
                (0.34, 0.46, (0, 1, 1)),  # bass dropout
                (0.46, 0.58, (0, 0, 0)),  # full silence
                (0.58, 0.76, (1, 0, 1)),  # only bass+perc
                (0.76, 1.00, (1, 1, 0)),  # only harmonic content
            ]
        else:
            segs = [
                (0.00, 0.30, (1, 1, 1)),
                (0.30, 0.50, (1, 0, 1)),
                (0.50, 0.70, (0, 1, 1)),
                (0.70, 1.00, (1, 1, 0)),
            ]

        def _mask_for(track_idx: int) -> torch.Tensor:
            m = torch.zeros(total_samples, dtype=torch.float32)
            for a, b, on in segs:
                if on[track_idx] == 0:
                    continue
                ia = int(a * total_samples)
                ib = int(b * total_samples)
                m[ia:ib] = 1.0
            return self._apply_fade_mask(m, fade_s=fade_s)

        mb = _mask_for(0)
        mm = _mask_for(1)
        mp = _mask_for(2)

        x = bass * mb + melody * mm + perc * mp
        x = self.normalize(x)
        return x.unsqueeze(0)
