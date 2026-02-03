import torch
import numpy as np
import random

class AudioGenerator:
    def __init__(self, sample_rate=16000):
        self.sr = sample_rate
        
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
            full_waveform[current_sample:end_sample] += wave[:end_sample-current_sample] * 0.9
            
            # Advance by note duration + pause
            current_sample += int((current_note_dur + current_pause) * self.sr)
            
            # Speed up logic
            # Every pair of notes (E-F), reduce duration and pause
            if note_idx % 2 == 1:
                current_note_dur = max(min_note_dur, current_note_dur * decay)
                current_pause = max(0, current_pause * decay - 0.05) # Pause shrinks faster
                
            note_idx += 1
            
        return full_waveform.unsqueeze(0)

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
            full_waveform[current_sample:end_sample] += wave[:end_sample-current_sample] * 0.6
            current_sample += int(note_dur * self.sr)

        # 2. Melody (Faster notes)
        current_sample = 0
        while current_sample < total_samples:
            note_dur = beat_dur * random.choice([0.5, 0.25, 1.0])
            freq = root_freq * random.choice(scale_ratios) * random.choice([1, 2])
            
            wave = self.generate_tone(freq, note_dur, wave_type='saw')
            
            end_sample = min(current_sample + wave.shape[0], total_samples)
            # Add with less volume
            full_waveform[current_sample:end_sample] += wave[:end_sample-current_sample] * 0.3
            current_sample += int(note_dur * self.sr)
            
        # Normalize
        if full_waveform.abs().max() > 0:
            full_waveform = full_waveform / full_waveform.abs().max()
            
        return full_waveform.unsqueeze(0) # (1, T)
