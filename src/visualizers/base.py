import torch

class Visualizer:
    def render(self, waveform, fps=30, height=128, width=128, sample_rate=16000):
        """
        Render audio waveform to video frames.
        waveform: (1, samples) tensor
        Returns: (num_frames, 3, height, width) tensor
        """
        raise NotImplementedError

    def get_frame_audio_chunks(self, waveform, fps, sample_rate=16000):
        duration = waveform.shape[1] / sample_rate
        num_frames = int(duration * fps)
        num_frames = max(1, num_frames)
        samples_per_frame = waveform.shape[1] // num_frames
        return num_frames, samples_per_frame
