import random
from .base import Visualizer
from .pulse import PulseVisualizer
from .spectrum import SpectrumVisualizer
from .waveform import WaveformVisualizer

REGISTRY = [PulseVisualizer, SpectrumVisualizer, WaveformVisualizer]

def get_random_visualizer():
    return random.choice(REGISTRY)()
