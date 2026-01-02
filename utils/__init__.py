"""
Utils Module
============
Funções auxiliares para manipulação de áudio.
"""

from .audio_utils import (
    load_audio,
    save_audio,
    get_audio_duration,
    generate_waveform,
    generate_comparison_waveform,
    audio_to_numpy,
    numpy_to_audio,
    format_duration
)

__all__ = [
    'load_audio',
    'save_audio', 
    'get_audio_duration',
    'generate_waveform',
    'generate_comparison_waveform',
    'audio_to_numpy',
    'numpy_to_audio',
    'format_duration'
]
