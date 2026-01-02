"""
Audio Utils
===========
Funções auxiliares para manipulação de áudio.
"""

import numpy as np
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
from typing import Tuple, Optional
import io


# Número máximo de pontos para visualização (evita overflow)
MAX_DISPLAY_POINTS = 10000


def load_audio(file_path: str, sr: Optional[int] = None, duration: Optional[float] = None) -> Tuple[np.ndarray, int]:
    """
    Carrega arquivo de áudio.
    
    Args:
        file_path: Caminho do arquivo
        sr: Sample rate desejado (None = original)
        duration: Duração máxima em segundos (None = completo)
        
    Returns:
        Tuple com (array_de_audio, sample_rate)
    """
    y, sr_loaded = librosa.load(file_path, sr=sr, duration=duration)
    return y, sr_loaded


def save_audio(y: np.ndarray, sr: int, file_path: str):
    """
    Salva array de áudio em arquivo.
    
    Args:
        y: Array de áudio
        sr: Sample rate
        file_path: Caminho de saída
    """
    sf.write(file_path, y, sr)


def get_audio_duration(file_path: str) -> float:
    """
    Retorna duração do áudio em segundos.
    """
    duration = librosa.get_duration(path=file_path)
    return duration


def _downsample_for_display(y: np.ndarray, max_points: int = MAX_DISPLAY_POINTS) -> np.ndarray:
    """
    Reduz o número de pontos para visualização eficiente.
    Usa max-pooling para preservar picos.
    """
    if len(y) <= max_points:
        return y
    
    # Calcular fator de downsampling
    factor = len(y) // max_points
    
    # Usar reshape e max para preservar picos
    # Truncar para múltiplo do fator
    truncated_length = (len(y) // factor) * factor
    y_truncated = y[:truncated_length]
    
    # Reshape e calcular envelope (max absoluto por bloco)
    y_reshaped = y_truncated.reshape(-1, factor)
    
    # Para waveform, queremos ver tanto máximos quanto mínimos
    y_max = np.max(y_reshaped, axis=1)
    y_min = np.min(y_reshaped, axis=1)
    
    # Intercalar máximos e mínimos para manter formato visual
    y_downsampled = np.empty(len(y_max) * 2)
    y_downsampled[0::2] = y_max
    y_downsampled[1::2] = y_min
    
    return y_downsampled


def generate_waveform(y: np.ndarray, sr: int, 
                      title: str = "Waveform",
                      figsize: Tuple[int, int] = (12, 4),
                      color: str = '#00D4FF') -> plt.Figure:
    """
    Gera visualização de waveform (otimizada para arquivos grandes).
    
    Args:
        y: Array de áudio
        sr: Sample rate
        title: Título do gráfico
        figsize: Tamanho da figura
        color: Cor da waveform
        
    Returns:
        Matplotlib Figure
    """
    # Downsample para visualização
    y_display = _downsample_for_display(y)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Criar eixo de tempo ajustado
    duration = len(y) / sr
    time = np.linspace(0, duration, len(y_display))
    
    # Plotar waveform
    ax.plot(time, y_display, color=color, linewidth=0.5, alpha=0.8)
    ax.fill_between(time, y_display, alpha=0.3, color=color)
    
    # Estilizar
    ax.set_xlabel('Tempo (s)', fontsize=10, color='#888888')
    ax.set_ylabel('Amplitude', fontsize=10, color='#888888')
    ax.set_title(title, fontsize=12, fontweight='bold', color='#FFFFFF')
    
    # Mostrar duração total no título se for longo
    if duration >= 60:
        minutes = int(duration // 60)
        seconds = int(duration % 60)
        ax.set_title(f"{title} ({minutes}:{seconds:02d})", fontsize=12, fontweight='bold', color='#FFFFFF')
    
    # Fundo escuro
    fig.patch.set_facecolor('#1E1E1E')
    ax.set_facecolor('#1E1E1E')
    ax.tick_params(colors='#888888')
    ax.spines['bottom'].set_color('#444444')
    ax.spines['left'].set_color('#444444')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Grid sutil
    ax.grid(True, alpha=0.2, color='#444444')
    
    plt.tight_layout()
    return fig


def generate_comparison_waveform(y_original: np.ndarray, 
                                  y_processed: np.ndarray,
                                  sr: int,
                                  figsize: Tuple[int, int] = (12, 6)) -> plt.Figure:
    """
    Gera visualização comparativa de waveforms (antes/depois).
    Otimizada para arquivos grandes.
    
    Args:
        y_original: Array de áudio original
        y_processed: Array de áudio processado
        sr: Sample rate
        figsize: Tamanho da figura
        
    Returns:
        Matplotlib Figure
    """
    # Downsample ambos para visualização
    y_orig_display = _downsample_for_display(y_original)
    y_proc_display = _downsample_for_display(y_processed)
    
    fig, axes = plt.subplots(2, 1, figsize=figsize)
    
    # Tempo para original e processado
    duration_original = len(y_original) / sr
    duration_processed = len(y_processed) / sr
    
    time_original = np.linspace(0, duration_original, len(y_orig_display))
    time_processed = np.linspace(0, duration_processed, len(y_proc_display))
    
    # Waveform original
    axes[0].plot(time_original, y_orig_display, color='#FF6B6B', linewidth=0.5, alpha=0.8)
    axes[0].fill_between(time_original, y_orig_display, alpha=0.3, color='#FF6B6B')
    
    # Título com duração
    if duration_original >= 60:
        minutes = int(duration_original // 60)
        seconds = int(duration_original % 60)
        axes[0].set_title(f'🎵 Áudio Original ({minutes}:{seconds:02d})', fontsize=11, fontweight='bold', color='#FF6B6B')
    else:
        axes[0].set_title('🎵 Áudio Original', fontsize=11, fontweight='bold', color='#FF6B6B')
    axes[0].set_ylabel('Amplitude', fontsize=9, color='#888888')
    
    # Waveform processado
    axes[1].plot(time_processed, y_proc_display, color='#00D4FF', linewidth=0.5, alpha=0.8)
    axes[1].fill_between(time_processed, y_proc_display, alpha=0.3, color='#00D4FF')
    
    if duration_processed >= 60:
        minutes = int(duration_processed // 60)
        seconds = int(duration_processed % 60)
        axes[1].set_title(f'✨ Áudio Processado ({minutes}:{seconds:02d})', fontsize=11, fontweight='bold', color='#00D4FF')
    else:
        axes[1].set_title('✨ Áudio Processado', fontsize=11, fontweight='bold', color='#00D4FF')
    axes[1].set_ylabel('Amplitude', fontsize=9, color='#888888')
    axes[1].set_xlabel('Tempo (s)', fontsize=9, color='#888888')
    
    # Estilizar ambos os gráficos
    fig.patch.set_facecolor('#1E1E1E')
    for ax in axes:
        ax.set_facecolor('#1E1E1E')
        ax.tick_params(colors='#888888', labelsize=8)
        ax.spines['bottom'].set_color('#444444')
        ax.spines['left'].set_color('#444444')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(True, alpha=0.2, color='#444444')
    
    plt.tight_layout()
    return fig


def generate_spectrogram(y: np.ndarray, sr: int,
                         title: str = "Espectrograma",
                         figsize: Tuple[int, int] = (12, 4),
                         max_duration: float = 60.0) -> plt.Figure:
    """
    Gera visualização de espectrograma.
    Para arquivos longos, mostra apenas os primeiros segundos.
    """
    # Limitar duração para espectrograma
    max_samples = int(max_duration * sr)
    if len(y) > max_samples:
        y = y[:max_samples]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Calcular espectrograma com parâmetros otimizados
    n_fft = 2048
    hop_length = 512
    
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length)), ref=np.max)
    
    # Plotar
    img = librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='hz', ax=ax, cmap='magma', hop_length=hop_length)
    
    ax.set_title(title, fontsize=12, fontweight='bold', color='#FFFFFF')
    ax.set_xlabel('Tempo (s)', fontsize=10, color='#888888')
    ax.set_ylabel('Frequência (Hz)', fontsize=10, color='#888888')
    
    # Estilizar
    fig.patch.set_facecolor('#1E1E1E')
    ax.set_facecolor('#1E1E1E')
    ax.tick_params(colors='#888888')
    
    # Colorbar
    cbar = fig.colorbar(img, ax=ax, format='%+2.0f dB')
    cbar.ax.yaxis.set_tick_params(color='#888888')
    cbar.outline.set_edgecolor('#444444')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='#888888')
    
    plt.tight_layout()
    return fig


def audio_to_numpy(audio_segment) -> Tuple[np.ndarray, int]:
    """
    Converte AudioSegment do pydub para numpy array.
    """
    samples = np.array(audio_segment.get_array_of_samples())
    
    if audio_segment.channels == 2:
        samples = samples.reshape((-1, 2))
        samples = samples.mean(axis=1)  # Converter para mono
    
    # Normalizar para float
    samples = samples.astype(np.float32)
    samples /= np.iinfo(np.int16).max
    
    return samples, audio_segment.frame_rate


def numpy_to_audio(y: np.ndarray, sr: int):
    """
    Converte numpy array para AudioSegment do pydub.
    """
    from pydub import AudioSegment
    
    # Converter para int16
    y_int = (y * np.iinfo(np.int16).max).astype(np.int16)
    
    # Criar AudioSegment
    audio = AudioSegment(
        y_int.tobytes(),
        frame_rate=sr,
        sample_width=2,
        channels=1
    )
    
    return audio


def format_duration(seconds: float) -> str:
    """Formata duração em formato legível."""
    if seconds >= 3600:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours}h {minutes}m {secs}s"
    elif seconds >= 60:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}m {secs}s"
    else:
        return f"{seconds:.1f}s"
