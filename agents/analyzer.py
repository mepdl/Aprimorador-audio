"""
Audio Analyzer Agent
====================
Responsável por analisar áudio e identificar:
- Ruídos de fundo
- Ecos e reverberação
- Defeitos (clipping, distorção)
- Espaços vazios maiores que 2 segundos
"""

import numpy as np
import librosa
from typing import Dict, List, Tuple
from dataclasses import dataclass


@dataclass
class AnalysisResult:
    """Resultado da análise de áudio."""
    has_noise: bool
    noise_level: float
    has_echo: bool
    echo_level: float
    has_clipping: bool
    clipping_percentage: float
    silent_segments: List[Tuple[float, float]]
    total_silence_duration: float
    overall_quality_score: float
    recommendations: List[str]


class AudioAnalyzer:
    """
    Agente responsável pela análise de qualidade do áudio.
    """
    
    # Limite de amostras para análise otimizada
    MAX_SAMPLES_FOR_ANALYSIS = 30 * 44100  # 30 segundos max
    
    def __init__(self, silence_threshold_db: float = -40.0, min_silence_duration: float = 2.0):
        """
        Args:
            silence_threshold_db: Limite em dB para considerar silêncio
            min_silence_duration: Duração mínima em segundos para detectar silêncio
        """
        self.silence_threshold_db = silence_threshold_db
        self.min_silence_duration = min_silence_duration
    
    def analyze(self, audio_path: str) -> AnalysisResult:
        """
        Analisa o áudio e retorna um relatório completo.
        
        Args:
            audio_path: Caminho do arquivo de áudio
            
        Returns:
            AnalysisResult com todos os problemas detectados
        """
        # Carregar áudio com sample rate fixo para performance
        y, sr = librosa.load(audio_path, sr=22050, mono=True)
        
        # Para análise rápida, usar apenas uma amostra do áudio
        y_sample = self._get_sample(y, sr)
        
        # Realizar análises otimizadas
        noise_analysis = self._analyze_noise_fast(y_sample, sr)
        echo_analysis = self._analyze_echo_fast(y_sample, sr)
        clipping_analysis = self._analyze_clipping(y)  # Usar áudio completo
        silence_analysis = self._analyze_silence_fast(y, sr)  # Usar áudio completo
        
        # Gerar recomendações
        recommendations = self._generate_recommendations(
            noise_analysis, echo_analysis, clipping_analysis, silence_analysis
        )
        
        # Calcular score de qualidade
        quality_score = self._calculate_quality_score(
            noise_analysis, echo_analysis, clipping_analysis, silence_analysis
        )
        
        return AnalysisResult(
            has_noise=noise_analysis['detected'],
            noise_level=noise_analysis['level'],
            has_echo=echo_analysis['detected'],
            echo_level=echo_analysis['level'],
            has_clipping=clipping_analysis['detected'],
            clipping_percentage=clipping_analysis['percentage'],
            silent_segments=silence_analysis['segments'],
            total_silence_duration=silence_analysis['total_duration'],
            overall_quality_score=quality_score,
            recommendations=recommendations
        )
    
    def _get_sample(self, y: np.ndarray, sr: int) -> np.ndarray:
        """Retorna uma amostra do áudio para análise rápida."""
        max_samples = min(len(y), self.MAX_SAMPLES_FOR_ANALYSIS)
        
        if len(y) <= max_samples:
            return y
        
        # Pegar amostras do início, meio e fim
        chunk_size = max_samples // 3
        start = y[:chunk_size]
        middle_start = len(y) // 2 - chunk_size // 2
        middle = y[middle_start:middle_start + chunk_size]
        end = y[-chunk_size:]
        
        return np.concatenate([start, middle, end])
    
    def _analyze_noise_fast(self, y: np.ndarray, sr: int) -> Dict:
        """
        Analisa nível de ruído de fundo (versão otimizada).
        """
        # Usar STFT com parâmetros menores para velocidade
        n_fft = 1024
        hop_length = 512
        
        S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
        
        # Estimar ruído usando os primeiros frames
        noise_frames = min(10, S.shape[1])
        if noise_frames == 0:
            return {'detected': False, 'level': 0, 'snr_db': 60}
        
        noise_profile = np.mean(S[:, :noise_frames], axis=1)
        
        # Calcular SNR aproximado
        signal_power = np.mean(S ** 2)
        noise_power = np.mean(noise_profile ** 2)
        
        if noise_power > 0:
            snr = 10 * np.log10(signal_power / noise_power)
        else:
            snr = 60
        
        noise_level = max(0, min(1, 1 - (snr / 60)))
        
        return {
            'detected': noise_level > 0.2,
            'level': float(noise_level),
            'snr_db': float(snr)
        }
    
    def _analyze_echo_fast(self, y: np.ndarray, sr: int) -> Dict:
        """
        Analisa presença de eco/reverberação (versão otimizada usando FFT).
        """
        # Usar análise de reverberação via decaimento de energia
        # Muito mais rápido que autocorrelação completa
        
        # Dividir em frames e calcular energia
        frame_length = int(0.05 * sr)  # 50ms frames
        hop_length = int(0.025 * sr)   # 25ms hop
        
        # Calcular RMS por frame
        frames = librosa.util.frame(y, frame_length=frame_length, hop_length=hop_length)
        rms = np.sqrt(np.mean(frames ** 2, axis=0))
        
        if len(rms) < 10:
            return {'detected': False, 'level': 0.0}
        
        # Detectar decaimento lento (indicativo de reverberação)
        # Normalizar RMS
        rms_norm = rms / (np.max(rms) + 1e-10)
        
        # Calcular taxa de decaimento média
        decay_rate = 0.0
        peak_indices = np.where(rms_norm > 0.5)[0]
        
        if len(peak_indices) > 1:
            # Verificar quanto tempo leva para o sinal decair após picos
            decay_times = []
            for idx in peak_indices[:10]:  # Limitar a 10 picos
                if idx + 5 < len(rms_norm):
                    decay = rms_norm[idx] - rms_norm[idx + 5]
                    decay_times.append(decay)
            
            if decay_times:
                avg_decay = np.mean(decay_times)
                # Decaimento lento indica eco
                echo_level = max(0, min(1, 1 - avg_decay * 2))
            else:
                echo_level = 0.0
        else:
            echo_level = 0.0
        
        return {
            'detected': echo_level > 0.3,
            'level': float(echo_level)
        }
    
    def _analyze_clipping(self, y: np.ndarray) -> Dict:
        """
        Analisa distorção/clipping no áudio.
        """
        threshold = 0.99
        clipped_samples = np.sum(np.abs(y) >= threshold)
        total_samples = len(y)
        
        clipping_percentage = (clipped_samples / total_samples) * 100
        
        return {
            'detected': clipping_percentage > 0.1,
            'percentage': float(clipping_percentage),
            'clipped_samples': int(clipped_samples)
        }
    
    def _analyze_silence_fast(self, y: np.ndarray, sr: int) -> Dict:
        """
        Detecta segmentos de silêncio (versão otimizada).
        """
        # Usar frames maiores para velocidade
        frame_length = int(0.1 * sr)   # 100ms frames
        hop_length = int(0.05 * sr)    # 50ms hop
        
        # Threshold de amplitude
        threshold_amplitude = 10 ** (self.silence_threshold_db / 20)
        
        # Calcular RMS de forma eficiente
        rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
        
        # Encontrar segmentos silenciosos
        silent_frames = rms < threshold_amplitude
        
        segments = []
        in_silence = False
        silence_start = 0
        frame_time = hop_length / sr
        
        for i, is_silent in enumerate(silent_frames):
            time = i * frame_time
            
            if is_silent and not in_silence:
                silence_start = time
                in_silence = True
            elif not is_silent and in_silence:
                silence_duration = time - silence_start
                if silence_duration >= self.min_silence_duration:
                    segments.append((silence_start, time))
                in_silence = False
        
        # Verificar silêncio no final
        if in_silence:
            end_time = len(y) / sr
            silence_duration = end_time - silence_start
            if silence_duration >= self.min_silence_duration:
                segments.append((silence_start, end_time))
        
        total_duration = sum(end - start for start, end in segments)
        
        return {
            'segments': segments,
            'total_duration': float(total_duration),
            'count': len(segments)
        }
    
    def _generate_recommendations(self, noise: Dict, echo: Dict, 
                                   clipping: Dict, silence: Dict) -> List[str]:
        """
        Gera recomendações baseadas na análise.
        """
        recommendations = []
        
        if noise['detected']:
            if noise['level'] > 0.5:
                recommendations.append(
                    "🔊 Alto nível de ruído detectado. Recomendado usar remoção de ruído em intensidade alta (0.7-1.0)."
                )
            else:
                recommendations.append(
                    "🔉 Ruído baixo detectado. Use remoção de ruído em intensidade moderada (0.3-0.5)."
                )
        
        if echo['detected']:
            recommendations.append(
                "🔁 Eco/reverberação detectado. O processamento pode ajudar, mas eco é difícil de remover completamente."
            )
        
        if clipping['detected']:
            recommendations.append(
                f"⚠️ Clipping detectado ({clipping['percentage']:.2f}% das amostras). O áudio original pode ter distorção permanente."
            )
        
        if silence['count'] > 0:
            recommendations.append(
                f"🔇 {silence['count']} segmento(s) de silêncio longo detectado(s) (total: {silence['total_duration']:.1f}s). "
                "Recomendado ativar remoção de silêncios."
            )
        
        if not recommendations:
            recommendations.append("✅ Áudio em boa qualidade! Ajustes mínimos podem ser necessários.")
        
        return recommendations
    
    def _calculate_quality_score(self, noise: Dict, echo: Dict,
                                  clipping: Dict, silence: Dict) -> float:
        """
        Calcula score de qualidade geral (0-100).
        """
        score = 100.0
        
        # Penalizar por ruído
        score -= noise['level'] * 30
        
        # Penalizar por eco
        score -= echo['level'] * 20
        
        # Penalizar por clipping
        score -= min(clipping['percentage'] * 5, 25)
        
        # Penalizar levemente por silêncios longos
        score -= min(silence['count'] * 2, 10)
        
        return max(0, min(100, score))
