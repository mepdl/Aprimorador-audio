"""
Audio Processor Agent
=====================
Responsável por aplicar filtros de processamento:
- Remoção de ruído (multi-passagem)
- Remoção de silêncios longos
- Normalização de volume
- Exportação em WAV
"""

import numpy as np
import librosa
import soundfile as sf
import noisereduce as nr
from scipy import signal
from scipy.ndimage import median_filter
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import tempfile
import os


@dataclass
class ProcessingSettings:
    """Configurações de processamento."""
    noise_reduce: bool = True
    noise_reduce_intensity: float = 0.5
    remove_silence: bool = True
    silence_threshold_db: float = -40.0
    min_silence_duration: float = 2.0
    normalize: bool = True
    target_loudness: float = -14.0  # LUFS
    sensitivity: float = 0.5


class AudioProcessor:
    """
    Agente responsável pelo processamento e aprimoramento do áudio.
    """
    
    def __init__(self):
        self.temp_dir = tempfile.mkdtemp(prefix="audio_processor_")
    
    def process(self, audio_path: str, settings: ProcessingSettings) -> Tuple[str, Dict]:
        """
        Processa o áudio aplicando todos os filtros configurados.
        
        Args:
            audio_path: Caminho do arquivo de áudio
            settings: Configurações de processamento
            
        Returns:
            Tuple com (caminho_do_audio_processado, metadados)
        """
        # Carregar áudio
        y, sr = librosa.load(audio_path, sr=None)
        
        processing_log = []
        
        # Aplicar remoção de ruído
        if settings.noise_reduce:
            y, noise_log = self._reduce_noise_advanced(
                y, sr, 
                settings.noise_reduce_intensity, 
                settings.sensitivity
            )
            processing_log.extend(noise_log)
        
        # Remover silêncios longos
        if settings.remove_silence:
            y, removed_count = self._remove_silence(
                y, sr, 
                settings.silence_threshold_db, 
                settings.min_silence_duration
            )
            processing_log.append(f"Silêncios removidos: {removed_count} segmento(s)")
        
        # Normalizar volume
        if settings.normalize:
            y = self._normalize(y, settings.target_loudness)
            processing_log.append(f"Volume normalizado para {settings.target_loudness} LUFS")
        
        # Salvar áudio processado
        output_path = os.path.join(self.temp_dir, "processed_audio.wav")
        sf.write(output_path, y, sr)
        
        metadata = {
            'sample_rate': sr,
            'duration_seconds': len(y) / sr,
            'processing_log': processing_log,
            'settings_applied': {
                'noise_reduce': settings.noise_reduce,
                'noise_reduce_intensity': settings.noise_reduce_intensity,
                'remove_silence': settings.remove_silence,
                'normalize': settings.normalize
            }
        }
        
        return output_path, metadata
    
    def _reduce_noise_advanced(self, y: np.ndarray, sr: int, 
                                intensity: float, sensitivity: float) -> Tuple[np.ndarray, List[str]]:
        """
        Remoção de ruído avançada com múltiplas técnicas.
        
        Args:
            y: Array de áudio
            sr: Sample rate
            intensity: Intensidade da remoção (0-1)
            sensitivity: Sensibilidade da detecção (0-1)
            
        Returns:
            Tuple com (áudio_limpo, log_de_processamento)
        """
        log = []
        
        # =========================================
        # ETAPA 1: Remoção de ruído estacionário
        # =========================================
        
        # Parâmetros mais agressivos baseados na intensidade
        prop_decrease = 0.6 + (intensity * 0.4)  # 0.6 a 1.0
        n_std_thresh = 1.2 - (sensitivity * 0.9)  # 1.2 a 0.3
        
        # Primeira passagem - ruído estacionário
        y_clean = nr.reduce_noise(
            y=y,
            sr=sr,
            prop_decrease=prop_decrease,
            n_std_thresh_stationary=n_std_thresh,
            stationary=True,
            n_fft=2048,
            win_length=2048,
            hop_length=512
        )
        log.append(f"Passagem 1: Ruído estacionário removido (prop: {prop_decrease:.2f})")
        
        # =========================================
        # ETAPA 2: Remoção de ruído não-estacionário
        # =========================================
        
        if intensity >= 0.5:
            # Segunda passagem para ruído variável
            y_clean = nr.reduce_noise(
                y=y_clean,
                sr=sr,
                prop_decrease=prop_decrease * 0.8,
                n_std_thresh_stationary=n_std_thresh * 1.2,
                stationary=False,
                n_fft=2048,
                win_length=2048,
                hop_length=512
            )
            log.append("Passagem 2: Ruído não-estacionário removido")
        
        # =========================================
        # ETAPA 3: Filtro de frequências problemáticas
        # =========================================
        
        if intensity >= 0.3:
            # Remover frequências muito baixas (rumble) e muito altas (hiss)
            y_clean = self._apply_frequency_filter(y_clean, sr, intensity)
            log.append("Filtro de frequências aplicado (rumble/hiss)")
        
        # =========================================
        # ETAPA 4: Spectral Gating adicional (alta intensidade)
        # =========================================
        
        if intensity >= 0.7:
            # Terceira passagem com gating espectral mais agressivo
            y_clean = nr.reduce_noise(
                y=y_clean,
                sr=sr,
                prop_decrease=1.0,  # Máximo
                n_std_thresh_stationary=0.5,  # Muito sensível
                stationary=True,
                n_fft=4096,  # Maior resolução de frequência
                win_length=4096,
                hop_length=1024
            )
            log.append("Passagem 3: Spectral gating agressivo aplicado")
        
        # =========================================
        # ETAPA 5: Suavização para intensidade máxima
        # =========================================
        
        if intensity >= 0.9:
            # Aplicar filtro mediano para remover artefatos restantes
            y_clean = self._apply_smoothing(y_clean, sr)
            log.append("Suavização final aplicada")
        
        # =========================================
        # ETAPA 6: Limpar artefatos residuais
        # =========================================
        
        # Limpar valores muito pequenos (ruído de quantização)
        noise_floor = 0.001 * (1 - intensity * 0.5)  # Mais agressivo com maior intensidade
        y_clean = np.where(np.abs(y_clean) < noise_floor, 0, y_clean)
        
        log.append(f"Remoção de ruído completa (intensidade: {intensity:.0%})")
        
        return y_clean, log
    
    def _apply_frequency_filter(self, y: np.ndarray, sr: int, intensity: float) -> np.ndarray:
        """
        Aplica filtros de frequência para remover rumble (baixas freq) e hiss (altas freq).
        """
        # Frequências de corte baseadas na intensidade
        # Mais intensidade = corte mais agressivo
        low_cut = 60 + (intensity * 80)   # 60-140 Hz
        high_cut = 12000 - (intensity * 4000)  # 12000-8000 Hz
        
        # Filtro passa-banda
        nyquist = sr / 2
        low = low_cut / nyquist
        high = min(high_cut / nyquist, 0.99)  # Garantir que não ultrapasse Nyquist
        
        if low >= high:
            return y
        
        # Criar filtro Butterworth
        try:
            b, a = signal.butter(4, [low, high], btype='band')
            y_filtered = signal.filtfilt(b, a, y)
            return y_filtered
        except Exception:
            return y
    
    def _apply_smoothing(self, y: np.ndarray, sr: int) -> np.ndarray:
        """
        Aplica suavização para remover artefatos de processamento.
        """
        # Filtro mediano para remover clicks e pops
        kernel_size = 3
        y_smoothed = median_filter(y, size=kernel_size)
        
        # Misturar com original para não perder detalhes
        alpha = 0.7  # 70% suavizado, 30% original
        y_result = alpha * y_smoothed + (1 - alpha) * y
        
        return y_result
    
    def _reduce_noise(self, y: np.ndarray, sr: int, 
                      intensity: float, sensitivity: float) -> np.ndarray:
        """
        Método legado para compatibilidade.
        """
        y_clean, _ = self._reduce_noise_advanced(y, sr, intensity, sensitivity)
        return y_clean
    
    def _remove_silence(self, y: np.ndarray, sr: int,
                        threshold_db: float, min_duration: float) -> Tuple[np.ndarray, int]:
        """
        Remove segmentos de silêncio maiores que a duração mínima.
        
        Returns:
            Tuple com (áudio_sem_silêncios, quantidade_removida)
        """
        # Converter threshold para amplitude
        threshold_amplitude = 10 ** (threshold_db / 20)
        
        # Calcular RMS por frame
        frame_length = int(0.025 * sr)
        hop_length = int(0.010 * sr)
        
        rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
        
        # Encontrar segmentos não-silenciosos
        non_silent_frames = rms >= threshold_amplitude
        
        # Expandir frames para amostras
        segments_to_keep = []
        in_sound = False
        sound_start = 0
        silence_start = 0
        removed_count = 0
        
        for i, is_sound in enumerate(non_silent_frames):
            sample_idx = i * hop_length
            
            if is_sound and not in_sound:
                # Início de som
                if in_sound is False and i > 0:
                    # Verificar se silêncio anterior era longo o suficiente para remover
                    silence_duration = (sample_idx - silence_start) / sr
                    if silence_duration >= min_duration:
                        removed_count += 1
                    else:
                        # Manter o silêncio curto
                        if segments_to_keep:
                            # Estender segmento anterior
                            pass
                sound_start = sample_idx
                in_sound = True
                
            elif not is_sound and in_sound:
                # Início de silêncio
                segments_to_keep.append((sound_start, sample_idx))
                silence_start = sample_idx
                in_sound = False
        
        # Adicionar último segmento se terminar em som
        if in_sound:
            segments_to_keep.append((sound_start, len(y)))
        
        # Reconstruir áudio sem silêncios longos
        if not segments_to_keep:
            return y, 0
        
        # Concatenar segmentos com pequeno fade para suavizar
        fade_samples = int(0.01 * sr)  # 10ms fade
        result_parts = []
        
        for i, (start, end) in enumerate(segments_to_keep):
            segment = y[start:end].copy()
            
            # Aplicar fade in no início (exceto primeiro segmento)
            if i > 0 and len(segment) > fade_samples:
                fade_in = np.linspace(0, 1, fade_samples)
                segment[:fade_samples] *= fade_in
            
            # Aplicar fade out no final (exceto último segmento)
            if i < len(segments_to_keep) - 1 and len(segment) > fade_samples:
                fade_out = np.linspace(1, 0, fade_samples)
                segment[-fade_samples:] *= fade_out
            
            result_parts.append(segment)
        
        if result_parts:
            y_trimmed = np.concatenate(result_parts)
        else:
            y_trimmed = y
        
        return y_trimmed, removed_count
    
    def _normalize(self, y: np.ndarray, target_loudness: float = -14.0) -> np.ndarray:
        """
        Normaliza o volume do áudio.
        
        Args:
            y: Array de áudio
            target_loudness: Loudness alvo em LUFS (aproximado)
        """
        # Calcular RMS atual
        rms = np.sqrt(np.mean(y ** 2))
        
        if rms == 0:
            return y
        
        # Calcular ganho necessário (aproximação simples)
        current_db = 20 * np.log10(rms)
        target_rms_db = target_loudness + 10  # Aproximação LUFS para RMS
        gain_db = target_rms_db - current_db
        gain = 10 ** (gain_db / 20)
        
        # Aplicar ganho com limitação para evitar clipping
        y_normalized = y * gain
        
        # Limitar para evitar clipping
        max_val = np.max(np.abs(y_normalized))
        if max_val > 0.99:
            y_normalized = y_normalized * (0.99 / max_val)
        
        return y_normalized
    
    def get_preview_comparison(self, original_path: str, 
                                processed_path: str,
                                duration: float = 10.0) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        Retorna arrays de áudio para comparação antes/depois.
        
        Args:
            original_path: Caminho do áudio original
            processed_path: Caminho do áudio processado
            duration: Duração do preview em segundos
            
        Returns:
            Tuple com (original_array, processed_array, sample_rate)
        """
        # Carregar áudios
        y_original, sr = librosa.load(original_path, sr=None, duration=duration)
        y_processed, _ = librosa.load(processed_path, sr=sr, duration=duration)
        
        return y_original, y_processed, sr
    
    def cleanup(self):
        """Remove arquivos temporários."""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
