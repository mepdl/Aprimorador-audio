"""
Audio Extractor Agent
=====================
Responsável por detectar formato e extrair áudio de arquivos de vídeo/áudio.
Suporta: mp3, mp4, mov, avi, aac, mkv, wav, ogg
"""

import os
import tempfile
import subprocess
from pathlib import Path
from typing import Optional, Tuple
from pydub import AudioSegment


class AudioExtractor:
    """
    Agente responsável pela extração de áudio de arquivos multimídia.
    """
    
    SUPPORTED_AUDIO = {'.mp3', '.wav', '.aac', '.ogg'}
    SUPPORTED_VIDEO = {'.mp4', '.mov', '.avi', '.mkv'}
    SUPPORTED_FORMATS = SUPPORTED_AUDIO | SUPPORTED_VIDEO
    
    def __init__(self):
        self.temp_dir = tempfile.mkdtemp(prefix="audio_enhancer_")
    
    def is_supported(self, file_path: str) -> bool:
        """Verifica se o formato do arquivo é suportado."""
        ext = Path(file_path).suffix.lower()
        return ext in self.SUPPORTED_FORMATS
    
    def get_file_type(self, file_path: str) -> str:
        """
        Retorna o tipo do arquivo: 'audio', 'video' ou 'unknown'.
        """
        ext = Path(file_path).suffix.lower()
        if ext in self.SUPPORTED_AUDIO:
            return 'audio'
        elif ext in self.SUPPORTED_VIDEO:
            return 'video'
        return 'unknown'
    
    def extract_audio(self, input_path: str, output_format: str = 'wav') -> Tuple[str, dict]:
        """
        Extrai áudio do arquivo de entrada.
        
        Args:
            input_path: Caminho do arquivo de entrada
            output_format: Formato de saída (padrão: wav)
            
        Returns:
            Tuple com (caminho_do_audio, metadados)
        """
        if not self.is_supported(input_path):
            raise ValueError(f"Formato não suportado: {Path(input_path).suffix}")
        
        file_type = self.get_file_type(input_path)
        input_name = Path(input_path).stem
        output_path = os.path.join(self.temp_dir, f"{input_name}.{output_format}")
        
        metadata = {
            'original_format': Path(input_path).suffix.lower(),
            'original_type': file_type,
            'extracted_format': output_format
        }
        
        if file_type == 'video':
            # Usar FFmpeg para extrair áudio de vídeo
            output_path = self._extract_from_video(input_path, output_path)
        else:
            # Converter áudio para o formato desejado
            output_path = self._convert_audio(input_path, output_path, output_format)
        
        # Obter informações adicionais do áudio extraído
        audio = AudioSegment.from_file(output_path)
        metadata.update({
            'duration_seconds': len(audio) / 1000.0,
            'sample_rate': audio.frame_rate,
            'channels': audio.channels,
            'sample_width': audio.sample_width
        })
        
        return output_path, metadata
    
    def _extract_from_video(self, video_path: str, output_path: str) -> str:
        """
        Extrai áudio de arquivo de vídeo usando FFmpeg.
        """
        try:
            cmd = [
                'ffmpeg',
                '-i', video_path,
                '-vn',  # Sem vídeo
                '-acodec', 'pcm_s16le',  # Codec WAV
                '-ar', '44100',  # Sample rate
                '-ac', '2',  # Stereo
                '-y',  # Sobrescrever se existir
                output_path
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            
            return output_path
            
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Erro ao extrair áudio com FFmpeg: {e.stderr}")
        except FileNotFoundError:
            raise RuntimeError(
                "FFmpeg não encontrado. Por favor, instale o FFmpeg e adicione ao PATH."
            )
    
    def _convert_audio(self, input_path: str, output_path: str, output_format: str) -> str:
        """
        Converte arquivo de áudio para o formato desejado.
        """
        try:
            audio = AudioSegment.from_file(input_path)
            audio.export(output_path, format=output_format)
            return output_path
        except Exception as e:
            raise RuntimeError(f"Erro ao converter áudio: {str(e)}")
    
    def cleanup(self):
        """Remove arquivos temporários."""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
