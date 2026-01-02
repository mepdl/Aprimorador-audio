"""
Audio Enhancer - Streamlit Web Application
==========================================
Interface web para processamento e aprimoramento de áudio.
"""

import streamlit as st
import tempfile
import os
from pathlib import Path
import numpy as np

# Importar agentes
from agents import AudioExtractor, AudioAnalyzer, AudioProcessor
from agents.processor import ProcessingSettings
from utils.audio_utils import (
    load_audio, 
    generate_waveform, 
    generate_comparison_waveform,
    format_duration
)
import matplotlib
matplotlib.use('Agg')  # Backend não-interativo para evitar memory leaks
import matplotlib.pyplot as plt

# Sample rate baixo para visualização (evita overflow em áudios longos)
VISUALIZATION_SR = 8000

# Configuração da página
st.set_page_config(
    page_title="Audio Enhancer",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado
st.markdown("""
<style>
    /* Tema escuro premium */
    .stApp {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    }
    
    /* Cards */
    .analysis-card {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 16px;
        padding: 20px;
        margin: 10px 0;
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
    }
    
    /* Headers estilizados */
    .gradient-text {
        background: linear-gradient(90deg, #00D4FF, #7B2CBF);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 10px;
    }
    
    .subtitle {
        color: #888888;
        text-align: center;
        font-size: 1.1rem;
        margin-bottom: 30px;
    }
    
    /* Métricas */
    .metric-container {
        background: rgba(0, 212, 255, 0.1);
        border-radius: 12px;
        padding: 15px;
        text-align: center;
        border: 1px solid rgba(0, 212, 255, 0.3);
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #00D4FF;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #888888;
    }
    
    /* Botões */
    .stButton > button {
        background: linear-gradient(90deg, #00D4FF, #7B2CBF);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 12px 30px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 20px rgba(0, 212, 255, 0.4);
    }
    
    /* Upload area */
    .uploadedFile {
        background: rgba(255, 255, 255, 0.05) !important;
        border-radius: 12px !important;
    }
    
    /* Sliders */
    .stSlider > div > div > div {
        background: linear-gradient(90deg, #00D4FF, #7B2CBF) !important;
    }
    
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: rgba(0, 0, 0, 0.3);
        backdrop-filter: blur(10px);
    }
    
    /* Recomendações */
    .recommendation {
        background: rgba(255, 255, 255, 0.03);
        border-left: 3px solid #00D4FF;
        padding: 10px 15px;
        margin: 8px 0;
        border-radius: 0 8px 8px 0;
    }
    
    /* Progress bar */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #00D4FF, #7B2CBF);
    }
</style>
""", unsafe_allow_html=True)


def init_session_state():
    """Inicializa variáveis de sessão."""
    if 'extractor' not in st.session_state:
        st.session_state.extractor = AudioExtractor()
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = AudioAnalyzer()
    if 'processor' not in st.session_state:
        st.session_state.processor = AudioProcessor()
    if 'audio_path' not in st.session_state:
        st.session_state.audio_path = None
    if 'processed_path' not in st.session_state:
        st.session_state.processed_path = None
    if 'analysis_result' not in st.session_state:
        st.session_state.analysis_result = None
    if 'metadata' not in st.session_state:
        st.session_state.metadata = None


def render_header():
    """Renderiza cabeçalho da aplicação."""
    st.markdown('<h1 class="gradient-text">🎵 Audio Enhancer</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Processamento inteligente de áudio com remoção de ruídos e aprimoramento automático</p>', unsafe_allow_html=True)


def render_upload_section():
    """Renderiza seção de upload."""
    st.markdown("### 📁 Upload de Arquivo")
    
    uploaded_file = st.file_uploader(
        "Arraste seu arquivo de áudio ou vídeo aqui",
        type=['mp3', 'mp4', 'mov', 'avi', 'aac', 'mkv', 'wav', 'ogg'],
        help="Formatos suportados: MP3, MP4, MOV, AVI, AAC, MKV, WAV, OGG"
    )
    
    if uploaded_file is not None:
        # Salvar arquivo temporário
        temp_dir = tempfile.mkdtemp()
        temp_path = os.path.join(temp_dir, uploaded_file.name)
        
        with open(temp_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())
        
        # Processar com extractor
        with st.spinner("🔄 Extraindo e preparando áudio..."):
            try:
                audio_path, metadata = st.session_state.extractor.extract_audio(temp_path)
                st.session_state.audio_path = audio_path
                st.session_state.metadata = metadata
                st.success("✅ Áudio extraído com sucesso!")
                
                # Mostrar informações do arquivo
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Duração", f"{metadata['duration_seconds']:.1f}s")
                with col2:
                    st.metric("Sample Rate", f"{metadata['sample_rate']} Hz")
                with col3:
                    st.metric("Canais", metadata['channels'])
                with col4:
                    st.metric("Formato Original", metadata['original_format'].upper())
                    
            except Exception as e:
                st.error(f"❌ Erro ao processar arquivo: {str(e)}")
                return False
    
    return st.session_state.audio_path is not None


def render_analysis_section():
    """Renderiza seção de análise."""
    if st.session_state.audio_path is None:
        return
    
    st.markdown("---")
    st.markdown("### 🔍 Análise do Áudio")
    
    if st.button("🔬 Analisar Áudio", use_container_width=True):
        with st.spinner("🔄 Analisando áudio..."):
            try:
                result = st.session_state.analyzer.analyze(st.session_state.audio_path)
                st.session_state.analysis_result = result
            except Exception as e:
                st.error(f"❌ Erro na análise: {str(e)}")
                return
    
    if st.session_state.analysis_result is not None:
        result = st.session_state.analysis_result
        
        # Métricas de qualidade
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            quality_color = "#00FF88" if result.overall_quality_score >= 70 else "#FFAA00" if result.overall_quality_score >= 40 else "#FF4444"
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-value" style="color: {quality_color}">{result.overall_quality_score:.0f}</div>
                <div class="metric-label">Score de Qualidade</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            noise_icon = "🔴" if result.has_noise else "🟢"
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-value">{noise_icon} {result.noise_level*100:.0f}%</div>
                <div class="metric-label">Nível de Ruído</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            echo_icon = "🔴" if result.has_echo else "🟢"
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-value">{echo_icon} {result.echo_level*100:.0f}%</div>
                <div class="metric-label">Eco Detectado</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-value">🔇 {result.total_silence_duration:.1f}s</div>
                <div class="metric-label">Silêncio Total</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Recomendações
        if result.recommendations:
            st.markdown("#### 💡 Recomendações")
            for rec in result.recommendations:
                st.markdown(f'<div class="recommendation">{rec}</div>', unsafe_allow_html=True)
        
        # Waveform original (carrega com sample rate reduzido para visualização)
        st.markdown("#### 📊 Waveform Original")
        try:
            y, sr = load_audio(st.session_state.audio_path, sr=VISUALIZATION_SR)
            fig = generate_waveform(y, sr, "Áudio Original")
            st.pyplot(fig)
            plt.close(fig)  # Liberar memória
        except Exception as e:
            st.warning(f"Não foi possível gerar waveform: {str(e)}")


def render_filters_sidebar():
    """Renderiza controles de filtros na sidebar."""
    st.sidebar.markdown("## ⚙️ Filtros de Processamento")
    
    # Remoção de ruído
    st.sidebar.markdown("### 🔊 Remoção de Ruído")
    noise_reduce = st.sidebar.checkbox("Ativar remoção de ruído", value=True)
    noise_intensity = st.sidebar.slider(
        "Intensidade",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.1,
        disabled=not noise_reduce,
        help="0 = suave, 1 = agressivo"
    )
    
    # Sensibilidade
    sensitivity = st.sidebar.slider(
        "Sensibilidade",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.1,
        help="Sensibilidade da detecção de ruído"
    )
    
    st.sidebar.markdown("---")
    
    # Remoção de silêncio
    st.sidebar.markdown("### 🔇 Remoção de Silêncios")
    remove_silence = st.sidebar.checkbox("Remover silêncios longos", value=True)
    silence_threshold = st.sidebar.slider(
        "Threshold de silêncio (dB)",
        min_value=-60.0,
        max_value=-20.0,
        value=-40.0,
        step=5.0,
        disabled=not remove_silence,
        help="Sons abaixo deste nível são considerados silêncio"
    )
    min_silence = st.sidebar.slider(
        "Duração mínima (s)",
        min_value=0.5,
        max_value=5.0,
        value=2.0,
        step=0.5,
        disabled=not remove_silence,
        help="Silêncios menores que isso serão mantidos"
    )
    
    st.sidebar.markdown("---")
    
    # Normalização
    st.sidebar.markdown("### 📊 Normalização")
    normalize = st.sidebar.checkbox("Normalizar volume", value=True)
    target_loudness = st.sidebar.slider(
        "Loudness alvo (LUFS)",
        min_value=-24.0,
        max_value=-6.0,
        value=-14.0,
        step=1.0,
        disabled=not normalize,
        help="-14 LUFS é o padrão para streaming"
    )
    
    return ProcessingSettings(
        noise_reduce=noise_reduce,
        noise_reduce_intensity=noise_intensity,
        remove_silence=remove_silence,
        silence_threshold_db=silence_threshold,
        min_silence_duration=min_silence,
        normalize=normalize,
        target_loudness=target_loudness,
        sensitivity=sensitivity
    )


def render_processing_section(settings: ProcessingSettings):
    """Renderiza seção de processamento."""
    if st.session_state.audio_path is None:
        return
    
    st.markdown("---")
    st.markdown("### 🚀 Processamento")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("⚡ Processar Áudio", use_container_width=True, type="primary"):
            with st.spinner("🔄 Processando áudio com filtros selecionados..."):
                try:
                    processed_path, proc_metadata = st.session_state.processor.process(
                        st.session_state.audio_path,
                        settings
                    )
                    st.session_state.processed_path = processed_path
                    st.success("✅ Áudio processado com sucesso!")
                    
                    # Mostrar log de processamento
                    with st.expander("📋 Log de Processamento"):
                        for log_item in proc_metadata['processing_log']:
                            st.write(f"✓ {log_item}")
                            
                except Exception as e:
                    st.error(f"❌ Erro no processamento: {str(e)}")


def render_preview_section():
    """Renderiza seção de preview antes/depois."""
    if st.session_state.processed_path is None:
        return
    
    st.markdown("---")
    st.markdown("### 🎧 Preview Antes/Depois")
    
    # Comparação de waveforms (carrega com sample rate reduzido)
    try:
        y_original, sr = load_audio(st.session_state.audio_path, sr=VISUALIZATION_SR)
        y_processed, _ = load_audio(st.session_state.processed_path, sr=VISUALIZATION_SR)
        
        fig = generate_comparison_waveform(y_original, y_processed, sr)
        st.pyplot(fig)
        plt.close(fig)  # Liberar memória
    except Exception as e:
        st.warning(f"Não foi possível gerar comparação visual: {str(e)}")
    
    # Players de áudio
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🎵 Áudio Original")
        with open(st.session_state.audio_path, 'rb') as f:
            st.audio(f.read(), format='audio/wav')
    
    with col2:
        st.markdown("#### ✨ Áudio Processado")
        with open(st.session_state.processed_path, 'rb') as f:
            st.audio(f.read(), format='audio/wav')


def render_download_section():
    """Renderiza seção de download."""
    if st.session_state.processed_path is None:
        return
    
    st.markdown("---")
    st.markdown("### 💾 Download")
    
    # Ler arquivo processado
    with open(st.session_state.processed_path, 'rb') as f:
        audio_bytes = f.read()
    
    # Gerar nome do arquivo
    original_name = Path(st.session_state.audio_path).stem
    download_name = f"{original_name}_enhanced.wav"
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.download_button(
            label="⬇️ Baixar Áudio Processado (WAV)",
            data=audio_bytes,
            file_name=download_name,
            mime="audio/wav",
            use_container_width=True
        )
        
        # Informações do arquivo
        file_size = len(audio_bytes) / (1024 * 1024)
        st.caption(f"📦 Tamanho: {file_size:.2f} MB")


def main():
    """Função principal da aplicação."""
    init_session_state()
    render_header()
    
    # Sidebar com filtros
    settings = render_filters_sidebar()
    
    # Conteúdo principal
    has_audio = render_upload_section()
    
    if has_audio:
        render_analysis_section()
        render_processing_section(settings)
        render_preview_section()
        render_download_section()
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style="text-align: center; color: #666666; padding: 20px;">
            <p>Audio Enhancer v2.0 | Desenvolvido com ❤️ usando Python, Streamlit e IA</p>
            <p style="font-size: 0.8rem;">Tecnologias: pydub, librosa, noisereduce, ffmpeg</p>
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
