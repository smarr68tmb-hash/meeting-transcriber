# -*- coding: utf-8 -*-
"""
Meeting Transcriber - Система записи и транскрипции совещаний.

Модули:
    - config: Конфигурация приложения
    - logging_setup: Настройка логирования
    - utils: Утилиты для работы с аудио
    - recorder: Запись аудио с микрофона
    - transcriber: Транскрипция аудио в текст
    - groq_backend: Groq API для быстрой транскрипции
    - summarizer: LLM суммаризация транскриптов
    - blackhole: Интеграция с BlackHole для записи системного звука
    - whisperx: Backend с диаризацией спикеров
    - cli: Командный интерфейс
"""

# Workaround для PyTorch 2.6+ совместимости с pyannote
# PyTorch 2.6+ по умолчанию использует weights_only=True что ломает pyannote
# ВАЖНО: патчим ПЕРЕД импортом любых библиотек
try:
    import torch
    
    # Сохраняем настоящую оригинальную функцию
    _real_torch_load = torch.load.__wrapped__ if hasattr(torch.load, '__wrapped__') else torch.load
    
    def _patched_load(f, map_location=None, pickle_module=None, *, weights_only=False, **kwargs):
        """Патч torch.load с weights_only=False по умолчанию."""
        if pickle_module is not None:
            kwargs['pickle_module'] = pickle_module
        return _real_torch_load(f, map_location=map_location, weights_only=weights_only, **kwargs)
    
    torch.load = _patched_load
    
    # Патчим lightning_fabric.utilities.cloud_io
    try:
        import lightning_fabric.utilities.cloud_io
        
        def _patched_cloud_load(path_or_url, map_location=None, weights_only=None):
            # ВСЕГДА используем weights_only=False для pyannote моделей
            return _patched_load(path_or_url, map_location=map_location, weights_only=False)
        
        lightning_fabric.utilities.cloud_io._load = _patched_cloud_load
        lightning_fabric.utilities.cloud_io.torch.load = _patched_load
    except ImportError:
        pass
        
except ImportError:
    pass

from .config import Config
from .recorder import MeetingRecorder
from .transcriber import EnhancedTranscriber
from .postprocess import postprocess_transcription, filter_hallucinations
from .audio_monitor import AudioLevelMonitor
from .groq_backend import GroqTranscriber, check_groq_available
from .summarizer import MeetingSummarizer, format_summary_text, check_summarizer_available
from .blackhole import (
    CaptureMode,
    check_blackhole_installed,
    find_blackhole_device,
    get_blackhole_status
)
from .cli import main, __version__

__all__ = [
    "Config",
    "MeetingRecorder", 
    "EnhancedTranscriber",
    "GroqTranscriber",
    "MeetingSummarizer",
    "AudioLevelMonitor",
    "CaptureMode",
    "postprocess_transcription",
    "filter_hallucinations",
    "format_summary_text",
    "check_groq_available",
    "check_summarizer_available",
    "check_blackhole_installed",
    "find_blackhole_device",
    "get_blackhole_status",
    "main",
    "__version__",
]

