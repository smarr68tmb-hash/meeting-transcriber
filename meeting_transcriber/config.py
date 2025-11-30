# -*- coding: utf-8 -*-
"""
Конфигурация системы записи и транскрипции.
Все параметры можно переопределить через переменные окружения.
"""

import os
from pathlib import Path


class Config:
    """Централизованная конфигурация приложения."""
    
    # Папки для данных
    RECORDINGS_FOLDER = Path.home() / "Meeting_Recordings"
    TRANSCRIPTS_FOLDER = Path.home() / "Meeting_Transcripts"
    LOGS_FOLDER = RECORDINGS_FOLDER / "logs"

    # Аудио запись
    DEFAULT_FORMAT = os.environ.get('REC_FORMAT', 'wav').lower()    # wav|flac
    DEFAULT_CHANNELS = os.environ.get('REC_CHANNELS', '2')          # '1'|'2'
    DEFAULT_SAMPLE_RATE = os.environ.get('REC_RATE', '48000')
    FLAC_LEVEL = os.environ.get('FLAC_LEVEL', '8')
    PRE_RECORD_PROBE = int(os.environ.get('PRE_RECORD_PROBE', '3'))  # сек; 0 = без пробы
    
    # Пресеты аудио фильтров (для избежания "квакания" от агрессивного шумодава)
    FILTER_PRESETS = {
        # raw: минимальная обработка, максимальное качество
        'raw': 'highpass=f=80',
        
        # soft: мягкая обработка без шумодава (рекомендуется)
        'soft': 'adeclick,highpass=f=80,lowpass=f=12000,'
                'acompressor=threshold=-24dB:ratio=2:attack=10:release=150,'
                'loudnorm=I=-16:TP=-1.5:LRA=11',
        
        # full: полная обработка с мягким шумодавом (anlmdn=s=3 вместо s=7)
        'full': 'adeclick,highpass=f=80,lowpass=f=12000,anlmdn=s=3,'
                'acompressor=threshold=-20dB:ratio=3:attack=5:release=100,'
                'loudnorm=I=-16:TP=-1.5:LRA=11',
        
        # legacy: старые настройки (может давать "квакание")
        'legacy': 'adeclick,highpass=f=80,lowpass=f=12000,anlmdn=s=7,'
                  'acompressor=threshold=-20dB:ratio=3:attack=5:release=100,'
                  'loudnorm=I=-16:TP=-1.5:LRA=11',
    }
    
    # Пресет фильтров по умолчанию: 'soft' — без шумодава, чистый звук
    FILTER_PRESET = os.environ.get('FILTER_PRESET', 'soft').lower()
    
    # Кастомные фильтры (переопределяют пресет)
    VOICE_FILTERS = os.environ.get('VOICE_FILTERS', FILTER_PRESETS.get(FILTER_PRESET, FILTER_PRESETS['soft']))

    # ASR (Automatic Speech Recognition)
    DEFAULT_MODEL = os.environ.get('WHISPER_MODEL', 'medium')
    ASR_BACKEND = os.environ.get('ASR_BACKEND', 'faster').lower()   # faster|whisper|whisperx|groq|auto
    ASR_DEVICE = os.environ.get('ASR_DEVICE', 'auto').lower()       # auto|cpu|cuda|mps|metal
    FORCE_RU = (os.environ.get('FORCE_RU', '0') == '1')             # принудительно русский язык
    
    # Groq API настройки
    GROQ_API_KEY = os.environ.get('GROQ_API_KEY', '')
    GROQ_MODEL = os.environ.get('GROQ_MODEL', 'whisper-large-v3')  # whisper-large-v3|whisper-large-v3-turbo
    GROQ_TIMEOUT = int(os.environ.get('GROQ_TIMEOUT', '300'))        # таймаут запроса в секундах
    GROQ_MAX_FILE_SIZE = 25 * 1024 * 1024                            # 25MB лимит Groq API
    ASR_FALLBACK = os.environ.get('ASR_FALLBACK', '1') == '1'        # fallback на локальный при ошибке API
    
    # LLM суммаризация через Groq
    SUMMARIZER_MODEL = os.environ.get('SUMMARIZER_MODEL', 'llama-3.3-70b-versatile')  # LLM модель для саммари
    SUMMARIZER_TIMEOUT = int(os.environ.get('SUMMARIZER_TIMEOUT', '120'))              # таймаут LLM запроса
    SUMMARIZER_MAX_TOKENS = int(os.environ.get('SUMMARIZER_MAX_TOKENS', '4096'))       # макс. токенов ответа
    AUTO_SUMMARIZE = os.environ.get('AUTO_SUMMARIZE', '0') == '1'                      # авто-суммаризация

    # faster-whisper специфичные настройки
    FASTER_COMPUTE = os.environ.get('FASTER_COMPUTE_TYPE', 'int8')  # int8|int8_float16|float16|float32
    FASTER_BEAM_SIZE = int(os.environ.get('FASTER_BEAM_SIZE', '5'))
    FASTER_VAD = os.environ.get('FASTER_VAD', '0') == '1'           # VAD по умолчанию выключен
    FASTER_CPU_THREADS = int(os.environ.get('FASTER_CPU_THREADS', '1'))

    # WhisperX специфичные настройки (диаризация)
    HF_TOKEN = os.environ.get('HF_TOKEN', '')                       # HuggingFace токен для pyannote
    WHISPERX_COMPUTE = os.environ.get('WHISPERX_COMPUTE', 'float16')  # float16|int8
    WHISPERX_BATCH_SIZE = int(os.environ.get('WHISPERX_BATCH_SIZE', '16'))
    WHISPERX_LANGUAGE = os.environ.get('WHISPERX_LANGUAGE', 'ru')   # язык по умолчанию
    DIARIZE_MIN_SPEAKERS = os.environ.get('DIARIZE_MIN_SPEAKERS')   # hint: мин. спикеров
    DIARIZE_MAX_SPEAKERS = os.environ.get('DIARIZE_MAX_SPEAKERS')   # hint: макс. спикеров
    
    # BlackHole интеграция (macOS)
    # Режим захвата: mic = микрофон, system = системный звук, both = оба
    CAPTURE_MODE = os.environ.get('CAPTURE_MODE', 'both').lower()   # mic|system|both (both для встреч)
    BLACKHOLE_DEVICE = os.environ.get('BLACKHOLE_DEVICE', '')       # авто или явный ID

    # Отладка
    DEBUG_SEGMENTS = os.environ.get('DEBUG_SEGMENTS', '0') == '1'
    
    @classmethod
    def ensure_directories(cls) -> None:
        """Создаёт необходимые директории если их нет."""
        cls.RECORDINGS_FOLDER.mkdir(parents=True, exist_ok=True)
        cls.TRANSCRIPTS_FOLDER.mkdir(parents=True, exist_ok=True)
        cls.LOGS_FOLDER.mkdir(parents=True, exist_ok=True)

