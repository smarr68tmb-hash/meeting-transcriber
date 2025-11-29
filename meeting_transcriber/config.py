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
    VOICE_FILTERS = os.environ.get(
        "VOICE_FILTERS",
        "adeclick,highpass=f=80,lowpass=f=12000,anlmdn=s=7,"
        "acompressor=threshold=-20dB:ratio=3:attack=5:release=100,"
        "loudnorm=I=-16:TP=-1.5:LRA=11"
    )

    # ASR (Automatic Speech Recognition)
    DEFAULT_MODEL = os.environ.get('WHISPER_MODEL', 'medium')
    ASR_BACKEND = os.environ.get('ASR_BACKEND', 'faster').lower()   # faster|whisper
    ASR_DEVICE = os.environ.get('ASR_DEVICE', 'auto').lower()       # auto|cpu|cuda|mps|metal
    FORCE_RU = (os.environ.get('FORCE_RU', '0') == '1')             # принудительно русский язык

    # faster-whisper специфичные настройки
    FASTER_COMPUTE = os.environ.get('FASTER_COMPUTE_TYPE', 'int8')  # int8|int8_float16|float16|float32
    FASTER_BEAM_SIZE = int(os.environ.get('FASTER_BEAM_SIZE', '5'))
    FASTER_VAD = os.environ.get('FASTER_VAD', '0') == '1'           # VAD по умолчанию выключен
    FASTER_CPU_THREADS = int(os.environ.get('FASTER_CPU_THREADS', '1'))

    # Отладка
    DEBUG_SEGMENTS = os.environ.get('DEBUG_SEGMENTS', '0') == '1'
    
    @classmethod
    def ensure_directories(cls) -> None:
        """Создаёт необходимые директории если их нет."""
        cls.RECORDINGS_FOLDER.mkdir(parents=True, exist_ok=True)
        cls.TRANSCRIPTS_FOLDER.mkdir(parents=True, exist_ok=True)
        cls.LOGS_FOLDER.mkdir(parents=True, exist_ok=True)

