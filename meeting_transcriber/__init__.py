# -*- coding: utf-8 -*-
"""
Meeting Transcriber - Система записи и транскрипции совещаний.

Модули:
    - config: Конфигурация приложения
    - logging_setup: Настройка логирования
    - utils: Утилиты для работы с аудио
    - recorder: Запись аудио с микрофона
    - transcriber: Транскрипция аудио в текст
    - cli: Командный интерфейс
"""

from .config import Config
from .recorder import MeetingRecorder
from .transcriber import EnhancedTranscriber
from .cli import main, __version__

__all__ = [
    "Config",
    "MeetingRecorder", 
    "EnhancedTranscriber",
    "main",
    "__version__",
]

