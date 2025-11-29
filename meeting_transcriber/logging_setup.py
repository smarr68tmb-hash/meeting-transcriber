# -*- coding: utf-8 -*-
"""
Настройка системы логирования с выводом в консоль и файл.
"""

import sys
import logging
from pathlib import Path
from logging.handlers import RotatingFileHandler

from .config import Config


def setup_logging(verbose: bool = False, debug: bool = False) -> logging.Logger:
    """
    Настройка системы логирования.
    
    Args:
        verbose: Если True, показывать INFO сообщения в консоли
        debug: Если True, показывать DEBUG сообщения в консоли
    
    Returns:
        Настроенный логгер
    """
    logger = logging.getLogger("meeting_transcriber")
    
    # Определяем уровень логирования для консоли
    if debug:
        level = logging.DEBUG
    elif verbose:
        level = logging.INFO
    else:
        level = logging.WARNING
    
    logger.setLevel(logging.DEBUG)  # Логгер принимает всё, фильтруют handlers
    
    # Формат для консоли (краткий)
    console_format = logging.Formatter(
        '%(asctime)s │ %(levelname)-7s │ %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # Формат для файла (подробный)
    file_format = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(console_format)
    
    # File handler с ротацией (5 MB, 3 бэкапа)
    Config.ensure_directories()
    log_file = Config.LOGS_FOLDER / "meeting_transcriber.log"
    
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=5 * 1024 * 1024,  # 5 MB
        backupCount=3,
        encoding='utf-8'
    )
    file_handler.setLevel(logging.DEBUG)  # В файл пишем всё
    file_handler.setFormatter(file_format)
    
    # Очищаем существующие handlers и добавляем новые
    logger.handlers.clear()
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger


def get_logger() -> logging.Logger:
    """Получить логгер приложения."""
    return logging.getLogger("meeting_transcriber")

