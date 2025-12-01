# -*- coding: utf-8 -*-
"""
Утилиты для работы с аудио и определения платформы.
"""

import shutil
import platform
import subprocess
from pathlib import Path
from typing import Dict

from .logging_setup import get_logger

logger = get_logger()


def ffprobe_ok(path: Path) -> bool:
    """
    Проверяет, что файл является валидным аудио.
    
    Args:
        path: Путь к аудио файлу
        
    Returns:
        True если файл валидный, False иначе
    """
    if not shutil.which('ffprobe'):
        # Если ffprobe нет, проверяем хотя бы размер файла
        return path.exists() and path.stat().st_size > 1000
    
    cmd = [
        'ffprobe', '-v', 'error',
        '-select_streams', 'a:0',
        '-show_entries', 'stream=codec_name',
        '-of', 'default=nokey=1:noprint_wrappers=1',
        str(path)
    ]
    try:
        result = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True).strip()
        return bool(result)
    except subprocess.CalledProcessError:
        return False


def get_audio_duration(path: Path) -> float:
    """
    Получает длительность аудио файла в секундах.
    
    Args:
        path: Путь к аудио файлу
        
    Returns:
        Длительность в секундах или 0.0 при ошибке
    """
    if not shutil.which('ffprobe'):
        logger.warning("ffprobe не найден, не могу определить длительность")
        return 0.0
    
    cmd = [
        'ffprobe', '-v', 'error',
        '-show_entries', 'format=duration',
        '-of', 'default=nokey=1:noprint_wrappers=1',
        str(path)
    ]
    try:
        result = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True).strip()
        return float(result)
    except Exception as e:
        logger.debug(f"Не удалось получить длительность: {e}")
        return 0.0


def get_platform_config() -> Dict[str, str]:
    """
    Получает конфигурацию ffmpeg для текущей платформы.
    
    Returns:
        Словарь с параметрами: format, dummy, list_cmd
    """
    system = platform.system()
    
    if system == "Darwin":
        return {
            'format': 'avfoundation',
            'dummy': '',
            'list_cmd': ['-list_devices', 'true']
        }
    
    if system == "Windows":
        return {
            'format': 'dshow',
            'dummy': 'dummy',
            'list_cmd': ['-list_devices', 'true']
        }
    
    # Linux
    use_pulse = shutil.which('pactl') is not None
    return {
        'format': 'pulse' if use_pulse else 'alsa',
        'dummy': 'default',
        'list_cmd': []
    }


def format_duration(seconds: float) -> str:
    """
    Форматирует длительность в читаемый вид.
    
    Args:
        seconds: Количество секунд
        
    Returns:
        Строка вида "1:23:45" или "23:45"
    """
    hours, remainder = divmod(int(seconds), 3600)
    minutes, secs = divmod(remainder, 60)
    
    if hours > 0:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def format_timestamp_srt(seconds: float) -> str:
    """
    Форматирует время для SRT субтитров.

    Args:
        seconds: Время в секундах

    Returns:
        Строка вида "00:01:23,456"
    """
    hours, remainder = divmod(int(seconds), 3600)
    minutes, secs = divmod(remainder, 60)
    ms = int((seconds - int(seconds)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"


def get_ffmpeg_device_name(device_id: str) -> str:
    """
    Получает имя устройства ffmpeg по его ID.

    Args:
        device_id: ID устройства (например ":0" для macOS)

    Returns:
        Имя устройства или device_id если не удалось определить
    """
    platform_config = get_platform_config()

    # Находим ffmpeg (может быть в /opt/homebrew/bin или в PATH)
    ffmpeg_path = shutil.which('ffmpeg')
    if not ffmpeg_path:
        # Попробуем стандартные пути для macOS
        if platform.system() == "Darwin":
            for path in ['/opt/homebrew/bin/ffmpeg', '/usr/local/bin/ffmpeg']:
                if Path(path).exists():
                    ffmpeg_path = path
                    break

        if not ffmpeg_path:
            logger.debug("ffmpeg не найден")
            return device_id

    try:
        # Запрашиваем список устройств ffmpeg
        cmd = [
            ffmpeg_path, '-f', platform_config['format'],
            *platform_config['list_cmd'],
            '-i', platform_config['dummy']
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='ignore'
        )

        # Парсим вывод ffmpeg (устройства в stderr)
        output = result.stderr

        # Для macOS avfoundation
        if platform_config['format'] == 'avfoundation':
            # Ищем секцию аудио устройств
            in_audio_section = False
            device_idx = device_id.lstrip(':')

            for line in output.split('\n'):
                if 'AVFoundation audio devices:' in line:
                    in_audio_section = True
                    continue

                if in_audio_section and f'[{device_idx}]' in line:
                    # Извлекаем имя устройства
                    # Формат: [AVFoundation indev @ 0x...] [0] Агрегатное устройство
                    parts = line.split(f'[{device_idx}]')
                    if len(parts) > 1:
                        name = parts[1].strip()
                        logger.debug(f"FFmpeg устройство {device_id} -> '{name}'")
                        return name

    except Exception as e:
        logger.debug(f"Не удалось получить имя устройства ffmpeg: {e}")

    return device_id

