# -*- coding: utf-8 -*-
"""
Интеграция с BlackHole для захвата системного аудио на macOS.

BlackHole — виртуальный аудио драйвер, позволяющий:
- Записывать системный звук (голоса собеседников в Zoom/Meet/Teams)
- Создавать Multi-Output Device для записи mic + system одновременно

Установка BlackHole:
    brew install blackhole-2ch
    # или скачать: https://existential.audio/blackhole/

Настройка для записи встреч:
    1. Audio MIDI Setup → Create Multi-Output Device
    2. Добавить: Built-in Output + BlackHole 2ch
    3. System Preferences → Sound → Output → Multi-Output Device
    4. Записывать с BlackHole 2ch (там будет системный звук)
"""

import platform
import subprocess
import shutil
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass
from enum import Enum

from .logging_setup import get_logger

logger = get_logger()


class CaptureMode(Enum):
    """Режим захвата аудио."""
    MIC = "mic"           # Только микрофон
    SYSTEM = "system"     # Только системный звук (через BlackHole)
    BOTH = "both"         # Микрофон + системный (требует Aggregate Device)


@dataclass
class AudioDevice:
    """Информация об аудио устройстве."""
    index: int
    name: str
    is_input: bool
    is_blackhole: bool = False
    channels: int = 2


def is_macos() -> bool:
    """Проверить, что мы на macOS."""
    return platform.system() == "Darwin"


def check_blackhole_installed() -> bool:
    """
    Проверить, установлен ли BlackHole.
    
    Returns:
        True если BlackHole найден в системе
    """
    if not is_macos():
        return False
    
    try:
        # Проверяем через system_profiler
        result = subprocess.run(
            ["system_profiler", "SPAudioDataType"],
            capture_output=True,
            text=True,
            timeout=10
        )
        return "blackhole" in result.stdout.lower()
    except Exception as e:
        logger.debug(f"Не удалось проверить BlackHole через system_profiler: {e}")
    
    # Альтернативная проверка через kextstat (устаревший способ)
    try:
        result = subprocess.run(
            ["kextstat"],
            capture_output=True,
            text=True,
            timeout=10
        )
        return "blackhole" in result.stdout.lower()
    except Exception:
        pass
    
    return False


def list_audio_devices_avfoundation() -> List[AudioDevice]:
    """
    Получить список аудио устройств через ffmpeg (avfoundation).
    
    Returns:
        Список AudioDevice
    """
    if not shutil.which("ffmpeg"):
        logger.warning("ffmpeg не найден")
        return []
    
    try:
        result = subprocess.run(
            ["ffmpeg", "-f", "avfoundation", "-list_devices", "true", "-i", ""],
            capture_output=True,
            text=True,
            timeout=10
        )
        output = result.stderr  # ffmpeg выводит в stderr
    except Exception as e:
        logger.error(f"Ошибка получения списка устройств: {e}")
        return []
    
    devices = []
    in_audio_section = False
    
    for line in output.split('\n'):
        # Ищем секцию аудио устройств
        if "AVFoundation audio devices:" in line:
            in_audio_section = True
            continue
        
        if not in_audio_section:
            continue
        
        # Парсим строку вида: [AVFoundation indev @ ...] [0] BlackHole 2ch
        if "] [" in line and "]" in line:
            try:
                # Извлекаем индекс и имя
                idx_start = line.rfind("] [") + 3
                idx_end = line.find("]", idx_start)
                if idx_start > 2 and idx_end > idx_start:
                    index = int(line[idx_start:idx_end])
                    name = line[idx_end + 2:].strip()
                    
                    is_blackhole = "blackhole" in name.lower()
                    
                    devices.append(AudioDevice(
                        index=index,
                        name=name,
                        is_input=True,
                        is_blackhole=is_blackhole
                    ))
            except (ValueError, IndexError) as e:
                logger.debug(f"Не удалось распарсить строку устройства: {line}, {e}")
    
    return devices


def find_blackhole_device() -> Optional[AudioDevice]:
    """
    Найти устройство BlackHole.
    
    Ищет в порядке приоритета:
    1. BlackHole 2ch (стандартный)
    2. BlackHole 16ch
    3. Любое устройство с "blackhole" в имени
    
    Returns:
        AudioDevice или None
    """
    if not is_macos():
        logger.debug("BlackHole доступен только на macOS")
        return None
    
    devices = list_audio_devices_avfoundation()
    blackhole_devices = [d for d in devices if d.is_blackhole]
    
    if not blackhole_devices:
        logger.debug("BlackHole устройства не найдены")
        return None
    
    # Приоритет: 2ch > 16ch > остальные
    for priority_name in ["BlackHole 2ch", "BlackHole 16ch"]:
        for device in blackhole_devices:
            if device.name == priority_name:
                logger.info(f"Найдено BlackHole устройство: {device.name} (:{device.index})")
                return device
    
    # Возвращаем первое найденное
    device = blackhole_devices[0]
    logger.info(f"Найдено BlackHole устройство: {device.name} (:{device.index})")
    return device


def find_aggregate_device() -> Optional[AudioDevice]:
    """
    Найти Aggregate/Multi-Output Device для записи mic + system.
    
    Ищет устройства с именами:
    - "Multi-Output Device"
    - "Aggregate Device"
    - "Агрегатное устройство" (русская локализация)
    - Содержащие "aggregate", "multi", "агрегат"
    
    Returns:
        AudioDevice или None
    """
    if not is_macos():
        return None
    
    devices = list_audio_devices_avfoundation()
    
    # Ищем по приоритету (включая русские названия)
    priority_names = [
        "Multi-Output Device",
        "Aggregate Device", 
        "Агрегатное устройство",
        "Устройство с несколькими выходами",
    ]
    
    for name in priority_names:
        for device in devices:
            if device.name == name:
                logger.info(f"Найдено Aggregate устройство: {device.name} (:{device.index})")
                return device
    
    # Ищем по ключевым словам (EN + RU)
    keywords = ["aggregate", "multi", "агрегат", "несколько"]
    for device in devices:
        name_lower = device.name.lower()
        if any(kw in name_lower for kw in keywords):
            logger.info(f"Найдено Aggregate устройство: {device.name} (:{device.index})")
            return device
    
    return None


def resolve_device_for_mode(
    capture_mode: CaptureMode,
    explicit_device: Optional[str] = None
) -> Tuple[Optional[str], str]:
    """
    Определить устройство записи для заданного режима.
    
    Args:
        capture_mode: Режим захвата (mic/system/both)
        explicit_device: Явно указанное устройство (переопределяет автоопределение)
        
    Returns:
        Tuple[device_id, description]
        device_id: ID устройства для ffmpeg (например ":0") или None при ошибке
        description: Описание для пользователя
    """
    # Если устройство указано явно — используем его
    if explicit_device:
        # Обрабатываем алиас "blackhole"
        if explicit_device.lower() == "blackhole":
            bh = find_blackhole_device()
            if bh:
                return f":{bh.index}", f"BlackHole ({bh.name})"
            return None, "BlackHole не найден (установите: brew install blackhole-2ch)"
        
        return explicit_device, f"Устройство {explicit_device}"
    
    # Автоопределение по режиму
    if capture_mode == CaptureMode.MIC:
        # Микрофон по умолчанию — :0 на macOS
        return ":0", "Микрофон (по умолчанию)"
    
    if capture_mode == CaptureMode.SYSTEM:
        bh = find_blackhole_device()
        if bh:
            return f":{bh.index}", f"Системный звук ({bh.name})"
        return None, "BlackHole не найден. Установите: brew install blackhole-2ch"
    
    if capture_mode == CaptureMode.BOTH:
        # Для записи mic + system нужен Aggregate Device
        agg = find_aggregate_device()
        if agg:
            return f":{agg.index}", f"Mic + System ({agg.name})"
        
        return None, (
            "Для записи mic + system нужен Aggregate Device.\n"
            "Создайте его в Audio MIDI Setup:\n"
            "1. Откройте 'Audio MIDI Setup'\n"
            "2. '+' → 'Create Aggregate Device'\n"
            "3. Включите микрофон и BlackHole 2ch"
        )
    
    return None, f"Неизвестный режим: {capture_mode}"


def get_blackhole_status() -> Dict[str, any]:
    """
    Получить статус BlackHole интеграции.
    
    Returns:
        Словарь с информацией о доступности
    """
    status = {
        "platform": platform.system(),
        "is_macos": is_macos(),
        "blackhole_installed": False,
        "blackhole_device": None,
        "aggregate_device": None,
        "available_modes": [CaptureMode.MIC.value],
    }
    
    if not is_macos():
        status["message"] = "BlackHole доступен только на macOS"
        return status
    
    # Проверяем BlackHole
    bh = find_blackhole_device()
    if bh:
        status["blackhole_installed"] = True
        status["blackhole_device"] = {
            "index": bh.index,
            "name": bh.name
        }
        status["available_modes"].append(CaptureMode.SYSTEM.value)
    
    # Проверяем Aggregate Device
    agg = find_aggregate_device()
    if agg:
        status["aggregate_device"] = {
            "index": agg.index,
            "name": agg.name
        }
        status["available_modes"].append(CaptureMode.BOTH.value)
    
    # Формируем сообщение
    if bh and agg:
        status["message"] = f"✅ BlackHole готов: {bh.name}, Aggregate: {agg.name}"
    elif bh:
        status["message"] = f"✅ BlackHole готов: {bh.name} (Aggregate Device не настроен)"
    else:
        status["message"] = "❌ BlackHole не найден. Установите: brew install blackhole-2ch"
    
    return status


def print_blackhole_status():
    """Вывести статус BlackHole в консоль."""
    status = get_blackhole_status()
    
    print("\n🔊 BlackHole Status")
    print("=" * 50)
    print(f"Platform: {status['platform']}")
    print(f"Status: {status['message']}")
    
    if status.get("blackhole_device"):
        bh = status["blackhole_device"]
        print(f"BlackHole: :{bh['index']} ({bh['name']})")
    
    if status.get("aggregate_device"):
        agg = status["aggregate_device"]
        print(f"Aggregate: :{agg['index']} ({agg['name']})")
    
    print(f"Available modes: {', '.join(status['available_modes'])}")
    print("=" * 50)
    
    if not status.get("blackhole_installed"):
        print("\n📦 Установка BlackHole:")
        print("   brew install blackhole-2ch")
        print("   # или: https://existential.audio/blackhole/")
    
    if status.get("blackhole_installed") and not status.get("aggregate_device"):
        print("\n🔧 Настройка записи Mic + System:")
        print("   1. Откройте 'Audio MIDI Setup' (Spotlight → Audio MIDI)")
        print("   2. Нажмите '+' → 'Create Aggregate Device'")
        print("   3. Включите галочки: микрофон + BlackHole 2ch")
        print("   4. Используйте: --capture-mode both")


def print_setup_instructions():
    """Вывести инструкции по настройке BlackHole."""
    print("""
╔══════════════════════════════════════════════════════════════════╗
║                    🎧 BlackHole Setup Guide                       ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                   ║
║  BlackHole позволяет записывать системный звук на macOS.         ║
║  Это полезно для транскрипции Zoom, Google Meet, Teams и др.     ║
║                                                                   ║
║  📦 УСТАНОВКА:                                                    ║
║     brew install blackhole-2ch                                    ║
║                                                                   ║
║  🔧 НАСТРОЙКА (для записи СВОЕГО голоса + СОБЕСЕДНИКОВ):         ║
║                                                                   ║
║  1. Откройте "Audio MIDI Setup" (через Spotlight)                 ║
║                                                                   ║
║  2. Создайте Multi-Output Device:                                 ║
║     • Нажмите "+" → "Create Multi-Output Device"                  ║
║     • Включите: Built-in Output ✓ + BlackHole 2ch ✓               ║
║     • Это позволит слышать звук И записывать его                  ║
║                                                                   ║
║  3. Создайте Aggregate Device (для mic + system):                 ║
║     • Нажмите "+" → "Create Aggregate Device"                     ║
║     • Включите: Built-in Microphone ✓ + BlackHole 2ch ✓           ║
║                                                                   ║
║  4. System Preferences → Sound → Output:                          ║
║     • Выберите "Multi-Output Device"                              ║
║                                                                   ║
║  🎙️ ИСПОЛЬЗОВАНИЕ:                                                ║
║                                                                   ║
║  # Только микрофон (по умолчанию):                                ║
║  meeting-transcriber record "Meeting" --device :0                 ║
║                                                                   ║
║  # Только системный звук (голоса собеседников):                   ║
║  meeting-transcriber record "Meeting" --capture-mode system       ║
║                                                                   ║
║  # Микрофон + системный звук (полная запись встречи):             ║
║  meeting-transcriber record "Meeting" --capture-mode both         ║
║                                                                   ║
║  # Явно указать BlackHole:                                        ║
║  meeting-transcriber record "Meeting" --device blackhole          ║
║                                                                   ║
╚══════════════════════════════════════════════════════════════════╝
""")

