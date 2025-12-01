# -*- coding: utf-8 -*-
"""
Мониторинг уровня звука в реальном времени.

Использует sounddevice для отображения уровня сигнала
во время записи через ffmpeg.
"""

import threading
import time
from typing import Optional, Callable

from .logging_setup import get_logger

logger = get_logger()

# Проверяем наличие sounddevice
HAS_SOUNDDEVICE = False
try:
    import sounddevice as sd
    import numpy as np
    HAS_SOUNDDEVICE = True
except ImportError:
    pass


class AudioLevelMonitor:
    """
    Мониторинг уровня звука с микрофона в реальном времени.
    
    Отображает визуальный индикатор уровня в консоли.
    """
    
    # Символы для визуализации уровня
    BAR_CHARS = "▁▂▃▄▅▆▇█"
    BAR_WIDTH = 30
    
    def __init__(
        self,
        device: Optional[str] = None,
        sample_rate: int = 16000,
        block_size: int = 1024
    ):
        """
        Инициализация монитора.
        
        Args:
            device: ID устройства ввода (None = по умолчанию)
            sample_rate: Частота дискретизации
            block_size: Размер блока для анализа
        """
        if not HAS_SOUNDDEVICE:
            logger.warning(
                "sounddevice не установлен. Мониторинг уровня недоступен. "
                "Установите: pip install sounddevice numpy"
            )
        
        self.device = device
        self.sample_rate = sample_rate
        self.block_size = block_size
        
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._stream = None
        self._current_level: float = 0.0
        self._peak_level: float = 0.0
        self._clipping = False
        
        # Callback для обновления UI (опционально)
        self.on_level_update: Optional[Callable[[float, float, bool], None]] = None
    
    def _get_device_index(self) -> Optional[int]:
        """Получить индекс устройства из строки."""
        if self.device is None:
            return None

        # Если это число, проверяем что устройство существует
        try:
            # Формат macOS avfoundation: ":0" или "0"
            device_str = self.device.lstrip(':')
            device_idx = int(device_str)

            # Проверяем что устройство существует и поддерживает ввод
            try:
                device_info = sd.query_devices(device_idx)
                if device_info['max_input_channels'] > 0:
                    return device_idx
                else:
                    logger.debug(f"Устройство {device_idx} не поддерживает ввод, используем дефолтное")
                    return None
            except (ValueError, IndexError):
                logger.debug(f"Устройство {device_idx} не найдено, используем дефолтное")
                return None
        except ValueError:
            pass

        # Поиск устройства по имени
        try:
            devices = sd.query_devices()
            for i, dev in enumerate(devices):
                if self.device.lower() in dev['name'].lower():
                    if dev['max_input_channels'] > 0:
                        return i
        except Exception as e:
            logger.debug(f"Ошибка поиска устройства: {e}")

        return None
    
    def _audio_callback(self, indata, frames, time_info, status):
        """Callback для обработки аудио данных."""
        if status:
            logger.debug(f"Audio status: {status}")

        # DEBUG: логируем первый callback
        if not hasattr(self, '_first_callback_logged'):
            logger.debug(f"Audio callback: shape={indata.shape}, dtype={indata.dtype}, max={np.max(np.abs(indata)):.6f}")
            self._first_callback_logged = True

        # Вычисляем RMS уровень для всех каналов (берём максимум)
        # Это нужно для multi-channel устройств (Агрегатное устройство)
        if indata.ndim > 1:
            # Многоканальный ввод - берём максимум по всем каналам
            rms = np.sqrt(np.max([np.mean(indata[:, ch] ** 2) for ch in range(indata.shape[1])]))
        else:
            # Одноканальный ввод
            rms = np.sqrt(np.mean(indata ** 2))
        
        # Конвертируем в dB (с защитой от log(0))
        if rms > 0:
            db = 20 * np.log10(rms)
        else:
            db = -100
        
        # Нормализуем в диапазон 0-1 (от -60dB до 0dB)
        normalized = max(0, min(1, (db + 60) / 60))
        
        self._current_level = normalized
        self._peak_level = max(self._peak_level, normalized)
        self._clipping = rms > 0.95
        
        # Вызываем callback если задан
        if self.on_level_update:
            self.on_level_update(normalized, self._peak_level, self._clipping)
    
    def _render_level_bar(self) -> str:
        """Создать строку визуализации уровня."""
        level = self._current_level
        peak = self._peak_level
        
        # Основная полоса
        filled = int(level * self.BAR_WIDTH)
        bar = ""
        
        for i in range(self.BAR_WIDTH):
            if i < filled:
                # Цветовая градация по уровню
                if i > self.BAR_WIDTH * 0.8:
                    bar += "█"  # Высокий уровень
                elif i > self.BAR_WIDTH * 0.5:
                    bar += "▓"
                else:
                    bar += "░"
            else:
                bar += " "
        
        # Пиковый индикатор
        peak_pos = int(peak * self.BAR_WIDTH)
        if peak_pos < self.BAR_WIDTH:
            bar = bar[:peak_pos] + "│" + bar[peak_pos + 1:]
        
        # Индикатор клиппинга
        clip_indicator = "🔴" if self._clipping else "🟢"
        
        # dB значение
        if self._current_level > 0:
            db = -60 + (self._current_level * 60)
            db_str = f"{db:+.0f}dB"
        else:
            db_str = "-∞dB"
        
        return f"{clip_indicator} [{bar}] {db_str:>6}"
    
    def _monitor_loop(self):
        """Основной цикл мониторинга."""
        device_idx = self._get_device_index()

        try:
            # Определяем параметры устройства
            if device_idx is not None:
                device_info = sd.query_devices(device_idx)
                channels = min(device_info['max_input_channels'], 2)  # Максимум 2 канала для мониторинга
                # Используем нативную частоту устройства
                samplerate = int(device_info['default_samplerate'])
            else:
                channels = 1
                samplerate = self.sample_rate

            logger.debug(f"Запуск мониторинга на устройстве: {device_idx}, каналов: {channels}, частота: {samplerate}")

            self._stream = sd.InputStream(
                device=device_idx,
                channels=channels,
                samplerate=samplerate,
                blocksize=self.block_size,
                callback=self._audio_callback
            )
            
            self._stream.start()
            
            while self._running:
                # Рендерим уровень в консоль
                bar = self._render_level_bar()
                print(f"\r🎙️  {bar}", end="", flush=True)
                
                # Сбрасываем пик постепенно
                self._peak_level *= 0.95
                
                time.sleep(0.05)  # 20 FPS
            
        except Exception as e:
            logger.error(f"Ошибка мониторинга аудио: {e}")
        finally:
            if self._stream:
                self._stream.stop()
                self._stream.close()
                self._stream = None
            print()  # Новая строка после завершения
    
    def start(self) -> bool:
        """
        Запустить мониторинг.
        
        Returns:
            True если успешно запущен
        """
        if not HAS_SOUNDDEVICE:
            logger.warning("Мониторинг недоступен (sounddevice не установлен)")
            return False
        
        if self._running:
            return True
        
        self._running = True
        self._peak_level = 0.0
        
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
        
        logger.debug("Мониторинг уровня запущен")
        return True
    
    def stop(self):
        """Остановить мониторинг."""
        self._running = False
        
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        
        self._thread = None
        logger.debug("Мониторинг уровня остановлен")
    
    def is_available(self) -> bool:
        """Проверить доступность мониторинга."""
        return HAS_SOUNDDEVICE
    
    def get_current_level(self) -> float:
        """Получить текущий уровень (0.0 - 1.0)."""
        return self._current_level
    
    def get_peak_level(self) -> float:
        """Получить пиковый уровень (0.0 - 1.0)."""
        return self._peak_level


def list_audio_devices():
    """Вывести список доступных аудио устройств."""
    if not HAS_SOUNDDEVICE:
        print("sounddevice не установлен")
        return
    
    print("\n📱 Доступные аудио устройства:")
    print("-" * 60)
    
    devices = sd.query_devices()
    for i, dev in enumerate(devices):
        inputs = dev['max_input_channels']
        outputs = dev['max_output_channels']
        
        if inputs > 0:  # Показываем только устройства ввода
            default = " (default)" if i == sd.default.device[0] else ""
            print(f"  [{i}] {dev['name']}{default}")
            print(f"      Входы: {inputs}, Частота: {dev['default_samplerate']:.0f} Hz")
    
    print("-" * 60)

