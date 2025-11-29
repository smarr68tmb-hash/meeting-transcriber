# -*- coding: utf-8 -*-
"""
Модуль записи аудио с микрофона через ffmpeg.
"""

import time
import subprocess
from pathlib import Path
from typing import Optional, List

from .config import Config
from .utils import get_platform_config, ffprobe_ok
from .logging_setup import get_logger
from .audio_monitor import AudioLevelMonitor

logger = get_logger()


class MeetingRecorder:
    """Класс для записи аудио с микрофона."""
    
    def __init__(self, enable_monitor: bool = True):
        """
        Инициализация рекордера.
        
        Args:
            enable_monitor: Включить мониторинг уровня звука
        """
        Config.ensure_directories()
        self.platform_config = get_platform_config()
        self.recording_process: Optional[subprocess.Popen] = None
        self.enable_monitor = enable_monitor
        self._audio_monitor: Optional[AudioLevelMonitor] = None

    def list_devices(self) -> None:
        """Показать список доступных аудио устройств."""
        fmt = self.platform_config['format']
        cmd = [
            'ffmpeg', '-f', fmt,
            *self.platform_config['list_cmd'],
            '-i', self.platform_config['dummy']
        ]
        logger.info(f"🔍 Выполняю: {' '.join(cmd)}")
        
        try:
            res = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='ignore'
            )
            output = (res.stderr or '') + "\n" + (res.stdout or '')
            print("\n" + output)  # Вывод устройств всегда в консоль
            logger.debug("Список устройств получен")
        except Exception as e:
            logger.error(f"Не удалось получить список устройств: {e}")

    def _record_probe(self, device: str) -> bool:
        """
        Пробная запись для проверки устройства.
        
        Args:
            device: ID устройства
            
        Returns:
            True если устройство работает, False иначе
        """
        if Config.PRE_RECORD_PROBE <= 0:
            logger.debug("Пробная запись отключена (PRE_RECORD_PROBE=0)")
            return True
        
        logger.info(f"🔎 Пробная запись ({Config.PRE_RECORD_PROBE} сек) — проверка устройства '{device}'...")
        
        probe_file = Config.LOGS_FOLDER / "_probe.wav"
        cmd = [
            'ffmpeg', '-y', '-hide_banner', '-nostdin',
            '-f', self.platform_config['format'],
            '-i', device,
            '-t', str(Config.PRE_RECORD_PROBE),
            '-c:a', 'pcm_s16le',
            str(probe_file)
        ]
        log_file = Config.LOGS_FOLDER / "_probe.log"
        logger.debug(f"Команда пробной записи: {' '.join(cmd)}")
        
        try:
            with open(log_file, 'w', encoding='utf-8') as log:
                p = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
                p.wait()
            
            ok = (p.returncode == 0) and ffprobe_ok(probe_file)
            if ok:
                logger.info("✅ Проба успешна")
            else:
                logger.error(f"Пробная запись не удалась. См. лог: {log_file}")
            return ok
        finally:
            if probe_file.exists():
                try:
                    probe_file.unlink()
                    logger.debug("Временный файл пробы удалён")
                except OSError as e:
                    logger.warning(f"Не удалось удалить временный файл пробы: {e}")

    def record(self, output_file: Path, device: str) -> Optional[List[Path]]:
        """
        Записать аудио с устройства.
        
        Args:
            output_file: Базовый путь для выходного файла (без расширения)
            device: ID устройства ввода
            
        Returns:
            Список путей к записанным файлам или None при ошибке
        """
        if not self._record_probe(device):
            return None
        
        # Определяем формат и путь
        suffix = '.wav' if Config.DEFAULT_FORMAT == 'wav' else '.flac'
        output_path = output_file.with_suffix(suffix)
        codec = 'pcm_s16le' if Config.DEFAULT_FORMAT == 'wav' else 'flac'
        
        # Собираем команду ffmpeg
        cmd = [
            'ffmpeg', '-y', '-hide_banner', '-nostdin',
            '-f', self.platform_config['format'],
            '-i', device,
            '-vn', '-ar', Config.DEFAULT_SAMPLE_RATE,
            '-ac', Config.DEFAULT_CHANNELS,
            '-acodec', codec
        ]
        
        if Config.DEFAULT_FORMAT == 'flac':
            cmd += ['-compression_level', Config.FLAC_LEVEL]
        else:
            cmd += ['-rf64', 'auto']
        
        cmd += ['-af', Config.VOICE_FILTERS, str(output_path)]
        
        log_file = Config.LOGS_FOLDER / f"{output_file.stem}.log"
        logger.debug(f"Команда записи: {' '.join(cmd)}")
        
        # Красивый вывод в консоль
        print("\n" + "=" * 52)
        print(f"🔴 ЗАПИСЬ НАЧАТА -> {output_path.name}")
        print("⏹  Остановка: Ctrl+C")
        print("=" * 52)
        
        logger.info(f"Запись начата: {output_path.name}")
        start = time.time()
        
        # Запускаем мониторинг уровня звука
        if self.enable_monitor:
            self._audio_monitor = AudioLevelMonitor(device=device)
            if self._audio_monitor.is_available():
                self._audio_monitor.start()
                print("🎙️  Мониторинг уровня: активен")
            else:
                print("⚠️  Мониторинг уровня недоступен (установите: pip install sounddevice numpy)")
        
        try:
            with open(log_file, 'w', encoding='utf-8') as log:
                self.recording_process = subprocess.Popen(
                    cmd, stdout=log, stderr=subprocess.STDOUT
                )
                while self.recording_process.poll() is None:
                    elapsed = int(time.time() - start)
                    # Если монитор активен, он сам выводит уровень
                    # Иначе показываем только время
                    monitor_active = self._audio_monitor and self._audio_monitor.is_available()
                    if not monitor_active:
                        print(f"\r⏱  Длительность: {elapsed // 60:02d}:{elapsed % 60:02d}", 
                              end="", flush=True)
                    time.sleep(0.1 if monitor_active else 1)
        except KeyboardInterrupt:
            print("\n⏸ Останавливаю запись...")
            logger.info("Запись остановлена пользователем (Ctrl+C)")
            if self.recording_process:
                self.recording_process.terminate()
                self.recording_process.wait()
        except Exception as e:
            logger.error(f"Ошибка записи: {e}", exc_info=True)
            return None
        finally:
            # Останавливаем монитор
            if self._audio_monitor:
                self._audio_monitor.stop()
                self._audio_monitor = None
        
        duration = time.time() - start
        logger.info(f"Запись завершена: {output_path.name}, длительность: {duration / 60:.1f} мин")
        print("\n✅ Запись завершена")
        
        if output_path.exists() and ffprobe_ok(output_path):
            file_size = output_path.stat().st_size / (1024 * 1024)
            logger.info(f"Файл создан: {output_path}, размер: {file_size:.1f} MB")
            return [output_path]
        
        logger.error(f"Файл записи не создан или повреждён. Лог: {log_file}")
        return None

