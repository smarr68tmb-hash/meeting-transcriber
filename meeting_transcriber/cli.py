#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CLI интерфейс для системы записи и транскрипции совещаний.
"""

import re
import sys
import datetime
import argparse
from pathlib import Path

from .config import Config
from .logging_setup import setup_logging, get_logger
from .recorder import MeetingRecorder
from .transcriber import EnhancedTranscriber

__version__ = "5.1.0"  # Добавлена диаризация спикеров


def main():
    """Главная точка входа CLI."""
    parser = argparse.ArgumentParser(
        description=f"Meeting Recorder & Transcriber v{__version__}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры:
  %(prog)s list-devices                        # Показать устройства
  %(prog)s record "Совещание" --device :0      # Записать и транскрибировать
  %(prog)s transcribe file.wav                 # Транскрибировать файл
  %(prog)s transcribe file.wav --diarize       # С определением спикеров
  %(prog)s transcribe file.wav -v              # С подробным выводом

Диаризация (определение спикеров):
  Требует установки: pip install whisperx
  Требует HuggingFace токен: export HF_TOKEN="hf_xxx"
  Лицензия pyannote: huggingface.co/pyannote/speaker-diarization-3.1
        """
    )
    
    # Версия
    parser.add_argument(
        "--version", "-V",
        action="version",
        version=f"%(prog)s {__version__}"
    )
    
    # Глобальные флаги логирования
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Подробный вывод (INFO уровень)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Отладочный вывод (DEBUG уровень)"
    )
    
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    # Команда: record
    p_rec = subparsers.add_parser(
        "record",
        help="Записать встречу и транскрибировать"
    )
    p_rec.add_argument(
        "name",
        help="Название/базовое имя файла"
    )
    p_rec.add_argument(
        "--device",
        required=True,
        help="ID устройства ввода (см. list-devices)"
    )
    p_rec.add_argument(
        "--no-transcribe",
        action="store_true",
        help="Только записать, без транскрипции"
    )
    p_rec.add_argument(
        "--diarize", "-d",
        action="store_true",
        help="Определять спикеров (требует whisperx и HF_TOKEN)"
    )
    p_rec.add_argument(
        "--speakers",
        type=int,
        metavar="N",
        help="Ожидаемое число спикеров (подсказка для диаризации)"
    )
    
    # Команда: transcribe
    p_tr = subparsers.add_parser(
        "transcribe",
        help="Транскрибировать готовые файлы"
    )
    p_tr.add_argument(
        "files",
        nargs='+',
        type=Path,
        help="Путь к аудио файлу(ам)"
    )
    p_tr.add_argument(
        "--diarize", "-d",
        action="store_true",
        help="Определять спикеров (требует whisperx и HF_TOKEN)"
    )
    p_tr.add_argument(
        "--speakers",
        type=int,
        metavar="N",
        help="Ожидаемое число спикеров (подсказка для диаризации)"
    )
    
    # Команда: list-devices
    subparsers.add_parser(
        "list-devices",
        help="Показать доступные аудио устройства"
    )
    
    args = parser.parse_args()
    
    # Настраиваем логирование
    logger = setup_logging(verbose=args.verbose, debug=args.debug)
    logger.debug(f"Запуск v{__version__} с аргументами: {args}")
    logger.debug(
        f"Конфигурация: model={Config.DEFAULT_MODEL}, "
        f"backend={Config.ASR_BACKEND}, device={Config.ASR_DEVICE}"
    )
    
    # Выполняем команду
    if args.command == "list-devices":
        MeetingRecorder().list_devices()
        sys.exit(0)
    
    if args.command == "record":
        _handle_record(args, logger)
    
    if args.command == "transcribe":
        _handle_transcribe(args, logger)


def _handle_record(args, logger):
    """Обработка команды record."""
    logger.info(f"Режим записи: '{args.name}', устройство: {args.device}")
    
    rec = MeetingRecorder()
    
    # Безопасное имя файла
    safe_name = re.sub(r'[^\w\s-]', '', args.name).strip().replace(' ', '_')
    base = Config.RECORDINGS_FOLDER / f"{safe_name}_{datetime.datetime.now():%Y%m%d_%H%M}"
    
    files = rec.record(base, args.device)
    
    if not files:
        logger.error("Запись не удалась")
        sys.exit(1)
    
    if args.no_transcribe:
        logger.info("Транскрипция пропущена (--no-transcribe)")
        print(f"\n✅ Запись сохранена: {files[0]}")
        sys.exit(0)
    
    diarize = getattr(args, 'diarize', False)
    speakers = getattr(args, 'speakers', None)
    logger.info(f"📝 Начинаем транскрипцию... (diarize={diarize}, speakers={speakers})")
    print("\n📝 Транскрипция..." + (" с диаризацией" if diarize else ""))
    
    # Передаём speakers как min и max для точного числа
    min_sp = max_sp = None
    if speakers is not None and speakers >= 1:
        min_sp = max_sp = speakers
    
    tr = EnhancedTranscriber(diarize=diarize, min_speakers=min_sp, max_speakers=max_sp)
    tr.transcribe_files(files)
    
    logger.info("Работа завершена успешно")
    sys.exit(0)


def _handle_transcribe(args, logger):
    """Обработка команды transcribe."""
    diarize = getattr(args, 'diarize', False)
    speakers = getattr(args, 'speakers', None)
    logger.info(f"Режим транскрипции: {len(args.files)} файл(ов), diarize={diarize}, speakers={speakers}")
    
    if diarize:
        print("🎭 Режим диаризации (определение спикеров)")
    
    # Передаём speakers как min и max для точного числа
    min_sp = max_sp = None
    if speakers is not None:
        if speakers < 1:
            print(f"⚠️ Некорректное число спикеров ({speakers}), игнорирую")
        else:
            min_sp = max_sp = speakers
    
    tr = EnhancedTranscriber(diarize=diarize, min_speakers=min_sp, max_speakers=max_sp)
    tr.transcribe_files(args.files)
    
    logger.info("Работа завершена")
    sys.exit(0)


if __name__ == "__main__":
    main()

