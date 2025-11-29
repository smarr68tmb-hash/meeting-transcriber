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
from .summarizer import check_summarizer_available
from .blackhole import (
    CaptureMode, 
    resolve_device_for_mode, 
    print_blackhole_status,
    print_setup_instructions,
    get_blackhole_status
)

__version__ = "5.5.0"  # BlackHole интеграция для записи системного звука


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
  %(prog)s transcribe file.wav --backend groq  # Через Groq API (быстро!)
  %(prog)s transcribe file.wav --diarize       # С определением спикеров
  %(prog)s transcribe file.wav -v              # С подробным выводом

Backends (ASR_BACKEND или --backend):
  groq    - Groq API (бесплатно, очень быстро, требует GROQ_API_KEY)
  auto    - Groq с fallback на faster-whisper при ошибках
  faster  - Локальный faster-whisper (по умолчанию)
  whisper - Локальный openai-whisper
  whisperx - WhisperX с диаризацией

Groq API:
  1. Получите ключ: https://console.groq.com
  2. export GROQ_API_KEY="gsk_xxx"
  3. Лимит: 28,800 секунд (8 часов) аудио в день — бесплатно!

Суммаризация (--summarize):
  Генерирует краткое саммари, action items, анализ спикеров.
  Использует Groq LLM API (Llama 3) — тот же ключ GROQ_API_KEY.
  meeting-transcriber transcribe file.wav --summarize

Диаризация (определение спикеров):
  Требует установки: pip install whisperx
  Требует HuggingFace токен: export HF_TOKEN="hf_xxx"
  Лицензия pyannote: huggingface.co/pyannote/speaker-diarization-3.1

BlackHole (запись системного звука, macOS):
  Установка: brew install blackhole-2ch
  
  Режимы записи (--capture-mode):
    mic    - только микрофон (по умолчанию)
    system - только системный звук (голоса собеседников)
    both   - микрофон + системный (требует Aggregate Device)
  
  Примеры:
    %(prog)s record "Zoom" --capture-mode system    # записать собеседников
    %(prog)s record "Meet" --device blackhole       # авто-выбор BlackHole
    %(prog)s blackhole-status                       # проверить настройку
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
        help="ID устройства ввода (см. list-devices) или 'blackhole' для авто"
    )
    p_rec.add_argument(
        "--capture-mode", "-c",
        choices=["mic", "system", "both"],
        default=None,
        help="Режим захвата: mic (микрофон), system (BlackHole), both (mic+system)"
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
    p_rec.add_argument(
        "--no-monitor",
        action="store_true",
        help="Отключить мониторинг уровня звука"
    )
    p_rec.add_argument(
        "--summarize", "-s",
        action="store_true",
        help="Сгенерировать саммари после транскрипции"
    )
    p_rec.add_argument(
        "--no-summarize",
        action="store_true",
        help="Отключить суммаризацию (даже если AUTO_SUMMARIZE=1)"
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
    p_tr.add_argument(
        "--no-filter",
        action="store_true",
        help="Отключить фильтрацию галлюцинаций Whisper"
    )
    p_tr.add_argument(
        "--backend", "-b",
        choices=["groq", "auto", "faster", "whisper", "whisperx"],
        help="Backend для транскрипции (по умолчанию из ASR_BACKEND)"
    )
    p_tr.add_argument(
        "--no-fallback",
        action="store_true",
        help="Отключить fallback на локальный backend при ошибках API"
    )
    p_tr.add_argument(
        "--summarize", "-s",
        action="store_true",
        help="Сгенерировать саммари встречи через LLM (требует GROQ_API_KEY)"
    )
    p_tr.add_argument(
        "--no-summarize",
        action="store_true",
        help="Отключить суммаризацию (даже если AUTO_SUMMARIZE=1)"
    )
    p_tr.add_argument(
        "--summary-lang",
        choices=["ru", "en"],
        default="ru",
        help="Язык саммари (по умолчанию: ru)"
    )
    
    # Команда: list-devices
    subparsers.add_parser(
        "list-devices",
        help="Показать доступные аудио устройства"
    )
    
    # Команда: blackhole-status
    p_bh = subparsers.add_parser(
        "blackhole-status",
        help="Проверить статус BlackHole интеграции"
    )
    p_bh.add_argument(
        "--setup",
        action="store_true",
        help="Показать инструкции по настройке"
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
    
    if args.command == "blackhole-status":
        if getattr(args, 'setup', False):
            print_setup_instructions()
        else:
            print_blackhole_status()
        sys.exit(0)
    
    if args.command == "record":
        _handle_record(args, logger)
    
    if args.command == "transcribe":
        _handle_transcribe(args, logger)


def _handle_record(args, logger):
    """Обработка команды record."""
    
    # Определяем режим захвата и устройство
    capture_mode_str = getattr(args, 'capture_mode', None) or Config.CAPTURE_MODE
    capture_mode = CaptureMode(capture_mode_str)
    explicit_device = getattr(args, 'device', None)
    
    # Резолвим устройство
    device, device_desc = resolve_device_for_mode(capture_mode, explicit_device)
    
    if device is None:
        print(f"❌ {device_desc}")
        logger.error(f"Не удалось определить устройство: {device_desc}")
        sys.exit(1)
    
    logger.info(f"Режим записи: '{args.name}', режим: {capture_mode.value}, устройство: {device}")
    print(f"🎙️  {device_desc}")
    
    enable_monitor = not getattr(args, 'no_monitor', False)
    rec = MeetingRecorder(enable_monitor=enable_monitor)
    
    # Безопасное имя файла
    safe_name = re.sub(r'[^\w\s-]', '', args.name).strip().replace(' ', '_')
    base = Config.RECORDINGS_FOLDER / f"{safe_name}_{datetime.datetime.now():%Y%m%d_%H%M}"
    
    files = rec.record(base, device)
    
    if not files:
        logger.error("Запись не удалась")
        sys.exit(1)
    
    if args.no_transcribe:
        logger.info("Транскрипция пропущена (--no-transcribe)")
        print(f"\n✅ Запись сохранена: {files[0]}")
        sys.exit(0)
    
    diarize = getattr(args, 'diarize', False)
    speakers = getattr(args, 'speakers', None)
    
    # Разрешаем summarize: --no-summarize > --summarize > None (использовать AUTO_SUMMARIZE)
    if getattr(args, 'no_summarize', False):
        summarize = False
    elif getattr(args, 'summarize', False):
        summarize = True
    else:
        summarize = None
    
    # Проверяем доступность суммаризации ДО вывода сообщения
    will_summarize = summarize if summarize is not None else Config.AUTO_SUMMARIZE
    if will_summarize and not check_summarizer_available():
        print("⚠️ Суммаризация запрошена, но GROQ_API_KEY не установлен — отключена")
        logger.warning("Суммаризация отключена: GROQ_API_KEY не установлен")
        summarize = False
        will_summarize = False
    
    logger.info(f"📝 Начинаем транскрипцию... (diarize={diarize}, speakers={speakers}, summarize={summarize})")
    
    summarize_text = " + саммари" if will_summarize else ""
    print("\n📝 Транскрипция..." + (" с диаризацией" if diarize else "") + summarize_text)
    
    # Передаём speakers как min и max для точного числа
    min_sp = max_sp = None
    if speakers is not None and speakers >= 1:
        min_sp = max_sp = speakers
    
    tr = EnhancedTranscriber(
        diarize=diarize, 
        min_speakers=min_sp, 
        max_speakers=max_sp,
        summarize=summarize
    )
    tr.transcribe_files(files)
    
    logger.info("Работа завершена успешно")
    sys.exit(0)


def _handle_transcribe(args, logger):
    """Обработка команды transcribe."""
    diarize = getattr(args, 'diarize', False)
    speakers = getattr(args, 'speakers', None)
    no_filter = getattr(args, 'no_filter', False)
    backend = getattr(args, 'backend', None)
    no_fallback = getattr(args, 'no_fallback', False)
    summary_lang = getattr(args, 'summary_lang', 'ru')
    
    # Разрешаем summarize: --no-summarize > --summarize > None (использовать AUTO_SUMMARIZE)
    if getattr(args, 'no_summarize', False):
        summarize = False
    elif getattr(args, 'summarize', False):
        summarize = True
    else:
        summarize = None  # Использовать Config.AUTO_SUMMARIZE
    
    # Переопределяем backend если указан в аргументах
    if backend:
        import os
        os.environ['ASR_BACKEND'] = backend
        # Перезагружаем Config чтобы подхватить изменение
        Config.ASR_BACKEND = backend
    
    # Отключаем fallback если указано
    if no_fallback:
        import os
        os.environ['ASR_FALLBACK'] = '0'
        Config.ASR_FALLBACK = False
    
    effective_backend = backend or Config.ASR_BACKEND
    logger.info(
        f"Режим транскрипции: {len(args.files)} файл(ов), "
        f"backend={effective_backend}, diarize={diarize}, "
        f"speakers={speakers}, filter={not no_filter}, summarize={summarize}"
    )
    
    # Информируем пользователя
    backend_names = {
        'groq': '🚀 Groq API (облако)',
        'auto': '🔄 Auto (Groq → локальный)',
        'faster': '💻 faster-whisper (локально)',
        'whisper': '💻 openai-whisper (локально)',
        'whisperx': '🎭 WhisperX (локально, с диаризацией)',
    }
    print(f"Backend: {backend_names.get(effective_backend, effective_backend)}")
    
    if diarize:
        print("🎭 Режим диаризации (определение спикеров)")
    
    # Проверяем доступность суммаризации ДО вывода сообщения
    will_summarize = summarize if summarize is not None else Config.AUTO_SUMMARIZE
    if will_summarize and not check_summarizer_available():
        print("⚠️ Суммаризация запрошена, но GROQ_API_KEY не установлен — отключена")
        logger.warning("Суммаризация отключена: GROQ_API_KEY не установлен")
        summarize = False
    elif summarize is True:
        print(f"🧠 Суммаризация: включена (язык: {summary_lang})")
    elif summarize is False:
        print("🧠 Суммаризация: отключена")
    elif Config.AUTO_SUMMARIZE:
        print(f"🧠 Суммаризация: авто (язык: {summary_lang})")
    
    if no_filter:
        print("⚠️ Фильтрация галлюцинаций отключена")
    
    # Передаём speakers как min и max для точного числа
    min_sp = max_sp = None
    if speakers is not None:
        if speakers < 1:
            print(f"⚠️ Некорректное число спикеров ({speakers}), игнорирую")
        else:
            min_sp = max_sp = speakers
    
    tr = EnhancedTranscriber(
        diarize=diarize, 
        min_speakers=min_sp, 
        max_speakers=max_sp,
        filter_hallucinations=not no_filter,
        summarize=summarize,
        summary_language=summary_lang
    )
    tr.transcribe_files(args.files)
    
    logger.info("Работа завершена")
    sys.exit(0)


if __name__ == "__main__":
    main()

