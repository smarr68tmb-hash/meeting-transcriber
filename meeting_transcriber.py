#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Meeting Transcriber v5.0 - Обёртка для обратной совместимости.

Этот файл сохранён для совместимости со старыми скриптами.
Вся логика теперь в пакете meeting_transcriber/

Использование:
    python meeting_transcriber.py list-devices
    python meeting_transcriber.py record "Совещание" --device :0
    python meeting_transcriber.py transcribe file.wav

Или как модуль:
    python -m meeting_transcriber list-devices
"""

import sys
from pathlib import Path

# Добавляем путь к пакету если запускаем напрямую
script_dir = Path(__file__).parent
if str(script_dir) not in sys.path:
    sys.path.insert(0, str(script_dir))

from meeting_transcriber import main

if __name__ == "__main__":
    main()
