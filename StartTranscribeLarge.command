#!/bin/bash
# --- РЕШЕНИЕ ПРОБЛЕМЫ ЗАВИСАНИЯ ---
export FASTER_CPU_THREADS=1

# --- Остальные настройки ---
export ASR_BACKEND=faster-whisper
export DEFAULT_MODEL=large-v2
export ASR_DEVICE=cpu
export FASTER_COMPUTE_TYPE=int8

# Указываем путь к основному скрипту Python
SCRIPT_PATH="$HOME/Scripts/meeting_transcriber.py"

echo "📂 Укажи путь к файлу для транскрипции:"
read FILE_PATH

# Запускаем транскрипцию
/usr/bin/python3 "$SCRIPT_PATH" transcribe "$FILE_PATH"

echo ""
echo "Нажми Enter, чтобы закрыть окно..."
read