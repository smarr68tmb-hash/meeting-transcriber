#!/bin/bash
cd "$(dirname "$0")"

# --- РЕШЕНИЕ ПРОБЛЕМЫ ЗАВИСАНИЯ ---
export FASTER_CPU_THREADS=1

# --- Настройки ASR ---
export ASR_BACKEND=faster
export WHISPER_MODEL=medium
export ASR_DEVICE=cpu
export FASTER_COMPUTE_TYPE=int8

echo "📂 Укажи путь к файлу для транскрипции:"
read FILE_PATH

# Запускаем транскрипцию с verbose режимом
/usr/bin/python3 -m meeting_transcriber -v transcribe "$FILE_PATH"

echo ""
echo "Нажми Enter, чтобы закрыть окно..."
read
