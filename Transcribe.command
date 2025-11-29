#!/bin/bash
# ============================================
# 📝 Транскрипция аудио файла
# ============================================
# Использование:
#   1. Двойной клик → ввести путь к файлу
#   2. Перетащить аудио файл на эту иконку
# ============================================

cd "$(dirname "$0")"

# Настройки
export FASTER_CPU_THREADS=1
export ASR_BACKEND=faster
export WHISPER_MODEL=medium
export ASR_DEVICE=auto
export FASTER_COMPUTE_TYPE=int8

# Определяем файл: либо аргумент (drag & drop), либо запрашиваем
if [ -n "$1" ]; then
    FILE_PATH="$1"
    echo "📂 Файл: $FILE_PATH"
else
    echo "╔══════════════════════════════════════════════════╗"
    echo "║        📝 ТРАНСКРИПЦИЯ АУДИО ФАЙЛА               ║"
    echo "╠══════════════════════════════════════════════════╣"
    echo "║  Совет: перетащи файл на иконку этого скрипта    ║"
    echo "║  для автоматического запуска                     ║"
    echo "╚══════════════════════════════════════════════════╝"
    echo ""
    echo "Введи путь к файлу (или перетащи сюда):"
    read -r FILE_PATH
fi

# Убираем кавычки если есть
FILE_PATH="${FILE_PATH%\"}"
FILE_PATH="${FILE_PATH#\"}"
FILE_PATH="${FILE_PATH%\'}"
FILE_PATH="${FILE_PATH#\'}"

# Проверяем существование файла
if [ ! -f "$FILE_PATH" ]; then
    echo ""
    echo "❌ Файл не найден: $FILE_PATH"
    echo ""
    echo "Нажми любую клавишу..."
    read -n 1 -s -r
    exit 1
fi

echo ""
echo "🎙️  Начинаю транскрипцию..."
echo "📁 Файл: $(basename "$FILE_PATH")"
echo ""

# Запускаем транскрипцию
/usr/bin/python3 -m meeting_transcriber transcribe "$FILE_PATH"

echo ""
echo "════════════════════════════════════════════════════"
echo "✅ Готово! Транскрипт сохранён в ~/Meeting_Transcripts/"
echo "════════════════════════════════════════════════════"
echo ""
echo "Нажми любую клавишу, чтобы закрыть..."
read -n 1 -s -r

