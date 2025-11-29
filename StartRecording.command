#!/bin/bash
cd "$(dirname "$0")"

# Настройки ASR
export ASR_BACKEND=faster
export ASR_DEVICE=auto
export FASTER_COMPUTE_TYPE=int8

MEETING_NAME="Meeting_$(date +%Y%m%d_%H%M)"

echo "🎙 Запускаю запись..."
echo "MEETING_NAME=$MEETING_NAME"
echo ""

/usr/bin/python3 -m meeting_transcriber record "$MEETING_NAME" --device ":0"

echo ""
echo "✅ Скрипт дошёл до конца"
echo ""
echo "Нажми любую клавишу, чтобы закрыть окно..."
read -n 1 -s -r
