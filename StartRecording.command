#!/bin/bash
export ASR_BACKEND=faster-whisper
export FASTER_DEVICE=auto
export FASTER_COMPUTE_TYPE=int8

SCRIPT_PATH="$HOME/Scripts/meeting_transcriber.py"
MEETING_NAME="Meeting_$(date +%Y%m%d_%H%M)"

echo "🎙 Запускаю запись..."
echo "SCRIPT_PATH=$SCRIPT_PATH"
echo "MEETING_NAME=$MEETING_NAME"

/usr/bin/python3 "$SCRIPT_PATH" record "$MEETING_NAME" --device ":0"

echo "✅ Скрипт дошёл до конца"
echo ""
echo "Нажми любую клавишу, чтобы закрыть окно..."
read
