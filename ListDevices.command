#!/bin/bash
cd "$(dirname "$0")"
/usr/bin/python3 -m meeting_transcriber list-devices
echo ""
read -n 1 -s -r -p "Нажми любую клавишу, чтобы закрыть окно…"
