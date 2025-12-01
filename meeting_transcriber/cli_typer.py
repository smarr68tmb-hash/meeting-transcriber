#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modern Typer-based CLI interface for Meeting Transcriber.

This is a new CLI implementation using Typer and Rich for better UX.
Eventually will replace the old argparse-based CLI.
"""

import os
import sys
from pathlib import Path
import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .blackhole import (
    get_blackhole_status,
)
from .recorder import MeetingRecorder
from .transcriber import EnhancedTranscriber
from .summarizer import check_summarizer_available
from .config import Config

__version__ = "5.6.0"

# Initialize Typer app and Rich console
app = typer.Typer(
    name="meeting-transcriber",
    help="Meeting Recorder & Transcriber with AI-powered transcription",
    add_completion=False,
)
console = Console()


@app.command(name="list-devices")
def list_devices():
    """
    Показать список доступных аудио устройств.

    Использует ffmpeg для получения списка устройств записи.
    """
    console.print()
    console.print(
        Panel(
            "[cyan]Получение списка аудио устройств...[/cyan]",
            title="🎤 Audio Devices",
            border_style="cyan"
        )
    )
    console.print()

    try:
        recorder = MeetingRecorder(enable_monitor=False)
        recorder.list_devices()
    except Exception as e:
        console.print(f"[red]❌ Ошибка:[/red] {e}")
        raise typer.Exit(code=1)


@app.command(name="transcribe")
def transcribe(
    files: list[Path] = typer.Argument(
        ...,
        help="Путь к аудио файлу(ам)",
        exists=True,
        file_okay=True,
        dir_okay=False,
    ),
    backend: str = typer.Option(
        None,
        "--backend", "-b",
        help="Backend для транскрипции (groq/auto/faster/whisper/whisperx)"
    ),
    diarize: bool = typer.Option(
        False,
        "--diarize", "-d",
        help="Определять спикеров (требует whisperx и HF_TOKEN)"
    ),
    speakers: int = typer.Option(
        None,
        "--speakers",
        help="Ожидаемое число спикеров (подсказка для диаризации)"
    ),
    no_filter: bool = typer.Option(
        False,
        "--no-filter",
        help="Отключить фильтрацию галлюцинаций Whisper"
    ),
    no_fallback: bool = typer.Option(
        False,
        "--no-fallback",
        help="Отключить fallback на локальный backend при ошибках API"
    ),
    summarize: bool = typer.Option(
        False,
        "--summarize", "-s",
        help="Сгенерировать саммари встречи через LLM (требует GROQ_API_KEY)"
    ),
    no_summarize: bool = typer.Option(
        False,
        "--no-summarize",
        help="Отключить суммаризацию (даже если AUTO_SUMMARIZE=1)"
    ),
    summary_lang: str = typer.Option(
        "ru",
        "--summary-lang",
        help="Язык саммари (ru/en)"
    ),
):
    """
    Транскрибировать готовые аудио файлы.

    Поддерживает несколько backend'ов:
    - groq: Groq API (облако, быстро, бесплатно)
    - auto: Groq с fallback на faster-whisper
    - faster: faster-whisper (локально)
    - whisper: openai-whisper (локально)
    - whisperx: WhisperX с диаризацией (локально)
    """
    # Разрешаем summarize: --no-summarize > --summarize > None
    if no_summarize:
        summarize_final = False
    elif summarize:
        summarize_final = True
    else:
        summarize_final = None  # Использовать Config.AUTO_SUMMARIZE

    # Переопределяем backend если указан
    if backend:
        os.environ['ASR_BACKEND'] = backend
        Config.ASR_BACKEND = backend

    # Отключаем fallback если указано
    if no_fallback:
        os.environ['ASR_FALLBACK'] = '0'
        Config.ASR_FALLBACK = False

    effective_backend = backend or Config.ASR_BACKEND

    # Красивый вывод информации о режиме
    console.print()

    # Backend info
    backend_info = {
        'groq': '🚀 Groq API (облако)',
        'auto': '🔄 Auto (Groq → локальный)',
        'faster': '💻 faster-whisper (локально)',
        'whisper': '💻 openai-whisper (локально)',
        'whisperx': '🎭 WhisperX (локально, с диаризацией)',
    }

    info_table = Table(show_header=False, box=None, padding=(0, 2))
    info_table.add_column("Label", style="cyan")
    info_table.add_column("Value", style="white")

    info_table.add_row("Backend", backend_info.get(effective_backend, effective_backend))
    info_table.add_row("Files", f"{len(files)} файл(ов)")

    if diarize:
        info_table.add_row("Диаризация", "🎭 Включена")
        if speakers:
            info_table.add_row("  Спикеров", str(speakers))

    # Проверяем доступность суммаризации
    will_summarize = summarize_final if summarize_final is not None else Config.AUTO_SUMMARIZE
    if will_summarize and not check_summarizer_available():
        console.print("[yellow]⚠️  Суммаризация запрошена, но GROQ_API_KEY не установлен — отключена[/yellow]")
        summarize_final = False
    elif summarize_final is True:
        info_table.add_row("Суммаризация", f"🧠 Включена (язык: {summary_lang})")
    elif summarize_final is False:
        info_table.add_row("Суммаризация", "🧠 Отключена")
    elif Config.AUTO_SUMMARIZE:
        info_table.add_row("Суммаризация", f"🧠 Авто (язык: {summary_lang})")

    if no_filter:
        info_table.add_row("Фильтрация", "⚠️  Отключена")

    console.print(info_table)
    console.print()

    # Передаём speakers как min и max для точного числа
    min_sp = max_sp = None
    if speakers is not None:
        if speakers < 1:
            console.print(f"[yellow]⚠️  Некорректное число спикеров ({speakers}), игнорирую[/yellow]")
        else:
            min_sp = max_sp = speakers

    try:
        tr = EnhancedTranscriber(
            diarize=diarize,
            min_speakers=min_sp,
            max_speakers=max_sp,
            filter_hallucinations=not no_filter,
            summarize=summarize_final,
            summary_language=summary_lang
        )
        tr.transcribe_files(files)

        console.print()
        console.print("[green]✅ Транскрипция завершена[/green]")
    except Exception as e:
        console.print()
        console.print(f"[red]❌ Ошибка:[/red] {e}")
        raise typer.Exit(code=1)


@app.command(name="blackhole-status")
def blackhole_status(
    setup: bool = typer.Option(
        False,
        "--setup",
        help="Показать подробные инструкции по настройке BlackHole"
    )
):
    """
    Проверить статус BlackHole интеграции.

    BlackHole позволяет записывать системный звук на macOS
    (Zoom, Google Meet, Teams и т.д.).
    """
    if setup:
        _print_setup_instructions()
    else:
        _print_blackhole_status()


def _print_blackhole_status():
    """Вывести статус BlackHole с красивым форматированием через Rich."""
    status = get_blackhole_status()

    # Создаём таблицу статуса
    table = Table(title="🔊 BlackHole Status", show_header=False, box=None)
    table.add_column("Key", style="cyan", width=20)
    table.add_column("Value", style="white")

    # Платформа
    table.add_row("Platform", status["platform"])

    # Статус с эмодзи
    status_text = Text()
    if status.get("blackhole_installed"):
        status_text.append("✅ ", style="green")
        status_text.append("Installed")
    else:
        status_text.append("❌ ", style="red")
        status_text.append("Not installed")
    table.add_row("BlackHole", status_text)

    # BlackHole device
    if status.get("blackhole_device"):
        bh = status["blackhole_device"]
        table.add_row("  Device", f":{bh['index']} ({bh['name']})")

    # Aggregate device
    if status.get("aggregate_device"):
        agg = status["aggregate_device"]
        agg_text = Text()
        agg_text.append("✅ ", style="green")
        agg_text.append(f":{agg['index']} ({agg['name']})")
        table.add_row("Aggregate Device", agg_text)
    else:
        agg_text = Text()
        agg_text.append("⚠️  ", style="yellow")
        agg_text.append("Not configured")
        table.add_row("Aggregate Device", agg_text)

    # Available modes
    modes = ", ".join(status["available_modes"])
    table.add_row("Available modes", modes)

    console.print(table)
    console.print()

    # Инструкции по установке
    if not status.get("blackhole_installed"):
        install_panel = Panel(
            "[yellow]brew install blackhole-2ch[/yellow]\n"
            "или: https://existential.audio/blackhole/",
            title="📦 Установка BlackHole",
            border_style="yellow"
        )
        console.print(install_panel)
        console.print()

    # Инструкции по настройке Aggregate Device
    if status.get("blackhole_installed") and not status.get("aggregate_device"):
        aggregate_panel = Panel(
            "1. Откройте 'Audio MIDI Setup' (Spotlight → Audio MIDI)\n"
            "2. Нажмите '+' → 'Create Aggregate Device'\n"
            "3. Включите галочки: микрофон + BlackHole 2ch\n"
            "4. Используйте: [cyan]--capture-mode both[/cyan]",
            title="🔧 Настройка записи Mic + System",
            border_style="blue"
        )
        console.print(aggregate_panel)
        console.print()

    # Важные советы по качеству
    if status.get("aggregate_device"):
        quality_panel = Panel(
            "• [yellow]Clock Source[/yellow]: выберите 'Built-in Microphone'\n"
            "• [yellow]Drift Correction[/yellow]: включите ТОЛЬКО для BlackHole 2ch",
            title="⚠️  Важно для качества звука (избежание 'квакания')",
            border_style="yellow"
        )
        console.print(quality_panel)


def _print_setup_instructions():
    """Вывести подробные инструкции по настройке BlackHole."""

    # Заголовок
    console.print(
        Panel(
            "[bold cyan]BlackHole позволяет записывать системный звук на macOS.[/bold cyan]\n"
            "Это полезно для транскрипции Zoom, Google Meet, Teams и др.",
            title="🎧 BlackHole Setup Guide",
            border_style="cyan"
        )
    )
    console.print()

    # Установка
    install_panel = Panel(
        "[yellow]brew install blackhole-2ch[/yellow]",
        title="📦 УСТАНОВКА",
        border_style="yellow"
    )
    console.print(install_panel)
    console.print()

    # Настройка Multi-Output
    multi_output_panel = Panel(
        "1. Откройте [cyan]\"Audio MIDI Setup\"[/cyan] (через Spotlight)\n\n"
        "2. Создайте [yellow]Multi-Output Device[/yellow]:\n"
        "   • Нажмите \"+\" → \"Create Multi-Output Device\"\n"
        "   • Включите: Built-in Output ✓ + BlackHole 2ch ✓\n"
        "   • Это позволит слышать звук И записывать его",
        title="🔧 НАСТРОЙКА: Multi-Output Device",
        border_style="blue"
    )
    console.print(multi_output_panel)
    console.print()

    # Настройка Aggregate
    aggregate_panel = Panel(
        "1. Нажмите \"+\" → \"Create Aggregate Device\"\n"
        "2. Включите: Built-in Microphone ✓ + BlackHole 2ch ✓\n\n"
        "[yellow]⚠️  ВАЖНО для избежания артефактов (\"квакания\"):[/yellow]\n"
        "   • [cyan]Clock Source[/cyan]: выберите \"Built-in Microphone\"\n"
        "   • [cyan]Drift Correction[/cyan]: включите ТОЛЬКО для BlackHole 2ch\n\n"
        "[green]✅ После этого используйте:[/green] [cyan]--capture-mode both[/cyan]",
        title="🔧 НАСТРОЙКА: Aggregate Device (для mic + system)",
        border_style="blue"
    )
    console.print(aggregate_panel)
    console.print()

    # Использование
    usage_table = Table(title="📝 Использование", show_header=True, border_style="green")
    usage_table.add_column("Режим", style="cyan", width=15)
    usage_table.add_column("Команда", style="yellow")
    usage_table.add_column("Описание", style="white")

    usage_table.add_row(
        "Только микрофон",
        "record \"Meeting\" --capture-mode mic",
        "Запись вашего голоса"
    )
    usage_table.add_row(
        "Системный звук",
        "record \"Meeting\" --capture-mode system",
        "Запись собеседников"
    )
    usage_table.add_row(
        "Mic + System",
        "record \"Meeting\" --capture-mode both",
        "Запись всех (рекомендуется)"
    )

    console.print(usage_table)
    console.print()

    # Системный звук через Multi-Output
    system_audio_panel = Panel(
        "1. System Preferences → Sound → Output\n"
        "2. Выберите созданный [cyan]Multi-Output Device[/cyan]\n"
        "3. Теперь звук из Zoom/Meet будет идти через BlackHole",
        title="🔊 Настройка системного звука",
        border_style="magenta"
    )
    console.print(system_audio_panel)


def version_callback(value: bool):
    """Print version and exit."""
    if value:
        console.print(f"Meeting Transcriber v{__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        None,
        "--version",
        "-v",
        callback=version_callback,
        is_eager=True,
        help="Show version and exit"
    )
):
    """
    Meeting Recorder & Transcriber v5.6.0

    Система записи и транскрипции совещаний с AI.
    """
    pass


if __name__ == "__main__":
    app()
