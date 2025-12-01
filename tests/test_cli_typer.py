#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Тесты для нового Typer-based CLI.
"""

import pytest
from typer.testing import CliRunner
from unittest.mock import patch, MagicMock

from meeting_transcriber.cli_typer import app


runner = CliRunner()


class TestBlackholeStatus:
    """Тесты команды blackhole-status."""

    @patch("meeting_transcriber.cli_typer.get_blackhole_status")
    def test_blackhole_status_not_installed(self, mock_status):
        """Тест статуса когда BlackHole не установлен."""
        mock_status.return_value = {
            "platform": "Darwin",
            "is_macos": True,
            "blackhole_installed": False,
            "blackhole_device": None,
            "aggregate_device": None,
            "available_modes": ["mic"],
            "message": "❌ BlackHole не найден"
        }

        result = runner.invoke(app, ["blackhole-status"])

        assert result.exit_code == 0
        assert "BlackHole Status" in result.stdout
        assert "Darwin" in result.stdout
        assert "Not installed" in result.stdout
        assert "mic" in result.stdout

    @patch("meeting_transcriber.cli_typer.get_blackhole_status")
    def test_blackhole_status_installed_no_aggregate(self, mock_status):
        """Тест статуса когда BlackHole установлен, но Aggregate Device нет."""
        mock_status.return_value = {
            "platform": "Darwin",
            "is_macos": True,
            "blackhole_installed": True,
            "blackhole_device": {"index": 1, "name": "BlackHole 2ch"},
            "aggregate_device": None,
            "available_modes": ["mic", "system"],
            "message": "✅ BlackHole готов"
        }

        result = runner.invoke(app, ["blackhole-status"])

        assert result.exit_code == 0
        assert "BlackHole Status" in result.stdout
        assert "BlackHole 2ch" in result.stdout
        assert "Not configured" in result.stdout  # Aggregate device

    @patch("meeting_transcriber.cli_typer.get_blackhole_status")
    def test_blackhole_status_full_setup(self, mock_status):
        """Тест статуса когда всё настроено."""
        mock_status.return_value = {
            "platform": "Darwin",
            "is_macos": True,
            "blackhole_installed": True,
            "blackhole_device": {"index": 1, "name": "BlackHole 2ch"},
            "aggregate_device": {"index": 2, "name": "My Aggregate"},
            "available_modes": ["mic", "system", "both"],
            "message": "✅ BlackHole готов"
        }

        result = runner.invoke(app, ["blackhole-status"])

        assert result.exit_code == 0
        assert "BlackHole 2ch" in result.stdout
        assert "My Aggregate" in result.stdout
        assert "mic, system, both" in result.stdout

    def test_blackhole_status_setup_flag(self):
        """Тест флага --setup."""
        result = runner.invoke(app, ["blackhole-status", "--setup"])

        assert result.exit_code == 0
        assert "BlackHole Setup Guide" in result.stdout
        assert "brew install blackhole-2ch" in result.stdout
        assert "Multi-Output Device" in result.stdout
        assert "Aggregate Device" in result.stdout
        assert "Audio MIDI Setup" in result.stdout


class TestListDevices:
    """Тесты команды list-devices."""

    @patch("meeting_transcriber.cli_typer.MeetingRecorder")
    def test_list_devices_success(self, mock_recorder_class):
        """Тест успешного получения списка устройств."""
        mock_recorder = MagicMock()
        mock_recorder_class.return_value = mock_recorder

        result = runner.invoke(app, ["list-devices"])

        assert result.exit_code == 0
        assert "Audio Devices" in result.stdout
        mock_recorder_class.assert_called_once_with(enable_monitor=False)
        mock_recorder.list_devices.assert_called_once()

    @patch("meeting_transcriber.cli_typer.MeetingRecorder")
    def test_list_devices_error(self, mock_recorder_class):
        """Тест обработки ошибки при получении списка устройств."""
        mock_recorder = MagicMock()
        mock_recorder.list_devices.side_effect = FileNotFoundError("ffmpeg not found")
        mock_recorder_class.return_value = mock_recorder

        result = runner.invoke(app, ["list-devices"])

        assert result.exit_code == 1
        assert "ffmpeg not found" in result.stdout


class TestTranscribe:
    """Тесты команды transcribe."""

    @patch("meeting_transcriber.cli_typer.EnhancedTranscriber")
    @patch("meeting_transcriber.cli_typer.check_summarizer_available")
    def test_transcribe_basic(self, mock_summarizer, mock_transcriber_class, tmp_path):
        """Тест базовой транскрипции файла."""
        # Создаём тестовый файл
        test_file = tmp_path / "test.wav"
        test_file.write_text("fake audio")

        mock_summarizer.return_value = False
        mock_transcriber = MagicMock()
        mock_transcriber_class.return_value = mock_transcriber

        result = runner.invoke(app, ["transcribe", str(test_file)])

        assert result.exit_code == 0
        assert "Backend" in result.stdout
        assert "завершена" in result.stdout
        mock_transcriber.transcribe_files.assert_called_once()

    @patch("meeting_transcriber.cli_typer.EnhancedTranscriber")
    @patch("meeting_transcriber.cli_typer.check_summarizer_available")
    def test_transcribe_with_backend(self, mock_summarizer, mock_transcriber_class, tmp_path):
        """Тест транскрипции с указанием backend."""
        test_file = tmp_path / "test.wav"
        test_file.write_text("fake audio")

        mock_summarizer.return_value = False
        mock_transcriber = MagicMock()
        mock_transcriber_class.return_value = mock_transcriber

        result = runner.invoke(app, ["transcribe", str(test_file), "--backend", "groq"])

        assert result.exit_code == 0
        assert "Groq API" in result.stdout

    @patch("meeting_transcriber.cli_typer.EnhancedTranscriber")
    @patch("meeting_transcriber.cli_typer.check_summarizer_available")
    def test_transcribe_with_diarization(self, mock_summarizer, mock_transcriber_class, tmp_path):
        """Тест транскрипции с диаризацией."""
        test_file = tmp_path / "test.wav"
        test_file.write_text("fake audio")

        mock_summarizer.return_value = False
        mock_transcriber = MagicMock()
        mock_transcriber_class.return_value = mock_transcriber

        result = runner.invoke(app, ["transcribe", str(test_file), "--diarize", "--speakers", "2"])

        assert result.exit_code == 0
        assert "Диаризация" in result.stdout
        mock_transcriber_class.assert_called_once()
        call_args = mock_transcriber_class.call_args
        assert call_args.kwargs["diarize"] is True
        assert call_args.kwargs["min_speakers"] == 2
        assert call_args.kwargs["max_speakers"] == 2

    @patch("meeting_transcriber.cli_typer.EnhancedTranscriber")
    @patch("meeting_transcriber.cli_typer.check_summarizer_available")
    def test_transcribe_with_summarize(self, mock_summarizer, mock_transcriber_class, tmp_path):
        """Тест транскрипции с суммаризацией."""
        test_file = tmp_path / "test.wav"
        test_file.write_text("fake audio")

        mock_summarizer.return_value = True  # GROQ_API_KEY доступен
        mock_transcriber = MagicMock()
        mock_transcriber_class.return_value = mock_transcriber

        result = runner.invoke(app, ["transcribe", str(test_file), "--summarize", "--summary-lang", "en"])

        assert result.exit_code == 0
        assert "Суммаризация" in result.stdout
        call_args = mock_transcriber_class.call_args
        assert call_args.kwargs["summarize"] is True
        assert call_args.kwargs["summary_language"] == "en"

    @patch("meeting_transcriber.cli_typer.EnhancedTranscriber")
    def test_transcribe_multiple_files(self, mock_transcriber_class, tmp_path):
        """Тест транскрипции нескольких файлов."""
        file1 = tmp_path / "test1.wav"
        file2 = tmp_path / "test2.wav"
        file1.write_text("fake audio 1")
        file2.write_text("fake audio 2")

        mock_transcriber = MagicMock()
        mock_transcriber_class.return_value = mock_transcriber

        result = runner.invoke(app, ["transcribe", str(file1), str(file2)])

        assert result.exit_code == 0
        assert "2 файл(ов)" in result.stdout

    @patch("meeting_transcriber.cli_typer.EnhancedTranscriber")
    def test_transcribe_error(self, mock_transcriber_class, tmp_path):
        """Тест обработки ошибки при транскрипции."""
        test_file = tmp_path / "test.wav"
        test_file.write_text("fake audio")

        mock_transcriber = MagicMock()
        mock_transcriber.transcribe_files.side_effect = Exception("Transcription failed")
        mock_transcriber_class.return_value = mock_transcriber

        result = runner.invoke(app, ["transcribe", str(test_file)])

        assert result.exit_code == 1
        assert "Transcription failed" in result.stdout


class TestRecord:
    """Тесты команды record."""

    @patch("meeting_transcriber.cli_typer.MeetingRecorder")
    @patch("meeting_transcriber.cli_typer.resolve_device_for_mode")
    def test_record_basic_with_no_transcribe(self, mock_resolve, mock_recorder_class, tmp_path):
        """Тест базовой записи без транскрипции."""
        # Мокаем resolve_device_for_mode
        mock_resolve.return_value = (":0", "Микрофон (по умолчанию)")

        # Мокаем MeetingRecorder
        mock_recorder = MagicMock()
        test_file = tmp_path / "test_20250101_1200.wav"
        mock_recorder.record.return_value = [test_file]
        mock_recorder_class.return_value = mock_recorder

        result = runner.invoke(app, ["record", "TestMeeting", "--no-transcribe"])

        assert result.exit_code == 0
        assert "Запись сохранена" in result.stdout
        assert "Транскрипция пропущена" in result.stdout
        mock_recorder.record.assert_called_once()

    @patch("meeting_transcriber.cli_typer.MeetingRecorder")
    @patch("meeting_transcriber.cli_typer.resolve_device_for_mode")
    def test_record_with_capture_mode(self, mock_resolve, mock_recorder_class, tmp_path):
        """Тест записи с указанием capture-mode."""
        mock_resolve.return_value = (":1", "BlackHole 2ch")
        mock_recorder = MagicMock()
        test_file = tmp_path / "test.wav"
        mock_recorder.record.return_value = [test_file]
        mock_recorder_class.return_value = mock_recorder

        result = runner.invoke(app, ["record", "Meeting", "--capture-mode", "system", "--no-transcribe"])

        assert result.exit_code == 0
        assert "system" in result.stdout

    @patch("meeting_transcriber.cli_typer.MeetingRecorder")
    @patch("meeting_transcriber.cli_typer.resolve_device_for_mode")
    def test_record_with_filter_preset(self, mock_resolve, mock_recorder_class, tmp_path):
        """Тест записи с filter-preset."""
        mock_resolve.return_value = (":0", "Микрофон")
        mock_recorder = MagicMock()
        test_file = tmp_path / "test.wav"
        mock_recorder.record.return_value = [test_file]
        mock_recorder_class.return_value = mock_recorder

        result = runner.invoke(app, ["record", "Meeting", "--filter-preset", "soft", "--no-transcribe"])

        assert result.exit_code == 0
        assert "soft" in result.stdout

    @patch("meeting_transcriber.cli_typer.resolve_device_for_mode")
    def test_record_device_resolution_error(self, mock_resolve):
        """Тест обработки ошибки при резолюции устройства."""
        # Device не найден
        mock_resolve.return_value = (None, "BlackHole не найден")

        result = runner.invoke(app, ["record", "Meeting", "--no-transcribe"])

        assert result.exit_code == 1
        assert "BlackHole не найден" in result.stdout

    @patch("meeting_transcriber.cli_typer.MeetingRecorder")
    @patch("meeting_transcriber.cli_typer.resolve_device_for_mode")
    def test_record_recording_failed(self, mock_resolve, mock_recorder_class):
        """Тест обработки ошибки при записи."""
        mock_resolve.return_value = (":0", "Микрофон")
        mock_recorder = MagicMock()
        mock_recorder.record.return_value = []  # Пустой список = запись не удалась
        mock_recorder_class.return_value = mock_recorder

        result = runner.invoke(app, ["record", "Meeting", "--no-transcribe"])

        assert result.exit_code == 1
        assert "не удалась" in result.stdout

    @patch("meeting_transcriber.cli_typer.EnhancedTranscriber")
    @patch("meeting_transcriber.cli_typer.MeetingRecorder")
    @patch("meeting_transcriber.cli_typer.resolve_device_for_mode")
    def test_record_with_diarization(self, mock_resolve, mock_recorder_class, mock_transcriber_class, tmp_path):
        """Тест записи с диаризацией."""
        mock_resolve.return_value = (":0", "Микрофон")
        mock_recorder = MagicMock()
        test_file = tmp_path / "test.wav"
        mock_recorder.record.return_value = [test_file]
        mock_recorder_class.return_value = mock_recorder

        mock_transcriber = MagicMock()
        mock_transcriber_class.return_value = mock_transcriber

        result = runner.invoke(app, ["record", "Meeting", "--diarize", "--speakers", "3"])

        assert result.exit_code == 0
        assert "🎭 С диаризацией спикеров (3)" in result.stdout
        mock_transcriber_class.assert_called_once()
        call_args = mock_transcriber_class.call_args
        assert call_args.kwargs["diarize"] is True
        assert call_args.kwargs["min_speakers"] == 3
        assert call_args.kwargs["max_speakers"] == 3

    @patch("meeting_transcriber.cli_typer.EnhancedTranscriber")
    @patch("meeting_transcriber.cli_typer.check_summarizer_available")
    @patch("meeting_transcriber.cli_typer.MeetingRecorder")
    @patch("meeting_transcriber.cli_typer.resolve_device_for_mode")
    def test_record_with_summarize(self, mock_resolve, mock_recorder_class, mock_summarizer, mock_transcriber_class, tmp_path):
        """Тест записи с суммаризацией."""
        mock_resolve.return_value = (":0", "Микрофон")
        mock_recorder = MagicMock()
        test_file = tmp_path / "test.wav"
        mock_recorder.record.return_value = [test_file]
        mock_recorder_class.return_value = mock_recorder

        mock_summarizer.return_value = True  # GROQ_API_KEY доступен
        mock_transcriber = MagicMock()
        mock_transcriber_class.return_value = mock_transcriber

        result = runner.invoke(app, ["record", "Meeting", "--summarize"])

        assert result.exit_code == 0
        assert "🧠 С суммаризацией" in result.stdout
        call_args = mock_transcriber_class.call_args
        assert call_args.kwargs["summarize"] is True

    @patch("meeting_transcriber.cli_typer.EnhancedTranscriber")
    @patch("meeting_transcriber.cli_typer.MeetingRecorder")
    @patch("meeting_transcriber.cli_typer.resolve_device_for_mode")
    def test_record_transcription_fails_gracefully(self, mock_resolve, mock_recorder_class, mock_transcriber_class, tmp_path):
        """Тест: запись успешна, но транскрипция падает — команда не должна упасть."""
        mock_resolve.return_value = (":0", "Микрофон")
        mock_recorder = MagicMock()
        test_file = tmp_path / "test.wav"
        mock_recorder.record.return_value = [test_file]
        mock_recorder_class.return_value = mock_recorder

        mock_transcriber = MagicMock()
        mock_transcriber.transcribe_files.side_effect = Exception("Transcription failed")
        mock_transcriber_class.return_value = mock_transcriber

        result = runner.invoke(app, ["record", "Meeting"])

        assert result.exit_code == 0  # Должна завершиться успешно
        assert "Запись сохранена" in result.stdout
        assert "Transcription failed" in result.stdout
        assert "не удалась" in result.stdout


class TestVersion:
    """Тесты версии."""

    def test_version_flag(self):
        """Тест флага --version."""
        result = runner.invoke(app, ["--version"])

        assert result.exit_code == 0
        assert "Meeting Transcriber v5.6.0" in result.stdout
