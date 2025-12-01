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


class TestVersion:
    """Тесты версии."""

    def test_version_flag(self):
        """Тест флага --version."""
        result = runner.invoke(app, ["--version"])

        assert result.exit_code == 0
        assert "Meeting Transcriber v5.6.0" in result.stdout
