# -*- coding: utf-8 -*-
"""
Groq API backend для транскрипции аудио.

Groq предоставляет бесплатный и очень быстрый Whisper API.
Лимиты: 28,800 секунд аудио в день (8 часов).

Использование:
    export GROQ_API_KEY="gsk_xxx"
    ASR_BACKEND=groq meeting-transcriber transcribe file.wav
"""

import os
import time
import json
import tempfile
import subprocess
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError
from http.client import HTTPResponse

from .config import Config
from .logging_setup import get_logger

logger = get_logger()

# Groq API endpoints
GROQ_API_URL = "https://api.groq.com/openai/v1/audio/transcriptions"


class GroqRateLimitError(Exception):
    """Исключение при превышении лимитов Groq API."""
    pass


class GroqAPIError(Exception):
    """Общая ошибка Groq API."""
    pass


class GroqTranscriber:
    """
    Транскрибер на базе Groq API.
    
    Преимущества:
    - Очень быстрая транскрипция (30 сек аудио ≈ 1 сек обработки)
    - Бесплатно до 28,800 секунд в день
    - whisper-large-v3-turbo качество
    
    Ограничения:
    - Максимальный размер файла: 25MB
    - Требует интернет-соединение
    """
    
    def __init__(self):
        self.api_key = Config.GROQ_API_KEY
        self.model = Config.GROQ_MODEL
        self.timeout = Config.GROQ_TIMEOUT
        
        if not self.api_key:
            logger.warning(
                "GROQ_API_KEY не установлен. Groq backend недоступен. "
                "Получите ключ: https://console.groq.com"
            )
    
    def is_available(self) -> bool:
        """Проверить доступность Groq API."""
        return bool(self.api_key)
    
    def _check_file_size(self, file_path: Path) -> bool:
        """Проверить размер файла (лимит 25MB)."""
        size = file_path.stat().st_size
        if size > Config.GROQ_MAX_FILE_SIZE:
            logger.warning(
                f"Файл слишком большой для Groq API: {size / 1024 / 1024:.1f}MB > 25MB"
            )
            return False
        return True
    
    def _prepare_audio(self, audio_path: Path) -> Tuple[Path, bool]:
        """
        Подготовить аудио для Groq API.
        
        Groq принимает: mp3, mp4, mpeg, mpga, m4a, wav, webm
        Конвертируем в mp3 для уменьшения размера.
        
        Returns:
            Tuple[путь к файлу, нужно ли удалять после]
        """
        # Если файл уже подходящего формата и размера
        suffix = audio_path.suffix.lower()
        if suffix in ('.mp3', '.m4a', '.webm') and self._check_file_size(audio_path):
            return audio_path, False
        
        # Конвертируем в mp3 для уменьшения размера
        logger.info("Конвертация в mp3 для Groq API...")
        
        temp_file = Path(tempfile.gettempdir()) / f"groq_upload_{int(time.time())}.mp3"
        
        try:
            subprocess.run([
                "ffmpeg", "-y", "-i", str(audio_path),
                "-ar", "16000",       # 16kHz достаточно для speech
                "-ac", "1",           # mono
                "-b:a", "64k",        # 64kbps достаточно для речи
                "-nostdin",
                str(temp_file)
            ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            if not self._check_file_size(temp_file):
                # Если всё ещё большой, пробуем более агрессивное сжатие
                subprocess.run([
                    "ffmpeg", "-y", "-i", str(audio_path),
                    "-ar", "16000", "-ac", "1", "-b:a", "32k",
                    "-nostdin",
                    str(temp_file)
                ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            return temp_file, True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Ошибка конвертации: {e}")
            # Пробуем отправить как есть
            return audio_path, False
    
    def _create_multipart_data(
        self, 
        file_path: Path, 
        language: Optional[str]
    ) -> Tuple[bytes, str]:
        """
        Создать multipart/form-data для запроса.
        
        Returns:
            Tuple[body bytes, content-type header]
        """
        boundary = f"----WebKitFormBoundary{int(time.time() * 1000)}"
        
        lines = []
        
        # Файл
        lines.append(f"--{boundary}")
        lines.append(f'Content-Disposition: form-data; name="file"; filename="{file_path.name}"')
        lines.append("Content-Type: audio/mpeg")
        lines.append("")
        
        with open(file_path, "rb") as f:
            file_data = f.read()
        
        # Модель
        lines_after = []
        lines_after.append(f"--{boundary}")
        lines_after.append('Content-Disposition: form-data; name="model"')
        lines_after.append("")
        lines_after.append(self.model)
        
        # Язык (опционально)
        if language:
            lines_after.append(f"--{boundary}")
            lines_after.append('Content-Disposition: form-data; name="language"')
            lines_after.append("")
            lines_after.append(language)
        
        # Response format
        lines_after.append(f"--{boundary}")
        lines_after.append('Content-Disposition: form-data; name="response_format"')
        lines_after.append("")
        lines_after.append("verbose_json")
        
        # Timestamp granularities
        lines_after.append(f"--{boundary}")
        lines_after.append('Content-Disposition: form-data; name="timestamp_granularities[]"')
        lines_after.append("")
        lines_after.append("segment")
        
        lines_after.append(f"--{boundary}--")
        lines_after.append("")
        
        # Собираем body
        header_part = "\r\n".join(lines).encode('utf-8') + b"\r\n"
        footer_part = b"\r\n" + "\r\n".join(lines_after).encode('utf-8')
        
        body = header_part + file_data + footer_part
        content_type = f"multipart/form-data; boundary={boundary}"
        
        return body, content_type
    
    def transcribe(
        self,
        audio_path: Path,
        language: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Транскрибировать аудио через Groq API.
        
        Args:
            audio_path: Путь к аудио файлу
            language: Язык аудио (None = автоопределение)
            
        Returns:
            Словарь с text и segments
            
        Raises:
            GroqRateLimitError: При превышении лимитов
            GroqAPIError: При других ошибках API
        """
        if not self.is_available():
            raise GroqAPIError("GROQ_API_KEY не установлен")
        
        # Подготавливаем файл
        upload_path, should_cleanup = self._prepare_audio(audio_path)
        
        try:
            logger.info(f"🚀 Отправка в Groq API ({self.model})...")
            print(f"🚀 Groq API: {upload_path.name} ({upload_path.stat().st_size / 1024 / 1024:.1f}MB)")
            
            t0 = time.time()
            
            # Создаём multipart запрос
            body, content_type = self._create_multipart_data(
                upload_path, 
                language or ('ru' if Config.FORCE_RU else None)
            )
            
            request = Request(
                GROQ_API_URL,
                data=body,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": content_type,
                },
                method="POST"
            )
            
            # Отправляем запрос
            try:
                response: HTTPResponse = urlopen(request, timeout=self.timeout)
                response_data = response.read().decode('utf-8')
                result = json.loads(response_data)
                
            except HTTPError as e:
                error_body = e.read().decode('utf-8') if e.fp else ""
                
                if e.code == 429:
                    # Rate limit exceeded
                    logger.warning(f"Groq rate limit: {error_body}")
                    raise GroqRateLimitError(
                        f"Превышен лимит Groq API. Попробуйте позже или используйте локальный backend."
                    )
                elif e.code == 413:
                    raise GroqAPIError(f"Файл слишком большой для Groq API")
                elif e.code == 401:
                    raise GroqAPIError("Неверный GROQ_API_KEY")
                else:
                    logger.error(f"Groq API error {e.code}: {error_body}")
                    raise GroqAPIError(f"Groq API error {e.code}: {error_body[:200]}")
                    
            except URLError as e:
                logger.error(f"Сетевая ошибка: {e}")
                raise GroqAPIError(f"Сетевая ошибка: {e}")
            
            elapsed = time.time() - t0
            
            # Парсим ответ
            text = result.get("text", "").strip()
            segments = self._parse_segments(result)
            detected_language = result.get("language", language or "unknown")
            
            logger.info(
                f"✅ Groq завершён за {elapsed:.1f} сек: "
                f"{len(segments)} сегментов, {len(text.split())} слов"
            )
            print(f"✅ Groq: {elapsed:.1f} сек, {len(text.split())} слов")
            
            return {
                "text": text,
                "segments": segments,
                "language": detected_language,
                "backend": "groq"
            }
            
        finally:
            # Удаляем временный файл
            if should_cleanup and upload_path.exists():
                try:
                    upload_path.unlink()
                except OSError:
                    pass
    
    def _parse_segments(self, result: Dict) -> List[Dict[str, Any]]:
        """Парсить сегменты из ответа Groq API."""
        segments = []
        
        # Groq возвращает segments в verbose_json формате
        for seg in result.get("segments", []):
            segments.append({
                "start": seg.get("start", 0),
                "end": seg.get("end", 0),
                "text": seg.get("text", "").strip()
            })
        
        return segments


def check_groq_available() -> bool:
    """Проверить доступность Groq API."""
    return bool(Config.GROQ_API_KEY)


def test_groq_connection() -> bool:
    """Тестовый запрос к Groq API для проверки ключа."""
    if not Config.GROQ_API_KEY:
        return False
    
    try:
        # Простой запрос к API для проверки ключа
        request = Request(
            "https://api.groq.com/openai/v1/models",
            headers={"Authorization": f"Bearer {Config.GROQ_API_KEY}"},
            method="GET"
        )
        response = urlopen(request, timeout=10)
        return response.status == 200
    except Exception as e:
        logger.debug(f"Groq connection test failed: {e}")
        return False

