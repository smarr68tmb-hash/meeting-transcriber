# -*- coding: utf-8 -*-
"""
WhisperX backend с поддержкой диаризации спикеров.

WhisperX добавляет:
- Быструю транскрипцию (faster-whisper)
- Точные word-level timestamps (forced alignment)
- Диаризацию спикеров (pyannote.audio)

Требования:
- pip install whisperx
- HuggingFace токен для pyannote (env: HF_TOKEN)
  Нужно принять лицензию: https://huggingface.co/pyannote/speaker-diarization-3.1
"""

import os
import time
from pathlib import Path
from typing import Optional, Dict, Any, List

from .config import Config
from .logging_setup import get_logger

logger = get_logger()


# Проверяем доступность whisperx
HAS_WHISPERX = False
try:
    import whisperx
    HAS_WHISPERX = True
except ImportError:
    pass

# Проверяем torch
HAS_TORCH = False
try:
    import torch
    HAS_TORCH = True
except ImportError:
    pass


class WhisperXTranscriber:
    """
    Транскрибер на базе WhisperX с диаризацией спикеров.
    
    Использует:
    - faster-whisper для быстрой транскрипции
    - wav2vec2 для forced alignment (точные timestamps)
    - pyannote для speaker diarization
    """
    
    def __init__(self):
        if not HAS_WHISPERX:
            raise ImportError(
                "whisperx не установлен. Установите: pip install whisperx"
            )
        
        self.model = None
        self.model_loaded = False
        self.model_size = Config.DEFAULT_MODEL
        self.device = self._resolve_device()
        # float16 не работает на CPU, используем int8
        self.compute_type = Config.WHISPERX_COMPUTE if self.device != 'cpu' else 'int8'
        self.hf_token = Config.HF_TOKEN
        
        # Кэшированные модели выравнивания
        self._align_model = None
        self._align_metadata = None
        self._align_language = None
    
    def _resolve_device(self) -> str:
        """Определить устройство для WhisperX."""
        d = Config.ASR_DEVICE
        
        if d == 'auto':
            if HAS_TORCH and torch.cuda.is_available():
                return 'cuda'
            # WhisperX на CPU работает, но медленно
            return 'cpu'
        
        if d == 'cuda':
            if HAS_TORCH and torch.cuda.is_available():
                return 'cuda'
            logger.warning("CUDA недоступна, использую CPU")
            return 'cpu'
        
        if d in ('mps', 'metal'):
            # WhisperX имеет ограниченную поддержку MPS
            logger.warning("WhisperX имеет ограниченную поддержку MPS, использую CPU")
            return 'cpu'
        
        return 'cpu'
    
    def load_model(self) -> None:
        """Загрузить модель WhisperX."""
        if self.model_loaded:
            logger.debug("Модель WhisperX уже загружена")
            return
        
        logger.info(
            f"🤖 Загрузка WhisperX модели '{self.model_size}' "
            f"(device={self.device}, compute={self.compute_type})..."
        )
        
        load_start = time.time()
        
        self.model = whisperx.load_model(
            self.model_size,
            device=self.device,
            compute_type=self.compute_type,
            language=Config.WHISPERX_LANGUAGE if Config.FORCE_RU else None
        )
        
        load_time = time.time() - load_start
        self.model_loaded = True
        logger.info(f"✅ WhisperX модель загружена за {load_time:.1f} сек")
    
    def transcribe(
        self,
        audio_path: Path,
        language: Optional[str] = None,
        diarize: bool = True,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Транскрибировать аудио с опциональной диаризацией.
        
        Args:
            audio_path: Путь к аудио файлу
            language: Язык аудио (None = автоопределение)
            diarize: Выполнять диаризацию спикеров
            min_speakers: Минимальное число спикеров (hint)
            max_speakers: Максимальное число спикеров (hint)
            
        Returns:
            Словарь с полями:
            - text: полный текст
            - segments: список сегментов с speaker
            - language: определённый язык
            - speakers: уникальные спикеры
        """
        self.load_model()
        
        audio_str = str(audio_path)
        
        # === Шаг 1: Транскрипция ===
        logger.info("📝 Шаг 1/3: Транскрипция...")
        print("📝 Шаг 1/3: Транскрипция...")
        
        t0 = time.time()
        
        # Загружаем аудио
        audio = whisperx.load_audio(audio_str)
        
        # Транскрибируем
        result = self.model.transcribe(
            audio,
            batch_size=Config.WHISPERX_BATCH_SIZE,
            language=language
        )
        
        detected_language = result.get("language", language or "ru")
        logger.info(f"Язык: {detected_language}, сегментов: {len(result['segments'])}")
        
        transcribe_time = time.time() - t0
        logger.debug(f"Транскрипция заняла {transcribe_time:.1f} сек")
        
        # === Шаг 2: Forced Alignment (точные timestamps) ===
        logger.info("⏱️  Шаг 2/3: Выравнивание timestamps...")
        print("⏱️  Шаг 2/3: Выравнивание timestamps...")
        
        t1 = time.time()
        
        # Загружаем модель выравнивания (кэшируем)
        if self._align_language != detected_language:
            logger.debug(f"Загрузка модели выравнивания для {detected_language}")
            self._align_model, self._align_metadata = whisperx.load_align_model(
                language_code=detected_language,
                device=self.device
            )
            self._align_language = detected_language
        
        # Выравниваем
        result = whisperx.align(
            result["segments"],
            self._align_model,
            self._align_metadata,
            audio,
            self.device,
            return_char_alignments=False
        )
        
        align_time = time.time() - t1
        logger.debug(f"Выравнивание заняло {align_time:.1f} сек")
        
        # === Шаг 3: Диаризация (опционально) ===
        speakers_found = set()
        
        if diarize:
            if not self.hf_token:
                logger.warning(
                    "⚠️ HF_TOKEN не задан, диаризация пропущена. "
                    "Установите env HF_TOKEN с токеном HuggingFace."
                )
                print("⚠️ Диаризация пропущена (нужен HF_TOKEN)")
            else:
                logger.info("🎭 Шаг 3/3: Диаризация спикеров...")
                print("🎭 Шаг 3/3: Диаризация спикеров...")
                
                t2 = time.time()
                
                try:
                    # Загружаем pipeline диаризации
                    from whisperx.diarize import DiarizationPipeline
                    diarize_model = DiarizationPipeline(
                        use_auth_token=self.hf_token,
                        device=self.device
                    )
                    
                    # Выполняем диаризацию
                    diarize_kwargs = {}
                    if min_speakers is not None:
                        diarize_kwargs["min_speakers"] = min_speakers
                    if max_speakers is not None:
                        diarize_kwargs["max_speakers"] = max_speakers
                    
                    diarize_segments = diarize_model(
                        audio,
                        **diarize_kwargs
                    )
                    
                    # Присваиваем спикеров сегментам
                    result = whisperx.assign_word_speakers(
                        diarize_segments,
                        result
                    )
                    
                    # Собираем уникальных спикеров
                    for seg in result.get("segments", []):
                        if "speaker" in seg:
                            speakers_found.add(seg["speaker"])
                    
                    diarize_time = time.time() - t2
                    logger.info(
                        f"✅ Диаризация завершена: {len(speakers_found)} спикер(ов), "
                        f"{diarize_time:.1f} сек"
                    )
                    print(f"✅ Найдено спикеров: {len(speakers_found)}")
                    
                except Exception as e:
                    logger.error(f"Ошибка диаризации: {e}", exc_info=True)
                    print(f"⚠️ Диаризация не удалась: {e}")
        else:
            logger.info("Диаризация отключена (--no-diarize)")
        
        # Формируем итоговый результат
        segments = result.get("segments", [])
        full_text = " ".join(seg.get("text", "").strip() for seg in segments)
        
        total_time = time.time() - t0
        logger.info(
            f"📊 WhisperX завершён: {len(segments)} сегментов, "
            f"{len(full_text.split())} слов, {total_time:.1f} сек"
        )
        
        return {
            "text": full_text,
            "segments": segments,
            "language": detected_language,
            "speakers": sorted(speakers_found) if speakers_found else []
        }
    
    def format_segments_with_speakers(
        self,
        segments: List[Dict],
        include_timestamps: bool = True
    ) -> str:
        """
        Форматировать сегменты с метками спикеров.
        
        Args:
            segments: Список сегментов
            include_timestamps: Включать временные метки
            
        Returns:
            Отформатированный текст
        """
        lines = []
        current_speaker = None
        current_text = []
        current_start = None
        
        for seg in segments:
            speaker = seg.get("speaker", "UNKNOWN")
            text = seg.get("text", "").strip()
            start = seg.get("start", 0)
            
            if not text:
                continue
            
            # Группируем последовательные реплики одного спикера
            if speaker == current_speaker:
                current_text.append(text)
            else:
                # Сохраняем предыдущую группу
                if current_text:
                    combined = " ".join(current_text)
                    if include_timestamps:
                        ts = self._format_time(current_start)
                        lines.append(f"[{ts}] {current_speaker}: {combined}")
                    else:
                        lines.append(f"{current_speaker}: {combined}")
                
                # Начинаем новую группу
                current_speaker = speaker
                current_text = [text]
                current_start = start
        
        # Добавляем последнюю группу
        if current_text:
            combined = " ".join(current_text)
            if include_timestamps:
                ts = self._format_time(current_start)
                lines.append(f"[{ts}] {current_speaker}: {combined}")
            else:
                lines.append(f"{current_speaker}: {combined}")
        
        return "\n\n".join(lines)
    
    @staticmethod
    def _format_time(seconds: float) -> str:
        """Форматировать время в MM:SS."""
        minutes = int(seconds) // 60
        secs = int(seconds) % 60
        return f"{minutes:02d}:{secs:02d}"


def check_whisperx_available() -> bool:
    """Проверить доступность WhisperX."""
    return HAS_WHISPERX


def check_diarization_ready() -> bool:
    """Проверить готовность к диаризации (наличие HF токена)."""
    return bool(Config.HF_TOKEN)

