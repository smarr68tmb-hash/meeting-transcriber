# -*- coding: utf-8 -*-
"""
Модуль транскрипции аудио с использованием Whisper.
Поддерживает faster-whisper, openai-whisper и whisperx (с диаризацией) backends.
"""

import os
import time
import json
import datetime
import platform
import subprocess
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

from .config import Config
from .utils import ffprobe_ok, get_audio_duration, format_timestamp_srt
from .logging_setup import get_logger

logger = get_logger()

# Проверяем доступность WhisperX
HAS_WHISPERX = False
try:
    from .whisperx import WhisperXTranscriber, check_whisperx_available
    HAS_WHISPERX = check_whisperx_available()
except ImportError:
    pass

# Проверяем наличие torch
HAS_TORCH = False
try:
    import torch
    HAS_TORCH = True
except ImportError:
    pass

# Проверяем наличие tqdm
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable=None, **kwargs):
        logger.warning("tqdm не установлен. Установите: pip install tqdm")
        return iterable if iterable is not None else []


class EnhancedTranscriber:
    """
    Класс для транскрипции аудио файлов.
    Поддерживает faster-whisper, openai-whisper и whisperx (с диаризацией) backends.
    """
    
    def __init__(
        self,
        diarize: bool = False,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None
    ):
        """
        Инициализация транскрибера.
        
        Args:
            diarize: Включить диаризацию спикеров (требует whisperx backend)
            min_speakers: Минимальное число спикеров (hint для диаризации)
            max_speakers: Максимальное число спикеров (hint для диаризации)
        """
        Config.ensure_directories()
        self.model = None
        self.model_loaded = False
        self.model_size = Config.DEFAULT_MODEL
        self.backend = Config.ASR_BACKEND
        self.device = 'cpu'
        self.use_fp16 = False
        self.diarize = diarize
        self.min_speakers = min_speakers
        self.max_speakers = max_speakers
        
        # WhisperX транскрибер (для диаризации)
        self.whisperx_transcriber = None
        
        # Автоматически переключаемся на whisperx если нужна диаризация
        if diarize and self.backend != 'whisperx':
            if HAS_WHISPERX:
                logger.info("Диаризация запрошена, переключаюсь на whisperx backend")
                self.backend = 'whisperx'
            else:
                logger.warning(
                    "Диаризация запрошена, но whisperx не установлен. "
                    "Установите: pip install whisperx. Диаризация отключена."
                )
                self.diarize = False  # Отключаем, чтобы не вводить в заблуждение

    def _resolve_device_whisper(self) -> Tuple[str, bool]:
        """Определить устройство для openai-whisper."""
        d = Config.ASR_DEVICE
        
        if d == 'auto':
            if HAS_TORCH and torch.cuda.is_available():
                return 'cuda', True
            if HAS_TORCH and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return 'mps', False
            return 'cpu', False
        
        if d == 'cuda':
            if HAS_TORCH and torch.cuda.is_available():
                return 'cuda', True
            return 'cpu', False
        
        if d in ('mps', 'metal'):
            ok = HAS_TORCH and hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
            return ('mps', False) if ok else ('cpu', False)
        
        return 'cpu', False

    def _resolve_device_faster(self) -> str:
        """Определить устройство для faster-whisper."""
        d = Config.ASR_DEVICE
        if d in ('auto', 'cpu', 'cuda', 'metal'):
            return d
        if d == 'mps':
            return 'metal'
        return 'auto'

    def _load_model(self) -> None:
        """Загрузить модель Whisper."""
        if self.model_loaded:
            logger.debug("Модель уже загружена, пропускаем")
            return
        
        logger.info(f"🤖 Загрузка модели '{self.model_size}' (backend={self.backend})...")
        load_start = time.time()
        
        if self.backend == 'whisperx':
            # WhisperX с диаризацией
            if not HAS_WHISPERX:
                raise ImportError(
                    "whisperx не установлен. Установите: pip install whisperx"
                )
            self.whisperx_transcriber = WhisperXTranscriber()
            self.whisperx_transcriber.load_model()
            self.device = self.whisperx_transcriber.device
        
        elif self.backend == 'faster':
            from faster_whisper import WhisperModel
            
            device = self._resolve_device_faster()
            cpu_threads = Config.FASTER_CPU_THREADS if device == 'cpu' else 0
            
            logger.debug(
                f"faster-whisper: device={device}, "
                f"compute_type={Config.FASTER_COMPUTE}, "
                f"cpu_threads={cpu_threads}"
            )
            
            self.model = WhisperModel(
                self.model_size,
                device=device,
                compute_type=Config.FASTER_COMPUTE,
                cpu_threads=cpu_threads
            )
            self.device = device
        else:
            import whisper
            
            device, fp16 = self._resolve_device_whisper()
            logger.debug(f"openai-whisper: device={device}, fp16={fp16}")
            
            self.model = whisper.load_model(self.model_size, device=device)
            self.device = device
            self.use_fp16 = fp16
        
        load_time = time.time() - load_start
        self.model_loaded = True
        logger.info(f"✅ Модель загружена (device={self.device}) за {load_time:.1f} сек")

    def transcribe_files(self, files: List[Path]) -> None:
        """
        Транскрибировать список файлов.
        
        Args:
            files: Список путей к аудио файлам
        """
        logger.info(f"Начинаем транскрипцию {len(files)} файл(ов)")
        self._load_model()
        
        success = 0
        total = len(files)
        
        for i, f in enumerate(files, 1):
            print(f"\n━━━ Файл {i}/{total}: {f.name} ━━━")
            logger.info(f"Обработка файла {i}/{total}: {f.name}")
            
            ok = self._transcribe_single(f, auto_open=(i == 1))
            success += 1 if ok else 0
        
        logger.info(f"📊 Итог: успешно {success}/{total}, ошибок {total - success}")
        print(f"\n📊 Итог: успешно {success}/{total}, ошибок {total - success}")

    def _transcribe_single(self, audio_file: Path, auto_open: bool = True) -> bool:
        """
        Транскрибировать один файл.
        
        Args:
            audio_file: Путь к аудио файлу
            auto_open: Открыть результат после завершения
            
        Returns:
            True при успехе, False при ошибке
        """
        if not audio_file.exists():
            logger.error(f"Файл не найден: {audio_file}")
            return False
        
        if not ffprobe_ok(audio_file):
            logger.error(f"Файл повреждён или не является аудио: {audio_file}")
            return False
        
        # Конвертируем в safe WAV (16kHz, mono)
        safe_file = audio_file.with_suffix(
            f".safe{datetime.datetime.now():%H%M%S}.wav"
        )
        
        logger.info("Подготовка аудио (конвертация в 16kHz mono WAV)...")
        print("Подготовка аудио...")
        
        try:
            subprocess.run([
                "ffmpeg", "-y", "-i", str(audio_file),
                "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", "-nostdin",
                str(safe_file)
            ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            logger.debug(f"Конвертация завершена: {safe_file}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Конвертация не удалась: {e}")
            return False
        
        t0 = time.time()
        language = 'ru' if Config.FORCE_RU else None
        
        try:
            # WhisperX backend (с диаризацией)
            if self.backend == 'whisperx':
                result = self._run_whisperx(safe_file, language=language)
            else:
                # Первый проход — БЕЗ VAD
                logger.debug(f"ASR проход 1: language={language}, vad=off")
                result = self._run_asr_once(safe_file, language=language, use_vad=False)
                
                # Fallback — с VAD и ru
                if not result or not result.get("segments"):
                    logger.warning("Первый проход пуст, пробуем с VAD...")
                    print("⚠️ Пусто без VAD, пробую с VAD...")
                    result = self._run_asr_once(
                        safe_file,
                        language=language or 'ru',
                        use_vad=True
                    )
            
            if not result or not result.get("text", "").strip():
                logger.error("Транскрипция не дала результата")
                return False
            
            elapsed = time.time() - t0
            word_count = len(result['text'].split())
            segment_count = len(result['segments'])
            speakers = result.get('speakers', [])
            
            if speakers:
                logger.info(
                    f"✅ Транскрипция завершена: {segment_count} сегментов, "
                    f"{word_count} слов, {len(speakers)} спикер(ов), {elapsed / 60:.1f} мин"
                )
                print(f"✅ Сегментов: {segment_count}, слов: {word_count}, "
                      f"спикеров: {len(speakers)}, время: {elapsed / 60:.1f} мин.")
            else:
                logger.info(
                    f"✅ Транскрипция завершена: {segment_count} сегментов, "
                    f"{word_count} слов, {elapsed / 60:.1f} мин"
                )
                print(f"✅ Сегментов: {segment_count}, слов: {word_count}, время: {elapsed / 60:.1f} мин.")
            
            # Сохраняем результаты
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            base = f"transcript_{audio_file.stem}_{ts}"
            
            # Если есть спикеры, сохраняем с диаризацией
            has_speakers = speakers or any(
                seg.get('speaker') for seg in result['segments']
            )
            
            if has_speakers:
                txt = self._save_txt_diarized(result, base)
            else:
                txt = self._save_txt(result, base)
            
            jsn = self._save_json(result, base, audio_file.name, language or 'auto')
            srt = self._save_srt(result, base, include_speaker=has_speakers)
            
            logger.info(f"📄 Сохранено: {txt.name}, {jsn.name}, {srt.name}")
            print("📄 Сохранено:", txt.name, jsn.name, srt.name)
            
            if auto_open:
                self._open_file(txt)
            
            return True
        finally:
            if safe_file.exists():
                try:
                    safe_file.unlink()
                    logger.debug("Временный safe WAV файл удалён")
                except OSError as e:
                    logger.warning(f"Не удалось удалить временный файл: {e}")

    def _run_asr_once(
        self,
        wav_file: Path,
        language: Optional[str],
        use_vad: bool
    ) -> Optional[Dict[str, Any]]:
        """
        Выполнить один проход ASR.
        
        Args:
            wav_file: Путь к WAV файлу
            language: Язык или None для автоопределения
            use_vad: Использовать Voice Activity Detection
            
        Returns:
            Словарь с text и segments или None при ошибке
        """
        total_sec = get_audio_duration(wav_file)
        logger.debug(f"Длительность аудио: {total_sec:.1f} сек")
        
        pbar = tqdm(
            total=int(total_sec) if total_sec > 0 else None,
            desc="Транскрипция",
            unit="s",
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'
        )
        
        segs: List[Dict] = []
        texts: List[str] = []
        last_progress = 0
        last_print = time.time()
        
        try:
            logger.info(f"ASR start (lang={language}, vad={'on' if use_vad else 'off'})")
            print(f" > ASR start (lang={language}, vad={'on' if use_vad else 'off'})")
            
            if self.backend == 'faster':
                segments_it, info = self.model.transcribe(
                    str(wav_file),
                    language=language,
                    vad_filter=use_vad,
                    beam_size=Config.FASTER_BEAM_SIZE,
                    word_timestamps=True
                )
                
                for s in segments_it:
                    segs.append({
                        'start': s.start,
                        'end': s.end,
                        'text': s.text
                    })
                    texts.append(s.text)
                    
                    # Обновляем прогресс
                    if total_sec and s.end is not None:
                        cur = int(s.end)
                        if cur > last_progress:
                            pbar.update(cur - last_progress)
                            last_progress = cur
                    
                    # Живой статус каждые ~2.5 сек
                    if time.time() - last_print > 2.5:
                        mm = int(s.end) // 60 if s.end else 0
                        ss = int(s.end) % 60 if s.end else 0
                        print(f"\r… t≈{mm:02d}:{ss:02d}, сегментов: {len(segs)}", 
                              end="", flush=True)
                        last_print = time.time()
                    
                    if Config.DEBUG_SEGMENTS:
                        logger.debug(f"[{s.start:.2f}-{s.end:.2f}] {s.text[:60]}")
                
                if language is None:
                    language = getattr(info, 'language', None)
                    logger.debug(f"Определён язык: {language}")
            
            else:  # openai/whisper
                import whisper
                
                res = self.model.transcribe(
                    str(wav_file),
                    language=language,
                    fp16=self.use_fp16,
                    word_timestamps=True
                )
                segs = res.get("segments", [])
                texts = [seg.get("text", "") for seg in segs]
                pbar.update(int(total_sec) if total_sec else 0)
            
            logger.debug(f"ASR завершён: {len(segs)} сегментов")
            return {'text': " ".join(texts).strip(), 'segments': segs}
        
        except Exception as e:
            logger.error(f"Ошибка ASR: {e}", exc_info=True)
            raise
        finally:
            if total_sec and pbar.n < int(total_sec):
                pbar.update(int(total_sec) - pbar.n)
            pbar.close()
            print()  # Перенос строки после прогресса

    def _run_whisperx(
        self,
        wav_file: Path,
        language: Optional[str]
    ) -> Optional[Dict[str, Any]]:
        """
        Выполнить транскрипцию через WhisperX с диаризацией.
        
        Args:
            wav_file: Путь к WAV файлу
            language: Язык или None для автоопределения
            
        Returns:
            Словарь с text, segments, speakers
        """
        if not self.whisperx_transcriber:
            raise RuntimeError("WhisperX транскрибер не инициализирован")
        
        return self.whisperx_transcriber.transcribe(
            wav_file,
            language=language,
            diarize=self.diarize,
            min_speakers=self.min_speakers,
            max_speakers=self.max_speakers
        )

    def _save_txt(self, result: Dict, base: str) -> Path:
        """Сохранить результат в TXT."""
        p = Config.TRANSCRIPTS_FOLDER / f"{base}.txt"
        with open(p, 'w', encoding='utf-8') as f:
            f.write(result["text"])
        return p

    def _save_txt_diarized(self, result: Dict, base: str) -> Path:
        """Сохранить результат с диаризацией в TXT."""
        p = Config.TRANSCRIPTS_FOLDER / f"{base}.txt"
        
        segments = result.get("segments", [])
        lines = []
        current_speaker = None
        current_text = []
        current_start = None
        
        for seg in segments:
            speaker = seg.get("speaker", "SPEAKER")
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
                    ts = self._format_time_short(current_start)
                    combined = " ".join(current_text)
                    lines.append(f"[{ts}] {current_speaker}:\n{combined}")
                
                # Начинаем новую группу
                current_speaker = speaker
                current_text = [text]
                current_start = start
        
        # Добавляем последнюю группу
        if current_text:
            ts = self._format_time_short(current_start)
            combined = " ".join(current_text)
            lines.append(f"[{ts}] {current_speaker}:\n{combined}")
        
        with open(p, 'w', encoding='utf-8') as f:
            f.write("\n\n".join(lines))
        
        return p

    @staticmethod
    def _format_time_short(seconds: float) -> str:
        """Форматировать время в MM:SS."""
        minutes = int(seconds) // 60
        secs = int(seconds) % 60
        return f"{minutes:02d}:{secs:02d}"

    def _save_json(
        self,
        result: Dict,
        base: str,
        audio_name: str,
        language: str
    ) -> Path:
        """Сохранить результат в JSON."""
        p = Config.TRANSCRIPTS_FOLDER / f"{base}.json"
        data = {
            'timestamp': datetime.datetime.now().isoformat(),
            'audio_file': audio_name,
            'language': language,
            'text': result['text'],
            'segments': result['segments']
        }
        with open(p, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return p

    def _save_srt(
        self,
        result: Dict,
        base: str,
        include_speaker: bool = False
    ) -> Path:
        """
        Сохранить результат в SRT (субтитры).
        
        Args:
            result: Результат транскрипции
            base: Базовое имя файла
            include_speaker: Добавлять метку спикера
        """
        p = Config.TRANSCRIPTS_FOLDER / f"{base}.srt"
        with open(p, 'w', encoding='utf-8') as f:
            for i, s in enumerate(result['segments'], 1):
                start = format_timestamp_srt(s.get('start', 0.0))
                end = format_timestamp_srt(s.get('end', 0.0))
                text = (s.get('text') or '').strip()
                
                if include_speaker and s.get('speaker'):
                    text = f"[{s['speaker']}] {text}"
                
                f.write(f"{i}\n{start} --> {end}\n{text}\n\n")
        return p

    def _open_file(self, path: Path) -> None:
        """Открыть файл в системном приложении."""
        try:
            logger.debug(f"Открываю файл: {path}")
            system = platform.system()
            
            if system == "Darwin":
                subprocess.run(["open", str(path)], check=False)
            elif system == "Windows":
                os.startfile(str(path))  # type: ignore
            else:
                subprocess.run(["xdg-open", str(path)], check=False)
        except Exception as e:
            logger.warning(f"Не удалось открыть файл {path}: {e}")

