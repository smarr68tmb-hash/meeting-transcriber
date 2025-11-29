#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Система записи и транскрипции совещаний v4.8 (all-in-one)
- Запись (ffmpeg) + список устройств
- Надёжная safe-конвертация (-nostdin, mono 16kHz PCM)
- faster-whisper с cpu_threads (устранение подвисаний на CPU)
- tqdm-прогресс по секундам + «живой» статус каждые 2–3 сек (t≈MM:SS, сегментов: N)
- Первый проход без VAD, fallback с VAD/ru при пустом результате
- Итоговая сводка и сохранение TXT/JSON/SRT
- Система логирования с ротацией файлов (-v/--verbose, --debug)
"""

import os, sys, re, shutil, subprocess, datetime, json, time, argparse, platform, logging
from pathlib import Path
from typing import List, Optional, Dict, Tuple
from logging.handlers import RotatingFileHandler

# ------------------- LOGGING SETUP -------------------
def setup_logging(verbose: bool = False, debug: bool = False) -> logging.Logger:
    """Настройка системы логирования с выводом в консоль и файл."""
    logger = logging.getLogger("meeting_transcriber")
    
    # Определяем уровень логирования
    if debug:
        level = logging.DEBUG
    elif verbose:
        level = logging.INFO
    else:
        level = logging.WARNING
    
    logger.setLevel(logging.DEBUG)  # Логгер принимает всё, фильтруют handlers
    
    # Формат для консоли (краткий)
    console_format = logging.Formatter(
        '%(asctime)s │ %(levelname)-7s │ %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # Формат для файла (подробный)
    file_format = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(console_format)
    
    # File handler с ротацией (5 MB, 3 бэкапа)
    log_dir = Path.home() / "Meeting_Recordings" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "meeting_transcriber.log"
    
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=5 * 1024 * 1024,  # 5 MB
        backupCount=3,
        encoding='utf-8'
    )
    file_handler.setLevel(logging.DEBUG)  # В файл пишем всё
    file_handler.setFormatter(file_format)
    
    # Очищаем существующие handlers и добавляем новые
    logger.handlers.clear()
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger

# Глобальный логгер (инициализируется в main)
logger = logging.getLogger("meeting_transcriber")

# ------------------- Torch (для whisper) -------------------
HAS_TORCH = False
try:
    import torch
    HAS_TORCH = True
except ImportError:
    pass

# ------------------- tqdm -------------------
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable=None, **kwargs):
        logger.warning("tqdm не установлен. Установите: pip install tqdm")
        return iterable if iterable is not None else []

# ------------------- CONFIG -------------------
class Config:
    RECORDINGS_FOLDER = Path.home() / "Meeting_Recordings"
    TRANSCRIPTS_FOLDER = Path.home() / "Meeting_Transcripts"
    LOGS_FOLDER = RECORDINGS_FOLDER / "logs"

    # Аудио запись
    DEFAULT_FORMAT = os.environ.get('REC_FORMAT', 'wav').lower()    # wav|flac
    DEFAULT_CHANNELS = os.environ.get('REC_CHANNELS', '2')          # '1'|'2'
    DEFAULT_SAMPLE_RATE = os.environ.get('REC_RATE', '48000')
    FLAC_LEVEL = os.environ.get('FLAC_LEVEL', '8')
    PRE_RECORD_PROBE = int(os.environ.get('PRE_RECORD_PROBE', '3')) # сек; 0 = без пробы
    VOICE_FILTERS = os.environ.get(
        "VOICE_FILTERS",
        "adeclick,highpass=f=80,lowpass=f=12000,anlmdn=s=7,"
        "acompressor=threshold=-20dB:ratio=3:attack=5:release=100,"
        "loudnorm=I=-16:TP=-1.5:LRA=11"
    )

    # ASR
    DEFAULT_MODEL = os.environ.get('WHISPER_MODEL', 'medium')
    ASR_BACKEND  = os.environ.get('ASR_BACKEND', 'faster').lower()  # faster|whisper
    ASR_DEVICE   = os.environ.get('ASR_DEVICE', 'auto').lower()      # auto|cpu|cuda|mps|metal
    FORCE_RU     = (os.environ.get('FORCE_RU', '0') == '1') or False # можно поставить True, если нужно жёстко

    # faster-whisper
    FASTER_COMPUTE     = os.environ.get('FASTER_COMPUTE_TYPE', 'int8')  # int8|int8_float16|float16|float32
    FASTER_BEAM_SIZE   = int(os.environ.get('FASTER_BEAM_SIZE', '5'))
    FASTER_VAD         = os.environ.get('FASTER_VAD', '0') == '1'       # по умолчанию off; включаем только fallback
    FASTER_CPU_THREADS = int(os.environ.get('FASTER_CPU_THREADS', '1')) # 1 поток на CPU — меньше шансов зависания

    # Логи сегментов во время распознавания
    DEBUG_SEGMENTS     = os.environ.get('DEBUG_SEGMENTS', '0') == '1'

# ------------------- УТИЛИТЫ -------------------
def ffprobe_ok(path: Path) -> bool:
    if not shutil.which('ffprobe'):
        return path.exists() and path.stat().st_size > 1000
    cmd = ['ffprobe','-v','error','-select_streams','a:0',
           '-show_entries','stream=codec_name',
           '-of','default=nokey=1:noprint_wrappers=1', str(path)]
    try:
        return bool(subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True).strip())
    except subprocess.CalledProcessError:
        return False

def get_audio_duration(path: Path) -> float:
    if not shutil.which('ffprobe'):
        return 0.0
    cmd = ['ffprobe','-v','error','-show_entries','format=duration',
           '-of','default=nokey=1:noprint_wrappers=1', str(path)]
    try:
        return float(subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True).strip())
    except Exception:
        return 0.0

def get_platform_config() -> Dict[str, str]:
    system = platform.system()
    if system == "Darwin":
        return {'format': 'avfoundation', 'dummy': '', 'list_cmd': ['-list_devices','true']}
    if system == "Windows":
        return {'format': 'dshow', 'dummy': 'dummy', 'list_cmd': ['-list_devices','true']}
    use_pulse = shutil.which('pactl') is not None
    return {'format': 'pulse' if use_pulse else 'alsa', 'dummy': 'default', 'list_cmd': []}

# ------------------- RECORDER -------------------
class MeetingRecorder:
    def __init__(self):
        Config.RECORDINGS_FOLDER.mkdir(exist_ok=True, parents=True)
        Config.LOGS_FOLDER.mkdir(exist_ok=True, parents=True)
        self.platform_config = get_platform_config()
        self.recording_process = None

    def list_devices(self) -> None:
        fmt = self.platform_config['format']
        cmd = ['ffmpeg','-f', fmt, *self.platform_config['list_cmd'], '-i', self.platform_config['dummy']]
        logger.info(f"🔍 Выполняю: {' '.join(cmd)}")
        try:
            res = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='ignore')
            output = (res.stderr or '') + "\n" + (res.stdout or '')
            print("\n" + output)  # Вывод устройств всегда в консоль
            logger.debug(f"Список устройств получен")
        except Exception as e:
            logger.error(f"Не удалось получить список устройств: {e}")

    def _record_probe(self, device: str) -> bool:
        if Config.PRE_RECORD_PROBE <= 0:
            logger.debug("Пробная запись отключена (PRE_RECORD_PROBE=0)")
            return True
        logger.info(f"🔎 Пробная запись ({Config.PRE_RECORD_PROBE} сек) — проверка устройства '{device}'...")
        probe_file = Config.LOGS_FOLDER / "_probe.wav"
        cmd = [
            'ffmpeg','-y','-hide_banner','-nostdin',
            '-f', self.platform_config['format'],
            '-i', device,
            '-t', str(Config.PRE_RECORD_PROBE),
            '-c:a','pcm_s16le',
            str(probe_file)
        ]
        log_file = Config.LOGS_FOLDER / "_probe.log"
        logger.debug(f"Команда пробной записи: {' '.join(cmd)}")
        try:
            with open(log_file,'w',encoding='utf-8') as log:
                p = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
                p.wait()
            ok = (p.returncode == 0) and ffprobe_ok(probe_file)
            if ok:
                logger.info("✅ Проба успешна")
            else:
                logger.error(f"Пробная запись не удалась. См. лог: {log_file}")
            return ok
        finally:
            if probe_file.exists():
                try: 
                    probe_file.unlink()
                    logger.debug("Временный файл пробы удалён")
                except OSError as e:
                    logger.warning(f"Не удалось удалить временный файл пробы: {e}")

    def record(self, output_file: Path, device: str) -> Optional[List[Path]]:
        if not self._record_probe(device):
            return None
        suffix = '.wav' if Config.DEFAULT_FORMAT == 'wav' else '.flac'
        output_path = output_file.with_suffix(suffix)
        codec = 'pcm_s16le' if Config.DEFAULT_FORMAT == 'wav' else 'flac'
        cmd = [
            'ffmpeg','-y','-hide_banner','-nostdin',
            '-f', self.platform_config['format'],
            '-i', device,
            '-vn','-ar', Config.DEFAULT_SAMPLE_RATE,
            '-ac', Config.DEFAULT_CHANNELS,
            '-acodec', codec
        ]
        if Config.DEFAULT_FORMAT == 'flac':
            cmd += ['-compression_level', Config.FLAC_LEVEL]
        else:
            cmd += ['-rf64','auto']
        cmd += ['-af', Config.VOICE_FILTERS, str(output_path)]

        log_file = Config.LOGS_FOLDER / f"{output_file.stem}.log"
        logger.debug(f"Команда записи: {' '.join(cmd)}")
        
        # Красивый вывод в консоль (всегда показываем)
        print("\n" + "="*52)
        print(f"🔴 ЗАПИСЬ НАЧАТА -> {output_path.name}")
        print("⏹  Остановка: Ctrl+C")
        print("="*52)
        
        logger.info(f"Запись начата: {output_path.name}")
        try:
            with open(log_file,'w',encoding='utf-8') as log:
                self.recording_process = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
                start = time.time()
                while self.recording_process.poll() is None:
                    elapsed = int(time.time() - start)
                    print(f"\r⏱  Длительность: {elapsed//60:02d}:{elapsed%60:02d}", end="", flush=True)
                    time.sleep(1)
        except KeyboardInterrupt:
            print("\n⏸ Останавливаю запись...")
            logger.info("Запись остановлена пользователем (Ctrl+C)")
            if self.recording_process:
                self.recording_process.terminate()
                self.recording_process.wait()
        except Exception as e:
            logger.error(f"Ошибка записи: {e}", exc_info=True)
            return None

        duration = time.time() - start if 'start' in locals() else 0
        logger.info(f"Запись завершена: {output_path.name}, длительность: {duration/60:.1f} мин")
        print("\n✅ Запись завершена")
        
        if output_path.exists() and ffprobe_ok(output_path):
            file_size = output_path.stat().st_size / (1024 * 1024)
            logger.info(f"Файл создан: {output_path}, размер: {file_size:.1f} MB")
            return [output_path]
        logger.error(f"Файл записи не создан или повреждён. Лог: {log_file}")
        return None

# ------------------- TRANSCRIBER -------------------
class EnhancedTranscriber:
    def __init__(self):
        Config.TRANSCRIPTS_FOLDER.mkdir(exist_ok=True, parents=True)
        self.model = None
        self.model_loaded = False
        self.model_size = Config.DEFAULT_MODEL
        self.backend = Config.ASR_BACKEND
        self.device = 'cpu'
        self.use_fp16 = False

    # --- выбор девайса ---
    def _resolve_device_whisper(self) -> Tuple[str, bool]:
        d = Config.ASR_DEVICE
        if d == 'auto':
            if HAS_TORCH and torch.cuda.is_available(): return 'cuda', True
            if HAS_TORCH and hasattr(torch.backends,"mps") and torch.backends.mps.is_available(): return 'mps', False
            return 'cpu', False
        if d == 'cuda':
            return ('cuda', True) if (HAS_TORCH and torch.cuda.is_available()) else ('cpu', False)
        if d in ('mps','metal'):
            ok = HAS_TORCH and hasattr(torch.backends,"mps") and torch.backends.mps.is_available()
            return ('mps', False) if ok else ('cpu', False)
        return 'cpu', False

    def _resolve_device_faster(self) -> str:
        d = Config.ASR_DEVICE
        if d in ('auto','cpu','cuda','metal'): return d
        if d == 'mps': return 'metal'
        return 'auto'

    # --- загрузка модели ---
    def _load_model(self) -> None:
        if self.model_loaded:
            logger.debug("Модель уже загружена, пропускаем")
            return
        logger.info(f"🤖 Загрузка модели '{self.model_size}' (backend={self.backend})...")
        load_start = time.time()
        
        if self.backend == 'faster':
            from faster_whisper import WhisperModel
            device = self._resolve_device_faster()
            cpu_threads = Config.FASTER_CPU_THREADS if device == 'cpu' else 0
            logger.debug(f"faster-whisper: device={device}, compute_type={Config.FASTER_COMPUTE}, cpu_threads={cpu_threads}")
            self.model = WhisperModel(self.model_size, device=device,
                                      compute_type=Config.FASTER_COMPUTE,
                                      cpu_threads=cpu_threads)
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

    # --- верхний уровень ---
    def transcribe_files(self, files: List[Path]) -> None:
        logger.info(f"Начинаем транскрипцию {len(files)} файл(ов)")
        self._load_model()
        success = 0
        total = len(files)
        for i, f in enumerate(files, 1):
            print(f"\n━━━ Файл {i}/{total}: {f.name} ━━━")
            logger.info(f"Обработка файла {i}/{total}: {f.name}")
            ok = self._transcribe_single(f, auto_open=(i==1))
            success += 1 if ok else 0
        
        logger.info(f"📊 Итог: успешно {success}/{total}, ошибок {total-success}")
        print(f"\n📊 Итог: успешно {success}/{total}, ошибок {total-success}")

    # --- обработка одного файла ---
    def _transcribe_single(self, audio_file: Path, auto_open: bool=True) -> bool:
        if not audio_file.exists():
            logger.error(f"Файл не найден: {audio_file}")
            return False
        if not ffprobe_ok(audio_file):
            logger.error(f"Файл повреждён или не является аудио: {audio_file}")
            return False

        # Safe WAV
        safe_file = audio_file.with_suffix(f".safe{datetime.datetime.now():%H%M%S}.wav")
        logger.info("Подготовка аудио (конвертация в 16kHz mono WAV)...")
        print("Подготовка аудио...")
        try:
            subprocess.run([
                "ffmpeg","-y","-i",str(audio_file),
                "-ar","16000","-ac","1","-c:a","pcm_s16le","-nostdin", str(safe_file)
            ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            logger.debug(f"Конвертация завершена: {safe_file}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Конвертация не удалась: {e}")
            return False

        t0 = time.time()
        language = 'ru' if Config.FORCE_RU else None

        try:
            # 1) Первый проход — БЕЗ VAD (чтобы не «ждал речи»)
            logger.debug(f"ASR проход 1: language={language}, vad=off")
            result = self._run_asr_once(safe_file, language=language, use_vad=False)
            
            # 2) Fallback — c VAD и ru
            if not result or not result.get("segments"):
                logger.warning("Первый проход пуст, пробуем с VAD...")
                print("⚠️ Пусто без VAD, пробую с VAD...")
                result = self._run_asr_once(safe_file, language=language or 'ru', use_vad=True)
            
            if not result or not result.get("text","").strip():
                logger.error("Транскрипция не дала результата")
                return False

            elapsed = time.time() - t0
            word_count = len(result['text'].split())
            segment_count = len(result['segments'])
            
            logger.info(f"✅ Транскрипция завершена: {segment_count} сегментов, {word_count} слов, {elapsed/60:.1f} мин")
            print(f"✅ Сегментов: {segment_count}, слов: {word_count}, время: {elapsed/60:.1f} мин.")

            ts  = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            base= f"transcript_{audio_file.stem}_{ts}"
            txt = self._save_txt(result, base, audio_file.name, (language or 'auto'))
            jsn = self._save_json(result, base, audio_file.name, (language or 'auto'))
            srt = self._save_srt(result, base)
            
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

    # --- единичный прогон ASR с прогрессом ---
    def _run_asr_once(self, wav_file: Path, language: Optional[str], use_vad: bool):
        total_sec = get_audio_duration(wav_file)
        logger.debug(f"Длительность аудио: {total_sec:.1f} сек")
        
        pbar = tqdm(total=int(total_sec) if total_sec>0 else None,
                    desc="Транскрипция", unit="s",
                    bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')
        segs, texts = [], []
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
                    segs.append({'start': s.start, 'end': s.end, 'text': s.text})
                    texts.append(s.text)

                    # обновляем прогресс по секундам
                    if total_sec and s.end is not None:
                        cur = int(s.end)
                        if cur > last_progress:
                            pbar.update(cur - last_progress)
                            last_progress = cur

                    # «живой» статус каждые ~2.5 сек
                    if time.time() - last_print > 2.5:
                        mm, ss = int(s.end)//60 if s.end else 0, int(s.end)%60 if s.end else 0
                        print(f"\r… t≈{mm:02d}:{ss:02d}, сегментов: {len(segs)}", end="", flush=True)
                        last_print = time.time()

                    if Config.DEBUG_SEGMENTS:
                        logger.debug(f"[{s.start:.2f}-{s.end:.2f}] {s.text[:60]}")

                if language is None:
                    language = getattr(info, 'language', None)
                    logger.debug(f"Определён язык: {language}")

            else:  # openai/whisper
                import whisper
                res = self.model.transcribe(str(wav_file),
                                            language=language,
                                            fp16=self.use_fp16,
                                            word_timestamps=True)
                segs = res.get("segments", [])
                texts = [seg.get("text","") for seg in segs]
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
            print()  # перенос строки после прогресса

    # --- сохранения ---
    def _save_txt(self, result, base, audio_name, language):
        p = Config.TRANSCRIPTS_FOLDER / f"{base}.txt"
        with open(p,'w',encoding='utf-8') as f:
            f.write(result["text"])
        return p

    def _save_json(self, result, base, audio_name, language):
        p = Config.TRANSCRIPTS_FOLDER / f"{base}.json"
        data = {
            'timestamp': datetime.datetime.now().isoformat(),
            'audio_file': audio_name,
            'language': language,
            'text': result['text'],
            'segments': result['segments']
        }
        with open(p,'w',encoding='utf-8') as f:
            json.dump(data,f,ensure_ascii=False,indent=2)
        return p

    def _save_srt(self, result, base):
        p = Config.TRANSCRIPTS_FOLDER / f"{base}.srt"
        with open(p,'w',encoding='utf-8') as f:
            for i,s in enumerate(result['segments'],1):
                st = self._fmt_ts(s.get('start',0.0))
                en = self._fmt_ts(s.get('end',0.0))
                txt = (s.get('text') or '').strip()
                f.write(f"{i}\n{st} --> {en}\n{txt}\n\n")
        return p

    def _fmt_ts(self, sec: float) -> str:
        h, r = divmod(int(sec), 3600); m, s = divmod(r, 60)
        ms = int((sec - int(sec))*1000)
        return f"{h:02}:{m:02}:{s:02},{ms:03}"

    def _open_file(self, path: Path):
        try:
            logger.debug(f"Открываю файл: {path}")
            if platform.system() == "Darwin": 
                subprocess.run(["open", str(path)], check=False)
            elif platform.system() == "Windows": 
                os.startfile(str(path))  # type: ignore
            else: 
                subprocess.run(["xdg-open", str(path)], check=False)
        except Exception as e:
            logger.warning(f"Не удалось открыть файл {path}: {e}")

# ------------------- MAIN -------------------
def main():
    global logger
    
    parser = argparse.ArgumentParser(
        description="Meeting Recorder & Transcriber v4.8",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры:
  %(prog)s list-devices                    # Показать устройства
  %(prog)s record "Совещание" --device :0  # Записать и транскрибировать
  %(prog)s transcribe file.wav             # Транскрибировать файл
  %(prog)s transcribe file.wav -v          # С подробным выводом
  %(prog)s transcribe file.wav --debug     # С отладочной информацией
        """
    )
    
    # Глобальные флаги логирования
    parser.add_argument("-v", "--verbose", action="store_true", 
                        help="Подробный вывод (INFO уровень)")
    parser.add_argument("--debug", action="store_true",
                        help="Отладочный вывод (DEBUG уровень)")
    
    subparsers = parser.add_subparsers(dest="command", required=True)

    # record
    p_rec = subparsers.add_parser("record", help="Записать встречу и транскрибировать")
    p_rec.add_argument("name", help="Название/базовое имя файла")
    p_rec.add_argument("--device", required=True, help="ID устройства ввода (см. list-devices)")

    # transcribe
    p_tr  = subparsers.add_parser("transcribe", help="Транскрибировать готовые файлы")
    p_tr.add_argument("files", nargs='+', type=Path)

    # list devices
    subparsers.add_parser("list-devices", help="Показать доступные устройства")

    args = parser.parse_args()
    
    # Настраиваем логирование
    logger = setup_logging(verbose=args.verbose, debug=args.debug)
    logger.debug(f"Запуск с аргументами: {args}")
    logger.debug(f"Конфигурация: model={Config.DEFAULT_MODEL}, backend={Config.ASR_BACKEND}, device={Config.ASR_DEVICE}")

    if args.command == "list-devices":
        MeetingRecorder().list_devices()
        sys.exit(0)

    if args.command == "record":
        logger.info(f"Режим записи: '{args.name}', устройство: {args.device}")
        rec = MeetingRecorder()
        safe_name = re.sub(r'[^\w\s-]', '', args.name).strip().replace(' ','_')
        base = Config.RECORDINGS_FOLDER / f"{safe_name}_{datetime.datetime.now():%Y%m%d_%H%M}"
        files = rec.record(base, args.device)
        if not files:
            logger.error("Запись не удалась")
            sys.exit(1)
        logger.info("📝 Начинаем транскрипцию...")
        print("\n📝 Транскрипция...")
        tr = EnhancedTranscriber()
        tr.transcribe_files(files)
        logger.info("Работа завершена успешно")
        sys.exit(0)

    if args.command == "transcribe":
        logger.info(f"Режим транскрипции: {len(args.files)} файл(ов)")
        tr = EnhancedTranscriber()
        tr.transcribe_files(args.files)
        logger.info("Работа завершена")
        sys.exit(0)

if __name__ == "__main__":
    main()
