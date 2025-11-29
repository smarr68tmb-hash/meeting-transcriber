#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Система записи и транскрипции совещаний v4.7 (all-in-one)
- Запись (ffmpeg) + список устройств
- Надёжная safe-конвертация (-nostdin, mono 16kHz PCM)
- faster-whisper с cpu_threads (устранение подвисаний на CPU)
- tqdm-прогресс по секундам + «живой» статус каждые 2–3 сек (t≈MM:SS, сегментов: N)
- Первый проход без VAD, fallback с VAD/ru при пустом результате
- Итоговая сводка и сохранение TXT/JSON/SRT
"""

import os, sys, re, shutil, subprocess, datetime, json, time, argparse, platform
from pathlib import Path
from typing import List, Optional, Dict, Tuple

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
        print("⚠️ tqdm не установлен. Установите: pip install tqdm")
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
        print(f"🔍 Выполняю: {' '.join(cmd)}")
        try:
            res = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='ignore')
            output = (res.stderr or '') + "\n" + (res.stdout or '')
            print("\n" + output)
        except Exception as e:
            print(f"❌ Не удалось получить список устройств: {e}")

    def _record_probe(self, device: str) -> bool:
        if Config.PRE_RECORD_PROBE <= 0:
            return True
        print(f"🔎 Пробная запись ({Config.PRE_RECORD_PROBE} сек) — проверка устройства '{device}'...")
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
        try:
            with open(log_file,'w',encoding='utf-8') as log:
                p = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
                p.wait()
            ok = (p.returncode == 0) and ffprobe_ok(probe_file)
            if ok:
                print("✅ Проба успешна")
            else:
                print("❌ Пробная запись не удалась. См. лог:", log_file)
            return ok
        finally:
            if probe_file.exists():
                try: probe_file.unlink()
                except: pass

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
        print("\n" + "="*52)
        print(f"🔴 ЗАПИСЬ НАЧАТА -> {output_path.name}")
        print("⏹  Остановка: Ctrl+C")
        print("="*52)
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
            if self.recording_process:
                self.recording_process.terminate()
                self.recording_process.wait()
        except Exception as e:
            print(f"\n❌ Ошибка записи: {e}")
            return None

        print("\n✅ Запись завершена")
        if output_path.exists() and ffprobe_ok(output_path):
            return [output_path]
        print(f"❌ Файл записи не создан/повреждён. Лог: {log_file}")
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
            return
        print(f"\n🤖 Загрузка модели '{self.model_size}' (backend={self.backend})...")
        if self.backend == 'faster':
            from faster_whisper import WhisperModel
            device = self._resolve_device_faster()
            cpu_threads = Config.FASTER_CPU_THREADS if device == 'cpu' else 0
            if device == 'cpu':
                print(f"   (cpu_threads={cpu_threads})")
            self.model = WhisperModel(self.model_size, device=device,
                                      compute_type=Config.FASTER_COMPUTE,
                                      cpu_threads=cpu_threads)
            self.device = device
        else:
            import whisper
            device, fp16 = self._resolve_device_whisper()
            self.model = whisper.load_model(self.model_size, device=device)
            self.device = device
            self.use_fp16 = fp16
        self.model_loaded = True
        print(f"✅ Модель загружена (device={self.device})")

    # --- верхний уровень ---
    def transcribe_files(self, files: List[Path]) -> None:
        self._load_model()
        success = 0
        total = len(files)
        for i, f in enumerate(files, 1):
            print(f"\n━━━ Файл {i}/{total}: {f.name} ━━━")
            ok = self._transcribe_single(f, auto_open=(i==1))
            success += 1 if ok else 0
        print(f"\n📊 Итог: успешно {success}/{total}, ошибок {total-success}")

    # --- обработка одного файла ---
    def _transcribe_single(self, audio_file: Path, auto_open: bool=True) -> bool:
        if not audio_file.exists() or not ffprobe_ok(audio_file):
            print(f"❌ Файл не найден или повреждён: {audio_file}")
            return False

        # Safe WAV
        safe_file = audio_file.with_suffix(f".safe{datetime.datetime.now():%H%M%S}.wav")
        print("Подготовка аудио...")
        try:
            subprocess.run([
                "ffmpeg","-y","-i",str(audio_file),
                "-ar","16000","-ac","1","-c:a","pcm_s16le","-nostdin", str(safe_file)
            ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except subprocess.CalledProcessError as e:
            print(f"❌ Конвертация не удалась: {e}")
            return False

        t0 = time.time()
        language = 'ru' if Config.FORCE_RU else None

        try:
            # 1) Первый проход — БЕЗ VAD (чтобы не «ждал речи»)
            result = self._run_asr_once(safe_file, language=language, use_vad=False)
            # 2) Fallback — c VAD и ru
            if not result or not result.get("segments"):
                print("⚠️ Пусто без VAD, пробую с VAD...")
                result = self._run_asr_once(safe_file, language=language or 'ru', use_vad=True)
            if not result or not result.get("text","").strip():
                print("❌ Транскрипция не дала результата")
                return False

            elapsed = time.time() - t0
            print(f"✅ Сегментов: {len(result['segments'])}, слов: {len(result['text'].split())}, время: {elapsed/60:.1f} мин.")

            ts  = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            base= f"transcript_{audio_file.stem}_{ts}"
            txt = self._save_txt(result, base, audio_file.name, (language or 'auto'))
            jsn = self._save_json(result, base, audio_file.name, (language or 'auto'))
            srt = self._save_srt(result, base)
            print("📄 Сохранено:", txt.name, jsn.name, srt.name)
            if auto_open:
                self._open_file(txt)
            return True
        finally:
            if safe_file.exists():
                try: safe_file.unlink()
                except: pass

    # --- единичный прогон ASR с прогрессом ---
    def _run_asr_once(self, wav_file: Path, language: Optional[str], use_vad: bool):
        total_sec = get_audio_duration(wav_file)
        pbar = tqdm(total=int(total_sec) if total_sec>0 else None,
                    desc="Транскрипция", unit="s",
                    bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')
        segs, texts = [], []
        last_progress = 0
        last_print = time.time()

        try:
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
                        print(f"\nDEBUG [{s.start:.2f}-{s.end:.2f}] {s.text[:60]}")

                if language is None:
                    language = getattr(info, 'language', None)

            else:  # openai/whisper
                import whisper
                res = self.model.transcribe(str(wav_file),
                                            language=language,
                                            fp16=self.use_fp16,
                                            word_timestamps=True)
                segs = res.get("segments", [])
                texts = [seg.get("text","") for seg in segs]
                pbar.update(int(total_sec) if total_sec else 0)

            return {'text': " ".join(texts).strip(), 'segments': segs}
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
            if platform.system() == "Darwin": subprocess.run(["open", str(path)], check=False)
            elif platform.system() == "Windows": os.startfile(str(path))  # type: ignore
            else: subprocess.run(["xdg-open", str(path)], check=False)
        except Exception: pass

# ------------------- MAIN -------------------
def main():
    parser = argparse.ArgumentParser(description="Meeting Recorder & Transcriber v4.7")
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

    if args.command == "list-devices":
        MeetingRecorder().list_devices()
        sys.exit(0)

    if args.command == "record":
        rec = MeetingRecorder()
        safe_name = re.sub(r'[^\w\s-]', '', args.name).strip().replace(' ','_')
        base = Config.RECORDINGS_FOLDER / f"{safe_name}_{datetime.datetime.now():%Y%m%d_%H%M}"
        files = rec.record(base, args.device)
        if not files:
            sys.exit(1)
        print("\n📝 Транскрипция...")
        tr = EnhancedTranscriber()
        tr.transcribe_files(files)
        sys.exit(0)

    if args.command == "transcribe":
        tr = EnhancedTranscriber()
        tr.transcribe_files(args.files)
        sys.exit(0)

if __name__ == "__main__":
    main()
