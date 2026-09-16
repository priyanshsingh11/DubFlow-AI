from __future__ import annotations

import time
from pathlib import Path
from typing import Callable, Optional

from app import ffmpeg, settings
from app.downloader import download, get_video_id
from app.models import load_segments, save_segments
from app.synchronizer import build_voice_track
from app.transcriber import transcribe
from app.translator import translate
from app.tts import synthesize


def stage(label: str, output: Optional[Path], fn: Callable[[], None]) -> None:
    """Run one step, skipping it if its output file already exists (so long runs can resume)."""
    if output is not None and output.exists():
        print(f"{label}  (cached)")
        return
    print(label)
    started = time.time()
    fn()
    print(f"       done in {time.time() - started:.1f}s")


def _transcribe(audio: Path, segments_json: Path, language_txt: Path) -> None:
    segments, language = transcribe(audio)
    language_txt.write_text(language)
    save_segments(segments, segments_json)


def _read(path: Path) -> Optional[str]:
    return path.read_text().strip() if path.exists() else None


def run(url: str) -> Path:
    started = time.time()
    video_id = get_video_id(url)
    work = settings.TEMP_DIR / video_id
    work.mkdir(parents=True, exist_ok=True)
    settings.OUTPUT_DIR.mkdir(exist_ok=True)

    video = work / "video.mp4"
    audio = work / "audio.wav"
    segments_json = work / "segments.json"
    language_txt = work / "language.txt"
    translated_json = work / "translated.json"
    tts_dir = work / "tts"
    voice = work / "voice_track.wav"
    final = settings.OUTPUT_DIR / f"{video_id}_en.mp4"

    stage("[1/7] Downloading video", video, lambda: download(url, work))
    stage("[2/7] Extracting audio", audio,
          lambda: ffmpeg.run("-i", video, "-vn", "-ac", "1", "-ar", "16000", audio))
    stage("[3/7] Transcribing", segments_json,
          lambda: _transcribe(audio, segments_json, language_txt))
    stage("[4/7] Translating to English", translated_json,
          lambda: save_segments(translate(load_segments(segments_json), _read(language_txt)), translated_json))
    stage("[5/7] Synthesizing English speech", None,
          lambda: synthesize(load_segments(translated_json), tts_dir,
                             settings.TTS_VOICE, settings.TTS_CONCURRENCY))
    stage("[6/7] Syncing voice to original timing", voice,
          lambda: build_voice_track(load_segments(translated_json), tts_dir, ffmpeg.duration(video),
                                    voice, settings.MAX_SPEEDUP, settings.SAMPLE_RATE))
    stage("[7/7] Replacing audio track", final,
          lambda: ffmpeg.run("-i", video, "-i", voice, "-map", "0:v:0", "-map", "1:a:0",
                             "-c:v", "copy", "-c:a", "aac", "-shortest", final))

    print(f"\nSaved {final}  (total {(time.time() - started) / 60:.1f} min)")
    return final
