from __future__ import annotations

from pathlib import Path

from faster_whisper import WhisperModel

from app.models import Segment


def transcribe(audio_path: Path, model_size: str) -> tuple[list[Segment], str]:
    model = WhisperModel(model_size, device="auto", compute_type="int8")
    raw_segments, info = model.transcribe(str(audio_path), vad_filter=True)
    print(f"       language: {info.language} ({info.language_probability:.0%}), "
          f"audio: {info.duration / 60:.1f} min")

    segments = []
    for i, s in enumerate(raw_segments):  # generator: transcription happens while iterating
        segments.append(Segment(i, round(s.start, 2), round(s.end, 2), s.text.strip()))
        print(f"\r       transcribed up to {s.end / 60:.1f} min", end="", flush=True)
    print()
    return segments, info.language
