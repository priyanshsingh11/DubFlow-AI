from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
from pydub import AudioSegment

from app import ffmpeg
from app.models import Segment
from app.tts import clip_path


def build_voice_track(segments: list[Segment], tts_dir: Path, total_seconds: float,
                      out_path: Path, max_speedup: float, sr: int) -> None:
    """Place each English clip at its original start time, sped up if it would run into the next line."""
    track = np.zeros(int(total_seconds * sr), dtype=np.int32)

    for i, seg in enumerate(segments):
        path = clip_path(tts_dir, seg)
        if not path.exists():
            continue

        # A clip may use the silence before the next line, not just its own segment.
        next_start = segments[i + 1].start if i + 1 < len(segments) else total_seconds
        slot = max(next_start - seg.start, 0.1)
        clip = AudioSegment.from_file(path)

        speed = clip.duration_seconds / slot
        if speed > 1.0:
            fitted = path.with_suffix(".fit.wav")
            ffmpeg.run("-i", path, "-filter:a", f"atempo={min(speed, max_speedup):.3f}", fitted)
            clip = AudioSegment.from_file(fitted)

        clip = clip.set_frame_rate(sr).set_channels(1).set_sample_width(2)
        samples = np.array(clip.get_array_of_samples(), dtype=np.int32)
        start = int(seg.start * sr)
        end = min(start + len(samples), len(track))
        track[start:end] += samples[: end - start]

    with wave.open(str(out_path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(np.clip(track, -32768, 32767).astype(np.int16).tobytes())
