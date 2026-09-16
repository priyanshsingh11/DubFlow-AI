from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
from pydub import AudioSegment
from pydub.silence import detect_leading_silence

from app import ffmpeg
from app.models import Segment
from app.tts import clip_path


def build_voice_track(segments: list[Segment], tts_dir: Path, total_seconds: float,
                      out_path: Path, max_speedup: float, sr: int) -> None:
    """Place each English clip at its original start time, sped up if it would run into the next line.

    Clips never overlap: if the previous clip is still playing, the next one waits for it and
    speeds up to catch back up with the original timing.
    """
    track = np.zeros(int(total_seconds * sr), dtype=np.int16)
    cursor = 0.0  # time the previous clip finished
    max_lag = 0.0

    for i, seg in enumerate(segments):
        path = clip_path(tts_dir, seg)
        if not path.exists():
            continue

        start = max(seg.start, cursor)
        next_start = segments[i + 1].start if i + 1 < len(segments) else total_seconds
        slot = max(next_start - start, 0.1)  # a clip may use the pause before the next line
        clip = _trim_silence(AudioSegment.from_file(path))

        speed = clip.duration_seconds / slot
        if speed > 1.0:
            trimmed = path.with_suffix(".trim.wav")
            fitted = path.with_suffix(".fit.wav")
            clip.export(trimmed, format="wav")
            ffmpeg.run("-i", trimmed, "-filter:a", f"atempo={min(speed, max_speedup):.3f}", fitted)
            clip = AudioSegment.from_file(fitted)

        clip = clip.set_frame_rate(sr).set_channels(1).set_sample_width(2)
        samples = np.array(clip.get_array_of_samples(), dtype=np.int16)
        first = int(start * sr)
        last = min(first + len(samples), len(track))
        track[first:last] = samples[: last - first]
        cursor = start + clip.duration_seconds
        max_lag = max(max_lag, start - seg.start)

    print(f"       largest delay behind the original: {max_lag:.1f}s")
    with wave.open(str(out_path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(track.tobytes())


def _trim_silence(clip: AudioSegment) -> AudioSegment:
    """TTS clips start and end with silence; cutting it leaves more room for speech."""
    start = detect_leading_silence(clip, silence_threshold=-45)
    end = detect_leading_silence(clip.reverse(), silence_threshold=-45)
    return clip[start:len(clip) - end]
