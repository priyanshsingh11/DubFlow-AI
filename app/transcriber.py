from __future__ import annotations

import io
import os
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import NamedTuple

import numpy as np
from faster_whisper import WhisperModel
from faster_whisper.audio import decode_audio
from faster_whisper.vad import VadOptions, get_speech_timestamps
from groq import Groq

from app import ffmpeg, settings
from app.models import Segment

SR = 16000


class Word(NamedTuple):
    text: str
    start: float
    end: float


def transcribe(audio_path: Path) -> tuple[list[Segment], str]:
    """Transcribe in chunks cut at silences: Groq's hosted Whisper if GROQ_API_KEY is set, else locally."""
    audio = decode_audio(str(audio_path), sampling_rate=SR)
    speech = [(s["start"] / SR, s["end"] / SR)
              for s in get_speech_timestamps(audio, VadOptions(min_silence_duration_ms=300))]
    chunks = _split_at_silence(speech, len(audio) / SR, settings.CHUNK_SECONDS)
    use_groq = bool(os.getenv("GROQ_API_KEY"))
    print(f"       audio: {len(audio) / SR / 60:.1f} min in {len(chunks)} chunks, "
          f"using {'Groq ' + settings.GROQ_WHISPER_MODEL if use_groq else 'local ' + settings.WHISPER_MODEL}")

    lines, languages = [], []
    for start, end in chunks:
        words, language = _transcribe_range(audio, start, end, use_groq)
        words = _fill_skipped_speech(audio, words, speech, start, end, use_groq)
        languages.append(language)
        lines += _split_into_lines(words)
        print(f"\r       transcribed up to {end / 60:.1f} min", end="", flush=True)
    print()

    language = Counter(languages).most_common(1)[0][0]
    print(f"       language: {language}")
    segments = [Segment(i, round(ws[0].start, 2), round(ws[-1].end, 2), " ".join(w.text for w in ws))
                for i, ws in enumerate(lines)]
    return segments, language


def _split_at_silence(speech: list[tuple[float, float]], total: float, max_seconds: float) -> list[tuple[float, float]]:
    """Chunk boundaries placed in the middle of pauses, so no word is cut in half."""
    chunks, chunk_start = [], 0.0
    for prev, cur in zip(speech, speech[1:]):
        if cur[1] - chunk_start > max_seconds:
            cut = (prev[1] + cur[0]) / 2
            chunks.append((chunk_start, cut))
            chunk_start = cut
    chunks.append((chunk_start, total))
    return chunks


def _transcribe_range(audio: np.ndarray, start: float, end: float, use_groq: bool) -> tuple[list[Word], str]:
    """Words between start and end (seconds), with timestamps relative to the whole audio."""
    clip = audio[int(start * SR):int(end * SR)]
    try:
        words, language = _groq(clip) if use_groq else _local(clip)
    except Exception as e:  # rate limit or network trouble: don't lose the run
        print(f"\n       Groq transcription failed ({e}), transcribing locally")
        words, language = _local(clip)
    return [w._replace(start=w.start + start, end=w.end + start) for w in words], language


def _fill_skipped_speech(audio: np.ndarray, words: list[Word], speech: list[tuple[float, float]],
                         start: float, end: float, use_groq: bool) -> list[Word]:
    """Whisper sometimes jumps over a stretch of speech; transcribe any long gap the VAD says has speech in it."""
    gaps = zip([start] + [w.end for w in words], [w.start for w in words] + [end])
    filled = list(words)
    for gap_start, gap_end in gaps:
        if gap_end - gap_start < settings.MIN_GAP_SECONDS:
            continue
        spoken = sum(max(0.0, min(b, gap_end) - max(a, gap_start)) for a, b in speech)
        if spoken >= settings.MIN_GAP_SECONDS / 2:
            retry, _ = _transcribe_range(audio, gap_start, gap_end, use_groq)
            filled += [w for w in retry if gap_start <= w.start < gap_end]
    return sorted(filled, key=lambda w: w.start)


def _groq(clip: np.ndarray) -> tuple[list[Word], str]:
    response = Groq(max_retries=6).audio.transcriptions.create(
        file=("chunk.mp3", io.BytesIO(ffmpeg.encode_mp3(clip, SR))),
        model=settings.GROQ_WHISPER_MODEL,
        response_format="verbose_json",
        timestamp_granularities=["word", "segment"],
        temperature=0,
    )
    # Whisper invents text over music and silence; drop words from segments it wasn't confident were speech.
    # (No compression-ratio check: non-Latin scripts like Hindi score high on it even when correct.)
    rejected = [(s["start"], s["end"]) for s in response.segments
                if s["no_speech_prob"] > 0.6 and s["avg_logprob"] < -1]
    words = [_word(w["word"], w["start"], w["end"]) for w in response.words]
    return [w for w in words if w.text and not any(a <= w.start < b for a, b in rejected)], \
        _language_code(response.language)


def _local(clip: np.ndarray) -> tuple[list[Word], str]:
    raw, info = _local_model().transcribe(clip, vad_filter=True, word_timestamps=True,
                                          condition_on_previous_text=False)  # avoids repetition loops
    words = [_word(w.word, w.start, w.end) for s in raw for w in s.words]
    return [w for w in words if w.text], info.language


def _word(text: str, start: float, end: float) -> Word:
    # A word before a pause is often stretched across the whole pause; no spoken word lasts this long.
    return Word(text.strip(), start, min(end, start + settings.MAX_WORD_SECONDS))


@lru_cache(maxsize=1)
def _local_model() -> WhisperModel:
    return WhisperModel(settings.WHISPER_MODEL, device="auto", compute_type="int8")


def _language_code(name: str) -> str:
    """Groq reports 'English', faster-whisper reports 'en'; only English is treated specially."""
    return "en" if name.lower() in ("en", "english") else name.lower()


def _split_into_lines(words: list[Word]) -> list[list[Word]]:
    """Group words into dub lines of MIN..MAX_LINE_SECONDS, ending at pauses or punctuation where possible."""
    lines, current = [], []
    for i, word in enumerate(words):
        current.append(word)
        duration = word.end - current[0].start
        next_gap = words[i + 1].start - word.end if i + 1 < len(words) else 0
        natural_break = word.text[-1] in ".?!,।" or next_gap >= 0.4
        if (duration >= settings.MAX_LINE_SECONDS or next_gap >= 2
                or (duration >= settings.MIN_LINE_SECONDS and natural_break)):
            lines.append(current)
            current = []
    if current:
        if (lines and current[-1].end - current[0].start < settings.MIN_LINE_SECONDS
                and current[0].start - lines[-1][-1].end < 2):
            lines[-1] += current  # don't leave a tiny fragment on its own
        else:
            lines.append(current)
    return lines
