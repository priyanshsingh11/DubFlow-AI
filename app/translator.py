from __future__ import annotations

import json
import os
import time

from deep_translator import GoogleTranslator
from deep_translator.exceptions import TooManyRequests
from groq import Groq

from app import settings
from app.models import Segment

SYSTEM_PROMPT = """You translate video dialogue into English for dubbing.
- Translate for meaning and natural spoken phrasing, not word-for-word. Keep the speaker's tone and energy.
- Each line has max_words so the English fits the original timing. Stay within it; rephrase more concisely if needed, but keep the key meaning.
- previous_lines are context only; do not translate them again.
Return JSON: {"translations": [{"id": <id>, "text": "<english>"}]} with exactly one entry per line id."""


def translate(segments: list[Segment], source_language: str | None = None) -> list[Segment]:
    if source_language == "en":
        print("       audio is already English, keeping original text")
        for seg in segments:
            seg.translated_text = seg.source_text
        return segments

    if not os.getenv("GROQ_API_KEY"):
        print("       GROQ_API_KEY not set, using Google Translate")
        _translate_google(segments)
        return segments

    client = Groq(max_retries=6)  # retries with backoff on rate limits
    max_words = _max_words(segments)
    for start in range(0, len(segments), settings.TRANSLATE_BATCH):
        batch = segments[start:start + settings.TRANSLATE_BATCH]
        context = segments[max(0, start - 3):start]
        try:
            _translate_llm(client, batch, context, max_words)
        except Exception as e:  # one bad batch shouldn't stop a 2-hour run
            print(f"\n       LLM batch failed ({e}), using Google Translate for it")
            _translate_google(batch)
        print(f"\r       {min(start + len(batch), len(segments))}/{len(segments)} segments", end="", flush=True)
    print()
    return segments


def _max_words(segments: list[Segment]) -> dict[int, int]:
    """Words that fit before the next line starts, at normal English speaking pace."""
    limits = {}
    for i, seg in enumerate(segments):
        slot_end = segments[i + 1].start if i + 1 < len(segments) else seg.end
        limits[seg.id] = max(2, int((slot_end - seg.start) * settings.WORDS_PER_SECOND))
    return limits


def _translate_llm(client: Groq, batch: list[Segment], context: list[Segment], max_words: dict[int, int]) -> None:
    payload = {
        "previous_lines": [{"source": s.source_text, "english": s.translated_text} for s in context],
        "lines": [{"id": s.id, "text": s.source_text, "max_words": max_words[s.id]}
                  for s in batch if s.source_text],
    }
    response = client.chat.completions.create(
        model=settings.GROQ_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
        response_format={"type": "json_object"},
        temperature=0.3,
        reasoning_effort="low",
    )
    translations = json.loads(response.choices[0].message.content)["translations"]
    by_id = {int(t["id"]): t["text"].strip() for t in translations}

    missing = []
    for seg in batch:
        if seg.id in by_id:
            seg.translated_text = by_id[seg.id]
        elif seg.source_text:
            missing.append(seg)
    _translate_google(missing)


def _translate_google(segments: list[Segment]) -> None:
    translator = GoogleTranslator(source="auto", target="en")
    for seg in segments:
        seg.translated_text = _google_one(translator, seg.source_text) if seg.source_text else ""


def _google_one(translator: GoogleTranslator, text: str, attempts: int = 5) -> str:
    """Free Google endpoint rate-limits (5 req/s); back off, and keep the original text rather than crash."""
    for attempt in range(attempts):
        try:
            time.sleep(0.25)
            return translator.translate(text) or ""
        except TooManyRequests:
            time.sleep(2 ** attempt * 5)  # 5, 10, 20, 40, 80s
        except Exception as e:
            print(f"\n       Google Translate failed ({e}), keeping original text")
            return text
    print("\n       Google Translate still rate-limited, keeping original text")
    return text
