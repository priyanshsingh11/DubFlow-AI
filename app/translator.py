from __future__ import annotations

import json
import os
import time

from deep_translator import GoogleTranslator
from deep_translator.exceptions import TooManyRequests
from groq import Groq, RateLimitError

from app import settings
from app.models import Segment

SYSTEM_PROMPT = """You translate video dialogue into English for dubbing.
- The lines come from automatic speech recognition, in order, from one continuous video. They may mix languages
  (e.g. Hindi with English words) and contain misheard words; use the surrounding lines to work out what was meant.
- One sentence is often split across several lines. Translate each line so the English lines read naturally one after another.
- Translate for meaning and natural spoken phrasing, not word-for-word. Keep the speaker's tone and energy.
- Keep names, products and technical terms in their usual English form.
- Only say what the speaker said. Never add content that isn't in the line.
- Each line has max_words so the English fits the original timing. Stay within it; rephrase more concisely if needed, but keep the key meaning.
- Every line is dubbed at its own time, so never merge lines: each id gets its own English text, even if short.
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
    models = list(settings.GROQ_MODELS)
    for start in range(0, len(segments), settings.TRANSLATE_BATCH):
        batch = segments[start:start + settings.TRANSLATE_BATCH]
        context = segments[max(0, start - 5):start]
        _translate_llm(client, models, batch, context, max_words)
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


def _translate_llm(client: Groq, models: list[str], batch: list[Segment], context: list[Segment],
                   max_words: dict[int, int], attempts: int = 3) -> None:
    """Ask the LLM again for any lines it skipped or merged; Google Translate is the last resort."""
    todo = [s for s in batch if s.source_text]
    for _ in range(attempts):
        if not todo:
            return
        try:
            by_id = _request_llm(client, models[0], todo, context, max_words)
        except RateLimitError as e:
            if "per day" in str(e) and len(models) > 1:
                print(f"\n       {models[0]} daily token limit reached, switching to {models[1]}")
                models.pop(0)  # for the rest of the run
            else:
                print(f"\n       LLM rate-limited ({e}), retrying")
            by_id = {}
        except Exception as e:  # e.g. invalid JSON; one bad reply shouldn't stop a 2-hour run
            print(f"\n       LLM request failed ({e}), retrying")
            by_id = {}
        for seg in todo:
            seg.translated_text = by_id.get(seg.id, "")
        todo = [s for s in todo if not s.translated_text]
    if todo:
        print(f"\n       LLM skipped {len(todo)} lines, using Google Translate for them")
        _translate_google(todo)


def _request_llm(client: Groq, model: str, lines: list[Segment], context: list[Segment],
                 max_words: dict[int, int]) -> dict[int, str]:
    payload = {
        "previous_lines": [{"source": s.source_text, "english": s.translated_text} for s in context],
        "lines": [{"id": s.id, "text": s.source_text, "max_words": max_words[s.id]} for s in lines],
    }
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
        response_format={"type": "json_object"},
        temperature=0.3,
        reasoning_effort="low",
    )
    data = json.loads(response.choices[0].message.content)
    translations = data["translations"] if isinstance(data, dict) else data  # smaller models may drop the wrapper
    return {int(t["id"]): t["text"].strip() for t in translations}


def _translate_google(segments: list[Segment]) -> None:
    translator = GoogleTranslator(source="auto", target="en")
    for seg in segments:
        seg.translated_text = _google_one(translator, seg.source_text) if seg.source_text else ""


def _google_one(translator: GoogleTranslator, text: str, attempts: int = 5) -> str:
    """Free Google endpoint rate-limits (5 req/s); back off, and leave the line silent rather than crash.

    An untranslated line would be read out by the English voice as gibberish, so failure returns "".
    """
    for attempt in range(attempts):
        try:
            time.sleep(0.25)
            return translator.translate(text) or ""
        except TooManyRequests:
            time.sleep(2 ** attempt * 5)  # 5, 10, 20, 40, 80s
        except Exception as e:
            print(f"\n       Google Translate failed ({e}), leaving line silent")
            return ""
    print("\n       Google Translate still rate-limited, leaving line silent")
    return ""
