from __future__ import annotations

import json
import os
import time
from pathlib import Path

from deep_translator import GoogleTranslator
from deep_translator.exceptions import TooManyRequests
from groq import BadRequestError, Groq, RateLimitError

from app import settings
from app.models import Segment, load_segments, save_segments

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

# Constrained decoding keeps replies to this shape (Groq can still reject a long reply as json_validate_failed).
RESPONSE_SCHEMA = {
    "name": "translations",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "translations": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"id": {"type": "integer"}, "text": {"type": "string"}},
                    "required": ["id", "text"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["translations"],
        "additionalProperties": False,
    },
}


def translate(segments: list[Segment], source_language: str | None = None,
              checkpoint: Path | None = None) -> list[Segment]:
    """checkpoint: progress is saved there after every batch, and a rerun continues from it."""
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
    if checkpoint and checkpoint.exists():
        done = {s.id: s.translated_text for s in load_segments(checkpoint)}
        for seg in segments:
            seg.translated_text = done.get(seg.id, "")
        print(f"       resuming from {sum(1 for t in done.values() if t)} translated segments")
    for start in range(0, len(segments), settings.TRANSLATE_BATCH):
        batch = segments[start:start + settings.TRANSLATE_BATCH]
        context = segments[max(0, start - 5):start]
        if any(s.source_text and not s.translated_text for s in batch):
            _translate_llm(client, models, batch, context, max_words)
            if checkpoint:
                save_segments(segments, checkpoint)
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
    todo = [s for s in batch if s.source_text and not s.translated_text]
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
        except BadRequestError as e:
            if "json_validate_failed" not in str(e) or len(todo) == 1:
                print(f"\n       LLM request failed ({e}), retrying")
                by_id = {}
            else:  # long replies are the ones that come back malformed, so ask for half at a time
                print(f"\n       LLM returned invalid JSON for {len(todo)} lines, retrying in two smaller batches")
                half = len(todo) // 2
                _translate_llm(client, models, todo[:half], context, max_words, attempts)
                _translate_llm(client, models, todo[half:], (context + todo[:half])[-5:], max_words, attempts)
                return
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
        response_format={"type": "json_schema", "json_schema": RESPONSE_SCHEMA},
        temperature=0.3,
        reasoning_effort="low",
    )
    data = json.loads(response.choices[0].message.content)
    return {int(t["id"]): t["text"].strip() for t in data["translations"]}


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
