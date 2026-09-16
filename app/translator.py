from __future__ import annotations

from deep_translator import GoogleTranslator

from app.models import Segment


def translate(segments: list[Segment]) -> list[Segment]:
    translator = GoogleTranslator(source="auto", target="en")
    for i, seg in enumerate(segments, 1):
        seg.translated_text = (translator.translate(seg.source_text) or "") if seg.source_text else ""
        print(f"\r       {i}/{len(segments)} segments", end="", flush=True)
    print()
    return segments
