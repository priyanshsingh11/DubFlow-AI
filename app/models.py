from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class Segment:
    id: int
    start: float
    end: float
    source_text: str
    translated_text: str = ""
    speaker: str = "SPEAKER_0"


def save_segments(segments: list[Segment], path: Path) -> None:
    path.write_text(json.dumps([asdict(s) for s in segments], ensure_ascii=False, indent=2))


def load_segments(path: Path) -> list[Segment]:
    return [Segment(**d) for d in json.loads(path.read_text())]
