from __future__ import annotations

import asyncio
from pathlib import Path

import edge_tts

from app.models import Segment


def clip_path(tts_dir: Path, seg: Segment) -> Path:
    return tts_dir / f"{seg.id:04d}.mp3"


def synthesize(segments: list[Segment], tts_dir: Path, voice: str, concurrency: int) -> None:
    tts_dir.mkdir(exist_ok=True)
    todo = [s for s in segments if s.translated_text and not clip_path(tts_dir, s).exists()]
    if not todo:
        return

    async def main():
        sem = asyncio.Semaphore(concurrency)
        done = 0

        async def one(seg: Segment):
            nonlocal done
            async with sem:
                await edge_tts.Communicate(seg.translated_text, voice).save(str(clip_path(tts_dir, seg)))
            done += 1
            print(f"\r       {done}/{len(todo)} clips", end="", flush=True)

        await asyncio.gather(*(one(s) for s in todo))
        print()

    asyncio.run(main())
