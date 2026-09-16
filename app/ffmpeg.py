import subprocess
from pathlib import Path


def run(*args) -> None:
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *map(str, args)], check=True)


def duration(path: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True,
    )
    return float(out.stdout.strip())


def encode_mp3(samples, sample_rate: int) -> bytes:
    """Compress float32 mono samples to MP3 in memory (keeps uploads small)."""
    out = subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-f", "f32le", "-ar", str(sample_rate), "-ac", "1", "-i", "-",
         "-b:a", "64k", "-f", "mp3", "-"],
        input=samples.astype("float32").tobytes(), capture_output=True, check=True,
    )
    return out.stdout
