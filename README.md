# DubFlow AI

Turns a YouTube video in any language into an English-dubbed video.

```
URL → download (yt-dlp) → transcribe (faster-whisper) → translate (Google)
    → synthesize (edge-tts) → sync to timestamps → replace audio (ffmpeg, video not re-encoded)
```

## Setup

```bash
brew install ffmpeg
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python main.py "https://www.youtube.com/watch?v=..."
```

Output: `output/<video_id>_en.mp4`. Intermediate files go to `temp/<video_id>/`;
re-running the same URL skips any step that already finished.

Settings (Whisper model, voice, max speed-up) are in `app/settings.py`.
