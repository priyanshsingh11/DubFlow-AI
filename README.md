# DubFlow AI

Turns a YouTube video in any language into an English-dubbed video.

```
URL → download (yt-dlp) → extract audio (ffmpeg) → transcribe (faster-whisper)
    → translate (Groq LLM, Google fallback) → synthesize (edge-tts)
    → sync to original timestamps → replace audio (ffmpeg, video not re-encoded)
```

Transcription runs locally on your machine with an open-source Whisper model.
Translation and text-to-speech use online services.

---

## Setup

```bash
brew install ffmpeg python@3.12
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Optional: install Node.js (`brew install node`). yt-dlp needs a JavaScript runtime to
unlock some YouTube formats; it uses Deno or Node, whichever is installed.

### API key (optional)

Create a `.env` file in the project root:

```
GROQ_API_KEY=your_key_here
```

With a key, translation uses a Groq-hosted LLM. Without one, it falls back to the free
Google Translate endpoint. `.env` is ignored by git.

## Run

```bash
python main.py "https://www.youtube.com/watch?v=..."
# or run without a URL and paste it when prompted
python main.py
```

Output: `output/<video_id>_en.mp4`.

---

## How it works

`app/pipeline.py` runs seven steps. Each one writes a file to `temp/<video_id>/`.
If that file already exists, the step is skipped and marked `(cached)`, so an interrupted
run resumes where it stopped when you rerun the same URL.

| # | Step | Code | Output |
|---|------|------|--------|
| 1 | Download video | `app/downloader.py` | `video.mp4` |
| 2 | Extract audio (mono, 16 kHz) | `app/ffmpeg.py` | `audio.wav` |
| 3 | Transcribe | `app/transcriber.py` | `segments.json`, `language.txt` |
| 4 | Translate to English | `app/translator.py` | `translated.json` |
| 5 | Synthesize speech | `app/tts.py` | `tts/0000.mp3`, `tts/0001.mp3`, … |
| 6 | Sync voice to timing | `app/synchronizer.py` | `voice_track.wav` |
| 7 | Replace audio track | `app/ffmpeg.py` | `output/<video_id>_en.mp4` |

### 1. Download (`app/downloader.py`)

- Prefers H.264 (`avc1`) video with M4A audio, merged into MP4, so the result plays everywhere.
- Tolerates slow connections: 60 s socket timeout, 10 retries, and 10 retries per fragment.

### 2. Extract audio

ffmpeg converts the audio to mono 16 kHz WAV, the format Whisper expects.

### 3. Transcribe (`app/transcriber.py`)

- Uses `faster-whisper` (a CTranslate2 build of OpenAI Whisper) with `compute_type="int8"`
  and `vad_filter=True`, which skips silent parts.
- Detects the spoken language automatically and saves it to `language.txt`.
- Each segment (`app/models.py`) stores `id`, `start`, `end`, `source_text`, `translated_text` and `speaker`.

### 4. Translate (`app/translator.py`)

- **English audio:** the original text is kept and no translation happens.
- **No `GROQ_API_KEY`:** each line goes through Google Translate (`deep-translator`).
  The free endpoint allows about 5 requests per second, so there is a 0.25 s delay between
  requests and a backoff of 5, 10, 20, 40 and 80 s when rate-limited. If it still fails,
  the original text is kept instead of stopping the run.
- **With `GROQ_API_KEY`:** lines are sent to `GROQ_MODEL` in batches of `TRANSLATE_BATCH`:
  - The prompt asks for natural spoken English that keeps the speaker's tone, not a word-for-word translation.
  - Each line gets a `max_words` limit: the time until the next line starts × `WORDS_PER_SECOND`.
    This keeps the English short enough to fit the original timing.
  - The previous 3 translated lines are sent as context for consistency.
  - The response is requested as JSON (`temperature=0.3`, `reasoning_effort="low"`).
  - The Groq client retries up to 6 times on rate limits. If a whole batch fails, that
    batch uses Google Translate. Lines missing from the LLM reply are also translated with Google.

### 5. Synthesize speech (`app/tts.py`)

- Uses Microsoft Edge TTS (`edge-tts`) with the voice in `TTS_VOICE`.
- Makes `TTS_CONCURRENCY` requests in parallel.
- Clips that already exist are skipped, so this step also resumes after an interruption.

### 6. Sync to original timing (`app/synchronizer.py`)

- Each English clip is placed at its segment's original start time.
- A clip may use the gap before the next line, not just its own segment.
- If a clip is still too long, ffmpeg's `atempo` speeds it up, but never beyond `MAX_SPEEDUP`
  so speech stays natural. Speed-up changes the tempo without changing the pitch.
- All clips are mixed into one mono track at `SAMPLE_RATE`, clipped to 16-bit range.

### 7. Replace audio

ffmpeg copies the original video stream unchanged (`-c:v copy`, fast and lossless) and adds
the new English audio as AAC.

---

## Whisper model details

### Which model is used

Set in `app/settings.py`:

```python
WHISPER_MODEL = "small"   # tiny | base | small | medium | large-v3
```

It is loaded in `app/transcriber.py`:

```python
model = WhisperModel(model_size, device="auto", compute_type="int8")
```

### Where the model is downloaded

The project has no download code of its own. On the first run, `faster-whisper` downloads
the model from Hugging Face automatically:

1. `WhisperModel.__init__` (`.venv/lib/python3.12/site-packages/faster_whisper/transcribe.py`)
   checks whether the name is a local folder. If not, it calls `download_model()`.
2. `download_model()` (`faster_whisper/utils.py`) looks up the name in `_MODELS`,
   e.g. `"small"` → `Systran/faster-whisper-small`.
3. It calls `huggingface_hub.snapshot_download()` and fetches only `config.json`,
   `preprocessor_config.json`, `model.bin`, `tokenizer.json` and `vocabulary.*`.
4. Files are saved to `~/.cache/huggingface/hub/models--Systran--faster-whisper-<size>/`.
   Later runs use this cache and don't download again.

To control this, pass options to `WhisperModel`:

```python
WhisperModel(model_size, ..., download_root="models/")     # save somewhere else
WhisperModel(model_size, ..., local_files_only=True)       # never go online
WhisperModel("/path/to/faster-whisper-small", ...)         # load a local folder
```

### Storage and memory

Measured on this project with `small`: **464 MB on disk**, and the whole `main.py` process
used about **1.2 GB of RAM** during a run. The model itself accounts for roughly 0.5–1 GB;
the rest is Python, the libraries and the audio data.

Approximate figures for other sizes with `int8`:

| Model | Disk | RAM | Notes |
|-------|------|-----|-------|
| tiny | ~75 MB | ~0.3 GB | fastest, least accurate |
| base | ~145 MB | ~0.4 GB | |
| **small** (default) | **~464 MB** | **~0.6–1 GB** | good balance |
| medium | ~1.5 GB | ~1.5–2 GB | |
| large-v3 | ~3 GB | ~3–4 GB | most accurate, much slower on CPU |

`int8` uses about half the memory of full precision, with little loss in accuracy.

On a Mac, `faster-whisper` runs on the **CPU** (CTranslate2 doesn't support Apple GPUs),
so `device="auto"` means CPU. Larger models fit in memory on a 16 GB Mac but take much
longer to transcribe. On a machine with an NVIDIA GPU, `device="auto"` uses CUDA.

To delete a downloaded model:

```bash
rm -rf ~/.cache/huggingface/hub/models--Systran--faster-whisper-small
```

---

## Settings (`app/settings.py`)

| Setting | Default | Meaning |
|---------|---------|---------|
| `WHISPER_MODEL` | `"small"` | Whisper model size (see above) |
| `TTS_VOICE` | `"en-US-GuyNeural"` | Edge TTS voice; list them with `edge-tts --list-voices` |
| `TTS_CONCURRENCY` | `8` | Parallel TTS requests |
| `MAX_SPEEDUP` | `1.25` | Maximum speed-up for a clip that's too long |
| `SAMPLE_RATE` | `24000` | Sample rate of the dubbed voice track |
| `GROQ_MODEL` | `"openai/gpt-oss-120b"` | LLM used when `GROQ_API_KEY` is set |
| `TRANSLATE_BATCH` | `30` | Segments per LLM request |
| `WORDS_PER_SECOND` | `2.6` | English speaking pace, used for `max_words` |

## Project layout

```
main.py              CLI entry point; loads .env and runs the pipeline
app/
  pipeline.py        the 7 steps, with caching and timing
  downloader.py      yt-dlp download and video ID lookup
  ffmpeg.py          ffmpeg / ffprobe helpers
  transcriber.py     faster-whisper transcription
  translator.py      Groq LLM translation with Google Translate fallback
  tts.py             edge-tts speech synthesis
  synchronizer.py    places and speeds up clips to match timing
  models.py          Segment dataclass and JSON save/load
  settings.py        configuration
temp/<video_id>/     intermediate files (safe to delete)
output/              finished dubbed videos
```

## Troubleshooting

- **Re-do a step:** delete its file in `temp/<video_id>/`, e.g. delete `translated.json`
  to translate again. Also delete the outputs of every later step (`tts/`, `voice_track.wav`
  and the output video), or they will be reused with the old text.
- **Start over completely:** delete `temp/<video_id>/`.
- **`ffmpeg` not found:** run `brew install ffmpeg`.
- **YouTube download fails or formats are missing:** update yt-dlp with
  `pip install -U "yt-dlp[default]"` and make sure Node or Deno is installed.
- **"Google Translate still rate-limited":** add a `GROQ_API_KEY`, or wait and rerun;
  finished steps are cached.
- **Transcription too slow:** use a smaller `WHISPER_MODEL` such as `base`.
- **Dubbed speech sounds rushed or overlaps:** lower `WORDS_PER_SECOND` so translations
  are shorter, or raise `MAX_SPEEDUP` slightly.
