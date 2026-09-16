from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TEMP_DIR = ROOT / "temp"
OUTPUT_DIR = ROOT / "output"

WHISPER_MODEL = "small"          # tiny | base | small | medium | large-v3
TTS_VOICE = "en-US-GuyNeural"    # list voices: edge-tts --list-voices
TTS_CONCURRENCY = 8
MAX_SPEEDUP = 1.25               # never speed a clip up more than this
SAMPLE_RATE = 24000

GROQ_MODEL = "openai/gpt-oss-120b"  # used when GROQ_API_KEY is set, else Google Translate
TRANSLATE_BATCH = 30                # segments per LLM request
WORDS_PER_SECOND = 2.6              # typical English TTS pace, used to cap line length
