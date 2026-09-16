from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TEMP_DIR = ROOT / "temp"
OUTPUT_DIR = ROOT / "output"

GROQ_WHISPER_MODEL = "whisper-large-v3"  # used when GROQ_API_KEY is set
WHISPER_MODEL = "large-v3-turbo"        # local fallback: tiny | base | small | medium | large-v3-turbo | large-v3
CHUNK_SECONDS = 600                     # audio per transcription request (Groq caps uploads at 25 MB)
MIN_LINE_SECONDS = 3                    # dub lines are split at pauses once they reach this length
MAX_LINE_SECONDS = 10                   # ...and always by this length
MAX_WORD_SECONDS = 1.5                  # caps Whisper's over-stretched word timestamps
MIN_GAP_SECONDS = 4                     # a gap this long with speech in it is transcribed again
TTS_VOICE = "en-US-GuyNeural"    # list voices: edge-tts --list-voices
TTS_CONCURRENCY = 8
MAX_SPEEDUP = 1.3                # never speed a clip up more than this
SAMPLE_RATE = 24000

# Translation LLMs when GROQ_API_KEY is set (else Google Translate); the next one is used once a daily token limit is hit.
GROQ_MODELS = ["openai/gpt-oss-120b", "openai/gpt-oss-20b"]
TRANSLATE_BATCH = 12                # segments per LLM request (small batches keep the model from merging lines)
WORDS_PER_SECOND = 2.6              # typical English TTS pace, used to cap line length
