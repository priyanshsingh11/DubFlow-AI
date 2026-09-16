import argparse

from dotenv import load_dotenv

from app.pipeline import run
from app.settings import ROOT

if __name__ == "__main__":
    load_dotenv(ROOT / ".env")  # reads GROQ_API_KEY
    parser = argparse.ArgumentParser(description="Dub a YouTube video into English.")
    parser.add_argument("url", nargs="?", help="YouTube URL (prompted if omitted)")
    args = parser.parse_args()
    run(args.url or input("YouTube URL: ").strip())
