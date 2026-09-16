import argparse

from app.pipeline import run

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dub a YouTube video into English.")
    parser.add_argument("url", nargs="?", help="YouTube URL (prompted if omitted)")
    args = parser.parse_args()
    run(args.url or input("YouTube URL: ").strip())
