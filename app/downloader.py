from pathlib import Path

import yt_dlp


# YouTube needs a JS runtime to unlock formats; Node is used if installed (deno is yt-dlp's default).
BASE_OPTS = {"quiet": True, "noprogress": True, "js_runtimes": {"deno": {}, "node": {}}}


def get_video_id(url: str) -> str:
    with yt_dlp.YoutubeDL(BASE_OPTS) as ydl:
        return ydl.extract_info(url, download=False)["id"]


def download(url: str, work_dir: Path) -> None:
    opts = {
        **BASE_OPTS,
        "format": "bv*[vcodec^=avc1]+ba[ext=m4a]/bv*[ext=mp4]+ba[ext=m4a]/b[ext=mp4]/b",  # prefer H.264: plays everywhere
        "outtmpl": str(work_dir / "video.%(ext)s"),
        "merge_output_format": "mp4",
        "socket_timeout": 60,    # slow connections stall for a while; don't give up at 20s
        "retries": 10,
        "fragment_retries": 10,
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        ydl.download([url])
