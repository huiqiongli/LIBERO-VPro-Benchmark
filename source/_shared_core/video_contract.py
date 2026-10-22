"""Dependency-light H.264/yuv420p validation for benchmark videos."""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import shutil
import subprocess



def _ffprobe(path: Path, executable: str) -> dict:
    result = subprocess.run(
        [
            executable, "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=codec_name,pix_fmt,nb_frames,width,height",
            "-of", "json", str(path),
        ],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode != 0:
        return {"passed": False, "error": result.stderr.strip(), "path": str(path)}
    try:
        stream = json.loads(result.stdout)["streams"][0]
        frames = int(stream.get("nb_frames") or 0)
        passed = (
            stream.get("codec_name") == "h264"
            and stream.get("pix_fmt") == "yuv420p"
            and frames > 0
            and int(stream.get("width") or 0) > 0
            and int(stream.get("height") or 0) > 0
        )
        return {"passed": passed, **stream, "path": str(path), "probe": "ffprobe"}
    except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError) as exc:
        return {"passed": False, "error": f"{type(exc).__name__}: {exc}", "path": str(path)}


def _ffmpeg_decode(path: Path, executable: str) -> dict:
    """Decode one frame and parse the authoritative input-stream description."""
    result = subprocess.run(
        [
            executable, "-hide_banner", "-i", str(path), "-map", "0:v:0",
            "-frames:v", "1", "-f", "null", "-",
        ],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    stream_line = next((
        line.strip() for line in result.stderr.splitlines()
        if "Stream #" in line and "Video:" in line and "Video: h264" in line
    ), "")
    dimensions = re.search(r"(?<![0-9])(\d{2,5})x(\d{2,5})(?![0-9])", stream_line)
    width = int(dimensions.group(1)) if dimensions else 0
    height = int(dimensions.group(2)) if dimensions else 0
    passed = (
        result.returncode == 0
        and bool(stream_line)
        and "yuv420p" in stream_line
        and width > 0
        and height > 0
    )
    return {
        "passed": passed,
        "codec_name": "h264" if "Video: h264" in stream_line else "",
        "pix_fmt": "yuv420p" if "yuv420p" in stream_line else "",
        "nb_frames": 1 if result.returncode == 0 else 0,
        "width": width,
        "height": height,
        "path": str(path),
        "probe": "ffmpeg_first_frame_decode",
        "error": "" if passed else result.stderr[-2000:].strip(),
    }


def video_probe(path: Path) -> dict:
    """Require a non-empty, decodable H.264/yuv420p video with dimensions."""
    path = Path(path)
    if not path.is_file() or path.stat().st_size <= 1024:
        return {"passed": False, "error": "missing_or_too_small", "path": str(path)}
    ffprobe = shutil.which("ffprobe")
    if ffprobe:
        return _ffprobe(path, ffprobe)
    configured = os.environ.get("WAM_FFMPEG", "").strip()
    ffmpeg = configured or shutil.which("ffmpeg")
    if not ffmpeg:
        try:
            import imageio_ffmpeg
            ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        except (ImportError, RuntimeError):
            ffmpeg = ""
    if not ffmpeg:
        return {"passed": False, "error": "no_ffprobe_or_ffmpeg", "path": str(path)}
    return _ffmpeg_decode(path, ffmpeg)
