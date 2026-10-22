"""H.264 recorder for A5 dual-slot fill controls."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping
import shutil
import subprocess

import cv2
import numpy as np


def _rgb(value: Any, size: int) -> np.ndarray:
    image = np.asarray(value, dtype=np.uint8)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"expected HWC RGB image, got {image.shape}")
    return cv2.resize(np.ascontiguousarray(image), (size, size), interpolation=cv2.INTER_AREA)


class H264VideoRecorder:
    def __init__(self, root: str | Path, *, fps: int = 15, size: int = 256):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.fps = fps
        self.size = size
        self.count = 0
        self.model_count = 0
        self._closed = False
        self.rollout_path = self.root / "rollout.mp4"
        self.model_input_path = self.root / "model_input.mp4"
        self._rollout_raw = self.root / ".rollout_mp4v.tmp.mp4"
        self._model_raw = self.root / ".model_input_mp4v.tmp.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        frame_size = (size * 2, size)
        self._rollout = cv2.VideoWriter(str(self._rollout_raw), fourcc, fps, frame_size)
        self._model = cv2.VideoWriter(str(self._model_raw), fourcc, fps, frame_size)
        if not self._rollout.isOpened() or not self._model.isOpened():
            raise RuntimeError("failed to open temporary MP4 writers")

    def add(self, raw: Mapping[str, Any], model_pair: tuple[Any, Any] | None = None) -> None:
        agent = _rgb(raw["agentview_image"], self.size)
        wrist = _rgb(raw["robot0_eye_in_hand_image"], self.size)
        self._rollout.write(cv2.cvtColor(np.hstack([agent, wrist]), cv2.COLOR_RGB2BGR))
        self.count += 1
        if model_pair is not None:
            model_agent = _rgb(model_pair[0], self.size)
            model_wrist = _rgb(model_pair[1], self.size)
            self._model.write(
                cv2.cvtColor(np.hstack([model_agent, model_wrist]), cv2.COLOR_RGB2BGR)
            )
            self.model_count += 1

    @staticmethod
    def _transcode(source: Path, destination: Path) -> None:
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            raise FileNotFoundError("ffmpeg is not available on PATH")
        subprocess.run(
            [ffmpeg, "-loglevel", "error", "-y", "-i", str(source), "-c:v", "libx264",
             "-preset", "veryfast", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
             str(destination)],
            check=True,
        )
        source.unlink()

    def close(self) -> dict[str, Any]:
        if self._closed:
            return self.summary()
        self._closed = True
        self._rollout.release()
        self._model.release()
        if self.count:
            self._transcode(self._rollout_raw, self.rollout_path)
        elif self._rollout_raw.exists():
            self._rollout_raw.unlink()
        if self.model_count:
            self._transcode(self._model_raw, self.model_input_path)
        elif self._model_raw.exists():
            self._model_raw.unlink()
        return self.summary()

    def summary(self) -> dict[str, Any]:
        return {
            "rollout_video_path": str(self.rollout_path) if self.rollout_path.is_file() else "",
            "model_input_video_path": (
                str(self.model_input_path) if self.model_input_path.is_file() else ""
            ),
            "video_frame_count": self.count,
            "model_input_frame_count": self.model_count,
            "video_fps": self.fps,
            "video_codec": "h264",
            "video_layout": "agentview|wrist",
        }

