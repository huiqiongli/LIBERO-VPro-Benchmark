#!/usr/bin/env python3
"""Validate A5 result artifacts and the recorded dual-camera model inputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


MODELS = ("lingbot_va", "fastwam", "lawam", "openvla_oft", "pi0", "pi05")
CASES = {
    "A5_agent_plus_blank_wrist": ("blank", "agentview"),
    "A5_wrist_plus_white_agent": ("white", "wrist"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--models", default=",".join(MODELS))
    parser.add_argument("--frames", type=int, default=12)
    return parser.parse_args()


def find_case_files(root: Path, model: str, case: str) -> tuple[list[Path], list[Path]]:
    case_dirs = [path for path in (root / model).rglob(case) if path.is_dir()]
    videos: list[Path] = []
    results: list[Path] = []
    for case_dir in case_dirs:
        videos.extend(case_dir.rglob("*model_input*.mp4"))
        results.extend(case_dir.rglob("episode.json"))
        results.extend(case_dir.rglob("summary.json"))
        results.extend(case_dir.rglob("*_results.json"))
        if model == "lingbot_va":
            results.extend(case_dir.rglob("libero_*.json"))
    return sorted(set(videos)), sorted(set(results))


def validate_video(path: Path, fill: str, kept_view: str, max_frames: int) -> dict:
    capture = cv2.VideoCapture(str(path))
    frame_stats = []
    try:
        for _ in range(max_frames):
            ok, frame = capture.read()
            if not ok:
                break
            midpoint = frame.shape[1] // 2
            agent, wrist = frame[:, :midpoint], frame[:, midpoint:]
            filled = wrist if kept_view == "agentview" else agent
            kept = agent if kept_view == "agentview" else wrist
            frame_stats.append((int(filled.min()), int(filled.max()), float(kept.std())))
    finally:
        capture.release()
    if not frame_stats:
        raise RuntimeError(f"no decodable frames: {path}")
    fill_min = min(item[0] for item in frame_stats)
    fill_max = max(item[1] for item in frame_stats)
    kept_std = max(item[2] for item in frame_stats)
    # H.264 YUV conversion shifts constant white by a few integer levels.
    valid_fill = fill_max <= 16 if fill == "blank" else fill_min >= 240
    if not valid_fill or kept_std <= 5.0:
        raise RuntimeError(
            f"bad model input {path}: fill_range={fill_min}..{fill_max}, "
            f"kept_std={kept_std:.2f}"
        )
    return {
        "path": str(path), "frames": len(frame_stats),
        "fill_range": [fill_min, fill_max], "kept_std_max": round(kept_std, 3),
    }


def main() -> int:
    args = parse_args()
    report = {"root": str(args.root.resolve()), "models": {}}
    failed = False
    for model in (item.strip() for item in args.models.split(",") if item.strip()):
        model_report = {}
        for case, (fill, kept_view) in CASES.items():
            videos, results = find_case_files(args.root, model, case)
            entry = {
                "result_files": [str(path) for path in results],
                "model_input_videos": [],
            }
            if not videos or not results:
                entry["error"] = f"videos={len(videos)}, results={len(results)}"
                failed = True
            for video in videos:
                try:
                    entry["model_input_videos"].append(
                        validate_video(video, fill, kept_view, args.frames)
                    )
                except Exception as exc:
                    entry.setdefault("validation_errors", []).append(str(exc))
                    failed = True
            model_report[case] = entry
        report["models"][model] = model_report
    report["passed"] = not failed
    report_path = args.root / "validation_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
