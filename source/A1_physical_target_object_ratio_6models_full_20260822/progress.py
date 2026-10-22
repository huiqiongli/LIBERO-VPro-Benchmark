#!/usr/bin/env python3
"""Strict progress accounting for the six-model A1 ratio experiment."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


HERE = Path(__file__).resolve().parent
MODELS = ("fastwam", "lingbot_va", "openvla_oft", "pi0", "pi05", "lawam")
MODEL_NAMES = {
    "fastwam": "FastWAM", "lingbot_va": "LingBot-VA", "openvla_oft": "OpenVLA-OFT",
    "pi0": "Pi0-LIBERO", "pi05": "Pi0.5-LIBERO", "lawam": "LaWAM",
}
RATIOS = (25, 50, 75, 100)
SUITES = ("libero_object", "libero_spatial", "libero_goal", "libero_10")
EXPECTED_PER_MODEL = 1600


def manifest() -> dict:
    return json.loads((HERE / "manifest.json").read_text(encoding="utf-8"))


def video_ok(path: str | None) -> bool:
    if not path:
        return False
    candidate = Path(path)
    try:
        return candidate.is_file() and candidate.stat().st_size > 1024
    except OSError:
        return False


def classify_record(record: dict, model: str) -> tuple[int, str, int, int] | None:
    metadata = record.get("metadata") or {}
    summary = record.get("summary") or {}
    if record.get("status") != "complete" or metadata.get("model_name") != MODEL_NAMES[model]:
        return None
    if metadata.get("track") != "A1" or metadata.get("subtask") != "physical_target_object_ratio":
        return None
    if metadata.get("condition") != "perturbed":
        return None
    suite = str(metadata.get("suite", ""))
    try:
        task = int(metadata.get("task_id", -1))
        trial = int(metadata.get("repeat_index", -1))
        init_state = int(metadata.get("init_state_index", -2))
    except (TypeError, ValueError):
        return None
    if suite not in SUITES or not (0 <= task < 10 and 0 <= trial < 10) or init_state != trial:
        return None
    spec = metadata.get("env_edit_spec") or {}
    try:
        ratio = round(float(spec.get("coverage", -1)) * 100)
    except (TypeError, ValueError):
        return None
    if ratio not in RATIOS or spec.get("coverage_definition") != "complete_3d_render_surface_area":
        return None
    if not spec.get("object_surface_black_fraction_overrides"):
        return None
    if metadata.get("visual_input_contract") not in ("official_dual_camera", "dual_agentview_plus_wrist"):
        return None
    if metadata.get("single_view") not in (None, "", False):
        return None
    rollout = summary.get("rollout_video_path") or summary.get("video_path")
    if not video_ok(rollout):
        return None
    if model in ("pi0", "pi05", "openvla_oft"):
        second = summary.get("model_input_video_path") or summary.get("masked_video_path")
        if not video_ok(second):
            return None
    return ratio, suite, task, trial


def fastwam_keys(root: Path) -> set[tuple[int, str, int, int]]:
    expected_spec = {
        (int(case["ratio"]), str(case["suite"]), int(case["task_id"])): str(case["spec_path"])
        for case in manifest()["cases"]
    }
    keys = set()
    for path in (root / "fastwam" / "batch" / "cases").glob("*/result.json"):
        try:
            result = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if result.get("status") != "complete" or int(result.get("total_episodes", 0)) != 1:
            continue
        suite = str(result.get("suite", "")); task = int(result.get("task_id", -1))
        spec_path = str(result.get("env_edit_source") or result.get("spec") or "")
        try:
            spec = json.loads(Path(spec_path).read_text(encoding="utf-8"))
            ratio = round(float(spec.get("coverage", -1)) * 100)
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            continue
        if expected_spec.get((ratio, suite, task)) != spec_path:
            continue
        trials = list(result.get("success_episodes") or []) + list(result.get("failure_episodes") or [])
        if len(trials) != 1:
            continue
        trial = int(trials[0])
        diagnostics = result.get("episode_decision_diagnostics") or []
        if not diagnostics or not all(bool((entry.get("environment_edit") or {}).get("applied")) for entry in diagnostics):
            continue
        videos = result.get("episode_video_paths") or []
        if len(videos) != 1 or not all(video_ok(videos[0].get(key)) for key in ("rollout", "model_input")):
            continue
        if result.get("visual_input_contract") != "official_dual_camera":
            continue
        if 0 <= trial < 10:
            keys.add((ratio, suite, task, trial))
    return keys


def record_keys(root: Path, model: str) -> tuple[set[tuple[int, str, int, int]], int, int]:
    seen: dict[tuple[int, str, int, int], list[Path]] = defaultdict(list)
    invalid = 0
    for path in (root / model).rglob("episode.json") if (root / model).exists() else ():
        relative = path.relative_to(root / model)
        if any(part.startswith(".") or part in ("superseded", "source_snapshot") for part in relative.parts):
            continue
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            invalid += 1
            continue
        key = classify_record(record, model)
        if key is None:
            if record.get("status") not in ("recording", "interrupted"):
                invalid += 1
            continue
        seen[key].append(path)
    duplicates = sum(max(0, len(paths) - 1) for paths in seen.values())
    return set(seen), duplicates, invalid


def summarize(output_root: str | Path) -> dict:
    root = Path(output_root)
    result = {"expected_total": EXPECTED_PER_MODEL * len(MODELS), "completed_total": 0, "models": {}}
    for model in MODELS:
        if model == "fastwam":
            keys = fastwam_keys(root); duplicates = invalid = 0
        else:
            keys, duplicates, invalid = record_keys(root, model)
        by_ratio = Counter(key[0] for key in keys)
        by_suite = Counter(key[1] for key in keys)
        result["models"][model] = {
            "expected": EXPECTED_PER_MODEL, "completed": len(keys),
            "remaining": EXPECTED_PER_MODEL - len(keys),
            "duplicates": duplicates, "invalid_records": invalid,
            "by_ratio": {f"{ratio}pct": by_ratio[ratio] for ratio in RATIOS},
            "by_suite": {suite: by_suite[suite] for suite in SUITES},
        }
        result["completed_total"] += len(keys)
    result["remaining_total"] = result["expected_total"] - result["completed_total"]
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--require-count", type=int)
    args = parser.parse_args()
    result = summarize(args.output_root)
    if args.model:
        count = result["models"][args.model]["completed"]
        print(count)
        if args.require_count is not None and count != args.require_count:
            raise SystemExit(1)
    else:
        print(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
