#!/usr/bin/env python3
"""Validate one six-model smoke/full state-benchmark episode output."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from video_contract import video_probe


MODEL_NAMES = {
    "fastwam": "FastWAM",
    "lingbot_va": "LingBot-VA",
    "lawam": "LaWAM",
    "pi0": "Pi0-LIBERO",
    "pi05": "Pi0.5-LIBERO",
    "openvla_oft": "OpenVLA-OFT",
}
DUAL_CONTRACTS = {"official_dual_camera", "dual_agentview_plus_wrist"}


def perturbation_enabled(step: dict) -> bool:
    value = step.get("perturbation") or {}
    return bool(value.get("enabled", value.get("active", False)))


def visual_perturbation_enabled(step: dict) -> bool:
    """Read the realized observation edit, not the environment declaration."""
    value = step.get("perturbation") or {}
    actual = value.get("actual") or {}
    observation = actual.get("observation_perturbation")
    if isinstance(observation, dict):
        return bool(observation.get("enabled", observation.get("active", False)))
    if isinstance(actual, dict) and ("enabled" in actual or "active" in actual):
        return bool(actual.get("enabled", actual.get("active", False)))
    # Some model adapters record the observation audit directly.
    return bool(value.get("enabled", value.get("active", False)))


def is_model_input_video(name: str) -> bool:
    normalized = str(name).lower().replace("-", "_")
    return "model_input" in normalized or "masked_input" in normalized


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=tuple(MODEL_NAMES), required=True)
    parser.add_argument("--case-json", required=True)
    parser.add_argument("--trial", type=int, required=True)
    parser.add_argument("--record-root", required=True)
    parser.add_argument("--video-root", required=True)
    parser.add_argument("--report", default="")
    args = parser.parse_args()

    case = json.loads(Path(args.case_json).read_text(encoding="utf-8"))
    records = list(Path(args.record_root).rglob("episode.json"))
    matching = []
    for path in records:
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
            metadata = record.get("metadata") or {}
            env_spec = metadata.get("env_edit_spec") or {}
            if (
                metadata.get("model_name") == MODEL_NAMES[args.model]
                and metadata.get("suite") == case["suite"]
                and int(metadata.get("task_id", -1)) == int(case["task_id"])
                and int(metadata.get("repeat_index", -1)) == int(args.trial)
                and env_spec.get("case_id") == case["case_id"]
            ):
                matching.append((path, record))
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            continue

    checks = []
    checks.append({"name": "exactly_one_matching_record", "passed": len(matching) == 1,
                   "details": [str(path) for path, _ in matching]})
    if len(matching) == 1:
        record_path, record = matching[0]
        metadata = record.get("metadata") or {}
        init_audit = metadata.get("initialization_audit") or metadata.get("environment_edit_audit")
        expected_case_ids = {
            case["case_id"], f"{case['case_id']}__trial{int(args.trial):02d}"
        }
        checks.extend([
            {"name": "record_complete", "passed": record.get("status") == "complete"},
            {"name": "trial_and_init_aligned", "passed": (
                int(metadata.get("repeat_index", -1)) == int(args.trial)
                and int(metadata.get("init_state_index", -1)) == int(args.trial) % 10
            )},
            {"name": "case_identity", "passed": metadata.get("case_id") in expected_case_ids,
             "details": metadata.get("case_id")},
            {"name": "track_subtask", "passed": (
                metadata.get("track") == case["track"]
                and metadata.get("subtask") == case.get("subtask", case.get("ambiguity_kind"))
            )},
            {"name": "official_dual_camera", "passed": (
                metadata.get("visual_input_contract") in DUAL_CONTRACTS
                and not metadata.get("view_copy_applied", False)
            ), "details": metadata.get("visual_input_contract")},
            {"name": "initialization_audit_present", "passed": isinstance(init_audit, dict) and bool(init_audit)},
            {"name": "replan_trace_present", "passed": len(record.get("replan_steps") or []) > 0,
             "details": len(record.get("replan_steps") or [])},
            {"name": "success_recorded", "passed": isinstance((record.get("summary") or {}).get("success"), bool)},
        ])
        replans = record.get("replan_steps") or []
        if case["track"] == "D2":
            checks.append({
                "name": "d2_physical_state_active",
                "passed": bool(replans) and perturbation_enabled(replans[0]),
            })
        elif case.get("observation_mode") == "obfuscated":
            checks.append({
                "name": "d3_obfuscation_active_for_full_episode",
                "passed": bool(replans) and all(
                    visual_perturbation_enabled(step) for step in replans
                ),
            })
        else:
            checks.append({
                "name": "d3_clean_has_no_visual_obfuscation",
                "passed": not any(visual_perturbation_enabled(step) for step in replans),
            })
    else:
        record_path = None

    videos = sorted(Path(args.video_root).rglob("*.mp4"))
    probes = [video_probe(path) for path in videos]
    names = [path.name.lower() for path in videos]
    model_input_flags = [is_model_input_video(name) for name in names]
    checks.extend([
        {"name": "exactly_two_evidence_videos", "passed": len(videos) == 2,
         "details": [str(path) for path in videos]},
        {"name": "rollout_and_model_input_present", "passed": (
            len(videos) == 2
            and any(model_input_flags)
            and any(not flag for flag in model_input_flags)
        ), "details": names},
        {"name": "videos_h264_yuv420p_decodable", "passed": bool(probes) and all(
            item.get("passed") for item in probes
        ), "details": probes},
    ])

    record_root = Path(args.record_root)
    observation_files = [
        str(path) for path in record_root.rglob("*")
        if path.is_file()
        and (
            path.suffix.lower() in {".png", ".jpg", ".jpeg", ".npy"}
            or "observation_frame" in path.name.lower()
        )
    ]
    checks.append({
        "name": "no_redundant_observation_frame_files",
        "passed": not observation_files,
        "details": observation_files[:20],
    })
    failed = [item for item in checks if not item["passed"]]
    payload = {
        "schema_version": "wam_syncbench.state_episode_output_validation.v1",
        "passed": not failed,
        "model": args.model,
        "case_id": case["case_id"],
        "trial": int(args.trial),
        "record": str(record_path) if record_path else "",
        "failed_count": len(failed),
        "failed_checks": failed,
        "checks": checks,
    }
    report = (
        Path(args.report)
        if args.report else Path(args.video_root).parent / "output_validation.json"
    )
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: payload[key] for key in (
        "passed", "model", "case_id", "trial", "failed_count"
    )}, indent=2))
    for failure in failed:
        print(json.dumps(failure, ensure_ascii=False))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
