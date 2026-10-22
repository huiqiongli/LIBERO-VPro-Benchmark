#!/usr/bin/env python3
"""Validate one C1 episode and its two evidence videos."""

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


def declared_case(metadata: dict) -> dict:
    env = metadata.get("env_edit_spec") or {}
    if env.get("track") == "C1":
        return env
    perturb = metadata.get("perturbation") or {}
    observation = perturb.get("observation") or {}
    return observation if isinstance(observation, dict) else {}


def actual_audits(step: dict) -> list[dict]:
    perturb = step.get("perturbation") or {}
    # Pi0, Pi0.5, and OpenVLA-OFT record the realized observation audit
    # directly in the replan's perturbation field.  WAM runners wrap the same
    # audit under `actual` (LaWAM nests it one level further).
    if (
        isinstance(perturb, dict)
        and "enabled" in perturb
        and ("mask_source" in perturb or "view_stats" in perturb)
    ):
        return [perturb]
    actual = perturb.get("actual") or {}
    nested = actual.get("observation_perturbation") if isinstance(actual, dict) else None
    candidate = nested if isinstance(nested, dict) else actual
    # LingBot-VA applies and audits C1 once per control frame inside a replan.
    # Preserve every realized audit instead of treating the aggregate wrapper
    # as though it were a single observation edit.
    if isinstance(candidate, dict):
        per_control = candidate.get("per_control")
        if isinstance(per_control, list) and per_control:
            return [item for item in per_control if isinstance(item, dict)]
        return [candidate]
    return []


def changed_pixels(audit: dict, view: str) -> int:
    stats = (audit.get("view_stats") or {}).get(view) or {}
    return int(stats.get("changed_pixels", stats.get("masked_pixels", 0)) or 0)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=tuple(MODEL_NAMES), required=True)
    parser.add_argument("--case-json", required=True)
    parser.add_argument("--trial", type=int, required=True)
    parser.add_argument("--record-root", required=True)
    parser.add_argument("--video-root", required=True)
    parser.add_argument("--report", default="")
    args = parser.parse_args()

    case = json.loads(Path(args.case_json).read_text())
    expected_case_ids = {
        case["case_id"], f"{case['case_id']}__trial{args.trial:02d}"
    }
    matching = []
    for path in Path(args.record_root).rglob("episode.json"):
        try:
            record = json.loads(path.read_text())
            metadata = record.get("metadata") or {}
            spec = declared_case(metadata)
            if (
                metadata.get("model_name") == MODEL_NAMES[args.model]
                and metadata.get("suite") == case["suite"]
                and int(metadata.get("task_id", -1)) == int(case["task_id"])
                and int(metadata.get("repeat_index", -1)) == args.trial
                and (
                    metadata.get("case_id") in expected_case_ids
                    or spec.get("case_id") == case["case_id"]
                )
            ):
                matching.append((path, record))
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            pass

    checks = [{
        "name": "exactly_one_matching_record",
        "passed": len(matching) == 1,
        "details": [str(path) for path, _ in matching],
    }]
    record_path = None
    if len(matching) == 1:
        record_path, record = matching[0]
        metadata = record.get("metadata") or {}
        replans = record.get("replan_steps") or []
        audits_by_replan = [actual_audits(step) for step in replans]
        enabled_by_replan = [
            [audit for audit in audits if audit.get("enabled")]
            for audits in audits_by_replan
        ]
        enabled = [audit for audits in enabled_by_replan for audit in audits]
        selected_view = case["corrupted_view"]
        truth_view = case["truth_view"]
        views_exact = [audit.get("views") == [selected_view] for audit in enabled]
        restoration = [
            bool(audit.get("sim_state_restored"))
            and bool(audit.get("physical_rollout_unchanged"))
            for audit in enabled
        ]
        total_changed = sum(changed_pixels(audit, selected_view) for audit in enabled)
        truth_changed = sum(changed_pixels(audit, truth_view) for audit in enabled)
        init_audit = (
            metadata.get("initialization_audit")
            or metadata.get("environment_edit_audit")
            or {"native_model_environment": True}
        )
        checks.extend([
            {"name": "record_complete", "passed": record.get("status") == "complete"},
            {"name": "trial_and_init_aligned", "passed": (
                int(metadata.get("repeat_index", -1)) == args.trial
                and int(metadata.get("init_state_index", -1)) == args.trial % 10
            )},
            {"name": "track_subtask", "passed": (
                metadata.get("track") == "C1"
                and metadata.get("subtask") == case["subtask"]
            )},
            {"name": "official_dual_camera", "passed": (
                metadata.get("visual_input_contract") in DUAL_CONTRACTS
                and not metadata.get("view_copy_applied", False)
            ), "details": metadata.get("visual_input_contract")},
            {"name": "initialization_audit_present", "passed": bool(init_audit)},
            {"name": "replan_trace_present", "passed": bool(replans), "details": len(replans)},
            {"name": "c1_active_for_all_replans", "passed": bool(replans) and all(enabled_by_replan),
             "details": {
                 "active_replans": sum(bool(items) for items in enabled_by_replan),
                 "total_replans": len(replans),
                 "realized_audits": len(enabled),
             }},
            {"name": "only_selected_view_counterfactual", "passed": bool(enabled) and all(views_exact),
             "details": {"expected": selected_view, "actual": [a.get("views") for a in enabled[:5]]}},
            {"name": "temporary_sim_state_restored", "passed": bool(enabled) and all(restoration)},
            {"name": "counterfactual_visually_effective", "passed": total_changed > 0,
             "details": {"view": selected_view, "changed_pixels": total_changed}},
            {"name": "truth_view_not_edited", "passed": truth_changed == 0,
             "details": {"view": truth_view, "changed_pixels": truth_changed}},
            {"name": "success_recorded", "passed": isinstance((record.get("summary") or {}).get("success"), bool)},
        ])

    videos = sorted(Path(args.video_root).rglob("*.mp4"))
    probes = [video_probe(path) for path in videos]
    model_input = [
        any(token in path.name.lower() for token in (
            "model_input", "model-input", "masked_input", "masked-input"
        ))
        for path in videos
    ]
    checks.extend([
        {"name": "exactly_two_evidence_videos", "passed": len(videos) == 2,
         "details": [str(path) for path in videos]},
        {"name": "rollout_and_model_input_present", "passed": (
            len(videos) == 2 and any(model_input) and any(not value for value in model_input)
        )},
        {"name": "videos_h264_yuv420p_decodable", "passed": bool(probes) and all(
            item.get("passed") for item in probes
        ), "details": probes},
    ])
    redundant = [
        str(path) for path in Path(args.record_root).rglob("*")
        if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".npy"}
    ]
    checks.append({
        "name": "no_redundant_observation_frames",
        "passed": not redundant,
        "details": redundant[:20],
    })
    failed = [check for check in checks if not check["passed"]]
    payload = {
        "schema_version": "wam_syncbench.c1_episode_validation.v2",
        "passed": not failed,
        "model": args.model,
        "case_id": case["case_id"],
        "trial": args.trial,
        "record": str(record_path) if record_path else "",
        "failed_count": len(failed),
        "failed_checks": failed,
        "checks": checks,
    }
    report = Path(args.report) if args.report else Path(args.video_root).parent / "output_validation.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({key: payload[key] for key in ("passed", "model", "case_id", "trial", "failed_count")}, indent=2))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
