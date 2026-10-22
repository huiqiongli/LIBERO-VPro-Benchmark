#!/usr/bin/env python3
"""Search one fixed, visible, collision-free C1 position edit per failed case.

The same world-frame offset must pass all ten official init states.  Semantic
visibility and displacement are judged only in the third-person agentview.
This script is diagnostic: it writes candidates but does not modify cases.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

import strict_revalidate_all_cases as validation


HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "C1_case_revalidation_20260825/validated_cases.json"
REPORT = HERE / "agentview_reference_revalidation/strict_revalidation_report.json"
OUTPUT = HERE / "agentview_reference_revalidation/position_reconfiguration_candidates.json"


def candidate_offsets() -> list[list[float]]:
    directions = (
        (1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0),
        (1.0, 1.0), (1.0, -1.0), (-1.0, 1.0), (-1.0, -1.0),
    )
    result: list[list[float]] = []
    for magnitude in (0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20, 0.22, 0.24):
        for dx, dy in directions:
            scale = magnitude / math.hypot(dx, dy)
            result.append([round(dx * scale, 6), round(dy * scale, 6), 0.0])
    return result


def semantic_rows() -> dict[str, dict]:
    source = json.loads(SOURCE.read_text())
    return {
        row["case_id"]: row
        for row in source["cases"]["position"]
    }


def task_relevant_candidates(
    env,
    current: str,
    objects: dict[str, str],
    interest: set[str],
    goal: set[str],
) -> list[str]:
    relevant = [name for name in objects if name in interest or name in goal]
    ordered = [current, *sorted(name for name in relevant if name != current)]
    return [
        name for name in ordered
        if validation.c1_runtime._free_joint_name(getattr(env, "env", env), name)
        is not None
    ]


def main() -> int:
    report = json.loads(REPORT.read_text())
    failed = {
        case_id for case_id in report["rejected_case_ids"]
        if case_id.startswith("C1_position_")
    }
    rows = semantic_rows()
    suites = validation.benchmark.get_benchmark_dict()
    results: dict[str, dict] = {}
    by_task: dict[tuple[str, int], list[str]] = defaultdict(list)
    for case_id in sorted(failed):
        row = rows[case_id]
        by_task[(row["suite"], int(row["task_id"]))].append(case_id)

    offsets = candidate_offsets()
    for task_number, ((suite_name, task_id), case_ids) in enumerate(
        sorted(by_task.items()), start=1
    ):
        suite = suites[suite_name]()
        states = suite.get_task_init_states(task_id)
        row0 = rows[case_ids[0]]
        bddl_text = Path(row0["bddl_file"]).read_text()
        objects = validation.parse_objects(bddl_text)
        interest = validation.named_tokens(
            validation.bddl_section(bddl_text, "obj_of_interest", ("init", "goal"))
        )
        goal = validation.named_tokens(
            validation.bddl_section(bddl_text, "goal", ("metric",))
        )
        env = validation.libero_client.construct_single_env(
            {
                "bddl_file_name": suite.get_task_bddl_file_path(task_id),
                "camera_heights": 256,
                "camera_widths": 256,
            },
            use_segmentation=True,
        )
        if env is None:
            for case_id in case_ids:
                results[case_id] = {"passed": False, "reason": "environment_construction_failed"}
            continue
        try:
            env.reset()
            for case_id in case_ids:
                row = rows[case_id]
                targets = task_relevant_candidates(
                    env, row["target"], objects, interest, goal
                )
                survivors = {
                    (target, tuple(offset)): {
                        "target": target,
                        "target_category": objects[target],
                        "offset_m": offset,
                        "minimum_clean_pixels": 10**9,
                        "minimum_edited_pixels": 10**9,
                        "minimum_centroid_shift_pixels": float("inf"),
                        "maximum_offset_norm_m": float(np.linalg.norm(offset[:2])),
                    }
                    for target in targets for offset in offsets
                }
                rejection_counts: defaultdict[str, int] = defaultdict(int)
                for init_index in range(10):
                    obs = env.set_init_state(states[init_index])
                    done = False
                    for _ in range(5):
                        obs, _, done, _ = env.step(validation.DUMMY)
                    for key in list(survivors):
                        target, offset = key
                        target_instance = validation.parent_instance(env, target)
                        clean_mask = validation.instance_mask(
                            obs, env, "agentview", target_instance
                        )
                        if done or int(clean_mask.sum()) < validation.MIN_VISIBLE_PIXELS:
                            rejection_counts["clean_target_not_visible_agentview"] += 1
                            del survivors[key]
                            continue
                        spec = {
                            "object_edits": [{
                                "op": "relocate", "object": target,
                                "offset": list(offset),
                            }]
                        }
                        edited_seg, metrics = validation.apply_edit_for_segmentation(
                            env, spec, "agentview", 256, 256, set(objects)
                        )
                        edited_mask = (
                            edited_seg == int(env.instance_to_id[target_instance])
                        )
                        old_center = validation.centroid(clean_mask)
                        new_center = validation.centroid(edited_mask)
                        shift = (
                            float(np.hypot(
                                old_center[0] - new_center[0],
                                old_center[1] - new_center[1],
                            ))
                            if old_center is not None and new_center is not None
                            else 0.0
                        )
                        reason = None
                        if int(edited_mask.sum()) < validation.MIN_VISIBLE_PIXELS:
                            reason = "alternate_not_visible_agentview"
                        elif shift < validation.MIN_POSITION_SHIFT_PIXELS:
                            reason = "insufficient_agentview_shift"
                        elif metrics.get("new_penetrating_object_contacts"):
                            reason = "new_object_penetration"
                        elif not metrics.get("finite_pose") or not metrics.get("z_preserved"):
                            reason = "invalid_pose"
                        if reason:
                            rejection_counts[reason] += 1
                            del survivors[key]
                            continue
                        item = survivors[key]
                        item["minimum_clean_pixels"] = min(
                            item["minimum_clean_pixels"], int(clean_mask.sum())
                        )
                        item["minimum_edited_pixels"] = min(
                            item["minimum_edited_pixels"], int(edited_mask.sum())
                        )
                        item["minimum_centroid_shift_pixels"] = min(
                            item["minimum_centroid_shift_pixels"], shift
                        )
                    if not survivors:
                        break
                ranked = sorted(
                    survivors.values(),
                    key=lambda item: (
                        item["target"] != row["target"],
                        item["maximum_offset_norm_m"],
                        -item["minimum_centroid_shift_pixels"],
                        -item["minimum_edited_pixels"],
                    ),
                )
                results[case_id] = {
                    "passed": bool(ranked),
                    "original_target": row["target"],
                    "selected": ranked[0] if ranked else None,
                    "valid_candidate_count": len(ranked),
                    "top_candidates": ranked[:10],
                    "rejection_counts": dict(rejection_counts),
                }
                print(
                    f"[{task_number}/{len(by_task)}] {case_id}: "
                    f"{results[case_id]['selected']}",
                    flush=True,
                )
        finally:
            env.close()

    payload = {
        "schema_version": "wam_syncbench.c1_position_reconfiguration_search.v1",
        "semantic_reference_view": "agentview",
        "all_ten_init_states_required": True,
        "failed_position_case_count": len(failed),
        "resolved_count": sum(bool(item["passed"]) for item in results.values()),
        "unresolved": [case_id for case_id, item in results.items() if not item["passed"]],
        "cases": results,
    }
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({
        "resolved": payload["resolved_count"],
        "total": len(failed),
        "unresolved": payload["unresolved"],
        "output": str(OUTPUT),
    }, indent=2))
    return 0 if not payload["unresolved"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
