#!/usr/bin/env python3
"""Build the auditable 40-task phase source / pre-place destination map."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from libero.libero import benchmark, get_libero_path
from libero.libero.envs.bddl_utils import robosuite_parse_problem


SUITES = ("libero_object", "libero_spatial", "libero_goal", "libero_10")
BINARY = {"in", "on", "stack"}
ARTICULATED = {"open", "close", "turnon", "turnoff"}


def task_row(suite_name: str, task_id: int) -> dict:
    suite = benchmark.get_benchmark_dict()[suite_name]()
    task = suite.get_task(task_id)
    bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    parsed = robosuite_parse_problem(str(bddl))
    goals = []
    for raw in parsed.get("goal_state", []):
        predicate = str(raw[0]).lower()
        goals.append(
            {
                "predicate": predicate,
                "source": str(raw[1]),
                "destination": str(raw[2]) if len(raw) > 2 else None,
            }
        )
    binary = [goal for goal in goals if goal["predicate"] in BINARY]
    selected = binary or [goal for goal in goals if goal["predicate"] in ARTICULATED]
    pre_place = []
    for goal in selected:
        if goal["destination"] is not None:
            destination = goal["destination"]
            mode = "exact_bddl_region" if "_region" in destination or "_side" in destination else "destination_object"
            standard = (
                "source enters the destination-aware lateral/vertical approach envelope "
                "after transport and measurable goal approach"
            )
        else:
            mode = "articulation_progress"
            standard = "articulated goal reaches at least 70% normalized joint progress after interaction transport"
        pre_place.append(
            {
                "source": goal["source"],
                "destination": goal["destination"],
                "predicate": goal["predicate"],
                "mapping_mode": mode,
                "standard": standard,
            }
        )
    return {
        "suite": suite_name,
        "task_id": task_id,
        "task_name": task.name,
        "instruction": task.language,
        "bddl_file": str(bddl),
        "all_goals": goals,
        "phase_sources": [goal["source"] for goal in selected],
        "pre_place": pre_place,
        "phase_semantics": {
            "reach": "end effector approaches the active BDDL source",
            "pre_grasp": "end effector is within the calibrated final-approach band without acquisition",
            "grasp": "physical finger contact or near-source gripper closure; held for a samplable edge window",
            "transport": "post-acquisition/contact manipulation before the exact goal approach envelope",
            "pre_place": "task-aware destination approach or near-complete articulated interaction",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default=str(Path(__file__).parent / "configs" / "phase_task_map_40.json"),
    )
    args = parser.parse_args()
    rows = [task_row(suite, task_id) for suite in SUITES for task_id in range(10)]
    payload = {
        "schema": "wam_syncbench.a4_phase_task_map.v2",
        "detector": "task_aware_v2",
        "suite_count": 4,
        "task_count": len(rows),
        "pre_place_mapping_standard": {
            "relational": "exact BDDL source-to-destination region/object proximity after transport",
            "articulated": "normalized joint progress >= 0.70 after interaction transport",
            "anti_false_positive": "low end-effector height alone can never trigger pre_place",
        },
        "tasks": rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(output)


if __name__ == "__main__":
    main()
