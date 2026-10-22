#!/usr/bin/env python3
"""Diagnose residual instance pixels for the three spatial count cases."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import strict_revalidate_all_cases as validation


HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "C1_case_revalidation_20260825/validated_cases.json"
OUTPUT = HERE / "agentview_reference_revalidation/count_disappearance_diagnostic.json"
CASE_IDS = {
    "C1_count_libero_spatial_t3",
    "C1_count_libero_spatial_t4",
    "C1_count_libero_spatial_t9",
}


def main() -> int:
    source = json.loads(SOURCE.read_text())
    rows = {
        row["case_id"]: row for row in source["cases"]["count"]
        if row["case_id"] in CASE_IDS
    }
    suites = validation.benchmark.get_benchmark_dict()
    results = {}
    for case_id, row in sorted(rows.items()):
        suite = suites[row["suite"]]()
        states = suite.get_task_init_states(int(row["task_id"]))
        bddl_text = Path(row["bddl_file"]).read_text()
        objects = validation.parse_objects(bddl_text)
        targets = [
            name for name, category in objects.items()
            if category == "akita_black_bowl"
        ]
        env = validation.libero_client.construct_single_env(
            {
                "bddl_file_name": suite.get_task_bddl_file_path(int(row["task_id"])),
                "camera_heights": 256,
                "camera_widths": 256,
            },
            use_segmentation=True,
        )
        env.reset()
        case_result = {}
        try:
            for target in targets:
                target_result = {}
                for hidden_z in (-10.0, -100.0, -1000.0):
                    per_init = []
                    for init_index in range(10):
                        obs = env.set_init_state(states[init_index])
                        for _ in range(5):
                            obs, _, _, _ = env.step(validation.DUMMY)
                        target_instance = validation.parent_instance(env, target)
                        clean = validation.instance_mask(
                            obs, env, "agentview", target_instance
                        )
                        edited, _ = validation.apply_edit_for_segmentation(
                            env,
                            {"object_edits": [{
                                "op": "disappear", "object": target,
                                "hidden_z": hidden_z,
                            }]},
                            "agentview", 256, 256, set(objects),
                        )
                        edited_pixels = int(np.count_nonzero(
                            edited == int(env.instance_to_id[target_instance])
                        ))
                        per_init.append({
                            "init_state_index": init_index,
                            "clean_pixels": int(clean.sum()),
                            "edited_pixels": edited_pixels,
                        })
                    target_result[str(hidden_z)] = per_init
                case_result[target] = target_result
        finally:
            env.close()
        results[case_id] = case_result
        print(case_id, flush=True)
    payload = {"cases": results}
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(OUTPUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
