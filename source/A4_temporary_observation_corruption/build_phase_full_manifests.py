#!/usr/bin/env python3
"""Build the immutable 4-suite x 10-trial manifests for A4 phase pulses."""
from __future__ import annotations

import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
CONFIGS = HERE / "configs"
PHASES = ("reach", "pre_grasp", "grasp", "transport", "pre_place")


def main() -> None:
    task_map = json.loads((CONFIGS / "phase_task_map_40.json").read_text())
    empty_spec = str((CONFIGS / "empty_env_edit.json").resolve())
    output_dir = CONFIGS / "phase_full"
    output_dir.mkdir(parents=True, exist_ok=True)
    for phase in PHASES:
        cases = []
        for task in task_map["tasks"]:
            for trial in range(10):
                suite = task["suite"]
                task_id = int(task["task_id"])
                cases.append(
                    {
                        "case_id": (
                            f"A4_phase_{phase}_3replan_once__{suite}__"
                            f"task{task_id:02d}__trial{trial:02d}"
                        ),
                        "suite": suite,
                        "task_id": task_id,
                        "trial_index": trial,
                        "phase": phase,
                        "condition": "fullscreen_noise_phase_3replan_once",
                        "instruction": task["instruction"],
                        "spec": empty_spec,
                    }
                )
        payload = {
            "schema_version": "wam_syncbench.a4_phase_full_manifest.v2",
            "detector": "task_aware_v2",
            "phase": phase,
            "suite_count": 4,
            "task_count": 40,
            "trials_per_task": 10,
            "case_count": len(cases),
            "cases": cases,
        }
        path = output_dir / f"{phase}.json"
        path.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"{path}: {len(cases)} cases")


if __name__ == "__main__":
    main()
