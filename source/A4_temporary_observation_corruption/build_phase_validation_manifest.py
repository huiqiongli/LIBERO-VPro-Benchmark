#!/usr/bin/env python3
"""Build a sharded FastWAM clean-rollout manifest for 40-task phase checks."""

from __future__ import annotations

import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
CONFIGS = HERE / "configs"


def main() -> None:
    task_map = json.loads((CONFIGS / "phase_task_map_40.json").read_text())
    empty_spec = (CONFIGS / "empty_env_edit.json").resolve()
    cases = []
    for row in task_map["tasks"]:
        cases.append(
            {
                "case_id": f"phase_validation_v2__{row['suite']}__task{row['task_id']:02d}",
                "suite": row["suite"],
                "task_id": row["task_id"],
                "trial_index": 0,
                "condition": "clean_phase_validation",
                "instruction": row["instruction"],
                "spec": str(empty_spec),
            }
        )
    payload = {
        "schema_version": "wam_syncbench.phase_validation_manifest.v2",
        "phase_detector": "task_aware_v2",
        "task_count": len(cases),
        "cases": cases,
    }
    output = CONFIGS / "phase_validation_40tasks_manifest.json"
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(output)


if __name__ == "__main__":
    main()
