#!/usr/bin/env python3
"""Build an isolated, no-perturbation 4-suite x 40-task x 10-trial manifest."""
from __future__ import annotations

import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
SUITES = ("libero_object", "libero_spatial", "libero_goal", "libero_10")


def main() -> None:
    case_root = HERE / "cases"
    episodes: list[dict] = []
    base_cases: list[dict] = []
    for suite in SUITES:
        suite_dir = case_root / suite
        suite_dir.mkdir(parents=True, exist_ok=True)
        for task_id in range(10):
            case_id = f"clean__{suite}__task{task_id:02d}"
            case = {
                "schema_version": "wam_syncbench.clean_baseline_case.v1",
                "track": "CLEAN",
                "subtask": "clean_baseline",
                "condition": "clean",
                "observation_mode": "clean",
                "mechanism": "none",
                "case_id": case_id,
                "suite": suite,
                "task_id": task_id,
                "instruction_source": "LIBERO task language",
                "environment_edit": False,
                "observation_perturbation": False,
                "decision_perturbation": False,
            }
            case_path = suite_dir / f"task_{task_id:02d}.json"
            case_path.write_text(json.dumps(case, indent=2) + "\n", encoding="utf-8")
            base_cases.append({**case, "case_json": str(case_path.resolve()), "spec": str(case_path.resolve())})
            for trial in range(10):
                episodes.append({
                    **case,
                    "case_json": str(case_path.resolve()),
                    "spec": str(case_path.resolve()),
                    "trial_index": trial,
                })

    payload = {
        "schema_version": "wam_syncbench.clean_baseline_manifest.v1",
        "name": "clean_baseline_6models_40task_10trial",
        "suites": list(SUITES),
        "task_count": 40,
        "trials_per_task": 10,
        "base_case_count": len(base_cases),
        "episode_count": len(episodes),
        "cases": base_cases,
        "episodes": episodes,
    }
    (HERE / "manifest.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    (HERE / "episodes.json").write_text(json.dumps({"episodes": episodes}, indent=2) + "\n", encoding="utf-8")
    (HERE / "fastwam_manifest.json").write_text(json.dumps({"cases": episodes}, indent=2) + "\n", encoding="utf-8")
    for suite in SUITES:
        rows = [row for row in episodes if row["suite"] == suite]
        (HERE / f"manifest_{suite}.json").write_text(
            json.dumps({"episodes": rows}, indent=2) + "\n", encoding="utf-8"
        )
    print(f"built {len(base_cases)} base cases and {len(episodes)} episodes")


if __name__ == "__main__":
    main()
