#!/usr/bin/env python3
"""Build the formal 4-suite x 4-ratio x 10-task x 10-trial A2 manifest."""

from __future__ import annotations

import csv
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "A2_physical_receptacle_ratio_20260822"
VALIDATION = Path(
    "/data/huiqiongli/Sim/Outputs/"
    "A2_physical_receptacle_ratio_complete_surface_40task_validation_20260822/"
    "validation_matrix.csv"
)
SUITES = ("libero_object", "libero_spatial", "libero_goal", "libero_10")
RATIOS = (25, 50, 75, 100)


def main() -> None:
    instruction: dict[tuple[str, int], str] = {}
    passed: set[tuple[str, int, int]] = set()
    with VALIDATION.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            suite = row["suite"]
            task = int(row["task_id"])
            ratio = round(float(row["coverage"]) * 100)
            instruction[(suite, task)] = row["instruction"]
            if str(row["passed"]).lower() == "true":
                passed.add((suite, task, ratio))

    expected = {(suite, task, ratio) for suite in SUITES for task in range(10) for ratio in RATIOS}
    missing = sorted(expected - passed)
    if missing:
        raise SystemExit(f"Refusing full launch: {len(missing)} environment cases are not validated: {missing[:10]}")

    base_cases = []
    episodes = []
    for ratio in RATIOS:
        label = f"physical_{ratio}pct"
        for suite in SUITES:
            for task in range(10):
                spec = SOURCE / "task_specs" / suite / f"task_{task:02d}" / f"{label}.json"
                if not spec.is_file():
                    raise SystemExit(f"Missing task spec: {spec}")
                base = {
                    "case_id": f"A2_ratio_{ratio}pct__{suite}__task{task:02d}",
                    "suite": suite,
                    "task_id": task,
                    "instruction": instruction[(suite, task)],
                    "condition": "perturbed",
                    "mechanism": "physical_receptacle_complete_surface_blackening",
                    "ratio": ratio,
                    "subclass": f"{ratio}pct",
                    "spec": str(spec.resolve()),
                    "spec_path": str(spec.resolve()),
                }
                base_cases.append(base)
                for trial in range(10):
                    episodes.append({**base, "trial_index": trial})

    payload = {
        "schema_version": "wam_syncbench.A2.physical_receptacle_ratio_full.v1",
        "coverage_definition": "complete_3d_render_surface_area",
        "suite_count": 4,
        "task_count": 40,
        "ratios": list(RATIOS),
        "trials_per_case": 10,
        "base_case_count": len(base_cases),
        "episode_count": len(episodes),
        "cases": base_cases,
    }
    (HERE / "manifest.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    fastwam = {
        "schema_version": "wam_syncbench.A2.physical_receptacle_ratio_fastwam_batch.v1",
        "cases": episodes,
    }
    (HERE / "fastwam_manifest.json").write_text(json.dumps(fastwam, indent=2) + "\n", encoding="utf-8")
    print(f"built {len(base_cases)} cases and {len(episodes)} episodes")


if __name__ == "__main__":
    main()
