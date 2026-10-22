#!/usr/bin/env python3
"""Aggregate the 40-task A1 physical-surface validation outputs."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np


SUITES = ("libero_object", "libero_spatial", "libero_goal", "libero_10")
VIEWS = ("agentview", "robot0_eye_in_hand")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    rows = []
    task_rows = []
    mode_counts: Counter[str] = Counter()
    coverage_counts: Counter[str] = Counter()
    suite_counts: Counter[str] = Counter()

    contact_dir = root / "contact_sheets"
    contact_dir.mkdir(parents=True, exist_ok=True)
    for suite in SUITES:
        suite_panels = {view: [] for view in VIEWS}
        for task_id in range(10):
            task_dir = root / suite / f"task_{task_id:02d}"
            result_path = task_dir / "validation.json"
            result = json.loads(result_path.read_text(encoding="utf-8"))
            task_passed = all(bool(row["passed"]) for row in result["rows"])
            task_rows.append(
                {
                    "suite": suite,
                    "task_id": task_id,
                    "instruction": result["rows"][0]["instruction"],
                    "passed": task_passed,
                    "passed_ratios": sum(bool(row["passed"]) for row in result["rows"]),
                    "total_ratios": len(result["rows"]),
                }
            )
            for row in result["rows"]:
                rows.append(row)
                mode = (
                    row.get("surface_audit", [{}])[0].get("mode", "missing")
                    if row.get("surface_audit")
                    else "missing"
                )
                mode_counts[mode] += 1
                coverage_counts[f"{int(round(float(row['coverage']) * 100))}pct"] += 1
                suite_counts[suite] += 1
            for view in VIEWS:
                panel_path = task_dir / f"{suite}_task{task_id:02d}_{view}_ratios.png"
                panel = cv2.imread(str(panel_path), cv2.IMREAD_COLOR)
                if panel is None:
                    raise FileNotFoundError(panel_path)
                suite_panels[view].append(panel)
        for view in VIEWS:
            cv2.imwrite(
                str(contact_dir / f"{suite}_{view}_10tasks_4ratios.png"),
                np.vstack(suite_panels[view]),
            )

    deviations = []
    for row in rows:
        for audit in row.get("surface_audit", []):
            deviations.append(
                abs(
                    float(audit["actual_surface_coverage"])
                    - float(row["coverage"])
                )
            )
    summary = {
        "schema_version": "wam_syncbench.A1.physical_surface_validation_summary.v1",
        "task_count": len(task_rows),
        "ratio_case_count": len(rows),
        "passed_task_count": sum(bool(row["passed"]) for row in task_rows),
        "passed_ratio_case_count": sum(bool(row["passed"]) for row in rows),
        "all_passed": all(bool(row["passed"]) for row in rows),
        "all_exact_complete_surface_ratio": all(
            bool(row["exact_complete_surface_ratio"]) for row in rows
        ),
        "all_physical_state_unchanged": all(
            bool(row["physical_state_unchanged"]) for row in rows
        ),
        "all_target_locality_ok": all(bool(row["target_locality_ok"]) for row in rows),
        "all_same_modified_world_surface": all(
            bool(row["same_modified_world_surface_for_both_views"]) for row in rows
        ),
        "max_absolute_complete_surface_coverage_error": max(deviations, default=None),
        "counts_by_suite": dict(suite_counts),
        "counts_by_coverage": dict(coverage_counts),
        "counts_by_implementation_mode": dict(mode_counts),
        "tasks": task_rows,
        "rows": rows,
    }
    (root / "validation_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    with (root / "validation_matrix.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=(
                "suite", "task_id", "instruction", "coverage", "target_objects",
                "implementation_mode", "actual_surface_coverage",
                "observed_agentview_overlay_coverage", "exact_complete_surface_ratio",
                "physical_state_unchanged", "target_locality_ok", "passed",
            ),
        )
        writer.writeheader()
        for row in rows:
            audits = row.get("surface_audit", [])
            writer.writerow(
                {
                    "suite": row["suite"],
                    "task_id": row["task_id"],
                    "instruction": row["instruction"],
                    "coverage": row["coverage"],
                    "target_objects": ";".join(str(item["object"]) for item in audits),
                    "implementation_mode": ";".join(str(item["mode"]) for item in audits),
                    "actual_surface_coverage": ";".join(
                        str(item["actual_surface_coverage"]) for item in audits
                    ),
                    "observed_agentview_overlay_coverage": ";".join(
                        str(item["observed_agentview_overlay_coverage"])
                        for item in audits
                    ),
                    "exact_complete_surface_ratio": row[
                        "exact_complete_surface_ratio"
                    ],
                    "physical_state_unchanged": row["physical_state_unchanged"],
                    "target_locality_ok": row["target_locality_ok"],
                    "passed": row["passed"],
                }
            )
    print(
        json.dumps(
            {
                key: summary[key]
                for key in (
                    "task_count", "ratio_case_count", "passed_task_count",
                    "passed_ratio_case_count", "all_passed",
                    "max_absolute_complete_surface_coverage_error",
                )
            }
        )
    )
    return 0 if summary["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
