#!/usr/bin/env python3
"""Validate D1 distractors on LIBERO initial states 1 through 9.

This initializes and renders environments only. It never loads a policy.
Each suite writes independent case reports so suites can run in parallel.
"""

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path

from PIL import Image

import render_and_validate as validator


HERE = Path(__file__).resolve().parent
PREVIEW_ROOT = HERE / "previews_trials_01_09"
VALIDATION_ROOT = HERE / "validation_trials_01_09"


def write_json(value: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def validate_suite(
    suite_name: str,
    init_state_indices: list[int],
    task_ids: set[int] | None,
    conditions: list[str],
    resume: bool,
) -> dict:
    manifest = validator.load_json(HERE / f"manifest_{suite_name}.json")
    suite = validator.benchmark.get_benchmark_dict()[suite_name]()
    suite_root = VALIDATION_ROOT / suite_name
    summary_path = suite_root / "summary.json"
    summary = {
        "schema_version": "d1_multistate_validation.v1",
        "suite": suite_name,
        "init_state_indices": init_state_indices,
        "conditions": conditions,
        "case_count": 0,
        "passed_count": 0,
        "failed_count": 0,
        "error_count": 0,
        "status": "running",
    }

    for task in manifest["tasks"]:
        task_id = int(task["task_id"])
        if task_ids is not None and task_id not in task_ids:
            continue
        init_states = suite.get_task_init_states(task_id)
        bddl = Path(suite.get_task_bddl_file_path(task_id))
        original_names = validator.original_scene_names(bddl)

        for init_state_index in init_state_indices:
            if init_state_index >= len(init_states):
                raise IndexError(
                    f"{suite_name} task {task_id} only has {len(init_states)} init states; "
                    f"requested {init_state_index}"
                )
            case_dir = (
                suite_root
                / f"task_{task_id:02d}"
                / f"init_state_{init_state_index:02d}"
            )
            report_path = case_dir / "validation.json"
            if resume and report_path.exists():
                existing = validator.load_json(report_path)
                if (
                    existing.get("status") == "complete"
                    and {item.get("condition") for item in existing.get("results", [])}
                    == set(conditions)
                ):
                    continue

            init_state = init_states[init_state_index]
            # render_case constructs paths below PREVIEWS/suite/task. Point PREVIEWS
            # at this init-state namespace so no trial overwrites another.
            validator.PREVIEWS = PREVIEW_ROOT / f"init_state_{init_state_index:02d}"
            preview_dir = validator.PREVIEWS / suite_name / f"task_{task_id:02d}"
            preview_dir.mkdir(parents=True, exist_ok=True)
            report = {
                "schema_version": "d1_multistate_case_validation.v1",
                "suite": suite_name,
                "task_id": task_id,
                "instruction": task["instruction"],
                "init_state_index": init_state_index,
                "status": "running",
                "results": [],
            }
            write_json(report, report_path)

            try:
                clean = validator.prepare_clean(
                    suite_name,
                    suite,
                    task_id,
                    init_state,
                    original_names,
                )
                for view in ("agentview", "wrist"):
                    Image.fromarray(clean["images"][view]).save(
                        preview_dir / f"clean_{view}.png"
                    )
            except Exception as error:
                traceback.print_exc()
                report.update(
                    status="error",
                    error=f"clean initialization: {type(error).__name__}: {error}",
                )
                write_json(report, report_path)
                continue

            for condition in conditions:
                print(
                    f"[D1-multistate] {suite_name} task={task_id:02d} "
                    f"init={init_state_index:02d} condition={condition}",
                    flush=True,
                )
                try:
                    result, _, _ = validator.render_case(
                        suite_name,
                        suite,
                        task,
                        init_state,
                        condition,
                        clean,
                        original_names,
                    )
                    result["init_state_index"] = init_state_index
                except Exception as error:
                    traceback.print_exc()
                    result = {
                        "suite": suite_name,
                        "task_id": task_id,
                        "init_state_index": init_state_index,
                        "condition": condition,
                        "passed": False,
                        "error": f"{type(error).__name__}: {error}",
                        "checks": [],
                    }
                report["results"].append(result)
                write_json(report, report_path)

            report["passed_count"] = sum(
                bool(item.get("passed")) for item in report["results"]
            )
            report["case_count"] = len(report["results"])
            report["all_passed"] = report["passed_count"] == report["case_count"]
            report["status"] = "complete"
            write_json(report, report_path)

    reports = list(suite_root.glob("task_*/init_state_*/validation.json"))
    completed = []
    for path in reports:
        data = validator.load_json(path)
        if data.get("status") == "complete":
            completed.append(data)
        elif data.get("status") == "error":
            summary["error_count"] += 1
    summary["init_state_count"] = len(completed)
    summary["case_count"] = sum(int(item.get("case_count", 0)) for item in completed)
    summary["passed_count"] = sum(int(item.get("passed_count", 0)) for item in completed)
    summary["failed_count"] = summary["case_count"] - summary["passed_count"]
    summary["all_passed"] = (
        summary["error_count"] == 0
        and summary["passed_count"] == summary["case_count"]
    )
    summary["status"] = "complete"
    write_json(summary, summary_path)
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", required=True)
    parser.add_argument("--init-state-indices", nargs="+", type=int, default=list(range(1, 10)))
    parser.add_argument("--tasks", nargs="*", type=int)
    parser.add_argument("--conditions", nargs="*", choices=validator.CONDITIONS)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    validate_suite(
        args.suite,
        list(args.init_state_indices),
        set(args.tasks) if args.tasks else None,
        list(args.conditions or validator.CONDITIONS),
        bool(args.resume),
    )


if __name__ == "__main__":
    main()
