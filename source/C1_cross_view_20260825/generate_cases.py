#!/usr/bin/env python3
"""Materialize the revalidated C1 cases as isolated executable specifications."""

from __future__ import annotations

import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
VALIDATED = Path(
    "/data/huiqiongli/Sim/reviewed_script/"
    "C1_case_revalidation_20260825/validated_cases.json"
)
DIRECTIONS = {
    "agent": ("agentview", "robot0_eye_in_hand"),
    "wrist": ("robot0_eye_in_hand", "agentview"),
}


def dump(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def executable_edit(case: dict) -> dict:
    edit = dict(case["edit"])
    operation = str(edit["op"])
    if operation == "relocate":
        return {
            "op": "relocate",
            "object": edit["object"],
            "offset": list(edit.get("offset_m", [0.12, 0.0, 0.0])),
        }
    if operation == "state_toggle":
        # The historical C1 implementation rendered the mirror point within
        # the compiled MuJoCo joint range on every policy observation.  Keeping
        # this dynamic is essential for all ten official init states.
        return {
            "op": "state",
            "joint": edit["joint"],
            "value_mode": edit.get("value_mode", "joint_range_mirror"),
        }
    return edit


def make_spec(subclass: str, case: dict, direction: str) -> dict:
    corrupted, truth = DIRECTIONS[direction]
    case_id = f"{case['case_id']}__{direction}"
    return {
        "schema_version": "wam_syncbench.c1_cross_view.v2",
        "status": "validated_semantic_definition",
        "track": "C1",
        "subtask": subclass,
        "case_id": case_id,
        "semantic_case_id": case["case_id"],
        "suite": case["suite"],
        "task_id": int(case["task_id"]),
        "instruction": case["instruction"],
        "bddl_file": case["bddl_file"],
        "direction": direction,
        "enabled": True,
        "observation_mode": "perturbed",
        "target_policy": "object_edit",
        "mask_source": "object_edit",
        "render_mode": "temporary_counterfactual_simulator_branch",
        "views": [corrupted],
        "corrupted_view": corrupted,
        "truth_view": truth,
        "target": case["target"],
        "object_edits": [executable_edit(case)],
        "save_masked_video": True,
        "visual_input_contract": "official_dual_camera",
        "simulator_state_contract": {
            "physical_rollout_unchanged": True,
            "temporary_edit_restored_after_render": True,
        },
        "source_validation_status": case["status"],
        "existing_scenario": case.get("existing_scenario", ""),
    }


def manifest(rows: list[dict], name: str) -> dict:
    return {
        "schema_version": "wam_syncbench.c1_episode_manifest.v2",
        "name": name,
        # FastWAM's resident-model batch path reads `cases`; the other five
        # model workers read `episodes`.  They intentionally reference the
        # same list so the experiment cannot drift between models.
        "cases": rows,
        "episodes": rows,
        "episode_count": len(rows),
    }


def main() -> None:
    validated = json.loads(VALIDATED.read_text())
    specs: dict[tuple[str, str, int, str], Path] = {}
    source_cases: dict[tuple[str, str, int], dict] = {}
    full_rows = []
    missing_rows = []
    for subclass, cases in validated["cases"].items():
        for case in cases:
            source_cases[(subclass, case["suite"], int(case["task_id"]))] = case
            for direction in DIRECTIONS:
                spec = make_spec(subclass, case, direction)
                path = (
                    HERE / "cases" / subclass / case["suite"]
                    / f"{spec['case_id']}.json"
                )
                dump(path, spec)
                specs[(subclass, case["suite"], int(case["task_id"]), direction)] = path
                for trial in range(10):
                    row = {
                        "track": "C1",
                        "subtask": subclass,
                        "case_id": spec["case_id"],
                        "suite": case["suite"],
                        "task_id": int(case["task_id"]),
                        "instruction": case["instruction"],
                        "direction": direction,
                        "trial_index": trial,
                        "case_json": str(path),
                        "spec": str(path),
                    }
                    full_rows.append(row)
                    if case["status"] == "new_case_needs_eval":
                        missing_rows.append(row)

    # One semantically representative task per subclass, both directions.
    # All four live in LIBERO-10 so suite-specific models load only once.
    smoke_keys = (
        ("existence", "libero_10", 0),
        ("position", "libero_10", 0),
        ("state", "libero_10", 2),
        ("count", "libero_10", 4),
    )
    smoke_rows = []
    for subclass, suite, task_id in smoke_keys:
        case = source_cases[(subclass, suite, task_id)]
        for direction in DIRECTIONS:
            path = specs[(subclass, suite, task_id, direction)]
            spec = json.loads(path.read_text())
            smoke_rows.append({
                "track": "C1",
                "subtask": subclass,
                "case_id": spec["case_id"],
                "suite": suite,
                "task_id": task_id,
                "instruction": case["instruction"],
                "direction": direction,
                "trial_index": 0,
                "case_json": str(path),
                "spec": str(path),
            })

    manifest_root = HERE / "episode_manifests"
    dump(manifest_root / "smoke" / "all.json", manifest(smoke_rows, "C1_6model_smoke"))
    dump(manifest_root / "full" / "all.json", manifest(full_rows, "C1_full_1900"))
    dump(manifest_root / "missing" / "all.json", manifest(missing_rows, "C1_missing_100"))
    for suite in ("libero_object", "libero_spatial", "libero_goal", "libero_10"):
        for kind, rows in (("smoke", smoke_rows), ("full", full_rows), ("missing", missing_rows)):
            selected = [row for row in rows if row["suite"] == suite]
            if selected:
                dump(
                    manifest_root / kind / f"{suite}.json",
                    manifest(selected, f"C1_{kind}_{suite}"),
                )
    report = {
        "schema_version": "wam_syncbench.c1_generation_report.v2",
        "semantic_case_count": sum(len(v) for v in validated["cases"].values()),
        "direction_configuration_count": len(specs),
        "full_episode_count": len(full_rows),
        "missing_semantic_case_count": len({(r['subtask'], r['suite'], r['task_id']) for r in missing_rows}),
        "missing_episode_count": len(missing_rows),
        "smoke_episode_count_per_model": len(smoke_rows),
        "subclass_counts": {key: len(value) for key, value in validated["cases"].items()},
    }
    dump(HERE / "generation_report.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
