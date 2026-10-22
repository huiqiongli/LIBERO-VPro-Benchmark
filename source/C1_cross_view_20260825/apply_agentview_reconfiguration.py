#!/usr/bin/env python3
"""Apply reviewed agentview-reference fixes and regenerate C1 case specs."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "C1_case_revalidation_20260825/validated_cases.json"
BACKUP = SOURCE.with_name("validated_cases.before_agentview_reconfiguration.json")
POSITION_CANDIDATES = (
    HERE / "agentview_reference_revalidation/position_reconfiguration_candidates.json"
)
REPORT = HERE / "agentview_reference_revalidation/reconfiguration_report.json"


def row_map(payload: dict) -> dict[str, tuple[str, dict]]:
    return {
        row["case_id"]: (subclass, row)
        for subclass, rows in payload["cases"].items()
        for row in rows
    }


def set_disappear(row: dict, target: str, category: str) -> None:
    row["target"] = target
    row["target_category"] = category
    row["edit"] = {
        "op": "disappear",
        "object": target,
        # Move well outside all camera frusta.  -10 m occasionally left a
        # single segmentation pixel for composite meshes.
        "hidden_z": -100.0,
    }


def main() -> int:
    source = json.loads(SOURCE.read_text())
    positions = json.loads(POSITION_CANDIDATES.read_text())
    if positions.get("unresolved"):
        raise RuntimeError(f"Unresolved position cases: {positions['unresolved']}")
    if not BACKUP.exists():
        shutil.copy2(SOURCE, BACKUP)

    rows = row_map(source)
    changes: list[dict] = []
    for case_id, result in sorted(positions["cases"].items()):
        selected = result.get("selected")
        if not selected:
            raise RuntimeError(f"Position case has no selected candidate: {case_id}")
        subclass, row = rows[case_id]
        before = {
            "target": row["target"],
            "target_category": row.get("target_category"),
            "edit": row["edit"],
        }
        row["target"] = selected["target"]
        row["target_category"] = selected["target_category"]
        row["edit"] = {
            "op": "relocate",
            "object": selected["target"],
            "offset_m": selected["offset_m"],
        }
        row["agentview_reconfiguration"] = {
            "all_ten_init_states_candidate_passed": True,
            "minimum_clean_pixels": selected["minimum_clean_pixels"],
            "minimum_edited_pixels": selected["minimum_edited_pixels"],
            "minimum_centroid_shift_pixels": selected[
                "minimum_centroid_shift_pixels"
            ],
            "new_object_penetration": False,
        }
        changes.append({
            "case_id": case_id, "subclass": subclass,
            "before": before,
            "after": {
                "target": row["target"],
                "target_category": row.get("target_category"),
                "edit": row["edit"],
            },
        })

    targeted_disappear = {
        "C1_existence_libero_10_t0": ("tomato_sauce_1", "tomato_sauce"),
        "C1_count_libero_10_t4": ("porcelain_mug_1", "porcelain_mug"),
        # bowl_2 carries a stable 1--4 pixel instance-map artifact even when
        # moved 1000 m away.  bowl_1 is the instruction-relevant bowl and its
        # complete disappearance was verified on all ten init states.
        "C1_count_libero_spatial_t3": ("akita_black_bowl_1", "akita_black_bowl"),
        "C1_count_libero_spatial_t4": ("akita_black_bowl_1", "akita_black_bowl"),
        "C1_count_libero_spatial_t9": ("akita_black_bowl_1", "akita_black_bowl"),
    }
    for case_id, (target, category) in targeted_disappear.items():
        subclass, row = rows[case_id]
        before = {
            "target": row["target"],
            "target_category": row.get("target_category"),
            "edit": row["edit"],
        }
        set_disappear(row, target, category)
        row["agentview_reconfiguration"] = {
            "semantic_reference_view": "agentview",
            "reason": (
                "select_task_relevant_visible_instance"
                if before["target"] != target
                else "eliminate_residual_composite_segmentation_pixel"
            ),
        }
        changes.append({
            "case_id": case_id, "subclass": subclass,
            "before": before,
            "after": {
                "target": row["target"],
                "target_category": row.get("target_category"),
                "edit": row["edit"],
            },
        })

    state_case_id = "C1_state_libero_10_t9"
    subclass, row = rows[state_case_id]
    before = dict(row["edit"])
    row["edit"] = {
        "op": "state_toggle",
        "joint": before["joint"],
        "value_mode": "farthest_joint_limit",
    }
    row["agentview_reconfiguration"] = {
        "semantic_reference_view": "agentview",
        "reason": "guarantee_at_least_half_joint_range_state_separation",
    }
    changes.append({
        "case_id": state_case_id, "subclass": subclass,
        "before": before, "after": row["edit"],
    })

    SOURCE.write_text(json.dumps(source, indent=2, ensure_ascii=False) + "\n")
    subprocess.run([sys.executable, str(HERE / "generate_cases.py")], check=True)
    payload = {
        "schema_version": "wam_syncbench.c1_agentview_reconfiguration.v1",
        "source": str(SOURCE),
        "backup": str(BACKUP),
        "changed_semantic_case_count": len(changes),
        "changes": changes,
    }
    REPORT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({
        "changed_semantic_cases": len(changes),
        "source": str(SOURCE),
        "backup": str(BACKUP),
        "report": str(REPORT),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
