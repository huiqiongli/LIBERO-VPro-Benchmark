#!/usr/bin/env python3
"""Generate physical target-receptacle surface-blackening configs."""

from __future__ import annotations

import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
ADAPTED = HERE.parent / "adapted"
SUITES = ("libero_object", "libero_spatial", "libero_goal", "libero_10")
COVERAGES = (0.25, 0.50, 0.75, 1.0)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    variant_manifest_path = (
        HERE.parents[1] / "assets/A2_target_receptacle"
        / "precompiled_variants_full_surface_v3/manifest.json"
    )
    variant_entries = json.loads(
        variant_manifest_path.read_text(encoding="utf-8")
    )["entries"]
    variants = {
        (
            str(entry["suite"]),
            int(entry["task_id"]),
            int(round(float(entry["coverage"]) * 100)),
            str(entry["object"]),
        ): entry
        for entry in variant_entries
    }
    manifest: list[dict] = []
    for suite in SUITES:
        legacy = json.loads(
            (ADAPTED / suite / "A2_receptacle_100pct.json").read_text(
                encoding="utf-8"
            )
        )
        targets_by_task = legacy["target_objects_by_task"]
        for coverage in COVERAGES:
            label = int(round(coverage * 100))
            task_edits = []
            for task_id in range(10):
                targets = targets_by_task[f"{suite}:{task_id}"]
                task_edit = {
                    "suite": suite,
                    "task_id": task_id,
                    "object_surface_black_fraction_overrides": [],
                }
                for target in targets:
                    override = {
                        "object": target,
                        "coverage": coverage,
                        "coverage_definition": "complete_3d_render_surface_area",
                        "selection_axis": "agentview_aligned_complete_surface_slice",
                        "rgba": [0.0, 0.0, 0.0, 1.0],
                        "require_exact": True,
                    }
                    variant = variants.get((suite, task_id, label, target))
                    if variant is not None:
                        override["precompiled_surface_audit"] = variant["audit"]
                        task_edit.setdefault("object_scale_overrides", []).append(
                            {
                                "object": target,
                                "category": variant["category"],
                                "scale": 1.0,
                                "scale_mode": "precompile_xml",
                                "xml_path": variant["xml_path"],
                            }
                        )
                    task_edit["object_surface_black_fraction_overrides"].append(override)
                task_edits.append(task_edit)
            spec = {
                "schema_version": "wam_syncbench.A2.physical_surface.v1",
                "track": "A2",
                "scenario": "A2_physical_target_receptacle_ratio",
                "variant": f"physical_surface_black_{label}pct",
                "description": (
                    "Blacken the requested fraction of the target receptacle's complete "
                    "three-dimensional render surface. A contiguous camera-aligned slice "
                    "is cut from the full surface, including inner, rear, initially hidden, "
                    "and later revealed regions that fall inside that slice."
                ),
                "status": "implemented",
                "enabled": True,
                "suite": suite,
                "coverage_definition": "complete_3d_render_surface_area",
                "coverage": coverage,
                "selection_axis": "agentview_aligned_complete_surface_slice",
                "render_edit_only": True,
                "collision_unchanged": True,
                "pose_unchanged": True,
                "observation_pixel_masking": False,
                "require_task_match": True,
                "task_env_edits": task_edits,
            }
            path = HERE / "configs" / suite / f"A2_physical_receptacle_{label}pct.json"
            write_json(path, spec)
            for task_edit in task_edits:
                resolved = {
                    key: value for key, value in spec.items() if key != "task_env_edits"
                }
                resolved.update(task_edit)
                write_json(
                    HERE
                    / "task_specs"
                    / suite
                    / f"task_{int(task_edit['task_id']):02d}"
                    / f"physical_{label}pct.json",
                    resolved,
                )
            manifest.append(
                {
                    "suite": suite,
                    "coverage": coverage,
                    "label": f"{label}pct",
                    "spec": str(path),
                    "tasks": 10,
                }
            )
    write_json(
        HERE / "manifest.json",
        {
            "schema_version": "wam_syncbench.A2.physical_surface_manifest.v1",
            "cases_per_model_per_trial": len(SUITES) * 10 * len(COVERAGES),
            "entries": manifest,
        },
    )


if __name__ == "__main__":
    main()
