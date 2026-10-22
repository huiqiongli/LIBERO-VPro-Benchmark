#!/usr/bin/env python3
"""Inspect render-surface components used by A2 target receptacles."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np


SIM = Path("/home/huiqiongli/Sim")
HERE = Path(__file__).resolve().parent
sys.path[:0] = [
    str(SIM),
    str(SIM / "Repos/LIBERO"),
    str(SIM / "Repos/lingbot-va/evaluation/libero"),
]

from libero.libero import benchmark  # noqa: E402
from libero.libero.envs import SegmentationRenderEnv  # noqa: E402
import robust_libero_perturbations as perturb  # noqa: E402


def primitive_area(geom_type: int, size: np.ndarray) -> float | None:
    a, b, c = [float(value) for value in size]
    if geom_type == 2:  # sphere
        return 4.0 * math.pi * a * a
    if geom_type == 3:  # capsule: radius, cylinder half-length
        return 4.0 * math.pi * a * a + 4.0 * math.pi * a * b
    if geom_type == 4:  # ellipsoid, Knud Thomsen approximation
        p = 1.6075
        return 4.0 * math.pi * (
            ((a * b) ** p + (a * c) ** p + (b * c) ** p) / 3.0
        ) ** (1.0 / p)
    if geom_type == 5:  # closed cylinder: radius, half-height
        return 4.0 * math.pi * a * b + 2.0 * math.pi * a * a
    if geom_type == 6:  # box half-sizes
        return 8.0 * (a * b + a * c + b * c)
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", required=True)
    parser.add_argument("--task-id", type=int, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    suite = benchmark.get_benchmark_dict()[args.suite]()
    state = suite.get_task_init_states(args.task_id)[0]
    spec_path = HERE / "configs" / args.suite / "A2_physical_receptacle_50pct.json"
    raw_spec = json.loads(spec_path.read_text(encoding="utf-8"))
    spec = perturb.resolve_env_edit_spec(raw_spec, args.suite, args.task_id)
    targets = [
        str(item["object"])
        for item in spec.get("object_surface_black_fraction_overrides", [])
    ]
    env = SegmentationRenderEnv(
        bddl_file_name=suite.get_task_bddl_file_path(args.task_id),
        camera_heights=256,
        camera_widths=256,
    )
    try:
        env.reset()
        env.set_init_state(state)
        base = env.env
        sim = base.sim
        perturb._forward_sim(sim)
        rows = []
        for target in targets:
            root = int(base.obj_body_id[target])
            bodies = set(perturb._descendant_body_ids(sim, root))
            geoms = []
            for geom_id in range(int(sim.model.ngeom)):
                if (
                    int(sim.model.geom_bodyid[geom_id]) not in bodies
                    or int(sim.model.geom_group[geom_id]) == 0
                    or float(sim.model.geom_rgba[geom_id][3]) <= 0.0
                ):
                    continue
                mesh = perturb._mesh_surface_description(base, geom_id)
                geom_type = int(sim.model.geom_type[geom_id])
                if mesh is not None:
                    area = float(np.asarray(mesh["areas"]).sum())
                elif geom_type == 7:
                    mesh_id = int(sim.model.geom_dataid[geom_id])
                    va = int(sim.model.mesh_vertadr[mesh_id])
                    fa = int(sim.model.mesh_faceadr[mesh_id])
                    vn = int(sim.model.mesh_vertnum[mesh_id])
                    fn = int(sim.model.mesh_facenum[mesh_id])
                    vertices = np.asarray(sim.model.mesh_vert[va : va + vn], dtype=float)
                    faces = np.asarray(sim.model.mesh_face[fa : fa + fn], dtype=np.int64)
                    triangles = vertices[faces]
                    area = float(
                        (
                            0.5
                            * np.linalg.norm(
                                np.cross(
                                    triangles[:, 1] - triangles[:, 0],
                                    triangles[:, 2] - triangles[:, 0],
                                ),
                                axis=1,
                            )
                        ).sum()
                    )
                else:
                    area = primitive_area(
                        geom_type, np.asarray(sim.model.geom_size[geom_id], dtype=float)
                    )
                geoms.append(
                    {
                        "geom_id": geom_id,
                        "name": sim.model.geom_id2name(geom_id) or f"geom_{geom_id}",
                        "type": geom_type,
                        "size": np.asarray(sim.model.geom_size[geom_id]).tolist(),
                        "area": area,
                        "editable_uv_mesh": mesh is not None,
                        "texture_id": perturb._surface_texture_id(sim, geom_id),
                        "world_centroid": np.asarray(sim.data.geom_xpos[geom_id]).tolist(),
                    }
                )
            rows.append(
                {
                    "suite": args.suite,
                    "task_id": args.task_id,
                    "instruction": suite.get_task(args.task_id).language,
                    "target": target,
                    "geoms": geoms,
                    "surface_area_total": sum(
                        float(item["area"]) for item in geoms if item["area"] is not None
                    ),
                    "editable_surface_area": sum(
                        float(item["area"])
                        for item in geoms
                        if item["area"] is not None and item["editable_uv_mesh"]
                    ),
                }
            )
    finally:
        env.close()

    result = {"rows": rows}
    text = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
