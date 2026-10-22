#!/usr/bin/env python3
"""Render and audit the four physical A1 surface-blackening ratios."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import cv2
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


VIEWS = ("agentview", "robot0_eye_in_hand")


def copy_images(obs: dict) -> dict[str, np.ndarray]:
    return {view: np.array(obs[f"{view}_image"], copy=True) for view in VIEWS}


def label(image: np.ndarray, text: str) -> np.ndarray:
    image = np.asarray(image, dtype=np.uint8)[::-1].copy()
    cv2.rectangle(image, (0, 0), (image.shape[1], 24), (245, 245, 245), -1)
    cv2.putText(image, text, (5, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1)
    return image


def physics_snapshot(base: object) -> dict[str, object]:
    """Capture state that can affect dynamics, initialization, or predicates."""
    sim = base.sim
    model = sim.model

    def values(array: object) -> list:
        return np.asarray(array).tolist()

    collision_geoms = []
    for geom_id in range(int(model.ngeom)):
        if int(model.geom_contype[geom_id]) == 0 and int(model.geom_conaffinity[geom_id]) == 0:
            continue
        body_id = int(model.geom_bodyid[geom_id])
        raw_name = model.geom_id2name(geom_id) or f"geom_{geom_id}"
        semantic_name = None if re.search(r"_g\d+$", raw_name) else raw_name
        collision_geoms.append(
            {
                "name": semantic_name,
                "body": model.body_id2name(body_id) or f"body_{body_id}",
                "type": int(model.geom_type[geom_id]),
                "pos": values(model.geom_pos[geom_id]),
                "quat": values(model.geom_quat[geom_id]),
                "size": values(model.geom_size[geom_id]),
                "contype": int(model.geom_contype[geom_id]),
                "conaffinity": int(model.geom_conaffinity[geom_id]),
                "friction": values(model.geom_friction[geom_id]),
                "solref": values(model.geom_solref[geom_id]),
                "solimp": values(model.geom_solimp[geom_id]),
                "margin": float(model.geom_margin[geom_id]),
                "gap": float(model.geom_gap[geom_id]),
            }
        )
    collision_geoms.sort(key=lambda item: json.dumps(item, sort_keys=True))
    joints = []
    for joint_id in range(int(model.njnt)):
        joints.append(
            {
                "name": model.joint_id2name(joint_id) or f"joint_{joint_id}",
                "type": int(model.jnt_type[joint_id]),
                "body": model.body_id2name(int(model.jnt_bodyid[joint_id])),
                "pos": values(model.jnt_pos[joint_id]),
                "axis": values(model.jnt_axis[joint_id]),
                "range": values(model.jnt_range[joint_id]),
                "damping": float(model.dof_damping[int(model.jnt_dofadr[joint_id])])
                if int(model.jnt_dofadr[joint_id]) >= 0
                else None,
            }
        )
    joints.sort(key=lambda item: item["name"])
    sites = []
    for site_id in range(int(model.nsite)):
        sites.append(
            {
                "name": model.site_id2name(site_id) or f"site_{site_id}",
                "body": model.body_id2name(int(model.site_bodyid[site_id])),
                "type": int(model.site_type[site_id]),
                "pos": values(model.site_pos[site_id]),
                "quat": values(model.site_quat[site_id]),
                "size": values(model.site_size[site_id]),
            }
        )
    sites.sort(key=lambda item: (item["body"], item["name"]))
    bodies = []
    for body_id in range(int(model.nbody)):
        bodies.append(
            {
                "name": model.body_id2name(body_id) or f"body_{body_id}",
                "parent": model.body_id2name(int(model.body_parentid[body_id]))
                if body_id > 0
                else None,
                "mass": float(model.body_mass[body_id]),
                "inertia": values(model.body_inertia[body_id]),
                "ipos": values(model.body_ipos[body_id]),
                "iquat": values(model.body_iquat[body_id]),
                "pos": values(model.body_pos[body_id]),
                "quat": values(model.body_quat[body_id]),
            }
        )
    bodies.sort(key=lambda item: item["name"])
    return {
        "collision_geoms": collision_geoms,
        "joints": joints,
        "sites": sites,
        "bodies": bodies,
    }


def snapshots_equal(left: object, right: object, atol: float = 1e-10) -> bool:
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(
            snapshots_equal(left[key], right[key], atol) for key in left
        )
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            snapshots_equal(a, b, atol) for a, b in zip(left, right)
        )
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return bool(np.isclose(left, right, atol=atol, rtol=0.0))
    return left == right


def snapshot_differences(
    left: object, right: object, path: str = "", limit: int = 20
) -> list[str]:
    differences: list[str] = []
    if isinstance(left, dict) and isinstance(right, dict):
        if left.keys() != right.keys():
            return [f"{path}: keys differ"]
        for key in left:
            differences.extend(
                snapshot_differences(left[key], right[key], f"{path}.{key}", limit)
            )
            if len(differences) >= limit:
                break
        return differences[:limit]
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            return [f"{path}: length {len(left)} != {len(right)}"]
        for index, (a, b) in enumerate(zip(left, right)):
            differences.extend(
                snapshot_differences(a, b, f"{path}[{index}]", limit)
            )
            if len(differences) >= limit:
                break
        return differences[:limit]
    if not snapshots_equal(left, right):
        return [f"{path}: {left!r} != {right!r}"]
    return []


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", default="libero_spatial")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--env-seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=HERE / "validation")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    suite = benchmark.get_benchmark_dict()[args.suite]()
    task = suite.get_task(args.task_id)
    state = suite.get_task_init_states(args.task_id)[0]
    rows = []
    panels = {view: [] for view in VIEWS}

    # Build a pristine official environment once for a static dynamics and
    # predicate-equivalence snapshot.  Images are compared within each single
    # renderer below so harmless cross-context rasterization noise is excluded.
    saved_override = os.environ.pop(perturb.LIBERO_OBJECT_XML_OVERRIDE_ENV, None)
    try:
        baseline_env = SegmentationRenderEnv(
            bddl_file_name=suite.get_task_bddl_file_path(args.task_id),
            camera_heights=256,
            camera_widths=256,
        )
        baseline_env.seed(args.env_seed)
        baseline_env.reset()
    finally:
        if saved_override is not None:
            os.environ[perturb.LIBERO_OBJECT_XML_OVERRIDE_ENV] = saved_override
    try:
        baseline_env.set_init_state(state)
        baseline_physics = physics_snapshot(baseline_env.env)
    finally:
        baseline_env.close()
    clean_reference: dict[str, np.ndarray] | None = None

    for label_value in (25, 50, 75, 100):
        spec_path = (
            HERE
            / "configs"
            / args.suite
            / f"A1_physical_target_object_{label_value}pct.json"
        )
        raw_spec = json.loads(spec_path.read_text(encoding="utf-8"))
        spec = perturb.resolve_env_edit_spec(raw_spec, args.suite, args.task_id)
        xml_overrides = perturb.object_xml_override_map(spec)
        old_xml_overrides = os.environ.get(perturb.LIBERO_OBJECT_XML_OVERRIDE_ENV)
        if xml_overrides:
            os.environ[perturb.LIBERO_OBJECT_XML_OVERRIDE_ENV] = json.dumps(xml_overrides)
        try:
            env = SegmentationRenderEnv(
                bddl_file_name=suite.get_task_bddl_file_path(args.task_id),
                camera_heights=256,
                camera_widths=256,
            )
            env.seed(args.env_seed)
            # LIBERO defaults to hard_reset=True, so reset may rebuild all
            # MujocoXMLObjects and must occur while the XML override is active.
            env.reset()
        finally:
            if old_xml_overrides is None:
                os.environ.pop(perturb.LIBERO_OBJECT_XML_OVERRIDE_ENV, None)
            else:
                os.environ[perturb.LIBERO_OBJECT_XML_OVERRIDE_ENV] = old_xml_overrides
        try:
            obs = env.set_init_state(state)
            base = env.env
            model = base.sim.model
            qpos_before = np.array(base.sim.data.qpos, copy=True)
            model_before = physics_snapshot(base)
            visual_style_before = {
                geom_id: {
                    "matid": int(model.geom_matid[geom_id]),
                    "rgba": np.asarray(model.geom_rgba[geom_id], dtype=float).copy(),
                }
                for geom_id in range(int(model.ngeom))
            }
            expected_target_names = [
                str(item["object"])
                for item in spec.get("object_surface_black_fraction_overrides", [])
            ]
            expected_target_body_ids: set[int] = set()
            for target_name in expected_target_names:
                expected_target_body_ids.update(
                    perturb._descendant_body_ids(
                        base.sim, int(base.obj_body_id[target_name])
                    )
                )

            if xml_overrides:
                # The variant already contains black render overlays at compile
                # time.  Hide only those non-colliding overlays to render the
                # matched clean frame in the same simulation and GL context.
                overlay_geom_ids = [
                    geom_id
                    for geom_id in range(int(model.ngeom))
                    if "a1_" in (model.geom_id2name(geom_id) or "")
                    and "_black" in (model.geom_id2name(geom_id) or "")
                ]
                saved_groups = {
                    geom_id: int(model.geom_group[geom_id])
                    for geom_id in overlay_geom_ids
                }
                for geom_id in overlay_geom_ids:
                    model.geom_group[geom_id] = 0
                base._update_observables(force=True)
                clean_obs = base._get_observations()
                clean = copy_images(clean_obs)
                for geom_id, group in saved_groups.items():
                    model.geom_group[geom_id] = group
                base._update_observables(force=True)
            else:
                clean = copy_images(obs)
            if clean_reference is None:
                clean_reference = clean
                for view in VIEWS:
                    panels[view].append(label(clean[view], f"{view} clean"))

            try:
                perturb.apply_libero_env_edit(env, spec)
            except Exception as exc:
                for view in VIEWS:
                    panels[view].append(
                        label(clean[view], f"physical {label_value}% ERROR")
                    )
                rows.append(
                    {
                        "suite": args.suite,
                        "task_id": args.task_id,
                        "instruction": task.language,
                        "coverage": label_value / 100.0,
                        "spec": str(spec_path),
                        "surface_audit": [],
                        "views": {},
                        "exact_surface_ratio": False,
                        "physical_state_unchanged": None,
                        "render_changed_in_both_views": False,
                        "render_changed_in_required_views": False,
                        "target_locality_ok": None,
                        "same_modified_world_surface_for_both_views": False,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "passed": False,
                    }
                )
                continue
            base._update_observables(force=True)
            edited_obs = base._get_observations()
            edited = copy_images(edited_obs)
            model_after = physics_snapshot(base)
            qpos_unchanged = np.array_equal(
                qpos_before, np.asarray(base.sim.data.qpos)
            )
            runtime_model_unchanged = snapshots_equal(model_before, model_after)
            official_model_equivalent = snapshots_equal(
                baseline_physics, model_before
            )
            physical_unchanged = (
                qpos_unchanged
                and runtime_model_unchanged
                and official_model_equivalent
            )
            audit = getattr(base, "_wam_syncbench_env_edit_audit", {})
            surface = list(audit.get("surface_blackening", []))
            changed_visual_geoms = []
            changed_visual_geoms_outside_target = []
            for geom_id, before in visual_style_before.items():
                changed = (
                    int(model.geom_matid[geom_id]) != before["matid"]
                    or not np.array_equal(
                        np.asarray(model.geom_rgba[geom_id], dtype=float),
                        before["rgba"],
                    )
                )
                if not changed:
                    continue
                geom_name = model.geom_id2name(geom_id) or f"geom_{geom_id}"
                changed_visual_geoms.append(geom_name)
                if int(model.geom_bodyid[geom_id]) not in expected_target_body_ids:
                    changed_visual_geoms_outside_target.append(geom_name)
            overlay_geoms_outside_target = []
            for item in surface:
                for geom_name in item.get("black_visual_geoms", []):
                    geom_id = int(model.geom_name2id(geom_name))
                    if int(model.geom_bodyid[geom_id]) not in expected_target_body_ids:
                        overlay_geoms_outside_target.append(geom_name)
            visual_edit_locality_ok = (
                not changed_visual_geoms_outside_target
                and not overlay_geoms_outside_target
            )
            requested = label_value / 100.0
            # Audit the actual visible black overlay from MuJoCo geom
            # segmentation.  This does not depend on RGB brightness, so a dark
            # target object is still counted in full before it is changed to black.
            raw_agent_segmentation = base.sim.render(
                camera_name="agentview",
                height=256,
                width=256,
                segmentation=True,
            )[::-1]
            per_target_visible_coverage = []
            per_target_observed_overlay_coverage = []
            overlay_rendered_for_all_targets = True
            for item in surface:
                original_geom_ids = [
                    int(model.geom_name2id(name))
                    for name in item.get("visual_geoms", [])
                ]
                black_geom_ids = [
                    int(model.geom_name2id(name))
                    for name in item.get("black_visual_geoms", [])
                ]
                if black_geom_ids:
                    target_mask_raw = np.isin(
                        raw_agent_segmentation[:, :, 1],
                        np.asarray(original_geom_ids + black_geom_ids, dtype=int),
                    )
                    black_mask_raw = np.isin(
                        raw_agent_segmentation[:, :, 1],
                        np.asarray(black_geom_ids, dtype=int),
                    )
                    visible_pixels = int(target_mask_raw.sum())
                    black_pixels = int(black_mask_raw.sum())
                    observed_overlay_coverage = black_pixels / float(max(1, visible_pixels))
                    if (
                        item.get("coverage_definition")
                        == "complete_3d_render_surface_area"
                    ):
                        visible_coverage = float(item["actual_surface_coverage"])
                    else:
                        visible_coverage = float(
                            item.get(
                                "predicted_agentview_visible_coverage",
                                observed_overlay_coverage,
                            )
                        )
                    overlay_rendered_for_all_targets = (
                        overlay_rendered_for_all_targets and black_pixels > 0
                    )
                else:
                    # The 100% variant directly changes every original visual
                    # geom to opaque black and therefore has no overlay geom.
                    visible_pixels = int(
                        np.isin(
                            raw_agent_segmentation[:, :, 1],
                            np.asarray(original_geom_ids, dtype=int),
                        ).sum()
                    )
                    black_pixels = visible_pixels
                    visible_coverage = 1.0
                    observed_overlay_coverage = 1.0
                item["actual_agentview_target_pixels"] = visible_pixels
                item["actual_agentview_black_pixels"] = black_pixels
                item["actual_agentview_visible_coverage"] = visible_coverage
                item["observed_agentview_overlay_coverage"] = observed_overlay_coverage
                per_target_visible_coverage.append(visible_coverage)
                per_target_observed_overlay_coverage.append(observed_overlay_coverage)
            complete_surface_mode = bool(surface) and all(
                item.get("coverage_definition")
                == "complete_3d_render_surface_area"
                for item in surface
            )
            # Complete-surface variants are audited geometrically and therefore
            # use the strict surface-area tolerance. The legacy agentview mode
            # retained a raster tolerance for small targets.
            visible_ratio_tolerance = 0.002 if complete_surface_mode else 0.03
            exact_ratio = bool(surface) and all(
                abs(float(value) - requested) <= visible_ratio_tolerance
                for value in per_target_visible_coverage
            )
            view_audit = {}
            target_names = [item["object"] for item in surface]
            target_ids = [env.instance_to_id[name] for name in target_names]
            for view in VIEWS:
                segmentation = np.asarray(
                    edited_obs[f"{view}_segmentation_instance"]
                ).squeeze(-1)
                target_mask = np.isin(segmentation, target_ids)
                changed = np.any(edited[view] != clean[view], axis=2)
                inside = int(np.logical_and(changed, target_mask).sum())
                outside = int(np.logical_and(changed, ~target_mask).sum())
                # Rendering an object's new albedo also changes antialiased edge
                # pixels and its immediately adjacent shadow.  Treat a three-pixel
                # halo as belonging to the target while still detecting a shared
                # texture accidentally changing a distant object.
                target_halo = cv2.dilate(
                    target_mask.astype(np.uint8), np.ones((7, 7), np.uint8)
                ).astype(bool)
                outside_halo = int(np.logical_and(changed, ~target_halo).sum())
                view_audit[view] = {
                    "target_pixels": int(target_mask.sum()),
                    "changed_inside_target": inside,
                    "changed_outside_target": outside,
                    "changed_outside_target_halo": outside_halo,
                    "outside_change_fraction": outside / float(max(1, int(changed.sum()))),
                    "outside_halo_change_fraction": outside_halo
                    / float(max(1, int(changed.sum()))),
                    "target_visible": bool(target_mask.any()),
                }
                panels[view].append(label(edited[view], f"physical {label_value}%"))
                cv2.imwrite(
                    str(args.output / f"{args.suite}_task{args.task_id:02d}_{label_value}pct_{view}.png"),
                    cv2.cvtColor(edited[view][::-1], cv2.COLOR_RGB2BGR),
                )
            render_changed = all(
                not view_audit[view]["target_visible"]
                or view_audit[view]["changed_inside_target"] > 0
                for view in VIEWS
            )
            # The ratio is defined in agentview and verified by segmentation,
            # rather than by RGB difference.  This is essential for receptacles
            # whose original material is already dark.
            render_changed_required = (
                view_audit["agentview"]["target_visible"]
                and exact_ratio
                and overlay_rendered_for_all_targets
            )
            raster_halo_leakage_within_1pct = all(
                view_audit[view]["outside_halo_change_fraction"] <= 0.01
                for view in VIEWS
            )
            rows.append(
                {
                    "suite": args.suite,
                    "task_id": args.task_id,
                    "instruction": task.language,
                    "coverage": requested,
                    "spec": str(spec_path),
                    "surface_audit": surface,
                    "views": view_audit,
                    "exact_surface_ratio": exact_ratio,
                    "exact_agentview_visible_ratio": exact_ratio,
                    "exact_complete_surface_ratio": exact_ratio
                    if complete_surface_mode
                    else None,
                    "agentview_visible_ratio_tolerance": visible_ratio_tolerance,
                    "per_target_agentview_visible_coverage": per_target_visible_coverage,
                    "per_target_observed_overlay_coverage": per_target_observed_overlay_coverage,
                    "overlay_rendered_for_all_targets": overlay_rendered_for_all_targets,
                    "physical_state_unchanged": physical_unchanged,
                    "qpos_unchanged": qpos_unchanged,
                    "runtime_model_unchanged": runtime_model_unchanged,
                    "official_model_equivalent": official_model_equivalent,
                    "official_model_differences": snapshot_differences(
                        baseline_physics, model_before
                    )
                    if not official_model_equivalent
                    else [],
                    "render_changed_in_both_views": render_changed,
                    "render_changed_in_required_views": render_changed_required,
                    "target_locality_ok": visual_edit_locality_ok,
                    "visual_edit_locality_ok": visual_edit_locality_ok,
                    "changed_visual_geoms": changed_visual_geoms,
                    "changed_visual_geoms_outside_target": (
                        changed_visual_geoms_outside_target
                    ),
                    "overlay_geoms_outside_target": overlay_geoms_outside_target,
                    "raster_halo_leakage_within_1pct": (
                        raster_halo_leakage_within_1pct
                    ),
                    "same_modified_world_surface_for_both_views": bool(surface),
                    "passed": exact_ratio
                    and physical_unchanged
                    and render_changed_required
                    and visual_edit_locality_ok,
                }
            )
        finally:
            env.close()

    for view in VIEWS:
        sheet = np.hstack(panels[view])
        cv2.imwrite(
            str(args.output / f"{args.suite}_task{args.task_id:02d}_{view}_ratios.png"),
            cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR),
        )
    summary = {
        "schema_version": "wam_syncbench.A1.physical_surface_validation.v1",
        "env_seed": args.env_seed,
        "all_passed": all(row["passed"] for row in rows),
        "passed": sum(row["passed"] for row in rows),
        "total": len(rows),
        "rows": rows,
    }
    (args.output / "validation.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({key: summary[key] for key in ("all_passed", "passed", "total")}))
    return 0 if summary["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
