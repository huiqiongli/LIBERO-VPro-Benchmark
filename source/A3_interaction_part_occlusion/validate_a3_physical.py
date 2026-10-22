#!/usr/bin/env python3
"""Bounded validation for A3 scene-level 3-D interaction patches."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import mujoco
import numpy as np


HERE = Path(__file__).resolve().parent
_DEFAULT_SIM = Path("/data/huiqiongli/Sim")
if not (_DEFAULT_SIM / "vla").exists():
    _DEFAULT_SIM = Path("/home/huiqiongli/Sim")
SIM_ROOT = Path(os.environ.get("LIBERO_VPRO_SIM_ROOT", str(_DEFAULT_SIM)))
LIBERO_ROOT = SIM_ROOT / "vla/openpi_official/third_party/libero"
if str(SIM_ROOT / "reviewed_script") not in sys.path:
    sys.path.insert(0, str(SIM_ROOT / "reviewed_script"))
if str(LIBERO_ROOT) not in sys.path:
    sys.path.insert(0, str(LIBERO_ROOT))

os.environ.setdefault("LIBERO_CONFIG_PATH", str(SIM_ROOT / ".libero_smoke"))
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

from A3_interaction_part_occlusion import maybe_wrap_env  # noqa: E402
from libero.libero import benchmark  # noqa: E402
from libero.libero.envs import OffScreenRenderEnv  # noqa: E402


IMAGE_KEYS = ("agentview_image", "robot0_eye_in_hand_image")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate A3 scene geoms and localization without model inference"
    )
    parser.add_argument(
        "--spec",
        type=Path,
        default=HERE / "specs/A3_all_interaction_parts_smoke.json",
    )
    parser.add_argument(
        "--suite",
        choices=("libero_spatial", "libero_object", "libero_goal", "libero_10"),
        default="libero_object",
    )
    parser.add_argument("--task", type=int, default=0)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--preview-dir", type=Path)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--force-contact-checks",
        action="store_true",
        help="Teleport one movable target to exercise real-contact localizers",
    )
    return parser.parse_args()


def image_differences(clean_obs: dict, patch_obs: dict) -> dict[str, int]:
    return {
        key: int(
            np.count_nonzero(
                np.any(np.asarray(clean_obs[key]) != np.asarray(patch_obs[key]), axis=-1)
            )
        )
        for key in IMAGE_KEYS
    }


def dynamic_state_difference(clean_env, wrapped) -> dict[str, float]:
    return {
        "qpos_max_abs": float(
            np.max(np.abs(np.asarray(clean_env.sim.data.qpos) - wrapped.sim.data.qpos))
        ),
        "qvel_max_abs": float(
            np.max(np.abs(np.asarray(clean_env.sim.data.qvel) - wrapped.sim.data.qvel))
        ),
    }


def source_geom_rgba_checks(clean_env, wrapped) -> dict:
    """Prove A3 adds black boxes without recoloring existing scene geoms."""

    clean_model = clean_env.sim.model
    patch_model = wrapped.sim.model
    changed = []
    compared = 0
    maximum_error = 0.0
    for clean_id in range(int(clean_model.ngeom)):
        name = clean_model.geom_id2name(clean_id)
        if not name or name.startswith("wam_a3_patch_"):
            continue
        try:
            patch_id = int(patch_model.geom_name2id(name))
        except ValueError:
            changed.append({"geom": name, "status": "missing_after_patch_install"})
            continue
        error = float(
            np.max(
                np.abs(
                    np.asarray(clean_model.geom_rgba[clean_id], dtype=np.float64)
                    - np.asarray(patch_model.geom_rgba[patch_id], dtype=np.float64)
                )
            )
        )
        compared += 1
        maximum_error = max(maximum_error, error)
        if error > 1.0e-12:
            changed.append(
                {"geom": name, "status": "rgba_changed", "max_abs_error": error}
            )
    return {
        "passed": not changed,
        "compared_existing_geoms": compared,
        "maximum_rgba_abs_error": maximum_error,
        "changed_existing_geoms": changed,
    }


def patch_pose_checks(wrapped) -> dict[str, dict]:
    record = wrapped.validation_report()["last_record"]
    result = {}
    for part in wrapped.parts:
        payload = record["parts"][part]
        if not payload["active"]:
            result[part] = {"active": False, "passed": True}
            continue
        if part == "gripper" and wrapped._gripper_source_geom_ids:
            patch_ids = wrapped._geom_ids_by_part[part]
            source_ids = wrapped._gripper_source_geom_ids
            geom_type = payload["patch_pose"]["geom_type"]
            pair_checks = []
            for source_id, patch_id in zip(source_ids, patch_ids):
                local_position_error = float(
                    np.max(
                        np.abs(
                            np.asarray(wrapped.sim.model.geom_pos[source_id])
                            - np.asarray(wrapped.sim.model.geom_pos[patch_id])
                        )
                    )
                )
                source_quaternion = np.asarray(
                    wrapped.sim.model.geom_quat[source_id], dtype=np.float64
                )
                patch_quaternion = np.asarray(
                    wrapped.sim.model.geom_quat[patch_id], dtype=np.float64
                )
                quaternion_alignment = abs(
                    float(np.dot(source_quaternion, patch_quaternion))
                )
                same_body = int(wrapped.sim.model.geom_bodyid[source_id]) == int(
                    wrapped.sim.model.geom_bodyid[patch_id]
                )
                correct_type = int(wrapped.sim.model.geom_type[patch_id]) == {
                    "source_mesh": int(mujoco.mjtGeom.mjGEOM_MESH),
                    "box": int(mujoco.mjtGeom.mjGEOM_BOX),
                }[geom_type]
                if geom_type == "source_mesh":
                    size_error = 0.0
                    world_position_error = float(
                        np.linalg.norm(
                            np.asarray(wrapped.sim.data.geom_xpos[source_id])
                            - np.asarray(wrapped.sim.data.geom_xpos[patch_id])
                        )
                    )
                    world_rotation_error = float(
                        np.max(
                            np.abs(
                                np.asarray(wrapped.sim.data.geom_xmat[source_id])
                                - np.asarray(wrapped.sim.data.geom_xmat[patch_id])
                            )
                        )
                    )
                else:
                    size_error = float(
                        np.max(
                            np.abs(
                                np.asarray(wrapped.sim.model.geom_size[patch_id])
                                - wrapped.gripper_half_extents_m
                            )
                        )
                    )
                    world_position_error = None
                    world_rotation_error = None
                passed = bool(
                    local_position_error < 1.0e-8
                    and quaternion_alignment > 1.0 - 1.0e-8
                    and size_error < 1.0e-12
                    and same_body
                    and correct_type
                    and (
                        geom_type != "source_mesh"
                        or (
                            world_position_error is not None
                            and world_position_error < 1.0e-12
                            and world_rotation_error is not None
                            and world_rotation_error < 1.0e-12
                        )
                    )
                )
                pair_checks.append(
                    {
                        "source_geom": wrapped.sim.model.geom_id2name(source_id),
                        "patch_geom": wrapped.sim.model.geom_id2name(patch_id),
                        "local_position_max_abs_error_m": local_position_error,
                        "local_quaternion_alignment_abs_dot": quaternion_alignment,
                        "half_extents_max_abs_error_m": size_error,
                        "world_position_error_m": world_position_error,
                        "world_rotation_matrix_max_abs_error": world_rotation_error,
                        "same_articulated_body": same_body,
                        "correct_geom_type": correct_type,
                        "passed": passed,
                    }
                )
            result[part] = {
                "active": True,
                "mode": (
                    "two_finger_source_mesh_surface"
                    if geom_type == "source_mesh"
                    else "two_finger_body_attached_rectangular_boxes"
                ),
                "finger_pairs": pair_checks,
                "passed": all(item["passed"] for item in pair_checks),
            }
            continue

        if part in wrapped._fixed_parts:
            template = wrapped._fixed_template_by_part[part]
            definitions = wrapped._fixed_patch_definitions_by_part[part]
            patch_ids = wrapped._geom_ids_by_part[part]
            expected_body = int(
                wrapped.sim.model.body_name2id(str(template["body_name"]))
            )
            patch_checks = []
            for definition, patch_id in zip(definitions, patch_ids):
                position_error = float(
                    np.max(
                        np.abs(
                            np.asarray(wrapped.sim.model.geom_pos[patch_id])
                            - definition["position"]
                        )
                    )
                )
                size_error = float(
                    np.max(
                        np.abs(
                            np.asarray(wrapped.sim.model.geom_size[patch_id])
                            - definition["size"]
                        )
                    )
                )
                actual_quaternion = np.asarray(
                    wrapped.sim.model.geom_quat[patch_id], dtype=np.float64
                )
                expected_quaternion = np.asarray(
                    definition["quaternion"], dtype=np.float64
                )
                quaternion_alignment = abs(
                    float(np.dot(actual_quaternion, expected_quaternion))
                )
                same_body = (
                    int(wrapped.sim.model.geom_bodyid[patch_id]) == expected_body
                )
                passed = (
                    position_error < 1.0e-8
                    and size_error < 1.0e-8
                    and quaternion_alignment > 1.0 - 1.0e-8
                    and same_body
                )
                patch_checks.append(
                    {
                        "geom": wrapped.sim.model.geom_id2name(patch_id),
                        "position_max_abs_error_m": position_error,
                        "size_max_abs_error_m": size_error,
                        "quaternion_alignment_abs_dot": quaternion_alignment,
                        "same_source_body": same_body,
                        "passed": passed,
                    }
                )
            result[part] = {
                "active": True,
                "mode": "frozen_task_local_multi_surface_region",
                "source_body": str(template["body_name"]),
                "patches": patch_checks,
                "passed": all(item["passed"] for item in patch_checks),
            }
            continue

        gid = wrapped._geom_ids[part]
        center = np.asarray(payload["patch_pose"]["center"], dtype=np.float64)
        normal = np.asarray(payload["world_normal"], dtype=np.float64)
        geom_center = np.asarray(wrapped.sim.data.geom_xpos[gid], dtype=np.float64)
        geom_z = np.asarray(wrapped.sim.data.geom_xmat[gid], dtype=np.float64).reshape(3, 3)[:, 2]
        center_error = float(np.linalg.norm(geom_center - center))
        normal_alignment = abs(float(np.dot(geom_z, normal)))
        result[part] = {
            "active": True,
            "center_error_m": center_error,
            "normal_alignment_abs_dot": normal_alignment,
            "passed": center_error < 1.0e-9 and normal_alignment > 1.0 - 1.0e-6,
        }
    return result


def semantic_location_checks(wrapped) -> dict[str, dict]:
    """Verify key points belong to the intended entity / BDDL goal region."""

    result: dict[str, dict] = {}
    if wrapped.localization_mode == "frozen_task_local_regions":
        pair = wrapped.target_receptacle_pairs[0]
        task_config = wrapped._fixed_task_config
        target_passed = bool(
            task_config is not None
            and pair[0].raw_name == task_config["target"]
            and (
                "object_contact" not in wrapped.parts
                or int(wrapped._body_ids["object_contact"])
                == int(
                    wrapped.sim.model.body_name2id(
                        wrapped._fixed_template_by_part["object_contact"]["body_name"]
                    )
                )
            )
        )
        receptacle_passed = bool(
            task_config is not None
            and pair[1].raw_name == task_config["receptacle"]
            and (
                "receptacle_contact" not in wrapped.parts
                or int(wrapped._body_ids["receptacle_contact"])
                == int(
                    wrapped.sim.model.body_name2id(
                        wrapped._fixed_template_by_part["receptacle_contact"]["body_name"]
                    )
                )
            )
        )
        return {
            "object_contact_manifest_entity": {
                "passed": target_passed,
                "resolved": pair[0].raw_name,
                "expected": None if task_config is None else task_config["target"],
            },
            "receptacle_contact_manifest_entity": {
                "passed": receptacle_passed,
                "resolved": pair[1].raw_name,
                "expected": (
                    None if task_config is None else task_config["receptacle"]
                ),
            },
        }

    locations = wrapped._locate_parts()
    target = wrapped._active_target
    if target is None:
        return {"target_resolved": {"passed": False}}

    nearest = wrapped._nearest_between(wrapped.gripper_geom_ids, target.geom_ids)
    if nearest is None:
        result["object_contact_on_target_surface"] = {"passed": False}
    else:
        point = np.asarray(locations["object_contact"].world_point)

        def bound_ratio(world_point: np.ndarray, geom_id: int) -> float:
            radius = max(float(wrapped.sim.model.geom_rbound[geom_id]), 1.0e-9)
            return float(
                np.linalg.norm(world_point - wrapped.sim.data.geom_xpos[geom_id])
            ) / radius

        target_ratio = bound_ratio(point, nearest.geom_b)
        gripper_ratio = bound_ratio(point, nearest.geom_a)
        point_error = float(np.linalg.norm(point - nearest.point_b))
        result["object_contact_on_target_surface"] = {
            "passed": (
                target_ratio <= 1.0 + 1.0e-6
                and point_error < 1.0e-9
                and gripper_ratio > 1.0 + 1.0e-3
            ),
            "target": target.raw_name,
            "target_geom": wrapped.sim.model.geom_id2name(nearest.geom_b),
            "gripper_geom": wrapped.sim.model.geom_id2name(nearest.geom_a),
            "target_bound_ratio": target_ratio,
            "gripper_bound_ratio": gripper_ratio,
            "point_error_m": point_error,
            "endpoint_order_repaired": nearest.endpoint_order_repaired,
        }

    pair = wrapped._pair_for_target(target)
    if pair is None:
        result["receptacle_contact_in_bddl_goal_region"] = {
            "passed": locations["receptacle_contact"].active is False,
            "not_applicable": True,
        }
    else:
        predicate = wrapped._placement_predicate(pair)
        receptacle = pair[1]
        support = wrapped._goal_region_support(receptacle, target)
        location = locations["receptacle_contact"]
        if support is None:
            result["receptacle_contact_in_bddl_goal_region"] = {
                "passed": bool(location.active),
                "predicate": predicate,
                "region_site": None,
                "source": location.source,
            }
        else:
            support_point, support_normal, support_method = support
            point = np.asarray(location.world_point)
            point_error = float(np.linalg.norm(point - support_point))
            normal_alignment = float(
                np.dot(
                    np.asarray(location.world_normal, dtype=np.float64),
                    support_normal,
                )
            )
            result["receptacle_contact_in_bddl_goal_region"] = {
                "passed": (
                    location.source
                    == f"bddl_{predicate}_goal_region_support_surface"
                    and point_error < 1.0e-9
                    and normal_alignment > 1.0 - 1.0e-6
                ),
                "predicate": predicate,
                "region_site": wrapped.sim.model.site_id2name(
                    receptacle.anchor_site_id
                ),
                "source": location.source,
                "point_error_m": point_error,
                "normal_alignment": normal_alignment,
                "support_method": support_method,
            }
    return result


def patch_segmentation_counts(wrapped) -> dict[str, dict[str, int]]:
    """Count each actual patch geom in validation-only camera segmentation."""

    result: dict[str, dict[str, int]] = {}
    for camera in ("agentview", "robot0_eye_in_hand"):
        segmentation = wrapped.sim.render(
            camera_name=camera,
            width=128,
            height=128,
            segmentation=True,
        )
        geom_ids = np.asarray(segmentation)[..., 1]
        result[camera] = {
            part: int(
                np.count_nonzero(
                    np.isin(geom_ids, wrapped._geom_ids_by_part[part])
                )
            )
            for part in wrapped.parts
        }
    return result


def _entity_render_geom_ids(wrapped, entity) -> set[int]:
    result = set(entity.geom_ids)
    if entity.resolved_name in wrapped.env.objects_dict:
        model_object = wrapped.env.objects_dict[entity.resolved_name]
    elif entity.resolved_name in wrapped.env.fixtures_dict:
        model_object = wrapped.env.fixtures_dict[entity.resolved_name]
    else:
        model_object = None
    for name in getattr(model_object, "visual_geoms", ()):
        try:
            result.add(int(wrapped.sim.model.geom_name2id(name)))
        except ValueError:
            pass
    return result


def gripper_occlusion_coverage_checks(wrapped) -> dict[str, dict]:
    """Measure how much of the clean gripper silhouette the 3-D patch hides."""

    if "gripper" not in wrapped.parts:
        return {
            camera: {"passed": True, "status": "not_applicable"}
            for camera in ("agentview", "robot0_eye_in_hand")
        }

    # The semantic target is the two grasping fingers.  A single rectangular
    # patch necessarily covers the gap between them; the palm and wrist are
    # not counted as required occlusion area.
    if wrapped._gripper_source_geom_ids:
        gripper_ids = set(wrapped._gripper_source_geom_ids)
    else:
        gripper_ids = set()
        for name in getattr(wrapped.env.robots[0].gripper, "visual_geoms", ()):
            if "finger" not in str(name).lower():
                continue
            try:
                gripper_ids.add(int(wrapped.sim.model.geom_name2id(name)))
            except ValueError:
                pass
        if len(gripper_ids) != 2:
            raise RuntimeError(
                "single-rectangle gripper validation requires exactly two "
                f"finger visual geoms, got {sorted(gripper_ids)}"
            )

    grip_position = np.asarray(
        wrapped.sim.data.site_xpos[wrapped.grip_site_id], dtype=np.float64
    )
    grip_rotation = np.asarray(
        wrapped.sim.data.site_xmat[wrapped.grip_site_id], dtype=np.float64
    ).reshape(3, 3)
    local_source_vertices = []
    for geom_id in sorted(gripper_ids):
        if int(wrapped.sim.model.geom_type[geom_id]) != int(
            mujoco.mjtGeom.mjGEOM_MESH
        ):
            continue
        mesh_id = int(wrapped.sim.model.geom_dataid[geom_id])
        vertex_start = int(wrapped.sim.model.mesh_vertadr[mesh_id])
        vertex_count = int(wrapped.sim.model.mesh_vertnum[mesh_id])
        vertices = np.asarray(
            wrapped.sim.model.mesh_vert[
                vertex_start : vertex_start + vertex_count
            ],
            dtype=np.float64,
        )
        geom_rotation = np.asarray(
            wrapped.sim.data.geom_xmat[geom_id], dtype=np.float64
        ).reshape(3, 3)
        geom_position = np.asarray(
            wrapped.sim.data.geom_xpos[geom_id], dtype=np.float64
        )
        world_vertices = vertices @ geom_rotation.T + geom_position
        local_source_vertices.append(
            (world_vertices - grip_position) @ grip_rotation
        )
    if local_source_vertices:
        stacked_source_vertices = np.concatenate(local_source_vertices, axis=0)
        source_bounds_in_grip_frame = {
            "minimum_m": stacked_source_vertices.min(axis=0).tolist(),
            "maximum_m": stacked_source_vertices.max(axis=0).tolist(),
        }
    else:
        source_bounds_in_grip_frame = None

    patch_ids = wrapped._geom_ids_by_part["gripper"]
    patch_masks: dict[str, np.ndarray] = {}
    for camera in ("agentview", "robot0_eye_in_hand"):
        segmentation = wrapped.sim.render(
            camera_name=camera, width=128, height=128, segmentation=True
        )
        patch_masks[camera] = np.isin(
            np.asarray(segmentation)[..., 1], patch_ids
        )

    clean_gripper_masks: dict[str, np.ndarray] = {}
    wrapped._hide_patches(forward=True)
    try:
        for camera in ("agentview", "robot0_eye_in_hand"):
            segmentation = wrapped.sim.render(
                camera_name=camera, width=128, height=128, segmentation=True
            )
            clean_gripper_masks[camera] = np.isin(
                np.asarray(segmentation)[..., 1], list(gripper_ids)
            )
    finally:
        wrapped._update_scene_patches()

    # MuJoCo's 128x128 segmentation occasionally emits one- or two-pixel
    # disconnected islands carrying a finger geom id tens of pixels away from
    # the projected gripper.  Keep the two real jaw components and report the
    # discarded raster noise explicitly instead of enlarging the physical box
    # toward unrelated scene pixels.
    minimum_component_pixels = int(
        wrapped.spec.get("gripper_min_semantic_component_pixels", 3)
    )
    if minimum_component_pixels < 1:
        raise ValueError("gripper_min_semantic_component_pixels must be positive")

    def remove_isolated_components(
        mask: np.ndarray, component_minimum_pixels: int
    ) -> tuple[np.ndarray, int]:
        source = np.asarray(mask, dtype=bool)
        kept = np.zeros_like(source)
        visited = np.zeros_like(source)
        for start_y, start_x in zip(*np.where(source)):
            if visited[start_y, start_x]:
                continue
            component = [(int(start_y), int(start_x))]
            visited[start_y, start_x] = True
            cursor = 0
            while cursor < len(component):
                y, x = component[cursor]
                cursor += 1
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        if dy == 0 and dx == 0:
                            continue
                        ny, nx = y + dy, x + dx
                        if (
                            0 <= ny < source.shape[0]
                            and 0 <= nx < source.shape[1]
                            and source[ny, nx]
                            and not visited[ny, nx]
                        ):
                            visited[ny, nx] = True
                            component.append((ny, nx))
            if len(component) >= component_minimum_pixels:
                ys, xs = zip(*component)
                kept[np.asarray(ys), np.asarray(xs)] = True
        return kept, int(np.count_nonzero(source) - np.count_nonzero(kept))

    discarded_by_camera = {}
    raw_clean_pixels_by_camera = {}
    for camera, raw_mask in tuple(clean_gripper_masks.items()):
        raw_clean_pixels_by_camera[camera] = int(np.count_nonzero(raw_mask))
        filtered, discarded = remove_isolated_components(
            raw_mask, minimum_component_pixels
        )
        clean_gripper_masks[camera] = filtered
        discarded_by_camera[camera] = discarded

    threshold = float(wrapped.spec.get("gripper_min_occluded_fraction", 0.0))
    precision_threshold = float(
        wrapped.spec.get("gripper_min_occlusion_precision", 0.0)
    )
    raster_edge_tolerance = int(
        wrapped.spec.get("gripper_max_raster_edge_background_pixels", 0)
    )
    uncovered_edge_tolerance = int(
        wrapped.spec.get("gripper_max_uncovered_semantic_edge_pixels", 0)
    )
    minimum_uncovered_component_pixels = int(
        wrapped.spec.get("gripper_min_uncovered_component_pixels", 1)
    )
    if minimum_uncovered_component_pixels < 1:
        raise ValueError(
            "gripper_min_uncovered_component_pixels must be positive"
        )
    edge_tolerance_fraction_floor = float(
        wrapped.spec.get(
            "gripper_min_occluded_fraction_with_edge_tolerance", threshold
        )
    )
    min_frame_fraction = wrapped.spec.get(
        "gripper_min_patch_frame_fraction",
        {"agentview": 0.0, "robot0_eye_in_hand": 0.0},
    )
    max_frame_fraction = wrapped.spec.get(
        "gripper_max_patch_frame_fraction",
        {"agentview": 1.0, "robot0_eye_in_hand": 1.0},
    )
    result: dict[str, dict] = {}
    for camera in ("agentview", "robot0_eye_in_hand"):
        patch_mask = patch_masks[camera]
        clean_mask = clean_gripper_masks[camera]
        clean_pixels = int(np.count_nonzero(clean_mask))
        patch_pixels = int(np.count_nonzero(patch_mask))
        raw_occluded_pixels = int(np.count_nonzero(patch_mask & clean_mask))
        raw_uncovered_mask = clean_mask & ~patch_mask
        raw_uncovered_pixels = int(np.count_nonzero(raw_uncovered_mask))
        effective_uncovered_mask, discarded_uncovered_pixels = (
            remove_isolated_components(
                raw_uncovered_mask, minimum_uncovered_component_pixels
            )
        )
        uncovered_pixels = int(np.count_nonzero(effective_uncovered_mask))
        occluded_pixels = clean_pixels - uncovered_pixels
        fraction = (
            float(occluded_pixels / clean_pixels) if clean_pixels else 1.0
        )
        precision = (
            float(occluded_pixels / patch_pixels) if patch_pixels else 0.0
        )
        background_pixels = patch_pixels - raw_occluded_pixels
        uncovered_yx = np.argwhere(effective_uncovered_mask)

        def mask_bbox_xyxy(mask: np.ndarray) -> list[int] | None:
            ys, xs = np.where(mask)
            if not len(xs):
                return None
            return [
                int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
            ]

        patch_frame_fraction = float(patch_pixels / patch_mask.size)
        minimum_frame_fraction = float(min_frame_fraction.get(camera, 0.0))
        maximum_frame_fraction = float(max_frame_fraction.get(camera, 1.0))
        area_passed = (
            minimum_frame_fraction
            <= patch_frame_fraction
            <= maximum_frame_fraction
        )
        precision_passed = precision >= precision_threshold or (
            bool(wrapped._gripper_source_geom_ids)
            and background_pixels <= raster_edge_tolerance
        )
        coverage_passed = fraction >= threshold or (
            uncovered_pixels <= uncovered_edge_tolerance
            and fraction >= edge_tolerance_fraction_floor
        )
        passed = coverage_passed and precision_passed and area_passed
        result[camera] = {
            "passed": passed,
            "status": (
                (
                    "exact_mesh_with_raster_edge_tolerance"
                    if passed and precision < precision_threshold
                    else (
                        "rectangular_box_with_semantic_edge_tolerance"
                        if passed and fraction < threshold
                        else "tight_gripper_silhouette_occlusion"
                    )
                )
                if passed
                else (
                    "insufficient_gripper_occlusion"
                    if not coverage_passed
                    else (
                        "patch_area_out_of_bounds"
                        if not area_passed
                        else "excessive_background_occlusion"
                    )
                )
            ),
            "clean_gripper_pixels": clean_pixels,
            "raw_clean_gripper_pixels": raw_clean_pixels_by_camera[camera],
            "discarded_isolated_segmentation_pixels": discarded_by_camera[camera],
            "minimum_semantic_component_pixels": minimum_component_pixels,
            "occluded_gripper_pixels": occluded_pixels,
            "raw_occluded_gripper_pixels": raw_occluded_pixels,
            "uncovered_gripper_pixels": uncovered_pixels,
            "raw_uncovered_gripper_pixels": raw_uncovered_pixels,
            "discarded_isolated_uncovered_pixels": discarded_uncovered_pixels,
            "minimum_uncovered_component_pixels": (
                minimum_uncovered_component_pixels
            ),
            "uncovered_semantic_pixels_xy": [
                [int(x), int(y)] for y, x in uncovered_yx
            ],
            "clean_gripper_bbox_xyxy": mask_bbox_xyxy(clean_mask),
            "patch_bbox_xyxy": mask_bbox_xyxy(patch_mask),
            "source_mesh_bounds_in_grip_frame": source_bounds_in_grip_frame,
            "occluded_fraction": fraction,
            "required_fraction": threshold,
            "maximum_uncovered_semantic_edge_pixels": uncovered_edge_tolerance,
            "edge_tolerance_fraction_floor": edge_tolerance_fraction_floor,
            "occlusion_precision": precision,
            "required_precision": precision_threshold,
            "background_pixels": background_pixels,
            "maximum_raster_edge_background_pixels": raster_edge_tolerance,
            "patch_pixels": patch_pixels,
            "patch_frame_fraction": patch_frame_fraction,
            "minimum_patch_frame_fraction": minimum_frame_fraction,
            "maximum_patch_frame_fraction": maximum_frame_fraction,
        }
    return result


def contact_patch_shape_checks(wrapped) -> dict[str, dict]:
    """Require thin, body-local-horizontal surface pads rather than 3-D balls."""

    record = wrapped.validation_report()["last_record"]
    max_ratio = float(
        wrapped.spec.get("contact_patch_max_normal_to_tangent_ratio", 0.25)
    )
    min_horizontal_ratio = float(
        wrapped.spec.get(
            "contact_patch_min_horizontal_to_vertical_ratio", 0.0
        )
    )
    result: dict[str, dict] = {}
    for part in ("object_contact", "receptacle_contact"):
        if part not in wrapped.parts:
            result[part] = {"passed": True, "status": "not_applicable"}
            continue
        payload = record["parts"][part]
        if not payload["active"]:
            result[part] = {"passed": True, "status": "not_applicable"}
            continue
        if part in wrapped._fixed_parts:
            patch_checks = []
            for patch, geom_id in zip(
                payload["patch_geoms"], wrapped._geom_ids_by_part[part]
            ):
                half_extents = np.asarray(
                    patch["half_extents_m"], dtype=np.float64
                )
                tangent_radius = float(max(half_extents[0], half_extents[1]))
                normal_half_extent = float(half_extents[2])
                aspect_ratio = normal_half_extent / max(
                    tangent_radius, 1.0e-12
                )
                horizontal_ratio = float(
                    half_extents[0] / max(half_extents[1], 1.0e-12)
                )
                rectangular = int(wrapped.sim.model.geom_type[geom_id]) == int(
                    mujoco.mjtGeom.mjGEOM_BOX
                )
                patch_checks.append(
                    {
                        "geom": patch["geom"],
                        "geom_type": "box" if rectangular else "non_box",
                        "half_extents_m": half_extents.tolist(),
                        "normal_to_tangent_ratio": aspect_ratio,
                        "horizontal_to_vertical_ratio": horizontal_ratio,
                        "passed": (
                            rectangular
                            and aspect_ratio <= max_ratio
                            and horizontal_ratio + 1.0e-12
                            >= min_horizontal_ratio
                        ),
                    }
                )
            result[part] = {
                "passed": all(item["passed"] for item in patch_checks),
                "status": "frozen_thin_rectangular_surface_region",
                "maximum_ratio": max_ratio,
                "minimum_horizontal_to_vertical_ratio": min_horizontal_ratio,
                "patches": patch_checks,
            }
            continue
        pose = payload["patch_pose"]
        half_extents = np.asarray(pose["half_extents_m"], dtype=np.float64)
        attachment = np.asarray(pose["attachment_point"], dtype=np.float64)
        centre = np.asarray(pose["center"], dtype=np.float64)
        normal = np.asarray(payload["world_normal"], dtype=np.float64)
        tangent_radius = float(max(half_extents[0], half_extents[1]))
        normal_half_extent = float(half_extents[2])
        aspect_ratio = normal_half_extent / max(tangent_radius, 1.0e-12)
        inner_pole = centre - normal * normal_half_extent
        attachment_error = float(np.linalg.norm(inner_pole - attachment))
        supported_shape = pose["geom_type"] in ("ellipsoid", "box")
        passed = (
            supported_shape
            and aspect_ratio <= max_ratio
            and attachment_error < 1.0e-9
        )
        result[part] = {
            "passed": passed,
            "status": "thin_surface_pad" if passed else "invalid_contact_patch_shape",
            "geom_type": pose["geom_type"],
            "half_extents_m": half_extents.tolist(),
            "normal_to_tangent_ratio": aspect_ratio,
            "maximum_ratio": max_ratio,
            "inner_pole_attachment_error_m": attachment_error,
        }
    return result


def patch_visibility_checks(
    wrapped, segmentation_counts: dict[str, dict[str, int]]
) -> dict[str, dict[str, dict]]:
    """Accept visible patches or intended self-occlusion by their own entity."""

    record = wrapped.validation_report()["last_record"]
    target = wrapped._active_target
    pair = None if target is None else wrapped._pair_for_target(target)
    gripper_geoms = set(wrapped.gripper_geom_ids)
    for name in getattr(wrapped.env.robots[0].gripper, "visual_geoms", ()):
        try:
            gripper_geoms.add(int(wrapped.sim.model.geom_name2id(name)))
        except ValueError:
            pass
    intended_by_part = {
        "gripper": gripper_geoms,
        "object_contact": (
            set() if target is None else _entity_render_geom_ids(wrapped, target)
        ),
        "receptacle_contact": (
            set() if pair is None else _entity_render_geom_ids(wrapped, pair[1])
        ),
    }
    all_patch_geom_ids = {
        gid for ids in wrapped._geom_ids_by_part.values() for gid in ids
    }

    result: dict[str, dict[str, dict]] = {}
    for camera in ("agentview", "robot0_eye_in_hand"):
        camera_id = wrapped.sim.model.camera_name2id(camera)
        camera_position = np.asarray(wrapped.sim.data.cam_xpos[camera_id])
        camera_rotation = np.asarray(
            wrapped.sim.data.cam_xmat[camera_id]
        ).reshape(3, 3)
        fovy = float(wrapped.sim.model.cam_fovy[camera_id])
        focal = 0.5 * 128.0 / np.tan(fovy * np.pi / 360.0)
        result[camera] = {}
        for part in wrapped.parts:
            payload = record["parts"][part]
            if not payload["active"]:
                result[camera][part] = {
                    "passed": True,
                    "status": "not_applicable",
                    "segmentation_pixels": 0,
                }
                continue
            pixels = segmentation_counts[camera][part]
            if pixels > 0:
                result[camera][part] = {
                    "passed": True,
                    "status": "patch_visible",
                    "segmentation_pixels": pixels,
                }
                continue

            centre = np.asarray(payload["patch_pose"]["center"], dtype=np.float64)
            relative = camera_rotation.T @ (centre - camera_position)
            depth = -float(relative[2])
            pixel_x = focal * float(relative[0]) / max(depth, 1.0e-12) + 64.0
            pixel_y = -focal * float(relative[1]) / max(depth, 1.0e-12) + 64.0
            in_view = depth > 0.0 and 0.0 <= pixel_x < 128.0 and 0.0 <= pixel_y < 128.0
            if not in_view:
                result[camera][part] = {
                    "passed": True,
                    "status": "key_region_outside_camera_frustum",
                    "segmentation_pixels": 0,
                    "projected_xy": [pixel_x, pixel_y],
                    "depth_m": depth,
                }
                continue

            direction = centre - camera_position
            centre_distance = float(np.linalg.norm(direction))
            direction /= max(centre_distance, 1.0e-12)
            hit_geom = np.empty(1, dtype=np.int32)
            # Ray only against physical source geometry (group 0), not the A3
            # visual boxes (group 1).  Fixed patches share their source body,
            # so excluding that body would also hide the very object whose
            # self-occlusion this check is meant to confirm.
            ray_geom_group = np.zeros(6, dtype=np.uint8)
            ray_geom_group[0] = 1
            hit_distance = float(
                mujoco.mj_ray(
                    wrapped.sim.model._model,
                    wrapped.sim.data._data,
                    np.ascontiguousarray(camera_position),
                    np.ascontiguousarray(direction),
                    ray_geom_group,
                    1,
                    -1,
                    hit_geom,
                )
            )
            hit_id = int(hit_geom[0])
            intended = hit_id in intended_by_part[part]
            another_patch = hit_id in all_patch_geom_ids
            occluded_before_centre = 0.0 <= hit_distance < centre_distance
            result[camera][part] = {
                "passed": bool(
                    (intended or another_patch) and occluded_before_centre
                ),
                "status": (
                    "naturally_occluded_by_intended_entity"
                    if intended and occluded_before_centre
                    else (
                        "occluded_by_another_a3_patch"
                        if another_patch and occluded_before_centre
                        else "unexpectedly_missing"
                    )
                ),
                "segmentation_pixels": 0,
                "projected_xy": [pixel_x, pixel_y],
                "depth_m": depth,
                "first_hit_geom": (
                    None
                    if hit_id < 0
                    else wrapped.sim.model.geom_id2name(hit_id)
                ),
                "first_hit_distance_m": hit_distance,
                "patch_centre_distance_m": centre_distance,
            }
    return result


def direct_physics_counterfactual(wrapped) -> dict[str, float | int | bool]:
    """Compare one identical physics step with patches in place vs out of scene."""

    model = wrapped.sim.model._model
    data = wrapped.sim.data._data
    state_spec = mujoco.mjtState.mjSTATE_INTEGRATION
    state = np.empty(mujoco.mj_stateSize(model, state_spec), dtype=np.float64)
    mujoco.mj_getState(model, data, state, state_spec)

    patch_geom_ids = {
        gid for ids in wrapped._geom_ids_by_part.values() for gid in ids
    }
    patch_contacts_before = sum(
        int(data.contact[index].geom1) in patch_geom_ids
        or int(data.contact[index].geom2) in patch_geom_ids
        for index in range(int(data.ncon))
    )
    try:
        mujoco.mj_step(model, data)
        visible_qpos = np.asarray(data.qpos).copy()
        visible_qvel = np.asarray(data.qvel).copy()

        mujoco.mj_setState(model, data, state, state_spec)
        for mocap_id in wrapped._mocap_ids.values():
            data.mocap_pos[mocap_id] = (0.0, 0.0, -10.0)
        mujoco.mj_forward(model, data)
        mujoco.mj_step(model, data)
        hidden_qpos = np.asarray(data.qpos).copy()
        hidden_qvel = np.asarray(data.qvel).copy()
    finally:
        mujoco.mj_setState(model, data, state, state_spec)
        mujoco.mj_forward(model, data)

    qpos_error = float(np.max(np.abs(visible_qpos - hidden_qpos)))
    qvel_error = float(np.max(np.abs(visible_qvel - hidden_qvel)))
    return {
        "patch_contacts_before_step": int(patch_contacts_before),
        "qpos_max_abs": qpos_error,
        "qvel_max_abs": qvel_error,
        "passed": (
            patch_contacts_before == 0
            and qpos_error < 1.0e-14
            and qvel_error < 1.0e-14
        ),
    }


def _fixed_contact_diagnostics(wrapped, part: str) -> dict | None:
    if part not in wrapped._fixed_parts:
        return None
    points = wrapped._fixed_region_contact_points(part)
    body_name = str(wrapped._fixed_template_by_part[part]["body_name"])
    body_id = int(wrapped.sim.model.body_name2id(body_name))
    body_position = np.asarray(wrapped.sim.data.body_xpos[body_id], dtype=np.float64)
    body_rotation = np.asarray(
        wrapped.sim.data.body_xmat[body_id], dtype=np.float64
    ).reshape(3, 3)
    local_points = [
        (body_rotation.T @ (np.asarray(point) - body_position)).tolist()
        for point in points
    ]
    covered = sum(
        wrapped._fixed_region_covers_point(part, point) for point in points
    )
    return {
        "source_body": body_name,
        "contact_points": len(points),
        "covered_contact_points": int(covered),
        "coverage_fraction": (
            None if not points else float(covered / len(points))
        ),
        "source_local_contact_points_m": local_points,
    }


def force_contact_checks(wrapped) -> dict:
    result = {
        "supported": False,
        "gripper_target": {"passed": False},
        "target_receptacle": {"passed": False},
        "target_occluded_goal_support": {"passed": False},
    }
    if not wrapped.targets or not wrapped.target_receptacle_pairs:
        result["reason"] = "task has no target/receptacle placement pair"
        return result
    target = wrapped.targets[0]
    if target.resolved_name not in wrapped.env.objects_dict:
        result["reason"] = "first target is not a movable LIBERO object"
        return result
    joints = tuple(getattr(wrapped.env.objects_dict[target.resolved_name], "joints", ()))
    if not joints:
        result["reason"] = "first target has no movable joint"
        return result

    joint = joints[0]
    original_qpos = wrapped.sim.data.get_joint_qpos(joint).copy()
    if original_qpos.shape != (7,):
        result["reason"] = "first target does not have a free joint"
        return result

    result["supported"] = True
    wrapped._hide_patches(forward=True)
    try:
        qpos = original_qpos.copy()
        qpos[:3] = wrapped.sim.data.site_xpos[wrapped.grip_site_id]
        wrapped.sim.data.set_joint_qpos(joint, qpos)
        wrapped.sim.forward()
        location = wrapped._locate_parts()["object_contact"]
        result["gripper_target"] = {
            "passed": (
                location.source == "active_mujoco_gripper_target_contact_centroid"
                and location.contact_count > 0
            ),
            "source": location.source,
            "contact_count": location.contact_count,
            "world_point": location.as_dict()["world_point"],
            "fixed_region": _fixed_contact_diagnostics(
                wrapped, "object_contact"
            ),
        }

        _, receptacle = wrapped.target_receptacle_pairs[0]
        anchor = wrapped._entity_anchor(receptacle)
        attempts = []
        if anchor is not None:
            for dz in (0.0, -0.01, -0.02, -0.03, -0.04, -0.05, -0.06):
                qpos = original_qpos.copy()
                qpos[:3] = anchor
                qpos[2] += dz
                wrapped.sim.data.set_joint_qpos(joint, qpos)
                wrapped.sim.forward()
                location = wrapped._locate_parts()["receptacle_contact"]
                attempts.append(
                    {
                        "vertical_offset_m": dz,
                        "source": location.source,
                        "contact_count": location.contact_count,
                    }
                )
                if (
                    location.source == "active_mujoco_target_receptacle_contact_centroid"
                    and location.contact_count > 0
                ):
                    result["target_receptacle"] = {
                        "passed": True,
                        "source": location.source,
                        "contact_count": location.contact_count,
                        "world_point": location.as_dict()["world_point"],
                        "vertical_offset_m": dz,
                        "attempts": attempts,
                        "fixed_region": _fixed_contact_diagnostics(
                            wrapped, "receptacle_contact"
                        ),
                    }
                    break
            else:
                result["target_receptacle"] = {
                    "passed": False,
                    "attempts": attempts,
                }
        else:
            result["target_receptacle"] = {
                "passed": False,
                "reason": "receptacle has no geometry centre or goal-region anchor",
            }

        occlusion_attempts = []
        if anchor is not None and receptacle.anchor_site_id is not None:
            for dz in (0.03, 0.05, 0.07, 0.09):
                qpos = original_qpos.copy()
                qpos[:3] = anchor
                qpos[2] += dz
                wrapped.sim.data.set_joint_qpos(joint, qpos)
                wrapped.sim.forward()
                unexcluded = wrapped._goal_region_support(receptacle)
                excluded = wrapped._goal_region_support(receptacle, target)
                attempt = {
                    "vertical_offset_m": dz,
                    "without_target_exclusion": (
                        None if unexcluded is None else unexcluded[2]
                    ),
                    "with_target_exclusion": (
                        None if excluded is None else excluded[2]
                    ),
                }
                occlusion_attempts.append(attempt)
                if (
                    unexcluded is not None
                    and excluded is not None
                    and unexcluded[2] == "goal_region_lower_face_fallback"
                    and excluded[2].startswith("top_down_ray_to_")
                ):
                    result["target_occluded_goal_support"] = {
                        "passed": True,
                        "support_point": excluded[0].tolist(),
                        "support_normal": excluded[1].tolist(),
                        "attempts": occlusion_attempts,
                    }
                    break
            else:
                result["target_occluded_goal_support"] = {
                    "passed": True,
                    "status": "not_applicable_target_does_not_block_support_ray",
                    "attempts": occlusion_attempts,
                }
        else:
            result["target_occluded_goal_support"] = {
                "passed": True,
                "status": "not_applicable_no_goal_region_site",
                "reason": "goal region or target anchor unavailable",
            }
    finally:
        wrapped.sim.data.set_joint_qpos(joint, original_qpos)
        wrapped.sim.forward()
    return result


def main() -> int:
    args = parse_args()
    if args.steps < 0 or args.steps > 10:
        raise ValueError("validation steps must be in [0, 10]")

    suite = benchmark.get_benchmark_dict()[args.suite]()
    if not 0 <= args.task < suite.get_num_tasks():
        raise ValueError(f"task index out of range: {args.task}")
    env_args = {
        "bddl_file_name": suite.get_task_bddl_file_path(args.task),
        "camera_heights": 128,
        "camera_widths": 128,
    }
    clean_env = OffScreenRenderEnv(**env_args)
    patch_base_env = OffScreenRenderEnv(**env_args)
    wrapped = maybe_wrap_env(
        patch_base_env,
        suite=args.suite,
        task_idx=args.task,
        spec_path=args.spec,
    )
    try:
        clean_env.seed(0)
        wrapped.seed(0)
        clean_env.reset()
        wrapped.reset()
        init_state = suite.get_task_init_states(args.task)[0]
        clean_obs = clean_env.set_init_state(init_state)
        patch_obs = wrapped.set_init_state(init_state)

        frame_differences = [image_differences(clean_obs, patch_obs)]
        state_differences = [dynamic_state_difference(clean_env, wrapped)]
        for _ in range(args.steps):
            action = np.zeros(7, dtype=np.float64)
            clean_obs, _, _, _ = clean_env.step(action)
            patch_obs, _, _, _ = wrapped.step(action)
            frame_differences.append(image_differences(clean_obs, patch_obs))
            state_differences.append(dynamic_state_difference(clean_env, wrapped))

        if args.preview_dir is not None:
            import imageio.v2 as imageio

            preview_dir = args.preview_dir.resolve()
            preview_dir.mkdir(parents=True, exist_ok=True)
            for key in IMAGE_KEYS:
                camera = key.removesuffix("_image")
                imageio.imwrite(preview_dir / f"{camera}_clean_control.png", clean_obs[key][::-1])
                imageio.imwrite(preview_dir / f"{camera}_scene_patch.png", patch_obs[key][::-1])

        report = wrapped.validation_report()
        report["control_comparison"] = {
            "method": "independent_clean_env_same_init_state_and_actions",
            "per_frame_changed_pixels": frame_differences,
            "per_frame_dynamic_state_difference": state_differences,
        }
        report["source_geom_rgba_checks"] = source_geom_rgba_checks(
            clean_env, wrapped
        )
        report["patch_pose_checks"] = patch_pose_checks(wrapped)
        report["semantic_location_checks"] = semantic_location_checks(wrapped)
        report["validation_only_patch_segmentation_pixels"] = patch_segmentation_counts(
            wrapped
        )
        report["patch_visibility_checks"] = patch_visibility_checks(
            wrapped, report["validation_only_patch_segmentation_pixels"]
        )
        report["gripper_occlusion_coverage_checks"] = (
            gripper_occlusion_coverage_checks(wrapped)
        )
        report["contact_patch_shape_checks"] = contact_patch_shape_checks(wrapped)
        report["direct_physics_counterfactual"] = direct_physics_counterfactual(
            wrapped
        )
        if args.force_contact_checks:
            report["forced_contact_checks"] = force_contact_checks(wrapped)

        model = wrapped.sim.model
        no_private_keys = not any(key.startswith("_wam_a3") for key in patch_obs)
        checks = {
            "dynamic_dimensions_unchanged": bool(
                report["model_invariants"]["dynamics_unchanged"]
            ),
            "all_patch_geoms_non_colliding": all(
                int(model.geom_contype[gid]) == 0
                and int(model.geom_conaffinity[gid]) == 0
                for part in wrapped.parts
                for gid in wrapped._geom_ids_by_part[part]
            ),
            "all_patches_use_configured_visual_geoms": all(
                int(model.geom_group[gid]) == 1
                and int(model.geom_type[gid])
                == {
                    "ellipsoid": int(mujoco.mjtGeom.mjGEOM_ELLIPSOID),
                    "box": int(mujoco.mjtGeom.mjGEOM_BOX),
                    "source_mesh": int(mujoco.mjtGeom.mjGEOM_MESH),
                }[wrapped.geom_type_by_part[part]]
                for part in wrapped.parts
                for gid in wrapped._geom_ids_by_part[part]
            ),
            "configured_rectangular_patch_count_per_interaction_part": all(
                len(wrapped._geom_ids_by_part[part])
                == int(wrapped.spec.get("expected_patch_count", {}).get(part, 1))
                and int(model.geom_type[wrapped._geom_ids_by_part[part][0]])
                == int(mujoco.mjtGeom.mjGEOM_BOX)
                for part in wrapped.parts
            ),
            "standard_observation_has_no_a3_side_channel_keys": no_private_keys,
            "standard_observation_passthrough": bool(
                report["standard_observation_passthrough"]
                and report["manual_camera_render_calls"] == 0
                and report["returned_image_replacements"] == 0
            ),
            "existing_scene_geoms_are_not_recolored": bool(
                report["source_geom_rgba_checks"]["passed"]
            ),
            "both_standard_camera_views_see_scene_patch": all(
                frame[key] > 0 for frame in frame_differences for key in IMAGE_KEYS
            ),
            "noncolliding_patches_leave_direct_physics_step_identical": bool(
                report["direct_physics_counterfactual"]["passed"]
            ),
            "all_active_patch_poses_match_scene_geoms": all(
                item["passed"] for item in report["patch_pose_checks"].values()
            ),
            "all_key_locations_match_intended_entities": all(
                item["passed"]
                for item in report["semantic_location_checks"].values()
            ),
            "each_patch_visible_or_physically_occluded_by_intended_entity": all(
                item["passed"]
                for camera in report["patch_visibility_checks"].values()
                for item in camera.values()
            ),
            "gripper_silhouette_occluded_in_each_view": all(
                item["passed"]
                for item in report["gripper_occlusion_coverage_checks"].values()
            ),
            "contact_occluders_are_thin_surface_pads": all(
                item["passed"]
                for item in report["contact_patch_shape_checks"].values()
            ),
            "full_model_inference_started": False,
        }
        if args.force_contact_checks:
            forced = report["forced_contact_checks"]
            checks["forced_gripper_target_contact_localized"] = bool(
                forced["supported"] and forced["gripper_target"]["passed"]
            )
            checks["forced_target_receptacle_contact_localized"] = bool(
                forced["supported"] and forced["target_receptacle"]["passed"]
            )
            fixed_receptacle = forced["target_receptacle"].get("fixed_region")
            checks["forced_receptacle_contacts_covered_by_fixed_region"] = bool(
                fixed_receptacle is None
                or (
                    fixed_receptacle["coverage_fraction"] is not None
                    and fixed_receptacle["coverage_fraction"]
                    >= float(wrapped.spec.get("minimum_contact_coverage", 0.95))
                )
            )
            checks["target_inside_region_does_not_hide_support_surface"] = bool(
                forced["supported"]
                and forced["target_occluded_goal_support"]["passed"]
            )
        report["checks"] = checks
        report["passed"] = all(
            value for key, value in checks.items() if key != "full_model_inference_started"
        ) and not checks["full_model_inference_started"]

        encoded = json.dumps(report, indent=2, ensure_ascii=False)
        if args.output is not None:
            output = args.output.resolve()
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(encoded + "\n", encoding="utf-8")
        if args.quiet:
            print(
                json.dumps(
                    {
                        "passed": report["passed"],
                        "suite": args.suite,
                        "task": args.task,
                        "changed_pixels": frame_differences[-1],
                        "checks": checks,
                    },
                    ensure_ascii=False,
                )
            )
        else:
            print(encoded)
        return 0 if report["passed"] else 1
    finally:
        wrapped.close()
        clean_env.close()


if __name__ == "__main__":
    raise SystemExit(main())
