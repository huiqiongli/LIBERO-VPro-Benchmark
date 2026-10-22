"""D3 state-evidence obfuscation and world-volume projection helpers.

The operator in this module never renders or constructs a counterfactual
visual state.  It degrades a fixed world-frame evidence volume in the RGB
render of the current truth state.  The same volume is projected into both
official cameras, while the depth buffer preserves real foreground occluders.
"""

from __future__ import annotations

import copy
from typing import Iterable

import cv2
import numpy as np


CAMERA_KEYS = {
    "agentview": "agentview_image",
    "wrist": "robot0_eye_in_hand_image",
}
CAMERA_NAMES = {
    "agentview": "agentview",
    "wrist": "robot0_eye_in_hand",
}


def _descendant_bodies(model, root_body_id: int) -> set[int]:
    selected = {int(root_body_id)}
    changed = True
    while changed:
        changed = False
        for body_id in range(int(model.nbody)):
            if body_id in selected:
                continue
            if int(model.body_parentid[body_id]) in selected:
                selected.add(body_id)
                changed = True
    return selected


def _geom_local_bounds(model, geom_id: int) -> tuple[np.ndarray, np.ndarray]:
    """Return compiled local lower / upper bounds for one MuJoCo geom."""
    geom_type = int(model.geom_type[geom_id])
    size = np.asarray(model.geom_size[geom_id], dtype=float)
    radius = float(model.geom_rbound[geom_id])
    # MuJoCo mGeom values: sphere=2, capsule=3, ellipsoid=4, cylinder=5,
    # box=6, mesh=7.  Plane / hfield are not expected in object subtrees.
    if geom_type == 2:
        half = np.repeat(size[0], 3)
        return -half, half
    if geom_type == 3:
        half = np.array([size[0], size[0], size[1] + size[0]], dtype=float)
        return -half, half
    if geom_type == 4:
        half = size[:3]
        return -half, half
    if geom_type == 5:
        half = np.array([size[0], size[0], size[1]], dtype=float)
        return -half, half
    if geom_type == 6:
        half = size[:3]
        return -half, half
    if geom_type == 7:
        mesh_id = int(model.geom_dataid[geom_id])
        if mesh_id >= 0:
            start = int(model.mesh_vertadr[mesh_id])
            count = int(model.mesh_vertnum[mesh_id])
            vertices = np.asarray(model.mesh_vert[start : start + count], dtype=float)
            if vertices.size:
                return vertices.min(axis=0), vertices.max(axis=0)
    half = np.repeat(max(radius, 1e-4), 3)
    return -half, half


def _oriented_aabb(position, rotation, half_extents) -> tuple[np.ndarray, np.ndarray]:
    position = np.asarray(position, dtype=float)
    rotation = np.asarray(rotation, dtype=float).reshape(3, 3)
    half_extents = np.asarray(half_extents, dtype=float)
    world_half = np.abs(rotation) @ half_extents
    return position - world_half, position + world_half


def _oriented_local_bounds_aabb(position, rotation, lower, upper):
    position = np.asarray(position, dtype=float)
    rotation = np.asarray(rotation, dtype=float).reshape(3, 3)
    corners = _box_corners(lower, upper)
    world = corners @ rotation.T + position
    return world.min(axis=0), world.max(axis=0)


def body_subtree_aabb(base_env, root_body_id: int) -> tuple[np.ndarray, np.ndarray]:
    sim = base_env.sim
    bodies = _descendant_bodies(sim.model, int(root_body_id))
    lower, upper = [], []
    for geom_id in range(int(sim.model.ngeom)):
        if int(sim.model.geom_bodyid[geom_id]) not in bodies:
            continue
        local_lower, local_upper = _geom_local_bounds(sim.model, geom_id)
        low, high = _oriented_local_bounds_aabb(
            sim.data.geom_xpos[geom_id],
            sim.data.geom_xmat[geom_id],
            local_lower,
            local_upper,
        )
        lower.append(low)
        upper.append(high)
    if not lower:
        raise ValueError(f"body subtree {root_body_id} has no render geoms")
    return np.min(np.stack(lower), axis=0), np.max(np.stack(upper), axis=0)


def object_aabb(base_env, object_name: str) -> tuple[np.ndarray, np.ndarray]:
    body_map = getattr(base_env, "obj_body_id", {})
    if object_name not in body_map:
        raise ValueError(f"object body is absent: {object_name}")
    return body_subtree_aabb(base_env, int(body_map[object_name]))


def joint_subtree_aabb(base_env, joint_name: str) -> tuple[np.ndarray, np.ndarray]:
    model = base_env.sim.model
    joint_id = int(model.joint_name2id(joint_name))
    body_id = int(model.jnt_bodyid[joint_id])
    return body_subtree_aabb(base_env, body_id)


def site_aabb(base_env, site_name: str) -> tuple[np.ndarray, np.ndarray]:
    sim = base_env.sim
    site_id = int(sim.model.site_name2id(site_name))
    size = np.asarray(sim.model.site_size[site_id], dtype=float)
    # Containment sites in LIBERO use box / ellipsoid extents.  Treat the three
    # declared size values as conservative local half extents for either type.
    half = np.maximum(size[:3], 1e-4)
    return _oriented_aabb(
        sim.data.site_xpos[site_id], sim.data.site_xmat[site_id], half
    )


def _union_bounds(bounds: Iterable[tuple[np.ndarray, np.ndarray]], margin_m: float):
    pairs = list(bounds)
    lower = np.min(np.stack([pair[0] for pair in pairs]), axis=0)
    upper = np.max(np.stack([pair[1] for pair in pairs]), axis=0)
    margin = np.repeat(float(margin_m), 3)
    return lower - margin, upper + margin


def evidence_volume_from_bounds(kind: str, bounds, source: str) -> dict:
    margins = {
        "object_support_state_ambiguity": 0.006,
        "articulation_state_ambiguity": 0.008,
        "receptacle_occupancy_ambiguity": 0.002,
    }
    coverage_scales = {
        "object_support_state_ambiguity": 0.86,
        "articulation_state_ambiguity": 0.90,
        "receptacle_occupancy_ambiguity": 0.86,
    }
    if kind not in margins:
        raise ValueError(f"unsupported D3 ambiguity kind: {kind}")
    margin = float(margins[kind])
    lower, upper = _union_bounds(bounds, margin)
    # V2 keeps the paired-state union centre but uses its diagnostic core, so
    # category / coarse identity remains visible around the softened region.
    # The scale is fixed by subclass and therefore cannot disclose truth A/B.
    coverage_scale = float(coverage_scales[kind])
    center = 0.5 * (lower + upper)
    half = 0.5 * (upper - lower) * coverage_scale
    lower = center - half
    upper = center + half
    return {
        "schema_version": "wam_syncbench.d3.evidence_volume.v1",
        "kind": kind.replace("_ambiguity", "_obfuscation"),
        "source": str(source),
        "lower_world_m": lower.tolist(),
        "upper_world_m": upper.tolist(),
        "margin_m": margin,
        "coverage_scale": coverage_scale,
        "state_invariant": True,
        "derived_from": "paired_truth_state_geometry_private_initialization_branch",
    }


def paired_evidence_volume(
    base_a,
    base_b,
    case: dict,
    *,
    occupancy_spec: dict | None = None,
) -> dict:
    """Build one A/B-invariant world AABB from a paired validation environment.

    Formal runs consume the resulting serialized volume; they must not derive a
    mask from the current truth silhouette.
    """
    kind = str(case["ambiguity_kind"])
    if kind == "object_support_state_ambiguity":
        source = "paired_target_pose_union"
        bounds = [
            object_aabb(base_a, str(case["target"])),
            object_aabb(base_b, str(case["target"])),
        ]
    elif kind == "articulation_state_ambiguity":
        source = "paired_joint_sweep_union"
        bounds = [
            joint_subtree_aabb(base_a, str(case["joint"])),
            joint_subtree_aabb(base_b, str(case["joint"])),
        ]
    elif kind == "receptacle_occupancy_ambiguity":
        if not occupancy_spec:
            raise ValueError("occupancy_spec is required for occupancy evidence")
        destination = occupancy_spec.get("destination") or {}
        destination_kind = str(destination.get("kind"))
        if destination_kind == "site":
            source = "paired_receiver_interior_union"
            site_name = str(destination["name"])
            bounds = [site_aabb(base_a, site_name), site_aabb(base_b, site_name)]
        elif destination_kind == "object_aabb":
            # Open support regions (plate, stove top, cabinet top) have no
            # containment site.  Their validated receiver surface is represented
            # by the receiver object's paired world AABB.
            source = "paired_receiver_support_surface_union"
            object_name = str(destination["name"])
            bounds = [
                object_aabb(base_a, object_name),
                object_aabb(base_b, object_name),
            ]
        else:
            raise ValueError(f"unsupported occupancy destination: {destination}")
    else:
        raise ValueError(f"unsupported D3 ambiguity kind: {kind}")
    volume = evidence_volume_from_bounds(kind, bounds, source)
    volume["derived_from"] = "paired_truth_state_geometry_offline"
    return volume


def _box_corners(lower, upper) -> np.ndarray:
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    return np.asarray([
        [x, y, z]
        for x in (lower[0], upper[0])
        for y in (lower[1], upper[1])
        for z in (lower[2], upper[2])
    ], dtype=float)


def _camera_geometry(sim, camera_name: str, height: int, width: int) -> dict:
    camera_id = int(sim.model.camera_name2id(camera_name))
    position = np.asarray(sim.data.cam_xpos[camera_id], dtype=float)
    rotation = np.asarray(sim.data.cam_xmat[camera_id], dtype=float).reshape(3, 3)
    fovy = float(sim.model.cam_fovy[camera_id])
    fy = 0.5 * float(height) / np.tan(np.deg2rad(fovy) * 0.5)
    fx = fy * float(width) / float(height)
    return {
        "camera_id": camera_id,
        "position": position,
        "rotation": rotation,
        "fx": float(fx),
        "fy": float(fy),
        "cx": 0.5 * float(width),
        "cy": 0.5 * float(height),
    }


def _projected_box_mask(sim, camera_name: str, height: int, width: int, lower, upper):
    camera = _camera_geometry(sim, camera_name, height, width)
    camera_points = (
        camera["rotation"].T
        @ (_box_corners(lower, upper) - camera["position"]).T
    ).T
    in_front = camera_points[:, 2] < -1e-6
    if int(np.count_nonzero(in_front)) < 3:
        return np.zeros((height, width), dtype=bool), camera, []
    points = camera_points[in_front]
    rows = camera["cy"] - camera["fy"] * points[:, 1] / (-points[:, 2])
    cols = camera["cx"] + camera["fx"] * points[:, 0] / (-points[:, 2])
    polygon = np.stack([cols, rows], axis=1)
    # Keep OpenCV coordinates bounded even when a volume crosses the frustum.
    polygon[:, 0] = np.clip(polygon[:, 0], -4 * width, 5 * width)
    polygon[:, 1] = np.clip(polygon[:, 1], -4 * height, 5 * height)
    hull = cv2.convexHull(np.rint(polygon).astype(np.int32))
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillConvexPoly(mask, hull, 1)
    return mask.astype(bool), camera, polygon.tolist()


def _metric_depth(sim, camera_name: str, height: int, width: int) -> np.ndarray:
    rendered = sim.render(
        width=int(width), height=int(height), camera_name=camera_name, depth=True
    )
    if not isinstance(rendered, tuple) or len(rendered) != 2:
        raise RuntimeError("MuJoCo depth render did not return (rgb, depth)")
    depth = np.asarray(rendered[1], dtype=float)
    if np.any(depth < 0.0) or np.any(depth > 1.0):
        raise RuntimeError("MuJoCo returned a non-normalized depth map")
    # Inline robosuite's conversion rather than importing camera_utils: that
    # module imports optional h5py even though depth conversion itself does not
    # need it, and several official evaluation environments omit h5py.
    extent = float(sim.model.stat.extent)
    far = float(sim.model.vis.map.zfar) * extent
    near = float(sim.model.vis.map.znear) * extent
    return near / (1.0 - depth * (1.0 - near / far))


def _depth_gate(mask, metric_depth, camera, lower, upper, foreground_margin_m=0.004):
    """Keep pixels whose first real surface is not in front of the world box."""
    rows, cols = np.where(mask)
    effective = np.zeros_like(mask, dtype=bool)
    if len(rows) == 0:
        return effective, 0
    directions_camera = np.stack([
        (cols.astype(float) - camera["cx"]) / camera["fx"],
        -(rows.astype(float) - camera["cy"]) / camera["fy"],
        -np.ones(len(rows), dtype=float),
    ], axis=1)
    directions_world = directions_camera @ camera["rotation"].T
    origin = np.asarray(camera["position"], dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    t_near = np.full(len(rows), -np.inf, dtype=float)
    t_far = np.full(len(rows), np.inf, dtype=float)
    valid = np.ones(len(rows), dtype=bool)
    for axis in range(3):
        direction = directions_world[:, axis]
        parallel = np.abs(direction) < 1e-10
        valid &= ~(parallel & ((origin[axis] < lower[axis]) | (origin[axis] > upper[axis])))
        safe = np.where(parallel, 1.0, direction)
        first = (lower[axis] - origin[axis]) / safe
        second = (upper[axis] - origin[axis]) / safe
        axis_near = np.minimum(first, second)
        axis_far = np.maximum(first, second)
        axis_near[parallel] = -np.inf
        axis_far[parallel] = np.inf
        t_near = np.maximum(t_near, axis_near)
        t_far = np.minimum(t_far, axis_far)
    t_near = np.maximum(t_near, 0.0)
    valid &= t_far >= t_near
    scene_depth = metric_depth[rows, cols]
    valid &= np.isfinite(scene_depth)
    valid &= scene_depth + float(foreground_margin_m) >= t_near
    effective[rows[valid], cols[valid]] = True
    return effective, int(len(rows) - np.count_nonzero(valid))


def _frosted_blur(image: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, dict]:
    output = np.array(image, copy=True)
    rows, cols = np.where(mask)
    if len(rows) == 0:
        return output, {"changed_pixels": 0, "crop": None}
    y0, y1 = int(rows.min()), int(rows.max()) + 1
    x0, x1 = int(cols.min()), int(cols.max()) + 1
    crop = np.asarray(image[y0:y1, x0:x1], dtype=np.uint8)
    crop_h, crop_w = crop.shape[:2]
    coarse_w = max(5, min(18, int(round(crop_w / 7.0))))
    coarse_h = max(5, min(18, int(round(crop_h / 7.0))))
    coarse = cv2.resize(crop, (coarse_w, coarse_h), interpolation=cv2.INTER_AREA)
    frosted = cv2.resize(coarse, (crop_w, crop_h), interpolation=cv2.INTER_CUBIC)
    sigma = max(1.5, 0.040 * float(min(crop_h, crop_w)))
    frosted = cv2.GaussianBlur(frosted, (0, 0), sigmaX=sigma, sigmaY=sigma)
    mean = np.asarray(frosted, dtype=np.float32).mean(axis=(0, 1), keepdims=True)
    frosted = np.clip(
        mean + 0.72 * (np.asarray(frosted, dtype=np.float32) - mean), 0, 255
    ).astype(np.uint8)
    local_mask = mask[y0:y1, x0:x1].astype(np.uint8)
    distance = cv2.distanceTransform(local_mask, cv2.DIST_L2, 3)
    feather_px = max(2.0, 0.022 * float(min(crop_h, crop_w)))
    alpha = np.clip(distance / feather_px, 0.0, 1.0)[..., None]
    blended = np.rint(
        alpha * frosted.astype(np.float32)
        + (1.0 - alpha) * crop.astype(np.float32)
    ).astype(np.uint8)
    local_output = output[y0:y1, x0:x1]
    local_output[local_mask.astype(bool)] = blended[local_mask.astype(bool)]
    output[y0:y1, x0:x1] = local_output
    changed = int(np.count_nonzero(np.any(output != image, axis=2)))
    return output, {
        "changed_pixels": changed,
        "crop": [x0, y0, x1, y1],
        "coarse_grid": [coarse_w, coarse_h],
        "gaussian_sigma_px": float(sigma),
        "contrast_scale": 0.72,
        "feather_px": float(feather_px),
    }


def obfuscate_truth_observation(
    base_env,
    true_observation: dict,
    evidence_volume: dict,
    *,
    return_masks: bool = False,
) -> tuple[dict, dict]:
    """Apply state-evidence obfuscation to both official truth RGB views."""
    model_observation = copy.deepcopy(true_observation)
    lower = np.asarray(evidence_volume["lower_world_m"], dtype=float)
    upper = np.asarray(evidence_volume["upper_world_m"], dtype=float)
    view_audits = {}
    masks = {}
    for view, image_key in CAMERA_KEYS.items():
        raw_image = np.asarray(true_observation[image_key])
        # LIBERO's raw observation arrays are vertically inverted relative to
        # the calibrated image coordinates consumed by all official policies.
        # Project and blur in policy-image coordinates, then restore the raw
        # array convention expected by the shared environment wrapper.
        image = np.ascontiguousarray(raw_image[::-1])
        height, width = image.shape[:2]
        camera_name = CAMERA_NAMES[view]
        projected, camera, polygon = _projected_box_mask(
            base_env.sim, camera_name, height, width, lower, upper
        )
        depth = np.ascontiguousarray(
            _metric_depth(base_env.sim, camera_name, height, width)[::-1]
        )
        effective, foreground_pixels = _depth_gate(
            projected, depth, camera, lower, upper
        )
        changed, blur_audit = _frosted_blur(image, effective)
        model_observation[image_key] = np.ascontiguousarray(changed[::-1])
        view_audits[view] = {
            "camera_name": camera_name,
            "projected_polygon_xy": polygon,
            "projected_mask_pixels": int(np.count_nonzero(projected)),
            "effective_mask_pixels": int(np.count_nonzero(effective)),
            "foreground_preserved_pixels": foreground_pixels,
            **blur_audit,
        }
        if return_masks:
            masks[view] = {
                # Return raw-observation-coordinate masks so callers can audit
                # exact outside-mask equality against true_observation.
                "projected": np.ascontiguousarray(projected[::-1]),
                "effective": np.ascontiguousarray(effective[::-1]),
            }
    audit = {
        "enabled": True,
        "type": "state_evidence_obfuscation",
        "operator": "depth_aware_localized_frosted_blur_v2",
        "truth_rgb_source": "current_simulator_state",
        "alternative_state_rendered": False,
        "physics_modified": False,
        "proprioception_modified": False,
        "camera_contract": "same_world_volume_projected_into_both_official_views",
        "evidence_volume": copy.deepcopy(evidence_volume),
        "views": view_audits,
    }
    if return_masks:
        audit["_masks"] = masks
    return model_observation, audit
