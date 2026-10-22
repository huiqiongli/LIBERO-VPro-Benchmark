"""D3 state-evidence suppression and shared evidence-volume helpers.

The image operator applies deterministic multi-scale edge suppression using
world-volume projection and depth gating. It does not write simulator state
or render a counterfactual state. The D3 runtime also reuses this module's
evidence-volume and edge-energy helpers.
"""

from __future__ import annotations

import copy

import cv2
import numpy as np

from d3_obfuscation import (
    CAMERA_KEYS,
    CAMERA_NAMES,
    _depth_gate,
    _metric_depth,
    _projected_box_mask,
    joint_subtree_aabb,
    object_aabb,
    site_aabb,
)


# Ordered from the least to the most destructive setting.  Calibration selects
# the first profile whose held-out state-probe accuracy reaches chance level.
PROFILES = {
    "gentle": {
        "coarse_scale": 0.14,
        "near_sigma_scale": 0.040,
        "far_sigma_scale": 0.080,
        "far_mix": 0.40,
        "contrast_scale": 0.64,
        "homogenize_mix": 0.08,
    },
    "moderate": {
        "coarse_scale": 0.10,
        "near_sigma_scale": 0.060,
        "far_sigma_scale": 0.120,
        "far_mix": 0.55,
        "contrast_scale": 0.48,
        "homogenize_mix": 0.18,
    },
    "strong": {
        "coarse_scale": 0.075,
        "near_sigma_scale": 0.085,
        "far_sigma_scale": 0.170,
        "far_mix": 0.68,
        "contrast_scale": 0.32,
        "homogenize_mix": 0.30,
    },
    "maximum": {
        "coarse_scale": 0.055,
        "near_sigma_scale": 0.110,
        "far_sigma_scale": 0.220,
        "far_mix": 0.80,
        "contrast_scale": 0.12,
        "homogenize_mix": 0.72,
    },
}


def _scaled_bounds(lower, upper, *, margin: float, scale: float = 1.0):
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    center = 0.5 * (lower + upper)
    half = 0.5 * (upper - lower) * float(scale) + float(margin)
    return center - half, center + half


def paired_evidence_volume_v3(
    base_a,
    base_b,
    case: dict,
    *,
    occupancy_spec: dict | None = None,
) -> dict:
    """Build a compact state-invariant union of world-space cue components.

    V2 collapsed both truth geometries into one large AABB.  V3 keeps the two
    endpoint components separate and, for articulation, adds only a narrow
    connector along the moving-part sweep.  This avoids hiding unrelated scene
    content between distant AABB corners.
    """
    kind = str(case["ambiguity_kind"])
    raw_components = []
    source = ""
    if kind == "object_support_state_ambiguity":
        source = "paired_target_endpoint_components"
        raw_components = [
            object_aabb(base_a, str(case["target"])),
            object_aabb(base_b, str(case["target"])),
        ]
        components = [
            _scaled_bounds(low, high, margin=0.004, scale=0.92)
            for low, high in raw_components
        ]
    elif kind == "articulation_state_ambiguity":
        source = "paired_moving_part_endpoints_plus_narrow_sweep"
        raw_components = [
            joint_subtree_aabb(base_a, str(case["joint"])),
            joint_subtree_aabb(base_b, str(case["joint"])),
        ]
        components = [
            _scaled_bounds(low, high, margin=0.005, scale=0.92)
            for low, high in raw_components
        ]
        centers = [0.5 * (np.asarray(low) + np.asarray(high)) for low, high in raw_components]
        halfs = [0.5 * (np.asarray(high) - np.asarray(low)) for low, high in raw_components]
        displacement = np.abs(centers[1] - centers[0])
        dominant_axis = int(np.argmax(displacement))
        connector_center = 0.5 * (centers[0] + centers[1])
        connector_half = np.minimum(halfs[0], halfs[1]) * 0.45 + 0.004
        connector_half[dominant_axis] = (
            0.5 * displacement[dominant_axis]
            + max(halfs[0][dominant_axis], halfs[1][dominant_axis])
            + 0.004
        )
        components.append((connector_center - connector_half, connector_center + connector_half))
    elif kind == "receptacle_occupancy_ambiguity":
        if not occupancy_spec:
            raise ValueError("occupancy_spec is required for occupancy evidence")
        destination = occupancy_spec.get("destination") or {}
        if destination.get("kind") == "site":
            source = "receiver_interior_component"
            raw_components = [site_aabb(base_a, str(destination["name"]))]
        elif destination.get("kind") == "object_aabb":
            source = "receiver_support_component"
            raw_components = [object_aabb(base_a, str(destination["name"]))]
        else:
            raise ValueError(f"unsupported occupancy destination: {destination}")
        components = [
            _scaled_bounds(low, high, margin=0.002, scale=0.88)
            for low, high in raw_components
        ]
    else:
        raise ValueError(f"unsupported D3 ambiguity kind: {kind}")

    lower = np.min(np.stack([pair[0] for pair in components]), axis=0)
    upper = np.max(np.stack([pair[1] for pair in components]), axis=0)
    return {
        "schema_version": "wam_syncbench.d3.evidence_volume.v3",
        "kind": kind.replace("_ambiguity", "_obfuscation"),
        "source": source,
        "lower_world_m": lower.tolist(),
        "upper_world_m": upper.tolist(),
        "components_world_m": [
            {"lower": low.tolist(), "upper": high.tolist()}
            for low, high in components
        ],
        "state_invariant": True,
        "derived_from": "paired_truth_state_geometry_offline",
    }


def _edge_energy(image: np.ndarray, mask: np.ndarray) -> float:
    gray = cv2.cvtColor(np.asarray(image, dtype=np.uint8), cv2.COLOR_RGB2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(gx, gy)
    valid = np.asarray(mask, dtype=bool)
    return float(magnitude[valid].mean()) if np.any(valid) else 0.0


def _suppress_evidence(
    image: np.ndarray,
    mask: np.ndarray,
    profile_name: str,
) -> tuple[np.ndarray, dict]:
    """Suppress high-frequency state cues while retaining coarse appearance."""
    if profile_name not in PROFILES:
        raise ValueError(f"unknown D3 V3 profile: {profile_name}")
    params = PROFILES[profile_name]
    output = np.array(image, copy=True)
    rows, cols = np.where(mask)
    if len(rows) == 0:
        return output, {
            "changed_pixels": 0,
            "crop": None,
            "profile": profile_name,
            "parameters": copy.deepcopy(params),
        }

    y0, y1 = int(rows.min()), int(rows.max()) + 1
    x0, x1 = int(cols.min()), int(cols.max()) + 1
    crop = np.asarray(image[y0:y1, x0:x1], dtype=np.uint8)
    local_mask = np.asarray(mask[y0:y1, x0:x1], dtype=bool)
    crop_h, crop_w = crop.shape[:2]
    extent = float(max(1, min(crop_h, crop_w)))

    coarse_w = max(4, min(24, int(round(crop_w * params["coarse_scale"]))))
    coarse_h = max(4, min(24, int(round(crop_h * params["coarse_scale"]))))
    coarse = cv2.resize(crop, (coarse_w, coarse_h), interpolation=cv2.INTER_AREA)
    coarse = cv2.resize(coarse, (crop_w, crop_h), interpolation=cv2.INTER_CUBIC)

    near_sigma = max(1.5, extent * params["near_sigma_scale"])
    far_sigma = max(3.0, extent * params["far_sigma_scale"])
    near = cv2.GaussianBlur(coarse, (0, 0), sigmaX=near_sigma, sigmaY=near_sigma)
    far = cv2.GaussianBlur(near, (0, 0), sigmaX=far_sigma, sigmaY=far_sigma)
    suppressed = (
        (1.0 - params["far_mix"]) * near.astype(np.float32)
        + params["far_mix"] * far.astype(np.float32)
    )

    # Use a narrow ring outside the fixed world-volume projection as the colour
    # anchor.  The ring does not contain the state cue and is therefore much
    # more stable across paired A/B renders than the state region's own median.
    ring_radius = max(3, int(round(0.06 * extent)))
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * ring_radius + 1, 2 * ring_radius + 1)
    )
    full_dilated = cv2.dilate(np.asarray(mask, dtype=np.uint8), kernel).astype(bool)
    full_ring = full_dilated & ~np.asarray(mask, dtype=bool)
    if np.any(full_ring):
        anchor_pixels = np.asarray(image, dtype=np.uint8)[full_ring]
    else:
        anchor_pixels = crop[local_mask]
    anchor = np.median(anchor_pixels.astype(np.float32), axis=0).reshape(1, 1, 3)
    suppressed = anchor + params["contrast_scale"] * (suppressed - anchor)
    suppressed = (
        (1.0 - params["homogenize_mix"]) * suppressed
        + params["homogenize_mix"] * anchor
    )
    suppressed = np.clip(suppressed, 0, 255).astype(np.uint8)

    distance = cv2.distanceTransform(local_mask.astype(np.uint8), cv2.DIST_L2, 3)
    feather_px = max(2.0, 0.025 * extent)
    alpha = np.clip(distance / feather_px, 0.0, 1.0)[..., None]
    blended = np.rint(
        alpha * suppressed.astype(np.float32)
        + (1.0 - alpha) * crop.astype(np.float32)
    ).astype(np.uint8)
    local_output = output[y0:y1, x0:x1]
    local_output[local_mask] = blended[local_mask]
    output[y0:y1, x0:x1] = local_output

    before_edge = _edge_energy(crop, local_mask)
    after_edge = _edge_energy(output[y0:y1, x0:x1], local_mask)
    low_sigma = max(3.0, 0.12 * extent)
    clean_low = cv2.GaussianBlur(crop, (0, 0), low_sigma)
    output_low = cv2.GaussianBlur(output[y0:y1, x0:x1], (0, 0), low_sigma)
    low_frequency_mae = float(
        np.abs(clean_low.astype(np.float32) - output_low.astype(np.float32))[local_mask].mean()
    )
    changed = int(np.count_nonzero(np.any(output != image, axis=2)))
    return output, {
        "changed_pixels": changed,
        "crop": [x0, y0, x1, y1],
        "profile": profile_name,
        "parameters": copy.deepcopy(params),
        "coarse_grid": [coarse_w, coarse_h],
        "near_gaussian_sigma_px": float(near_sigma),
        "far_gaussian_sigma_px": float(far_sigma),
        "feather_px": float(feather_px),
        "edge_energy_before": before_edge,
        "edge_energy_after": after_edge,
        "edge_energy_ratio": after_edge / max(before_edge, 1e-8),
        "low_frequency_mae": low_frequency_mae,
    }


def suppress_truth_observation_v3(
    base_env,
    true_observation: dict,
    evidence_volume: dict,
    *,
    profile_name: str,
    return_masks: bool = False,
) -> tuple[dict, dict]:
    """Apply deterministic V3 suppression to both official RGB views."""
    model_observation = copy.deepcopy(true_observation)
    declared_components = evidence_volume.get("components_world_m") or [{
        "lower": evidence_volume["lower_world_m"],
        "upper": evidence_volume["upper_world_m"],
    }]
    view_audits: dict[str, dict] = {}
    masks: dict[str, dict] = {}
    for view, image_key in CAMERA_KEYS.items():
        raw_image = np.asarray(true_observation[image_key])
        image = np.ascontiguousarray(raw_image[::-1])
        height, width = image.shape[:2]
        camera_name = CAMERA_NAMES[view]
        projected = np.zeros((height, width), dtype=bool)
        effective = np.zeros((height, width), dtype=bool)
        polygons = []
        camera = None
        depth = np.ascontiguousarray(
            _metric_depth(base_env.sim, camera_name, height, width)[::-1]
        )
        for component in declared_components:
            lower = np.asarray(component["lower"], dtype=float)
            upper = np.asarray(component["upper"], dtype=float)
            component_projected, camera, polygon = _projected_box_mask(
                base_env.sim, camera_name, height, width, lower, upper
            )
            component_effective, _ = _depth_gate(
                component_projected, depth, camera, lower, upper
            )
            projected |= component_projected
            effective |= component_effective
            polygons.append(polygon)
        foreground_pixels = int(np.count_nonzero(projected & ~effective))
        changed, operator_audit = _suppress_evidence(image, effective, profile_name)
        model_observation[image_key] = np.ascontiguousarray(changed[::-1])
        view_audits[view] = {
            "camera_name": camera_name,
            "projected_component_polygons_xy": polygons,
            "projected_mask_pixels": int(np.count_nonzero(projected)),
            "effective_mask_pixels": int(np.count_nonzero(effective)),
            "foreground_preserved_pixels": int(foreground_pixels),
            **operator_audit,
        }
        if return_masks:
            masks[view] = {
                "projected": np.ascontiguousarray(projected[::-1]),
                "effective": np.ascontiguousarray(effective[::-1]),
            }

    audit = {
        "enabled": True,
        "type": "state_evidence_suppression",
        "operator": "depth_aware_multiscale_evidence_suppression_v3",
        "profile": profile_name,
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
