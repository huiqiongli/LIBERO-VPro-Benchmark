"""Symmetric paired-state exposure for D3 visual state ambiguity.

The formal D3 observation runtime uses this operator to blend both endpoint
renders with equal weight inside a fixed world-space evidence volume. Pixels
outside the volume and non-RGB observations retain the active truth state.
The caller restores simulator state after rendering the alternate endpoint.
"""

from __future__ import annotations

import copy

import cv2
import numpy as np

from d3_obfuscation import CAMERA_KEYS, CAMERA_NAMES, _projected_box_mask
from d3_evidence_suppression_v3 import _edge_energy


PROFILES = {
    # Equal exposure is deliberate: unequal weights reveal the true endpoint.
    "balanced": {"truth_a_weight": 0.5, "truth_b_weight": 0.5},
}


def _projected_union(base_env, image_shape, evidence_volume):
    height, width = image_shape[:2]
    components = evidence_volume.get("components_world_m") or [{
        "lower": evidence_volume["lower_world_m"],
        "upper": evidence_volume["upper_world_m"],
    }]
    masks, polygons = {}, {}
    for view in CAMERA_KEYS:
        camera_name = CAMERA_NAMES[view]
        union = np.zeros((height, width), dtype=bool)
        view_polygons = []
        for component in components:
            component_mask, _, polygon = _projected_box_mask(
                base_env.sim,
                camera_name,
                height,
                width,
                np.asarray(component["lower"], dtype=float),
                np.asarray(component["upper"], dtype=float),
            )
            union |= component_mask
            view_polygons.append(polygon)
        masks[view] = union
        polygons[view] = view_polygons
    return masks, polygons


def _balanced_exposure(image_a, image_b, mask):
    image_a = np.asarray(image_a, dtype=np.uint8)
    image_b = np.asarray(image_b, dtype=np.uint8)
    mask = np.asarray(mask, dtype=bool)
    # Integer arithmetic makes A/B ordering exactly symmetric.
    paired = (
        image_a.astype(np.uint16) + image_b.astype(np.uint16) + 1
    ) // 2
    paired = paired.astype(np.uint8)
    output_a = np.array(image_a, copy=True)
    output_b = np.array(image_b, copy=True)
    output_a[mask] = paired[mask]
    output_b[mask] = paired[mask]
    changed_a = int(np.count_nonzero(np.any(output_a != image_a, axis=2)))
    changed_b = int(np.count_nonzero(np.any(output_b != image_b, axis=2)))
    mean_absolute_change = 0.0
    if np.any(mask):
        mean_absolute_change = 0.5 * (
            float(np.abs(output_a.astype(np.float32) - image_a.astype(np.float32))[mask].mean())
            + float(np.abs(output_b.astype(np.float32) - image_b.astype(np.float32))[mask].mean())
        )
    before_edge = 0.5 * (
        _edge_energy(image_a, mask) + _edge_energy(image_b, mask)
    )
    after_edge = _edge_energy(paired, mask)
    return output_a, output_b, {
        "changed_pixels_truth_a": changed_a,
        "changed_pixels_truth_b": changed_b,
        "changed_pixels": changed_a + changed_b,
        "edge_energy_before": float(before_edge),
        "edge_energy_after": float(after_edge),
        "edge_energy_ratio": float(after_edge / max(before_edge, 1e-8)),
        "low_frequency_mae": float(mean_absolute_change),
        "truth_a_b_roi_mae_after": float(
            np.abs(output_a.astype(np.float32) - output_b.astype(np.float32))[mask].mean()
        ) if np.any(mask) else 0.0,
    }


def paired_state_exposure_v5(
    base_a,
    true_a: dict,
    base_b,
    true_b: dict,
    evidence_volume: dict,
    *,
    profile_name: str = "balanced",
    return_masks: bool = False,
) -> tuple[dict, dict, dict, dict]:
    """Return symmetric ambiguous observations for the paired truth states."""
    if profile_name not in PROFILES:
        raise ValueError(f"unknown D3 V5 profile: {profile_name}")
    changed_a = copy.deepcopy(true_a)
    changed_b = copy.deepcopy(true_b)
    first_image = np.asarray(true_a[next(iter(CAMERA_KEYS.values()))])
    display_masks, polygons = _projected_union(
        base_a, first_image.shape, evidence_volume
    )
    audits_a, audits_b, raw_masks = {}, {}, {}
    for view, image_key in CAMERA_KEYS.items():
        # LIBERO observations are vertically inverted relative to display RGB.
        display_a = np.ascontiguousarray(np.asarray(true_a[image_key])[::-1])
        display_b = np.ascontiguousarray(np.asarray(true_b[image_key])[::-1])
        display_mask = display_masks[view]
        out_a, out_b, metrics = _balanced_exposure(
            display_a, display_b, display_mask
        )
        changed_a[image_key] = np.ascontiguousarray(out_a[::-1])
        changed_b[image_key] = np.ascontiguousarray(out_b[::-1])
        raw_mask = np.ascontiguousarray(display_mask[::-1])
        raw_masks[view] = {"projected": raw_mask, "effective": raw_mask}
        common = {
            "camera_name": CAMERA_NAMES[view],
            "projected_component_polygons_xy": polygons[view],
            "projected_mask_pixels": int(np.count_nonzero(display_mask)),
            "effective_mask_pixels": int(np.count_nonzero(display_mask)),
            "foreground_preserved_pixels": 0,
            "profile": profile_name,
            "parameters": copy.deepcopy(PROFILES[profile_name]),
            **metrics,
        }
        audits_a[view] = copy.deepcopy(common)
        audits_b[view] = copy.deepcopy(common)
        # The generic validator sums one changed_pixels value per truth audit.
        audits_a[view]["changed_pixels"] = metrics["changed_pixels_truth_a"]
        audits_b[view]["changed_pixels"] = metrics["changed_pixels_truth_b"]

    common_audit = {
        "enabled": True,
        "type": "paired_state_exposure",
        "operator": "symmetric_paired_state_exposure_v5",
        "profile": profile_name,
        "truth_rgb_source": "equal_exposure_of_validated_truth_a_and_truth_b",
        "alternative_state_replaces_truth": False,
        "physics_modified": False,
        "proprioception_modified": False,
        "camera_contract": "same_paired_world_states_rendered_in_both_official_views",
        "evidence_volume": copy.deepcopy(evidence_volume),
    }
    audit_a = {**copy.deepcopy(common_audit), "active_truth": "truth_a", "views": audits_a}
    audit_b = {**copy.deepcopy(common_audit), "active_truth": "truth_b", "views": audits_b}
    if return_masks:
        audit_a["_masks"] = copy.deepcopy(raw_masks)
        audit_b["_masks"] = copy.deepcopy(raw_masks)
    return changed_a, audit_a, changed_b, audit_b
