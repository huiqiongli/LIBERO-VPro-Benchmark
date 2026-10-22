"""Shared, model-independent state resolution for isolated D2/D3 runners."""

from __future__ import annotations

import copy

import numpy as np


OPEN_AT_MIN = {"drawer", "microwave_door"}
OPEN_AT_MAX = {"stove_button"}


def joint_id(model, name: str) -> int:
    value = model.joint_name2id(name)
    if value is None or int(value) < 0:
        raise ValueError(f"MuJoCo joint not found: {name}")
    return int(value)


def semantic_joint_endpoints(model, name: str, family: str) -> tuple[float, float]:
    """Return `(closed_or_off, open_or_on)` from the compiled joint range."""
    jid = joint_id(model, name)
    low, high = (float(x) for x in model.jnt_range[jid])
    if not high > low:
        raise ValueError(f"Joint {name} has invalid range [{low}, {high}]")
    if family in OPEN_AT_MIN:
        return high, low
    if family in OPEN_AT_MAX:
        return low, high
    raise ValueError(f"Unsupported articulation family: {family}")


def qpos_for_open_fraction(model, name: str, family: str, fraction: float) -> float:
    fraction = float(fraction)
    if not 0.0 <= fraction <= 1.0:
        raise ValueError(f"Open fraction must be in [0,1], got {fraction}")
    closed, opened = semantic_joint_endpoints(model, name, family)
    return closed + fraction * (opened - closed)


def resolve_normalized_articulations(base_env, spec: dict) -> tuple[dict, list[dict]]:
    """Convert normalized articulation edits into legacy qpos overrides.

    The returned spec is a deep copy and can safely be passed to the existing
    LIBERO client without modifying a shared configuration object.
    """
    resolved = copy.deepcopy(spec)
    legacy = list(resolved.get("articulation_qpos_overrides", []))
    audit: list[dict] = []
    model = base_env.sim.model
    for item in resolved.pop("normalized_articulation_overrides", []):
        name = str(item["joint"])
        family = str(item["joint_family"])
        fraction = float(item["open_fraction"])
        closed, opened = semantic_joint_endpoints(model, name, family)
        qpos = qpos_for_open_fraction(model, name, family, fraction)
        legacy.append({"joint": name, "qpos": qpos})
        audit.append(
            {
                "joint": name,
                "joint_family": family,
                "requested_open_fraction": fraction,
                "closed_endpoint": closed,
                "open_endpoint": opened,
                "resolved_qpos": qpos,
            }
        )
    resolved["articulation_qpos_overrides"] = legacy
    resolved["_normalized_articulation_audit"] = audit
    return resolved, audit


SUPPORT_ROTATIONS_DEG = {
    "axial_container": {
        "adjacent_support_face": (90.0, 0.0, 0.0),
        "opposite_support_face": (-90.0, 0.0, 0.0),
        # A bottle/carton/can lying on its side is common after a minor bump.
        "lying_on_side": (88.0, 0.0, 0.0),
    },
    "handled_vessel": {
        "adjacent_support_face": (90.0, 0.0, 0.0),
        "opposite_support_face": (180.0, 0.0, 0.0),
        # Requested at 75 degrees; contact settling may bring the vessel close
        # to its stable side face, but it can never become base-up / inverted.
        "tipped_on_side": (75.0, 0.0, 0.0),
    },
    "open_vessel": {
        "adjacent_support_face": (90.0, 0.0, 0.0),
        "opposite_support_face": (180.0, 0.0, 0.0),
        # A bowl placed rim-down for drying/storage is physically and
        # semantically plausible; a rim-edge pose is not stable without a prop.
        "rim_down_storage": (180.0, 0.0, 0.0),
    },
    "box": {
        "adjacent_support_face": (90.0, 0.0, 0.0),
        "opposite_support_face": (0.0, 90.0, 0.0),
        "laid_on_side": (90.0, 0.0, 0.0),
    },
    "planar": {
        "adjacent_support_face": (180.0, 0.0, 0.0),
        "opposite_support_face": (0.0, 180.0, 0.0),
        "face_down_storage": (0.0, 180.0, 0.0),
    },
}


def _axis_angle_quat(axis: tuple[float, float, float], degrees: float) -> np.ndarray:
    axis_array = np.asarray(axis, dtype=float)
    axis_array /= np.linalg.norm(axis_array)
    half = np.deg2rad(float(degrees)) * 0.5
    return np.concatenate([[np.cos(half)], axis_array * np.sin(half)])


def _quat_mul(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Hamilton product for MuJoCo `wxyz` quaternions."""
    w1, x1, y1, z1 = np.asarray(left, dtype=float)
    w2, x2, y2, z2 = np.asarray(right, dtype=float)
    result = np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=float,
    )
    return result / np.linalg.norm(result)


def _euler_delta_quat(degrees_xyz: tuple[float, float, float]) -> np.ndarray:
    q = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    for axis, degrees in zip(((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)), degrees_xyz):
        if degrees:
            q = _quat_mul(q, _axis_angle_quat(axis, degrees))
    return q


def quaternion_distance_deg(first: np.ndarray, second: np.ndarray) -> float:
    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)
    first /= np.linalg.norm(first)
    second /= np.linalg.norm(second)
    cosine = float(np.clip(abs(np.dot(first, second)), 0.0, 1.0))
    return float(np.rad2deg(2.0 * np.arccos(cosine)))


def _snapshot_sim(sim) -> dict:
    return {
        "time": float(sim.data.time),
        "qpos": np.array(sim.data.qpos, copy=True),
        "qvel": np.array(sim.data.qvel, copy=True),
        "act": np.array(sim.data.act, copy=True),
        "ctrl": np.array(sim.data.ctrl, copy=True),
        "warm": np.array(sim.data.qacc_warmstart, copy=True),
        "mocap_pos": np.array(sim.data.mocap_pos, copy=True),
        "mocap_quat": np.array(sim.data.mocap_quat, copy=True),
    }


def _restore_sim(sim, state: dict) -> None:
    sim.data.time = state["time"]
    sim.data.qpos[:] = state["qpos"]
    sim.data.qvel[:] = state["qvel"]
    if sim.data.act.size:
        sim.data.act[:] = state["act"]
    sim.data.ctrl[:] = state["ctrl"]
    if sim.data.mocap_pos.size:
        sim.data.mocap_pos[:] = state["mocap_pos"]
        sim.data.mocap_quat[:] = state["mocap_quat"]
    sim.forward()
    # Preserve exact policy state after mj_forward normalization.
    sim.data.time = state["time"]
    sim.data.qpos[:] = state["qpos"]
    sim.data.qvel[:] = state["qvel"]
    if sim.data.act.size:
        sim.data.act[:] = state["act"]
    sim.data.ctrl[:] = state["ctrl"]
    sim.data.qacc_warmstart[:] = state["warm"]
    if sim.data.mocap_pos.size:
        sim.data.mocap_pos[:] = state["mocap_pos"]
        sim.data.mocap_quat[:] = state["mocap_quat"]


def _joint_qvel(sim, joint_name: str) -> np.ndarray:
    address = sim.model.get_joint_qvel_addr(joint_name)
    if isinstance(address, tuple):
        return np.asarray(sim.data.qvel[address[0]:address[1]], dtype=float)
    return np.asarray([sim.data.qvel[int(address)]], dtype=float)


def apply_object_support_state(
    base_env,
    libero_client,
    *,
    object_name: str,
    morphology: str,
    state: str,
) -> dict:
    """Rotate a target onto a morphology-specific support face.

    The collision lower surface is realigned to the target's pre-edit support
    plane before physics is resumed; this prevents the legacy floating/sinking
    behavior caused by keeping the original center Z after a 90-degree rotate.
    """
    try:
        degrees = SUPPORT_ROTATIONS_DEG[morphology][state]
    except KeyError as exc:
        raise ValueError(f"Unsupported support state {morphology}/{state}") from exc
    free_joint = libero_client._free_joint_name(base_env, object_name)
    if free_joint is None:
        raise ValueError(f"Object has no free joint: {object_name}")
    sim = base_env.sim
    before = np.asarray(libero_client._get_joint_qpos(sim, free_joint), dtype=float)
    support_world_z = libero_client._object_collision_lower_world_z(base_env, object_name)
    if support_world_z is None:
        raise ValueError(f"Cannot resolve collision support plane: {object_name}")
    requested_quat = _quat_mul(before[3:], _euler_delta_quat(degrees))
    provisional = np.concatenate([before[:3], requested_quat])
    libero_client._set_joint_qpos(sim, free_joint, provisional)
    libero_client._zero_joint_velocity(sim, free_joint)
    libero_client._forward_sim(sim)
    rotated_lower = libero_client._object_collision_lower_world_z(base_env, object_name)
    if rotated_lower is None:
        raise ValueError(f"Cannot resolve rotated collision support plane: {object_name}")
    requested_pos = before[:3].copy()
    requested_pos[2] += float(support_world_z) - float(rotated_lower)
    requested = np.concatenate([requested_pos, requested_quat])
    preserved = _snapshot_sim(sim)
    # Try the original XY first.  Small deterministic offsets are only used
    # when a rotated footprint overlaps a nearby object in a particular init
    # state.  Every candidate is privately settled and the full scene is then
    # restored, so no probe step enters the formal episode.
    offsets = (
        (0.0, 0.0),
        (0.012, 0.0), (-0.012, 0.0), (0.0, 0.012), (0.0, -0.012),
        (0.024, 0.0), (-0.024, 0.0), (0.0, 0.024), (0.0, -0.024),
        (0.024, 0.024), (0.024, -0.024),
        (-0.024, 0.024), (-0.024, -0.024),
    )
    probe_steps = 1500
    candidate_audit = []
    selected_qpos = None
    selected_offset = None
    for offset_x, offset_y in offsets:
        _restore_sim(sim, preserved)
        candidate = requested.copy()
        candidate[0] += float(offset_x)
        candidate[1] += float(offset_y)
        libero_client._set_joint_qpos(sim, free_joint, candidate)
        libero_client._zero_joint_velocity(sim, free_joint)
        libero_client._forward_sim(sim)
        for _ in range(probe_steps):
            sim.step()
        probed = np.asarray(
            libero_client._get_joint_qpos(sim, free_joint), dtype=float
        ).copy()
        velocity = _joint_qvel(sim, free_joint).copy()
        orientation_change = quaternion_distance_deg(before[3:], probed[3:])
        orientation_error = quaternion_distance_deg(requested_quat, probed[3:])
        lower_z = libero_client._object_collision_lower_world_z(base_env, object_name)
        support_delta = (
            float(lower_z) - float(support_world_z)
            if lower_z is not None else float("inf")
        )
        support_error = abs(support_delta)
        minimum_contact = (
            min(float(sim.data.contact[index].dist) for index in range(int(sim.data.ncon)))
            if int(sim.data.ncon) else None
        )
        max_orientation_error = 35.0 if morphology == "handled_vessel" else 15.0
        accepted = bool(
            orientation_change >= 30.0
            # A non-axisymmetric mesh may have a stable contact face a few
            # degrees away from the ideal Euler transform.  This is only a
            # candidate-search bound; the independent validator still checks
            # final orientation class, last-window drift, penetration, and
            # task feasibility.
            and orientation_error <= max_orientation_error
            and float(np.linalg.norm(velocity)) <= 0.05
            # Targets initially stacked on small movable objects (for example
            # a bowl on a ramekin) may need to settle onto the underlying table
            # after changing support face.  Permit a recorded downward support
            # transition, but never a comparable upward / floating shift.
            and -0.040 <= support_delta <= 0.015
            and (minimum_contact is None or minimum_contact >= -0.01)
            and not bool(base_env._check_success())
        )
        candidate_audit.append({
            "offset_xy_m": [float(offset_x), float(offset_y)],
            "requested_qpos": candidate.tolist(),
            "probed_qpos": probed.tolist(),
            "probed_qvel_norm": float(np.linalg.norm(velocity)),
            "orientation_change_from_official_deg": orientation_change,
            "orientation_error_from_requested_deg": orientation_error,
            "maximum_orientation_error_deg": max_orientation_error,
            "support_plane_error_m": support_error,
            "support_plane_delta_m": support_delta,
            "minimum_contact_distance_m": minimum_contact,
            "accepted": accepted,
        })
        if accepted:
            selected_qpos = probed
            selected_offset = [float(offset_x), float(offset_y)]
            break
    _restore_sim(sim, preserved)
    if selected_qpos is None:
        raise RuntimeError(
            f"No stable support-state pose for {object_name}: {candidate_audit}"
        )
    libero_client._set_joint_qpos(sim, free_joint, selected_qpos)
    libero_client._zero_joint_velocity(sim, free_joint)
    libero_client._forward_sim(sim)
    return {
        "object": object_name,
        "free_joint": free_joint,
        "morphology": morphology,
        "state": state,
        "rotation_degrees_xyz": list(degrees),
        "before_qpos": before.tolist(),
        "requested_qpos": requested.tolist(),
        "installed_qpos": selected_qpos.tolist(),
        "selected_offset_xy_m": selected_offset,
        "placement_probe_steps": probe_steps,
        "candidate_audit": candidate_audit,
        "placement_contract": "zero_policy_step_probe_restore_stable_pose",
        "support_transition_contract": (
            "same_support_plane"
            if abs(candidate_audit[-1]["support_plane_delta_m"]) <= 0.015
            else "downward_to_underlying_physical_support"
        ),
        "support_world_z_before": float(support_world_z),
        "support_world_z_after_alignment": float(
            libero_client._object_collision_lower_world_z(base_env, object_name)
        ),
    }
