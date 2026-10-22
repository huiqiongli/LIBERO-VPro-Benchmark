"""Physical neutral-object occupancy helpers shared by D2 and D3 runners."""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np


BLOCK_NAME = "occupancy_block_d2"
ROBOT_JOINT_TOKENS = ("robot", "gripper", "panda", "finger")


def _quat_wxyz_to_matrix(quat: np.ndarray) -> np.ndarray:
    w, x, y, z = np.asarray(quat, dtype=float)
    norm = float(np.linalg.norm([w, x, y, z]))
    if norm <= 0.0:
        raise ValueError("zero-norm quaternion")
    w, x, y, z = (value / norm for value in (w, x, y, z))
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=float)


def _matrix_to_quat_wxyz(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=float).reshape(3, 3)
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = np.sqrt(trace + 1.0) * 2.0
        quat = np.array([
            0.25 * scale,
            (matrix[2, 1] - matrix[1, 2]) / scale,
            (matrix[0, 2] - matrix[2, 0]) / scale,
            (matrix[1, 0] - matrix[0, 1]) / scale,
        ])
    else:
        axis = int(np.argmax(np.diag(matrix)))
        if axis == 0:
            scale = np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
            quat = np.array([
                (matrix[2, 1] - matrix[1, 2]) / scale,
                0.25 * scale,
                (matrix[0, 1] + matrix[1, 0]) / scale,
                (matrix[0, 2] + matrix[2, 0]) / scale,
            ])
        elif axis == 1:
            scale = np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
            quat = np.array([
                (matrix[0, 2] - matrix[2, 0]) / scale,
                (matrix[0, 1] + matrix[1, 0]) / scale,
                0.25 * scale,
                (matrix[1, 2] + matrix[2, 1]) / scale,
            ])
        else:
            scale = np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
            quat = np.array([
                (matrix[1, 0] - matrix[0, 1]) / scale,
                (matrix[0, 2] + matrix[2, 0]) / scale,
                (matrix[1, 2] + matrix[2, 1]) / scale,
                0.25 * scale,
            ])
    quat /= np.linalg.norm(quat)
    return quat if quat[0] >= 0.0 else -quat


def destination_frame(base_env, geometry: dict) -> tuple[np.ndarray, np.ndarray]:
    """Return a destination-attached frame for probe/restore placement."""
    if geometry["kind"] == "site":
        return (
            np.asarray(geometry["center"], dtype=float).copy(),
            np.asarray(geometry["rotation"], dtype=float).copy(),
        )
    name = str(geometry["name"])
    if name not in getattr(base_env, "obj_body_id", {}):
        raise ValueError(f"Destination object body is absent: {name}")
    body_id = int(base_env.obj_body_id[name])
    return (
        np.asarray(base_env.sim.data.body_xpos[body_id], dtype=float).copy(),
        np.asarray(base_env.sim.data.body_xmat[body_id], dtype=float).reshape(3, 3).copy(),
    )


def site_geometry(sim, name: str) -> dict | None:
    try:
        site_id = int(sim.model.site_name2id(name))
    except (KeyError, ValueError):
        return None
    if site_id < 0:
        return None
    sim.forward()
    rotation = np.asarray(sim.data.site_xmat[site_id], dtype=float).reshape(3, 3)
    center = np.asarray(sim.data.site_xpos[site_id], dtype=float)
    half_local = np.asarray(sim.model.site_size[site_id], dtype=float)
    half_world = np.abs(rotation) @ half_local
    return {
        "kind": "site",
        "name": name,
        "center": center,
        "rotation": rotation,
        "half_local": half_local,
        "low": center - half_world,
        "high": center + half_world,
    }


def destination_geometry(base_env, libero_client, destination: dict) -> dict:
    name = str(destination["name"])
    if destination["kind"] == "site":
        geometry = site_geometry(base_env.sim, name)
        if geometry is None:
            raise ValueError(f"Destination site disappeared from compiled model: {name}")
        return geometry
    low, high = libero_client._object_collision_aabb_bounds(base_env, name)
    return {
        "kind": "object_aabb",
        "name": name,
        "low": np.asarray(low, dtype=float),
        "high": np.asarray(high, dtype=float),
    }


def block_half_extents(base_env, libero_client) -> np.ndarray:
    low, high = libero_client._object_collision_aabb_bounds(base_env, BLOCK_NAME)
    return 0.5 * (np.asarray(high, dtype=float) - np.asarray(low, dtype=float))


def block_candidate_positions(
    geometry: dict,
    half: np.ndarray,
    *,
    receiver_kind: str,
    freeze_destination: bool = False,
) -> list[np.ndarray]:
    """Place the block at the +X side, reserving the opposite side for target placement."""
    margin = 0.003
    contact_offset = -0.0002
    if geometry["kind"] == "site":
        rotation = np.asarray(geometry["rotation"], dtype=float)
        # Containment sites are not guaranteed to use local Z as world-up
        # (drawer regions commonly rotate local X into world Z).  Select the
        # site axis most aligned with gravity and reserve the other two axes
        # for horizontal occupancy placement.
        vertical_axis = int(np.argmax(np.abs(rotation[2, :])))
        horizontal_axes = [axis for axis in range(3) if axis != vertical_axis]
        block_half_local = np.abs(rotation.T) @ np.asarray(half, dtype=float)
        clearance = {
            axis: max(
                0.0,
                float(geometry["half_local"][axis])
                - float(block_half_local[axis]) - margin,
            )
            for axis in horizontal_axes
        }
        horizontal_axes.sort(key=lambda axis: clearance[axis], reverse=True)
        primary, secondary = horizontal_axes
        fractions = (
            (0.85, 0.0), (0.55, 0.0), (0.0, 0.0),
            (0.0, 0.55), (0.0, -0.55), (-0.55, 0.0),
        )
        desired_z = (
            float(geometry["low"][2]) + float(half[2]) + 0.0002
            if receiver_kind == "enclosed"
            else float(geometry["center"][2]) + float(half[2]) + contact_offset
        )
        positions = []
        for first_fraction, second_fraction in fractions:
            local = np.zeros(3, dtype=float)
            local[primary] = first_fraction * clearance[primary]
            local[secondary] = second_fraction * clearance[secondary]
            vertical_gain = float(rotation[2, vertical_axis])
            if abs(vertical_gain) < 1e-8:
                raise ValueError("destination site has no gravity-aligned local axis")
            horizontal_z = float(np.dot(rotation[2, :], local))
            local[vertical_axis] = (
                desired_z - float(geometry["center"][2]) - horizontal_z
            ) / vertical_gain
            positions.append(geometry["center"] + rotation @ local)
        return positions
    center = 0.5 * (geometry["low"] + geometry["high"])
    # Movable LIBERO plates have a recessed center: their collision-AABB top is
    # the rim, not the resting height of a small object in the center.  Seed at
    # the audited body/AABB center height so the block and plate settle together
    # without an artificial 1 cm relative drop.
    z = (
        float(geometry["high"][2]) + float(half[2]) - 0.0002
        if freeze_destination
        else float(center[2]) - 0.0005
    )
    span_x = max(0.0, 0.5 * float(geometry["high"][0] - geometry["low"][0]) - float(half[0]))
    span_y = max(0.0, 0.5 * float(geometry["high"][1] - geometry["low"][1]) - float(half[1]))
    return [
        np.array([center[0] + fx * span_x, center[1] + fy * span_y, z], dtype=float)
        for fx, fy in (
            (0.38, 0.0), (0.55, 0.0), (0.0, 0.38), (0.0, -0.38),
            (-0.38, 0.0), (0.85, 0.0), (0.0, 0.0),
        )
    ]


def _snapshot(sim) -> dict:
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


def _restore(sim, state: dict) -> None:
    sim.data.time = state["time"]
    sim.data.qpos[:] = state["qpos"]
    sim.data.qvel[:] = state["qvel"]
    if sim.data.act.size:
        sim.data.act[:] = state["act"]
    sim.data.ctrl[:] = state["ctrl"]
    if sim.data.mocap_pos.size:
        sim.data.mocap_pos[:] = state["mocap_pos"]
        sim.data.mocap_quat[:] = state["mocap_quat"]
    try:
        sim.forward()
    except AttributeError:
        pass
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


def _joint_slice(address) -> slice:
    if isinstance(address, (tuple, list)):
        return slice(int(address[0]), int(address[1]))
    return slice(int(address), int(address) + 1)


def _freeze_non_block_state(
    sim,
    preserved: dict,
    *,
    movable_slices: list[tuple[slice, slice]],
) -> None:
    """Keep a private contact probe local to the injected occupancy block.

    Long raw MuJoCo probes must not integrate the policy controller, robot, or
    unrelated task objects.  Preserve the newly integrated state of an explicit
    allow-list (the block and, when movable, its destination) while restoring
    every other generalized coordinate to the authoritative official init
    state.  This helper is used only by D3 specs that opt in; D2 retains its
    already validated unrestricted probe.
    """
    probe_time = float(sim.data.time)
    movable_state = [(
        np.array(sim.data.qpos[qpos_slice], copy=True),
        np.array(sim.data.qvel[qvel_slice], copy=True),
    ) for qpos_slice, qvel_slice in movable_slices]
    sim.data.qpos[:] = preserved["qpos"]
    sim.data.qvel[:] = 0.0
    for (qpos_slice, qvel_slice), (joint_qpos, joint_qvel) in zip(
        movable_slices, movable_state
    ):
        sim.data.qpos[qpos_slice] = joint_qpos
        sim.data.qvel[qvel_slice] = joint_qvel
    if sim.data.act.size:
        sim.data.act[:] = preserved["act"]
    sim.data.ctrl[:] = preserved["ctrl"]
    sim.data.qacc_warmstart[:] = 0.0
    if sim.data.mocap_pos.size:
        sim.data.mocap_pos[:] = preserved["mocap_pos"]
        sim.data.mocap_quat[:] = preserved["mocap_quat"]
    sim.data.time = probe_time
    sim.forward()


def apply_occupancy_block(base_env, libero_client, spec: dict, *, env=None) -> dict:
    geometry = destination_geometry(base_env, libero_client, spec["destination"])
    half = block_half_extents(base_env, libero_client)
    freeze_other_joints = bool(spec.get("placement_probe_freeze_other_joints", False))
    joint = libero_client._free_joint_name(base_env, BLOCK_NAME)
    if joint is None:
        raise ValueError(f"Injected occupancy object has no free joint: {BLOCK_NAME}")
    movable_joint_names = [joint]
    if freeze_other_joints:
        # Preserve all free task-object joints so contacts around a movable
        # receiver (for example a bowl already beside a plate) match the real
        # scene.  Articulated fixtures and the full robot/controller chain stay
        # frozen, which is the source of isolation this D3 probe requires.
        for joint_name in base_env.sim.model.joint_names:
            lowered = str(joint_name).lower()
            if any(token in lowered for token in ROBOT_JOINT_TOKENS):
                continue
            try:
                if np.asarray(
                    libero_client._get_joint_qpos(base_env.sim, joint_name)
                ).size != 7:
                    continue
            except (KeyError, ValueError):
                continue
            if joint_name not in movable_joint_names:
                movable_joint_names.append(joint_name)
    positions = block_candidate_positions(
        geometry,
        half,
        receiver_kind=str(spec["receiver_kind"]),
        freeze_destination=(freeze_other_joints and len(movable_joint_names) == 1),
    )
    selected_index = 0
    position = positions[0]
    preserved = _snapshot(base_env.sim)
    relation = str(spec["primary_goal_states"][0][0])
    accepted = False
    selected_qpos = None
    # Resolve the contact pose in a private MuJoCo probe, then restore the
    # complete official state before installing only the settled block qpos.
    # This consumes zero environment / policy steps and prevents the first
    # model observation from containing an interpenetrating, still-moving
    # occupancy object.
    probe_steps = int(spec.get("placement_probe_steps", 500))
    max_probe_velocity = spec.get("placement_probe_max_velocity_norm")
    max_probe_velocity = (
        None if max_probe_velocity is None else float(max_probe_velocity)
    )
    max_probe_linear_velocity = spec.get("placement_probe_max_linear_velocity_norm")
    max_probe_linear_velocity = (
        None
        if max_probe_linear_velocity is None
        else float(max_probe_linear_velocity)
    )
    full_scene_probe_steps = int(spec.get("placement_full_scene_probe_steps", 0))
    full_scene_max_relative_drift = float(
        spec.get("placement_full_scene_max_relative_drift_m", 0.008)
    )
    if probe_steps < 1:
        raise ValueError("placement_probe_steps must be positive")
    movable_slices = [(
        _joint_slice(base_env.sim.model.get_joint_qpos_addr(joint_name)),
        _joint_slice(base_env.sim.model.get_joint_qvel_addr(joint_name)),
    ) for joint_name in movable_joint_names]
    candidate_audit = []
    for selected_index, candidate in enumerate(positions):
        _restore(base_env.sim, preserved)
        qpos = np.concatenate([candidate, np.array([1.0, 0.0, 0.0, 0.0])])
        libero_client._set_joint_qpos(base_env.sim, joint, qpos)
        libero_client._zero_joint_velocity(base_env.sim, joint)
        libero_client._forward_sim(base_env.sim)
        # Probe contact dynamics directly in MuJoCo, never through env.step().
        # The complete simulator arrays are restored after every probe, so no
        # policy/control step or episode time leaks into the formal initial state.
        for _ in range(probe_steps):
            base_env.sim.step()
            if freeze_other_joints:
                _freeze_non_block_state(
                    base_env.sim,
                    preserved,
                    movable_slices=movable_slices,
                )
        predicate = [relation, BLOCK_NAME, str(spec["destination"]["name"])]
        predicate_true = bool(base_env._eval_predicate(predicate))
        probed_qpos = np.asarray(
            libero_client._get_joint_qpos(base_env.sim, joint), dtype=float
        ).copy()
        probed_geometry = destination_geometry(
            base_env, libero_client, spec["destination"]
        )
        probed_origin, probed_rotation = destination_frame(
            base_env, probed_geometry
        )
        relative_position = probed_rotation.T @ (probed_qpos[:3] - probed_origin)
        relative_rotation = (
            probed_rotation.T @ _quat_wxyz_to_matrix(probed_qpos[3:])
        )
        probed_joint = base_env.sim.model.get_joint_qvel_addr(joint)
        if isinstance(probed_joint, tuple):
            probed_qvel = np.asarray(
                base_env.sim.data.qvel[probed_joint[0]:probed_joint[1]], dtype=float
            ).copy()
        else:
            probed_qvel = np.asarray(
                [base_env.sim.data.qvel[int(probed_joint)]], dtype=float
            )
        geometric_support_occupancy = False
        if relation == "on" and geometry["kind"] == "object_aabb":
            geometric_support_occupancy = bool(
                geometry["low"][0] <= probed_qpos[0] <= geometry["high"][0]
                and geometry["low"][1] <= probed_qpos[1] <= geometry["high"][1]
                and probed_qpos[2] >= geometry["low"][2]
            )
        geometric_site_occupancy = False
        if geometry["kind"] == "site":
            local = geometry["rotation"].T @ (probed_qpos[:3] - geometry["center"])
            geometric_site_occupancy = bool(
                abs(local[0]) <= geometry["half_local"][0]
                and abs(local[1]) <= geometry["half_local"][1]
                and (
                    spec["receiver_kind"] == "support"
                    or abs(local[2]) <= geometry["half_local"][2]
                )
            )
        destination_relation_ok = (
            predicate_true or geometric_support_occupancy or geometric_site_occupancy
        )
        probed_qvel_norm = float(np.linalg.norm(probed_qvel))
        probed_linear_qvel_norm = float(np.linalg.norm(probed_qvel[:3]))
        probe_velocity_ok = (
            max_probe_velocity is None or probed_qvel_norm <= max_probe_velocity
        ) and (
            max_probe_linear_velocity is None
            or probed_linear_qvel_norm <= max_probe_linear_velocity
        )
        accepted_candidate = destination_relation_ok and probe_velocity_ok
        full_scene_audit = None
        if accepted_candidate and full_scene_probe_steps > 0:
            # Re-install the isolated-probe pose into the untouched official
            # scene, then run a short raw MuJoCo stability check with every
            # scene object enabled.  The entire state is restored afterwards,
            # so this selection step consumes no formal environment step.
            _restore(base_env.sim, preserved)
            official_geometry = destination_geometry(
                base_env, libero_client, spec["destination"]
            )
            official_origin, official_rotation = destination_frame(
                base_env, official_geometry
            )
            mapped_position = official_origin + official_rotation @ relative_position
            mapped_orientation = _matrix_to_quat_wxyz(
                official_rotation @ relative_rotation
            )
            libero_client._set_joint_qpos(
                base_env.sim,
                joint,
                np.concatenate([mapped_position, mapped_orientation]),
            )
            libero_client._zero_joint_velocity(base_env.sim, joint)
            libero_client._forward_sim(base_env.sim)
            full_start_geometry = destination_geometry(
                base_env, libero_client, spec["destination"]
            )
            full_start_origin, full_start_rotation = destination_frame(
                base_env, full_start_geometry
            )
            full_start_relative = full_start_rotation.T @ (
                mapped_position - full_start_origin
            )
            for _ in range(full_scene_probe_steps):
                base_env.sim.step()
            full_qpos = np.asarray(
                libero_client._get_joint_qpos(base_env.sim, joint), dtype=float
            ).copy()
            full_end_geometry = destination_geometry(
                base_env, libero_client, spec["destination"]
            )
            full_end_origin, full_end_rotation = destination_frame(
                base_env, full_end_geometry
            )
            full_end_relative = full_end_rotation.T @ (
                full_qpos[:3] - full_end_origin
            )
            full_relative_drift = float(np.linalg.norm(
                full_end_relative - full_start_relative
            ))
            full_predicate_true = bool(base_env._eval_predicate(predicate))
            full_geometric_occupancy = False
            if relation == "on" and full_end_geometry["kind"] == "object_aabb":
                full_geometric_occupancy = bool(
                    full_end_geometry["low"][0] <= full_qpos[0] <= full_end_geometry["high"][0]
                    and full_end_geometry["low"][1] <= full_qpos[1] <= full_end_geometry["high"][1]
                    and full_qpos[2] >= full_end_geometry["low"][2]
                )
            elif full_end_geometry["kind"] == "site":
                local = full_end_geometry["rotation"].T @ (
                    full_qpos[:3] - full_end_geometry["center"]
                )
                full_geometric_occupancy = bool(
                    abs(local[0]) <= full_end_geometry["half_local"][0]
                    and abs(local[1]) <= full_end_geometry["half_local"][1]
                    and (
                        spec["receiver_kind"] == "support"
                        or abs(local[2]) <= full_end_geometry["half_local"][2]
                    )
                )
            full_scene_ok = (
                (full_predicate_true or full_geometric_occupancy)
                and full_relative_drift <= full_scene_max_relative_drift
            )
            full_scene_audit = {
                "steps": full_scene_probe_steps,
                "relative_position_before": full_start_relative.tolist(),
                "relative_position_after": full_end_relative.tolist(),
                "relative_drift_m": full_relative_drift,
                "max_relative_drift_m": full_scene_max_relative_drift,
                "predicate_true": full_predicate_true,
                "geometric_occupancy": full_geometric_occupancy,
                "accepted": full_scene_ok,
            }
            accepted_candidate = accepted_candidate and full_scene_ok
        probe_audit = {
            "candidate_index": int(selected_index),
            "requested_position_world": np.asarray(candidate).tolist(),
            "probed_qpos": probed_qpos.tolist(),
            "probed_qvel": probed_qvel.tolist(),
            "probed_qvel_norm": probed_qvel_norm,
            "probed_linear_qvel_norm": probed_linear_qvel_norm,
            "probe_velocity_limit": max_probe_velocity,
            "probe_linear_velocity_limit": max_probe_linear_velocity,
            "probe_velocity_ok": probe_velocity_ok,
            "full_scene_probe": full_scene_audit,
            "probed_destination_relative_position": relative_position.tolist(),
            "predicate_true": predicate_true,
            "geometric_support_occupancy": geometric_support_occupancy,
            "geometric_site_occupancy": geometric_site_occupancy,
            "accepted": accepted_candidate,
        }
        destination_name = str(spec["destination"]["name"])
        if destination_name in getattr(base_env, "obj_body_id", {}):
            destination_body = np.asarray(
                base_env.sim.data.body_xpos[base_env.obj_body_id[destination_name]],
                dtype=float,
            )
            block_body = np.asarray(
                base_env.sim.data.body_xpos[base_env.obj_body_id[BLOCK_NAME]],
                dtype=float,
            )
            probe_audit.update({
                "destination_body_position_world": destination_body.tolist(),
                "block_body_position_world": block_body.tolist(),
                "body_xy_distance_m": float(
                    np.linalg.norm(destination_body[:2] - block_body[:2])
                ),
                "objects_in_contact": bool(base_env.check_contact(
                    base_env.get_object(destination_name),
                    base_env.get_object(BLOCK_NAME),
                )),
            })
        candidate_audit.append(probe_audit)
        if accepted_candidate:
            selected_qpos = probed_qpos
            selected_relative_position = relative_position.copy()
            selected_relative_rotation = relative_rotation.copy()
            accepted = True
            break
    _restore(base_env.sim, preserved)
    if not accepted:
        # MuJoCo / robosuite builds used by the VLA runners can diverge from
        # the LingBot validation environment when a long unrestricted private
        # probe integrates the whole initialized scene.  In particular, a
        # marginally unstable task object can explode before the injected
        # occupancy block settles, even though the same physical placement is
        # valid.  Retry the *private placement probe only* with unrelated
        # joints frozen.  The installed asset, candidate coordinates, official
        # init state, and formal rollout physics are unchanged.
        if not freeze_other_joints:
            fallback_spec = copy.deepcopy(spec)
            fallback_spec["placement_probe_freeze_other_joints"] = True
            fallback_spec.setdefault(
                "placement_probe_max_linear_velocity_norm", 0.002
            )
            fallback_spec.setdefault("placement_full_scene_probe_steps", 20)
            fallback_spec.setdefault(
                "placement_full_scene_max_relative_drift_m", 0.008
            )
            fallback_audit = apply_occupancy_block(
                base_env, libero_client, fallback_spec, env=env
            )
            fallback_audit["cross_runtime_probe_fallback"] = {
                "used": True,
                "reason": "unrestricted_private_probe_rejected_all_candidates",
                "rejected_candidate_count": len(candidate_audit),
            }
            return fallback_audit
        raise RuntimeError(
            f"No zero-control-step occupancy placement satisfied {relation} "
            f"for {spec['destination']['name']}: {candidate_audit}"
        )
    restored_geometry = destination_geometry(
        base_env, libero_client, spec["destination"]
    )
    restored_origin, restored_rotation = destination_frame(
        base_env, restored_geometry
    )
    position = restored_origin + restored_rotation @ selected_relative_position
    orientation = _matrix_to_quat_wxyz(
        restored_rotation @ selected_relative_rotation
    )
    selected_qpos = np.concatenate([position, orientation])
    libero_client._set_joint_qpos(base_env.sim, joint, selected_qpos)
    libero_client._zero_joint_velocity(base_env.sim, joint)
    libero_client._forward_sim(base_env.sim)
    return {
        "object": BLOCK_NAME,
        "joint": joint,
        "requested_position_world": position.tolist(),
        "installed_qpos": selected_qpos.tolist(),
        "installed_destination_relative_position": selected_relative_position.tolist(),
        "candidate_index": int(selected_index),
        "candidate_count": len(positions),
        "probe_steps": probe_steps,
        "probe_max_velocity_norm": max_probe_velocity,
        "probe_max_linear_velocity_norm": max_probe_linear_velocity,
        "full_scene_probe_steps": full_scene_probe_steps,
        "full_scene_max_relative_drift_m": full_scene_max_relative_drift,
        "probe_freeze_other_joints": freeze_other_joints,
        "probe_movable_joints": movable_joint_names,
        "candidate_audit": candidate_audit,
        "placement_contract": "zero_policy_step_probe_restore_stable_pose",
        "placement_acceptance": (
            "libero_relation_predicate"
            if candidate_audit[selected_index]["predicate_true"]
            else "geometric_destination_footprint"
        ),
        "placement_accepted": True,
        "half_extents_world_aabb": half.tolist(),
        "destination_kind": geometry["kind"],
        "destination_name": geometry["name"],
        "receiver_kind": spec["receiver_kind"],
        "severity_pct": int(spec["severity_pct"]),
        "achieved_nominal_footprint_fraction": float(
            spec["achieved_nominal_footprint_fraction"]
        ),
    }


def env_construction_spec(spec: dict) -> dict:
    """Return the edit subset used while initializing an extended BDDL env."""
    value = copy.deepcopy(spec)
    value["_base_bddl_file_name_for_init_state"] = value["base_bddl_file_name"]
    value.pop("object_pose_overrides", None)
    value["settle_steps"] = 0
    return value


def write_block_xml(
    path: Path,
    half_xyz: tuple[float, float, float],
    *,
    visual_rgba: tuple[float, float, float, float] = (0.36, 0.36, 0.40, 1.0),
    density: float = 220.0,
    friction: tuple[float, float, float] = (0.95, 0.3, 0.1),
) -> None:
    hx, hy, hz = (float(value) for value in half_xyz)
    rgba = " ".join(f"{float(value):.4f}" for value in visual_rgba)
    friction_text = " ".join(f"{float(value):.4f}" for value in friction)
    text = f'''<mujoco model="occupancy_block">
  <worldbody>
    <body>
      <body name="object">
        <geom name="occupancy_visual" type="box" pos="0 0 0" size="{hx:.7f} {hy:.7f} {hz:.7f}" group="1" contype="0" conaffinity="0" rgba="{rgba}"/>
        <geom name="occupancy_collision" type="box" pos="0 0 0" size="{hx:.7f} {hy:.7f} {hz:.7f}" group="0" density="{density:.4f}" friction="{friction_text}" solimp="0.998 0.998 0.001" solref="0.001 1" rgba="0.8 0.8 0.8 0.25"/>
      </body>
      <site rgba="0 0 0 0" size="0.003" pos="0 0 -{hz:.7f}" name="bottom_site"/>
      <site rgba="0 0 0 0" size="0.003" pos="0 0 {hz:.7f}" name="top_site"/>
      <site rgba="0 0 0 0" size="0.003" pos="{hx:.7f} {hy:.7f} 0" name="horizontal_radius_site"/>
    </body>
  </worldbody>
</mujoco>
'''
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
