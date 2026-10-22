"""Render-only state ambiguity with exact simulator restoration."""

from __future__ import annotations

import copy

import numpy as np

from d3_paired_state_exposure_v5 import paired_state_exposure_v5
from occupancy_runtime import BLOCK_NAME


ROBOT_JOINT_TOKENS = ("robot", "gripper", "panda", "finger")


def canonical_scene_joint_qpos(base_env, libero_client) -> dict[str, list[float]]:
    """Capture all named non-robot joints for a canonical render-only scene."""
    model = base_env.sim.model
    result: dict[str, list[float]] = {}
    for joint_id in range(int(model.njnt)):
        name = model.joint_id2name(joint_id)
        if not name:
            continue
        lowered = str(name).lower()
        if any(token in lowered for token in ROBOT_JOINT_TOKENS):
            continue
        value = np.asarray(
            libero_client._get_joint_qpos(base_env.sim, str(name)), dtype=float
        ).reshape(-1)
        result[str(name)] = value.tolist()
    return result


def _apply_canonical_scene(base_env, libero_client, context: dict) -> int:
    named = context.get("canonical_scene_joint_qpos") or {}
    applied = 0
    for name, value in named.items():
        try:
            libero_client._set_joint_qpos(
                base_env.sim, name, np.asarray(value, dtype=float)
            )
        except (KeyError, ValueError):
            # Truth-B occupancy adds one joint; truth-A canonical dictionaries
            # intentionally contain only the common base-task joint names.
            continue
        applied += 1
    libero_client._forward_sim(base_env.sim)
    return applied


def snapshot_sim(sim) -> dict:
    return {
        "time": float(sim.data.time),
        "qpos": np.array(sim.data.qpos, copy=True),
        "qvel": np.array(sim.data.qvel, copy=True),
        "act": np.array(sim.data.act, copy=True),
        "ctrl": np.array(sim.data.ctrl, copy=True),
        "qacc_warmstart": np.array(sim.data.qacc_warmstart, copy=True),
        "mocap_pos": np.array(sim.data.mocap_pos, copy=True),
        "mocap_quat": np.array(sim.data.mocap_quat, copy=True),
    }


def restore_sim(sim, state: dict) -> None:
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
    # mj_forward can normalize free-joint quaternions at machine precision and
    # recomputes the solver warm-start cache.  Reapply the authoritative arrays
    # without another forward pass so the next real policy step starts from the
    # byte-exact truth state; the derived kinematic cache already corresponds to
    # the same state up to that normalization roundoff.
    sim.data.time = state["time"]
    sim.data.qpos[:] = state["qpos"]
    sim.data.qvel[:] = state["qvel"]
    if sim.data.act.size:
        sim.data.act[:] = state["act"]
    sim.data.ctrl[:] = state["ctrl"]
    sim.data.qacc_warmstart[:] = state["qacc_warmstart"]
    if sim.data.mocap_pos.size:
        sim.data.mocap_pos[:] = state["mocap_pos"]
        sim.data.mocap_quat[:] = state["mocap_quat"]


def state_restore_differences(sim, state: dict) -> dict:
    """Return exact differences in policy-relevant MuJoCo state arrays."""
    current = snapshot_sim(sim)
    differences = {}
    for key in (
        "qpos", "qvel", "act", "ctrl", "qacc_warmstart",
        "mocap_pos", "mocap_quat",
    ):
        expected = np.asarray(state[key])
        actual = np.asarray(current[key])
        if np.array_equal(actual, expected, equal_nan=True):
            continue
        finite = np.isfinite(actual) & np.isfinite(expected)
        max_abs = (
            float(np.max(np.abs(actual[finite] - expected[finite])))
            if np.any(finite) else None
        )
        differences[key] = {
            "changed_values": int(np.count_nonzero(~np.isclose(
                actual, expected, rtol=0.0, atol=0.0, equal_nan=True
            ))),
            "max_abs_delta": max_abs,
            "actual_has_nonfinite": bool(np.any(~np.isfinite(actual))),
            "expected_has_nonfinite": bool(np.any(~np.isfinite(expected))),
        }
    if float(current["time"]) != float(state["time"]):
        differences["time"] = {
            "expected": float(state["time"]),
            "actual": float(current["time"]),
        }
    return differences


def _wrapper_clock_snapshot(base_env) -> dict:
    """Capture the episode clocks touched by a forced private render."""
    return {
        name: copy.deepcopy(getattr(base_env, name))
        for name in ("timestep", "cur_time", "done")
        if hasattr(base_env, name)
    }


def _restore_wrapper_clocks(base_env, state: dict) -> None:
    for name, value in state.items():
        setattr(base_env, name, value)


def _install_paired_endpoint(base_env, libero_client, endpoint: dict) -> dict:
    """Install one already validated endpoint without advancing physics.

    Endpoint poses are calibrated during environment preparation.  Runtime
    replans only install the stored pose long enough to render both cameras;
    they never repeat the settling probe and never execute an environment step.
    """
    joint = str(endpoint["joint"])
    qpos = np.asarray(endpoint["qpos"], dtype=float)
    libero_client._set_joint_qpos(base_env.sim, joint, qpos)
    if endpoint["kind"] in {"object_support", "occupancy"}:
        libero_client._zero_joint_velocity(base_env.sim, joint)
    libero_client._forward_sim(base_env.sim)
    return {
        "endpoint_truth_state": str(endpoint["truth_state"]),
        "kind": str(endpoint["kind"]),
        "joint": joint,
        "qpos": qpos.tolist(),
        "source": str(endpoint.get("source", "validated_initialization_endpoint")),
    }


def paired_v5_model_observation(
    base_env,
    libero_client,
    true_observation: dict,
    spec: dict,
    context: dict,
) -> tuple[dict, dict]:
    """Render the other V5 endpoint privately and restore truth byte-exactly.

    The active simulator state supplies physics, rewards, proprioception and
    one RGB endpoint.  The other endpoint is rendered at the same robot pose,
    camera pose and replan instant.  Only the RGB pixels inside the shared
    world-space evidence volume are replaced by the symmetric 50/50 exposure.
    """
    if spec.get("ambiguity_schedule") != {
        "mode": "full_episode",
        "operator": "symmetric_paired_state_exposure_v5",
    }:
        raise ValueError(f"D3 V5 runner received a non-V5 case: {spec.get('case_id')}")
    active_truth = str(spec["truth_state"])
    other_truth = "truth_b" if active_truth == "truth_a" else "truth_a"
    endpoints = context.get("paired_endpoints") or {}
    if set(endpoints) != {"truth_a", "truth_b"}:
        raise ValueError(
            f"D3 V5 endpoints are incomplete for {spec.get('case_id')}: "
            f"{sorted(endpoints)}"
        )

    sim_before = snapshot_sim(base_env.sim)
    wrapper_before = _wrapper_clock_snapshot(base_env)
    alternate_raw = None
    endpoint_audit = None
    try:
        endpoint_audit = _install_paired_endpoint(
            base_env, libero_client, endpoints[other_truth]
        )
        alternate_raw = base_env._get_observations(force_update=True)
        # Own the arrays before the truth cache is restored below.
        alternate_raw = {
            key: np.array(value, copy=True) if isinstance(value, np.ndarray) else copy.deepcopy(value)
            for key, value in alternate_raw.items()
        }
    finally:
        restore_sim(base_env.sim, sim_before)
        _restore_wrapper_clocks(base_env, wrapper_before)
        # Put truth RGB back into robosuite's observable cache, then reapply the
        # authoritative arrays because rendering can call mj_forward.
        base_env._get_observations(force_update=True)
        restore_sim(base_env.sim, sim_before)
        _restore_wrapper_clocks(base_env, wrapper_before)

    restore_differences = state_restore_differences(base_env.sim, sim_before)
    if restore_differences:
        raise RuntimeError(
            "D3 V5 private endpoint render failed exact simulator restoration: "
            f"{restore_differences}"
        )
    if alternate_raw is None:
        raise RuntimeError("D3 V5 alternate endpoint did not render")

    if active_truth == "truth_a":
        changed_a, audit_a, _, _ = paired_state_exposure_v5(
            base_env,
            true_observation,
            base_env,
            alternate_raw,
            context["evidence_volume"],
            profile_name="balanced",
        )
        model_observation, paired_audit = changed_a, audit_a
    else:
        _, _, changed_b, audit_b = paired_state_exposure_v5(
            base_env,
            alternate_raw,
            base_env,
            true_observation,
            context["evidence_volume"],
            profile_name="balanced",
        )
        model_observation, paired_audit = changed_b, audit_b
    paired_audit.update({
        "active_truth": active_truth,
        "temporarily_rendered_truth": other_truth,
        "temporary_endpoint": endpoint_audit,
        "temporary_render_policy_steps": 0,
        "exact_state_restore": True,
        "state_restore_differences": {},
        "active_for_full_episode": True,
    })
    return model_observation, paired_audit


def object_segmentation_pixels(
    base_env,
    object_name: str,
    *,
    width: int = 256,
    height: int = 256,
) -> dict[str, int]:
    """Count pixels of one object in both official camera views.

    MuJoCo's segmentation buffer reports the physical geom ids directly, so
    this audit is independent of object colour, illumination, antialiasing,
    and background texture.  It is used only when explicitly requested by a
    validator; normal policy inference does not pay for segmentation renders.
    """
    sim = base_env.sim
    geom_ids = {
        geom_id
        for geom_id in range(int(sim.model.ngeom))
        if str(sim.model.geom_id2name(geom_id) or "").startswith(object_name)
    }
    if not geom_ids:
        return {"agentview": 0, "wrist": 0}
    counts: dict[str, int] = {}
    for view, camera_name in (
        ("agentview", "agentview"),
        ("wrist", "robot0_eye_in_hand"),
    ):
        segmentation = np.asarray(sim.render(
            width=int(width),
            height=int(height),
            camera_name=camera_name,
            segmentation=True,
        ))
        geom_channel = segmentation[:, :, 1]
        counts[view] = int(sum(
            np.count_nonzero(geom_channel == geom_id) for geom_id in geom_ids
        ))
    return counts


def _set_canonical_state(base_env, libero_client, spec: dict, context: dict) -> dict:
    kind = str(spec["ambiguity_kind"])
    sim = base_env.sim
    common_joint_count = _apply_canonical_scene(base_env, libero_client, context)
    if kind == "articulation_state_ambiguity":
        joint = str(spec["joint"])
        qpos = float(context["canonical_joint_qpos"])
        libero_client._set_joint_qpos(sim, joint, np.array([qpos], dtype=float))
        libero_client._forward_sim(sim)
        return {"joint": joint, "canonical_qpos": qpos,
                "canonical_scene_joint_count": common_joint_count}
    if kind == "object_support_state_ambiguity":
        target = str(spec["target"])
        joint = libero_client._free_joint_name(base_env, target)
        qpos = np.asarray(context["canonical_target_qpos"], dtype=float)
        libero_client._set_joint_qpos(sim, joint, qpos)
        libero_client._zero_joint_velocity(sim, joint)
        libero_client._forward_sim(sim)
        return {"target": target, "joint": joint, "canonical_qpos": qpos.tolist(),
                "canonical_scene_joint_count": common_joint_count}
    if kind == "receptacle_occupancy_ambiguity":
        # The truth-B extended environment keeps its injected object, but the
        # canonical empty rendering returns that object to its BDDL parking pose.
        if BLOCK_NAME not in getattr(base_env, "obj_body_id", {}):
            return {"object": BLOCK_NAME, "present": False, "canonical": "empty",
                    "canonical_scene_joint_count": common_joint_count}
        joint = libero_client._free_joint_name(base_env, BLOCK_NAME)
        qpos = np.asarray(context["canonical_block_parking_qpos"], dtype=float)
        libero_client._set_joint_qpos(sim, joint, qpos)
        libero_client._zero_joint_velocity(sim, joint)
        libero_client._forward_sim(sim)
        return {
            "object": BLOCK_NAME,
            "joint": joint,
            "present": True,
            "canonical": "empty_parking_pose",
            "canonical_qpos": qpos.tolist(),
            "canonical_scene_joint_count": common_joint_count,
        }
    raise ValueError(f"Unsupported D3 ambiguity kind: {kind}")


def canonical_model_observation(
    base_env,
    libero_client,
    true_observation: dict,
    spec: dict,
    context: dict,
) -> tuple[dict, dict]:
    """Render both cameras from one temporary canonical state, then restore truth.

    Only raw RGB camera arrays are replaced.  Proprioception and every other
    observation field stay tied to the true state.
    """
    sim = base_env.sim
    before = snapshot_sim(sim)
    try:
        canonical_audit = _set_canonical_state(base_env, libero_client, spec, context)
        # Direct qpos edits do not advance the observable clock.  Force both
        # camera observables to render from the temporary canonical state;
        # otherwise robosuite returns the cached truth-state frames.
        canonical_raw = base_env._get_observations(force_update=True)
        model_observation = copy.deepcopy(true_observation)
        for key in ("agentview_image", "robot0_eye_in_hand_image"):
            model_observation[key] = np.array(canonical_raw[key], copy=True)
        if (
            spec["ambiguity_kind"] == "receptacle_occupancy_ambiguity"
            and bool(context.get("capture_segmentation_audit", False))
        ):
            height, width = np.asarray(canonical_raw["agentview_image"]).shape[:2]
            canonical_audit["segmentation_pixels"] = object_segmentation_pixels(
                base_env, BLOCK_NAME, width=int(width), height=int(height)
            )
    finally:
        restore_sim(sim, before)
        # Restore observable caches as well as MuJoCo arrays, so a later clean
        # replan cannot inherit canonical RGB from this temporary render.
        base_env._get_observations(force_update=True)
        # Observable refreshes may invoke mj_forward and normalize quaternions;
        # restore once more after the truth frames have entered the cache.
        restore_sim(sim, before)
    restore_differences = state_restore_differences(sim, before)
    if restore_differences:
        raise RuntimeError(
            "D3 temporary render failed exact simulator-state restoration: "
            f"{restore_differences}"
        )
    return model_observation, {
        "enabled": True,
        "type": "render_time_canonical_state",
        "ambiguity_kind": spec["ambiguity_kind"],
        "camera_render_contract": "both_official_cameras_same_temporary_state",
        "proprioception_source": "true_state",
        "physics_source": "true_state",
        "exact_state_restore": True,
        "state_restore_differences": {},
        "canonical_state": canonical_audit,
    }


class RevealLatch:
    """Reveal true visual evidence once the gripper reaches the relevant entity."""

    def __init__(self, threshold_m: float = 0.10):
        self.threshold_m = float(threshold_m)
        self.revealed = False

    def update(self, base_env, entity_name: str) -> tuple[bool, float]:
        if entity_name not in getattr(base_env, "obj_body_id", {}):
            raise ValueError(f"D3 reveal entity is absent: {entity_name}")
        sim = base_env.sim
        entity_pos = np.asarray(
            sim.data.body_xpos[int(base_env.obj_body_id[entity_name])], dtype=float
        )
        eef_site = int(sim.model.site_name2id("gripper0_grip_site"))
        eef_pos = np.asarray(sim.data.site_xpos[eef_site], dtype=float)
        distance = float(np.linalg.norm(entity_pos - eef_pos))
        self.revealed = self.revealed or distance <= self.threshold_m
        return self.revealed, distance
