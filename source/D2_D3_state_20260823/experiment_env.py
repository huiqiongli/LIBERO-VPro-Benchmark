"""Shared, model-independent D2/D3 environment preparation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from d3_obfuscation import (
    evidence_volume_from_bounds,
    joint_subtree_aabb,
    object_aabb,
)
from d3_evidence_suppression_v3 import paired_evidence_volume_v3
from d3_runtime import paired_v5_model_observation, restore_sim, snapshot_sim
from occupancy_runtime import (
    BLOCK_NAME,
    apply_occupancy_block,
    env_construction_spec,
)
from state_runtime import (
    apply_object_support_state,
    qpos_for_open_fraction,
    resolve_normalized_articulations,
)


DUMMY = [0.0] * 7


def object_qpos(base_env, libero_client, name: str) -> np.ndarray:
    joint = libero_client._free_joint_name(base_env, name)
    if joint is None:
        raise ValueError(f"No free joint for {name}")
    return np.asarray(libero_client._get_joint_qpos(base_env.sim, joint), dtype=float)


def step_noop(env, steps: int) -> dict:
    observation = None
    for _ in range(int(steps)):
        observation, _, done, _ = env.step(DUMMY)
        if done:
            raise RuntimeError("Perturbed initial state became successful during settle")
    return observation


def _snapshot_wrapper_state(base_env) -> dict:
    """Capture robosuite counters changed by private counterfactual settling."""
    return {
        "timestep": int(base_env.timestep),
        "cur_time": float(base_env.cur_time),
        "done": bool(base_env.done),
    }


def _restore_private_branch(base_env, sim_state: dict, wrapper_state: dict) -> None:
    """Restore simulator, observation cache, and wrapper clocks exactly."""
    restore_sim(base_env.sim, sim_state)
    for name, value in wrapper_state.items():
        setattr(base_env, name, value)
    # Refresh observables from the restored branch.  Rendering can invoke a
    # forward pass, so reapply authoritative simulator arrays and counters.
    base_env._get_observations(force_update=True)
    restore_sim(base_env.sim, sim_state)
    for name, value in wrapper_state.items():
        setattr(base_env, name, value)


@dataclass
class PreparedEnvironment:
    env: object
    base_env: object
    case: dict
    context: dict
    initial_observation: dict
    initialization_audit: dict

    def observation_for_model(self, libero_client, true_observation: dict) -> tuple[dict, dict]:
        if self.case["track"] == "D2":
            return true_observation, {
                "enabled": True,
                "type": "physical_simulator_state",
                "track": "D2",
                "subtask": self.case["subtask"],
                "initialization_audit": self.initialization_audit,
            }
        if self.case["observation_mode"] == "clean":
            return true_observation, {
                "enabled": False,
                "type": "clean_truth_observation",
                "truth_state": self.case["truth_state"],
            }
        model_observation, audit = paired_v5_model_observation(
            self.base_env,
            libero_client,
            true_observation,
            self.case,
            self.context,
        )
        audit.update({
            "truth_state": self.case["truth_state"],
            "observation_mode": "obfuscated",
            "active_for_full_episode": True,
        })
        return model_observation, audit


def _construct(
    libero_client,
    bddl: str,
    spec: dict | None = None,
    *,
    camera_size: int = 256,
):
    xml_map = libero_client._libero_object_xml_override_map(spec or {})
    env = libero_client.construct_single_env(
        {
            "bddl_file_name": bddl,
            "camera_heights": int(camera_size),
            "camera_widths": int(camera_size),
        },
        object_xml_override_map=xml_map or None,
    )
    if env is None:
        raise RuntimeError(f"Failed to construct LIBERO environment: {bddl}")
    return env


def _prepare_d2(
    libero_client, suite, case: dict, init_state, *, camera_size: int
) -> PreparedEnvironment:
    base_bddl = suite.get_task_bddl_file_path(int(case["task_id"]))
    subtask = case["subtask"]
    if subtask in {"receptacle_occupancy", "support_region_occupancy"}:
        init_spec = env_construction_spec(case)
        env = _construct(
            libero_client, case["bddl_file_name_override"], init_spec,
            camera_size=camera_size,
        )
        libero_client.init_single_env(
            env, init_state, init_spec, None, case["suite"], int(case["task_id"])
        )
        base_env = getattr(env, "env", env)
        parking = object_qpos(base_env, libero_client, BLOCK_NAME).tolist()
        occupancy_audit = apply_occupancy_block(
            base_env, libero_client, case, env=env
        )
        raw = base_env._get_observations(force_update=True)
        audit = {"occupancy": occupancy_audit, "parking_qpos": parking}
    elif subtask == "articulation_precondition":
        env = _construct(libero_client, base_bddl, camera_size=camera_size)
        base_env = getattr(env, "env", env)
        spec = dict(case)
        spec["settle_steps"] = 20
        resolved, articulation_audit = resolve_normalized_articulations(base_env, spec)
        _, _, _ = libero_client.init_single_env(
            env, init_state, resolved, None, case["suite"], int(case["task_id"])
        )
        raw = base_env._get_observations()
        audit = {"articulation": articulation_audit}
    elif subtask == "object_support_state":
        env = _construct(libero_client, base_bddl, camera_size=camera_size)
        env.reset()
        env.set_init_state(init_state)
        base_env = getattr(env, "env", env)
        step_noop(env, 20)
        support_audit = apply_object_support_state(
            base_env,
            libero_client,
            object_name=case["target"],
            morphology=case["morphology"],
            state=case["support_state"],
        )
        raw = step_noop(env, 50)
        audit = {"object_support_state": support_audit}
    else:
        raise ValueError(f"Unsupported D2 subtask: {subtask}")
    if bool(base_env._check_success()):
        env.close()
        raise RuntimeError("D2 prepared environment is already successful")
    return PreparedEnvironment(env, base_env, case, {}, raw, audit)


def _prepare_d3(
    libero_client, suite, case: dict, init_state, *, camera_size: int
) -> PreparedEnvironment:
    base_bddl = suite.get_task_bddl_file_path(int(case["task_id"]))
    kind = case["ambiguity_kind"]
    truth_b = case["truth_state"] == "truth_b"
    occupancy_spec = None
    if kind == "receptacle_occupancy_ambiguity":
        occupancy_spec = json.loads(Path(
            case["truth_b_definition"]["occupancy_spec_path"]
        ).read_text(encoding="utf-8"))
    if kind == "receptacle_occupancy_ambiguity":
        init_spec = env_construction_spec(occupancy_spec)
        env = _construct(
            libero_client, occupancy_spec["bddl_file_name_override"], init_spec,
            camera_size=camera_size,
        )
        libero_client.init_single_env(
            env, init_state, init_spec, None, case["suite"], int(case["task_id"])
        )
    else:
        env = _construct(libero_client, base_bddl, camera_size=camera_size)
        env.reset()
        env.set_init_state(init_state)
    base_env = getattr(env, "env", env)
    step_noop(env, 20)
    context: dict = {}
    audit: dict = {"truth_state": case["truth_state"], "ambiguity_kind": kind}
    if kind == "articulation_state_ambiguity":
        joint = case["joint"]
        branch_state = snapshot_sim(base_env.sim)
        wrapper_state = _snapshot_wrapper_state(base_env)
        step_noop(env, 30)
        truth_a_qpos = float(np.asarray(
            libero_client._get_joint_qpos(base_env.sim, joint), dtype=float
        ).reshape(-1)[0])
        bounds_a = joint_subtree_aabb(base_env, joint)
        _restore_private_branch(base_env, branch_state, wrapper_state)
        fraction = float(case["truth_b_definition"]["open_fraction_pct"]) / 100.0
        alternate = qpos_for_open_fraction(
            base_env.sim.model, joint, case["joint_family"], fraction
        )
        libero_client._set_joint_qpos(base_env.sim, joint, np.array([alternate]))
        libero_client._forward_sim(base_env.sim)
        raw_b = step_noop(env, 30)
        truth_b_qpos = float(np.asarray(
            libero_client._get_joint_qpos(base_env.sim, joint), dtype=float
        ).reshape(-1)[0])
        bounds_b = joint_subtree_aabb(base_env, joint)
        context["evidence_volume"] = evidence_volume_from_bounds(
            kind, [bounds_a, bounds_b], "paired_joint_sweep_union"
        )
        context["paired_endpoints"] = {
            "truth_a": {
                "truth_state": "truth_a", "kind": "articulation", "joint": joint,
                "qpos": [truth_a_qpos], "source": "settled_official_initial_articulation",
            },
            "truth_b": {
                "truth_state": "truth_b", "kind": "articulation", "joint": joint,
                "qpos": [truth_b_qpos], "source": "settled_realistic_open_articulation",
            },
        }
        if truth_b:
            raw = raw_b
            audit["truth_b_joint_qpos"] = alternate
        else:
            _restore_private_branch(base_env, branch_state, wrapper_state)
            raw = step_noop(env, 30)
    elif kind == "object_support_state_ambiguity":
        branch_state = snapshot_sim(base_env.sim)
        wrapper_state = _snapshot_wrapper_state(base_env)
        step_noop(env, 50)
        truth_a_qpos = object_qpos(base_env, libero_client, case["target"])
        bounds_a = object_aabb(base_env, case["target"])
        _restore_private_branch(base_env, branch_state, wrapper_state)
        truth_b_support_audit = apply_object_support_state(
            base_env,
            libero_client,
            object_name=case["target"],
            morphology=case["morphology"],
            state=case["truth_b_definition"]["support_state"],
        )
        raw_b = step_noop(env, 50)
        truth_b_qpos = object_qpos(base_env, libero_client, case["target"])
        bounds_b = object_aabb(base_env, case["target"])
        context["evidence_volume"] = evidence_volume_from_bounds(
            kind, [bounds_a, bounds_b], "paired_target_pose_union"
        )
        target_joint = libero_client._free_joint_name(base_env, case["target"])
        context["paired_endpoints"] = {
            "truth_a": {
                "truth_state": "truth_a", "kind": "object_support", "joint": target_joint,
                "qpos": truth_a_qpos.tolist(), "source": "settled_official_initial_support_pose",
            },
            "truth_b": {
                "truth_state": "truth_b", "kind": "object_support", "joint": target_joint,
                "qpos": truth_b_qpos.tolist(), "source": "settled_realistic_alternative_support_pose",
            },
        }
        if truth_b:
            raw = raw_b
            audit["truth_b_support_state"] = truth_b_support_audit
        else:
            _restore_private_branch(base_env, branch_state, wrapper_state)
            raw = step_noop(env, 50)
    elif kind == "receptacle_occupancy_ambiguity":
        block_joint = libero_client._free_joint_name(base_env, BLOCK_NAME)
        if block_joint is None:
            env.close()
            raise RuntimeError("D3 V5 occupancy environment is missing its paired-state object")
        parking_qpos = object_qpos(base_env, libero_client, BLOCK_NAME)
        parking_qpos[2] = -10.0
        libero_client._set_joint_qpos(base_env.sim, block_joint, parking_qpos)
        libero_client._zero_joint_velocity(base_env.sim, block_joint)
        libero_client._forward_sim(base_env.sim)
        # Build the physical truth-B endpoint in a private branch, then either
        # keep it (truth B) or restore the invisible parking endpoint (truth A).
        branch_state = snapshot_sim(base_env.sim)
        wrapper_state = _snapshot_wrapper_state(base_env)
        occupancy_audit = apply_occupancy_block(
            base_env, libero_client, occupancy_spec, env=env
        )
        occupied_qpos = object_qpos(base_env, libero_client, BLOCK_NAME)
        context["evidence_volume"] = paired_evidence_volume_v3(
            base_env, base_env, case, occupancy_spec=occupancy_spec
        )
        context["paired_endpoints"] = {
            "truth_a": {
                "truth_state": "truth_a", "kind": "occupancy", "joint": block_joint,
                "qpos": parking_qpos.tolist(), "source": "outside_camera_empty_endpoint",
            },
            "truth_b": {
                "truth_state": "truth_b", "kind": "occupancy", "joint": block_joint,
                "qpos": occupied_qpos.tolist(), "source": "validated_realistic_occupancy_endpoint",
            },
        }
        if truth_b:
            audit["truth_b_occupancy"] = occupancy_audit
            raw = base_env._get_observations(force_update=True)
        else:
            _restore_private_branch(base_env, branch_state, wrapper_state)
            raw = base_env._get_observations(force_update=True)
    else:
        env.close()
        raise ValueError(f"Unsupported D3 ambiguity kind: {kind}")
    if bool(base_env._check_success()):
        env.close()
        raise RuntimeError("D3 truth environment is already successful")
    audit["evidence_volume"] = context["evidence_volume"]
    return PreparedEnvironment(
        env,
        base_env,
        case,
        context,
        raw,
        audit,
    )


def prepare_environment(
    libero_client,
    suite,
    case: dict,
    init_state,
    *,
    camera_size: int = 256,
) -> PreparedEnvironment:
    if case["track"] == "D2":
        return _prepare_d2(
            libero_client, suite, case, init_state, camera_size=camera_size
        )
    if case["track"] == "D3":
        return _prepare_d3(
            libero_client, suite, case, init_state, camera_size=camera_size
        )
    raise ValueError(f"Unsupported experiment track: {case.get('track')}")
