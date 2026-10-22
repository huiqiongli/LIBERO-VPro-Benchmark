"""Clean LIBERO environment plus temporary C1 counterfactual camera rendering."""

from __future__ import annotations

from dataclasses import dataclass

import robust_libero_perturbations as c1_runtime
from source._shared_core.simulator_state import (
    runtime_state_differences,
    snapshot_runtime_state,
)


DUMMY = [0.0] * 7


def _snapshot(base_env) -> dict:
    return snapshot_runtime_state(base_env)


def _same_state(base_env, before: dict) -> bool:
    return not runtime_state_differences(base_env, before)


@dataclass
class PreparedEnvironment:
    env: object
    base_env: object
    case: dict
    initial_observation: dict
    initialization_audit: dict

    def observation_for_model(self, _libero_client, true_observation: dict) -> tuple[dict, dict]:
        before = _snapshot(self.base_env)
        model_observation, audit = c1_runtime.apply_obs_noise_to_raw_obs(
            true_observation,
            self.env,
            self.case,
            self.case["suite"],
            int(self.case["task_id"]),
        )
        if not _same_state(self.base_env, before):
            raise RuntimeError("C1 counterfactual render changed the physical rollout state")
        audit = dict(audit)
        audit.update({
            "sim_state_restored": True,
            "physical_rollout_unchanged": True,
        })
        if self.case.get("track") == "C1":
            audit.update({"direction": self.case["direction"], "truth_view": self.case["truth_view"]})
        else:
            audit.update({"interaction_parts": self.case.get("interaction_parts", [])})
        return model_observation, audit


def prepare_environment(
    libero_client,
    suite,
    case: dict,
    init_state,
    *,
    camera_size: int = 256,
) -> PreparedEnvironment:
    supported = (
        case.get("track") == "C1" and case.get("mask_source") == "object_edit"
    ) or (
        case.get("track") == "A3" and case.get("mask_source") == "interaction_part"
    )
    if not supported:
        raise ValueError(f"Unsupported isolated visual case: {case.get('case_id')}")
    task_id = int(case["task_id"])
    bddl = suite.get_task_bddl_file_path(task_id)
    env = libero_client.construct_single_env({
        "bddl_file_name": bddl,
        "camera_heights": int(camera_size),
        "camera_widths": int(camera_size),
    })
    if env is None:
        raise RuntimeError(f"Failed to construct LIBERO environment: {bddl}")
    env.reset()
    observation = env.set_init_state(init_state)
    # Match the official LingBot initialization path: five no-op settle steps.
    for _ in range(5):
        observation, _, done, _ = env.step(DUMMY)
        if done:
            env.close()
            raise RuntimeError("Clean initial state became successful during settle")
    base_env = getattr(env, "env", env)
    before = _snapshot(base_env)
    _, preflight = c1_runtime.apply_obs_noise_to_raw_obs(
        observation, env, case, case["suite"], task_id
    )
    if not _same_state(base_env, before):
        env.close()
        raise RuntimeError("C1 initialization preflight changed the simulator state")
    audit = {
        "clean_official_init_state": True,
        "settle_steps": 5,
        "counterfactual_render_preflight": preflight,
        "physical_rollout_unchanged": True,
        "sim_state_restored": True,
    }
    return PreparedEnvironment(env, base_env, case, observation, audit)
