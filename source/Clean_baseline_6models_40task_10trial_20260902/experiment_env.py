"""Clean LIBERO environment plus temporary C1 counterfactual camera rendering."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import robust_libero_perturbations as c3_runtime


DUMMY = [0.0] * 7


def _snapshot(base_env) -> dict:
    sim = base_env.sim
    return {
        "qpos": np.array(sim.data.qpos, copy=True),
        "qvel": np.array(sim.data.qvel, copy=True),
        "timestep": int(getattr(base_env, "timestep", 0)),
    }


def _same_state(base_env, before: dict) -> bool:
    sim = base_env.sim
    return bool(
        np.allclose(np.asarray(sim.data.qpos), before["qpos"], rtol=0.0, atol=1e-10)
        and np.allclose(np.asarray(sim.data.qvel), before["qvel"], rtol=0.0, atol=1e-10)
        and int(getattr(base_env, "timestep", 0)) == before["timestep"]
    )


@dataclass
class PreparedEnvironment:
    env: object
    base_env: object
    case: dict
    initial_observation: dict
    initialization_audit: dict

    def observation_for_model(self, _libero_client, true_observation: dict) -> tuple[dict, dict]:
        # Clean baseline: return the simulator observation untouched.  Keeping
        # this explicit branch prevents any perturbation helper from
        # being inherited by the baseline runner.
        if self.case.get("track") == "CLEAN":
            return true_observation, {
                "enabled": False,
                "clean_observation": True,
                "sim_state_restored": True,
                "physical_rollout_unchanged": True,
            }
        before = _snapshot(self.base_env)
        model_observation, audit = c3_runtime.apply_obs_noise_to_raw_obs(
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
            "direction": self.case["direction"],
            "truth_view": self.case["truth_view"],
        })
        return model_observation, audit


def prepare_environment(
    libero_client,
    suite,
    case: dict,
    init_state,
    *,
    camera_size: int = 256,
) -> PreparedEnvironment:
    if case.get("track") == "CLEAN":
        task_id = int(case["task_id"])
        bddl = suite.get_task_bddl_file_path(task_id)
        env = libero_client.construct_single_env({
            "bddl_file_name": bddl,
            "camera_heights": int(camera_size),
            "camera_widths": int(camera_size),
        })
        if env is None:
            raise RuntimeError(f"Failed to construct clean LIBERO environment: {bddl}")
        env.reset()
        observation = env.set_init_state(init_state)
        for _ in range(5):
            observation, _, done, _ = env.step(DUMMY)
            if done:
                env.close()
                raise RuntimeError("Clean baseline initial state became successful during settle")
        audit = {
            "clean_official_init_state": True,
            "settle_steps": 5,
            "environment_edited": False,
            "observation_perturbation": False,
            "physical_rollout_unchanged": True,
        }
        return PreparedEnvironment(env, getattr(env, "env", env), case, observation, audit)
    if case.get("track") != "C1" or case.get("mask_source") != "object_edit":
        raise ValueError(f"Unsupported isolated C1 case: {case.get('case_id')}")
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
            raise RuntimeError("Clean C1 initial state became successful during settle")
    base_env = getattr(env, "env", env)
    before = _snapshot(base_env)
    _, preflight = c3_runtime.apply_obs_noise_to_raw_obs(
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
