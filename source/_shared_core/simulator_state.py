"""Exact restoration of simulation inputs after a private render-only branch."""

from __future__ import annotations

import copy

import numpy as np


# MjSim.get_state() omits solver warm starts, controls, mocap and applied forces.
# These are integration inputs, not merely derived rendering quantities.
STATE_ARRAYS = (
    "qpos", "qvel", "act", "ctrl", "qacc_warmstart", "mocap_pos", "mocap_quat",
    "qfrc_applied", "xfrc_applied", "userdata", "plugin_state",
)
WRAPPER_FIELDS = ("timestep", "cur_time", "done")


def snapshot_runtime_state(base_env) -> dict:
    data = base_env.sim.data
    return {
        "time": float(data.time),
        "arrays": {
            name: np.array(getattr(data, name), copy=True)
            for name in STATE_ARRAYS if hasattr(data, name)
        },
        "wrapper": {
            name: copy.deepcopy(getattr(base_env, name))
            for name in WRAPPER_FIELDS if hasattr(base_env, name)
        },
    }


def runtime_state_differences(base_env, state: dict) -> dict:
    data = base_env.sim.data
    differences = {}
    if float(data.time) != state["time"]:
        differences["time"] = {"expected": state["time"], "actual": float(data.time)}
    for name, expected in state["arrays"].items():
        actual = np.asarray(getattr(data, name))
        if np.array_equal(actual, expected, equal_nan=True):
            continue
        if actual.shape != expected.shape:
            differences[name] = {"expected_shape": expected.shape, "actual_shape": actual.shape}
            continue
        finite = np.isfinite(actual) & np.isfinite(expected)
        differences[name] = {
            "max_abs_delta": float(np.max(np.abs(actual[finite] - expected[finite])))
            if np.any(finite) else None,
        }
    for name, expected in state["wrapper"].items():
        actual = getattr(base_env, name)
        if actual != expected:
            differences[name] = {"expected": expected, "actual": actual}
    return differences


def restore_runtime_state(base_env, state: dict) -> None:
    """Restore kinematics, then reapply integration inputs byte-exactly.

    mj_forward refreshes derived geometry but can modify warm starts and
    normalize quaternion coordinates. Reapplying saved arrays afterwards
    preserves the exact inputs to the next real physics step.
    """
    data = base_env.sim.data

    def assign():
        data.time = state["time"]
        for name, value in state["arrays"].items():
            getattr(data, name)[:] = value
        for name, value in state["wrapper"].items():
            setattr(base_env, name, copy.deepcopy(value))

    assign()
    try:
        base_env.sim.forward()
    finally:
        assign()
    differences = runtime_state_differences(base_env, state)
    if differences:
        raise RuntimeError(f"Private render failed exact state restoration: {differences}")
