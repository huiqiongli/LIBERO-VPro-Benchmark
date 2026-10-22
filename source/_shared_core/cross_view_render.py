"""Single authoritative C1 counterfactual-view implementation."""

from __future__ import annotations

import numpy as np

from source._shared_core.simulator_state import restore_runtime_state, snapshot_runtime_state


def _finite_scalar(value, label: str) -> float:
    array = np.asarray(value, dtype=float)
    if array.size != 1 or not np.all(np.isfinite(array)):
        raise ValueError(f"C1 {label} must be one finite scalar")
    return float(array.reshape(-1)[0])


def _joint_edit(sim, edit, get_joint_qpos) -> tuple[str, float, dict]:
    joint = edit.get("joint")
    if not isinstance(joint, str) or not joint.strip():
        raise ValueError("C1 state edit requires a nonempty joint name")
    current = _finite_scalar(get_joint_qpos(sim, joint), f"joint {joint!r} qpos")
    mode = edit.get("value_mode", "explicit")
    if mode not in ("explicit", "joint_range_mirror", "farthest_joint_limit"):
        raise ValueError(f"Unsupported C1 joint value_mode: {mode!r}")
    low = high = None
    if mode == "explicit":
        if type(edit.get("value")) not in (int, float):
            raise ValueError("C1 explicit joint edit requires a finite numeric value")
        value = _finite_scalar(edit["value"], "joint value")
    elif mode in {"joint_range_mirror", "farthest_joint_limit"}:
        index = int(sim.model.joint_name2id(joint))
        low, high = map(float, sim.model.jnt_range[index])
        if not np.isfinite([low, high]).all() or not high > low:
            raise ValueError(f"C1 joint {joint!r} has invalid range [{low}, {high}]")
        if mode == "farthest_joint_limit":
            value = low if abs(current - low) >= abs(high - current) else high
        else:
            value = low + high - current
        value = float(np.clip(value, low, high))
    else:
        raise ValueError(f"Unsupported C1 joint value_mode: {mode!r}")
    return joint, value, {
        "joint": joint, "op": "state", "value_mode": mode,
        "original_value": current, "value": value,
        "joint_range": [low, high] if low is not None else None, "ok": True,
    }


def apply_cross_view_object_edit(
    raw_obs, env_in, spec, *, normalize_views, view_configs,
    free_joint_name, get_joint_qpos, set_joint_qpos,
):
    """Render edits only in selected views; never advance the simulator.

    Both shared and C1-specific helper entrypoints delegate here. All mutable
    integration inputs and episode clocks are restored on success or error.
    """
    base_env = getattr(env_in, "env", env_in)
    sim = base_env.sim
    views = normalize_views(spec.get("views"))
    edits = spec.get("object_edits") or spec.get("edits") or []
    if not isinstance(edits, (list, tuple)) or not edits:
        raise ValueError("object_edit perturbation requires nonempty object_edits")
    before = snapshot_runtime_state(base_env)
    audit = []
    try:
        for edit in edits:
            if not isinstance(edit, dict):
                raise ValueError("C1 object_edits entries must be objects")
            operation = str(edit.get("op", "")).lower()
            if operation in {"disappear", "remove", "hide", "relocate", "move"}:
                name = str(edit["object"])
                joint = free_joint_name(base_env, name)
                if joint is None:
                    raise ValueError(f"C1 object {name!r} has no movable free joint")
                qpos = np.array(get_joint_qpos(sim, joint), dtype=float, copy=True)
                if qpos.shape != (7,):
                    raise ValueError(f"C1 object {name!r} root is not a free joint")
                if operation in {"disappear", "remove", "hide"}:
                    qpos[2] = _finite_scalar(edit.get("hidden_z", -10.0), "hidden_z")
                    canonical_op = "disappear"
                else:
                    offset = np.asarray(edit.get("offset", [0.12, 0.0, 0.0]), dtype=float)
                    if offset.shape != (3,) or not np.isfinite(offset).all():
                        raise ValueError("C1 relocation offset must contain three finite values")
                    qpos[:3] += offset
                    canonical_op = "relocate"
                set_joint_qpos(sim, joint, qpos)
                audit.append({"object": name, "op": canonical_op, "ok": True})
            elif operation in {"state", "set_joint"}:
                joint, value, entry = _joint_edit(sim, edit, get_joint_qpos)
                set_joint_qpos(sim, joint, np.array([value], dtype=float))
                audit.append(entry)
            else:
                raise ValueError(f"Unsupported C1 object edit operation: {operation}")
        sim.forward()
        output = dict(raw_obs)
        view_stats = {}
        for view in views:
            key = view_configs[view]["image_key"]
            source = np.asarray(raw_obs[key])
            if source.ndim != 3 or source.shape[-1] != 3:
                raise ValueError(f"C1 expected HWC RGB observation for {key}")
            height, width = source.shape[:2]
            edited = np.asarray(sim.render(camera_name=view, width=width, height=height))
            if edited.shape != source.shape:
                raise ValueError(f"C1 counterfactual render shape mismatch for {view}")
            changed_pixels = int(np.count_nonzero(np.any(edited != source, axis=-1)))
            output[key] = np.array(edited, copy=True)
            view_stats[view] = {
                "changed_pixels": changed_pixels,
                "mask_fraction": changed_pixels / float(height * width), "edits": audit,
            }
    finally:
        restore_runtime_state(base_env, before)
    # This audit is reachable only after exact restoration succeeds.
    return output, {
        "enabled": True, "target_policy": "object_edit",
        "target_objects": [str(item["object"]) for item in edits if item.get("object")],
        "mask_source": "object_edit", "mask_geometry": "sim_perview", "views": views,
        "object_edits": audit, "view_stats": view_stats,
        "physical_rollout_unchanged": True, "sim_state_restored": True,
        "exact_state_restore": True, "state_restore_differences": {},
    }
