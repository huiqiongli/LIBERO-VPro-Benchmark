"""Sidecar episode recording for LIBERO-VPro analysis.

The recorder is deliberately passive:

* disabled unless ``WAM_RECORD_EPISODES=1``;
* never invokes a model or mutates model inputs;
* copies values to host memory before writing;
* stores large arrays outside JSON so records remain inspectable.

Evaluation adapters should call :func:`make_episode_recorder` once per episode,
then ``record_replan`` / ``record_control`` and finally ``finish``.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import re
import tempfile
import uuid
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np


SCHEMA_VERSION = "wam_syncbench_episode_record.v2"
RECORDER_VERSION = "2.0.0"
TRUE_VALUES = {"1", "true", "yes", "y", "on"}
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in TRUE_VALUES


def recording_enabled() -> bool:
    return _as_bool(
        os.getenv(
            "WAM_RECORD_EPISODES",
            os.getenv("WAM_EPISODE_RECORDING", "0"),
        )
    )


def _slug(value: Any, maximum: int = 80) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "")).strip("_.")
    return (text or "unknown")[:maximum]


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "detach"):
        try:
            return value.detach().cpu().numpy().tolist()
        except Exception:
            pass
    return str(value)


def _array(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    elif hasattr(value, "convert") and hasattr(value, "size"):
        value = np.asarray(value)
    else:
        value = np.asarray(value)
    if value.dtype == object:
        return None
    return np.array(value, copy=True)


def _load_json_or_path(raw: str | None) -> dict[str, Any]:
    text = str(raw or "").strip()
    if not text:
        return {}
    try:
        path = Path(os.path.expandvars(os.path.expanduser(text)))
        if path.is_file():
            value = json.loads(path.read_text(encoding="utf-8"))
            return value if isinstance(value, dict) else {"value": value}
        value = json.loads(text)
        return value if isinstance(value, dict) else {"value": value}
    except Exception:
        return {"source": text}


def load_first_env_spec(names: Iterable[str]) -> tuple[dict[str, Any], str]:
    for name in names:
        raw = os.getenv(name, "").strip()
        if raw:
            return _load_json_or_path(raw), name
    return {}, ""


def _hash_json(value: Any) -> str:
    payload = json.dumps(
        _jsonable(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _spec_has_effect(value: Any) -> bool:
    if value in (None, False, "", [], {}):
        return False
    if isinstance(value, Mapping):
        if "enabled" in value and not _as_bool(value.get("enabled")):
            return False
        return any(_spec_has_effect(item) for item in value.values())
    return True


def _view_arrays(prefix: str, observations: Any) -> dict[str, np.ndarray]:
    """Extract camera tensors without changing their orientation or dtype."""
    if observations is None:
        return {}
    if not isinstance(observations, Mapping):
        array = _array(observations)
        return {} if array is None else {prefix: array}

    aliases = {
        "third_person": (
            "agentview_image",
            "observation.images.agentview_rgb",
            "image",
            "primary_image",
            "agentview",
            "third_person",
        ),
        "wrist": (
            "robot0_eye_in_hand_image",
            "observation.images.eye_in_hand_rgb",
            "wrist_image",
            "wrist",
            "eye_in_hand",
        ),
    }
    arrays: dict[str, np.ndarray] = {}
    for view_name, keys in aliases.items():
        for key in keys:
            if key not in observations:
                continue
            value = observations[key]
            if isinstance(value, (list, tuple)) and len(value) == 1:
                value = value[0]
            array = _array(value)
            if array is not None:
                arrays[f"{prefix}_{view_name}"] = array
            break
    return arrays


def _artifact_ref(relative_path: str, key: str, array: np.ndarray) -> dict[str, Any]:
    return {
        "path": relative_path,
        "key": key,
        "shape": list(array.shape),
        "dtype": str(array.dtype),
    }


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(_jsonable(value), handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _object_names_from_spec(env_edit_spec: Mapping[str, Any] | None) -> list[str]:
    spec = dict(env_edit_spec or {})
    definition = spec.get("definition") or {}
    names = [
        definition.get("target"),
        definition.get("distractor"),
        definition.get("receiver"),
        spec.get("target"),
    ]
    for rule in spec.get("same_position_replacements", []) or []:
        names.extend([rule.get("target"), rule.get("replacement")])
    return list(dict.fromkeys(str(name) for name in names if name))


def capture_libero_state(
    env: Any,
    observation: Mapping[str, Any] | None = None,
    object_names: Iterable[str] | None = None,
    *,
    include_sim_state: bool = False,
) -> dict[str, Any]:
    """Capture simulator state read-only.

    Missing fields are omitted so this helper is safe across LIBERO wrappers.
    """
    state: dict[str, Any] = {}
    obs = observation or {}
    for source, destination in (
        ("robot0_eef_pos", "ee_pos"),
        ("robot0_eef_quat", "ee_quat_xyzw"),
        ("robot0_gripper_qpos", "gripper_qpos"),
        ("robot0_joint_pos", "robot_joint_positions"),
    ):
        if source in obs:
            state[destination] = _jsonable(np.asarray(obs[source]))

    base_env = getattr(env, "env", env)
    sim = getattr(base_env, "sim", None)
    if sim is None:
        return state
    try:
        state["sim_time"] = float(sim.data.time)
    except Exception:
        pass
    if include_sim_state:
        try:
            state["sim_qpos"] = np.asarray(sim.data.qpos, dtype=float).tolist()
            state["sim_qvel"] = np.asarray(sim.data.qvel, dtype=float).tolist()
        except Exception:
            pass

    if "ee_pos" not in state:
        for site_name in ("gripper0_grip_site", "robot0_grip_site"):
            try:
                site_id = int(sim.model.site_name2id(site_name))
                state["ee_pos"] = np.asarray(
                    sim.data.site_xpos[site_id], dtype=float
                ).tolist()
                break
            except Exception:
                continue

    if "robot_joint_positions" not in state:
        try:
            robot = base_env.robots[0]
            joint_names = list(getattr(robot, "robot_joints", []) or [])
            positions = [
                float(np.asarray(sim.data.get_joint_qpos(name)).reshape(-1)[0])
                for name in joint_names
            ]
            if joint_names:
                state["robot_joint_names"] = joint_names
                state["robot_joint_positions"] = positions
        except Exception:
            pass

    poses: dict[str, Any] = {}
    body_ids = getattr(base_env, "obj_body_id", {}) or {}
    for name in object_names or []:
        try:
            body_id = int(body_ids[name])
            poses[str(name)] = {
                "pos": np.asarray(sim.data.body_xpos[body_id], dtype=float).tolist(),
                "quat_wxyz": np.asarray(
                    sim.data.body_xquat[body_id], dtype=float
                ).tolist(),
            }
        except Exception:
            continue
    if poses:
        state["object_poses"] = poses

    # Preserve bounded internal-dynamics joints used by geometry-preserving
    # grasp tests. This is a read-only side channel and never feeds the model.
    extra_joint_qpos: dict[str, float] = {}
    try:
        for joint_name in list(sim.model.joint_names):
            if "payload_slide" not in str(joint_name):
                continue
            extra_joint_qpos[str(joint_name)] = float(
                np.asarray(sim.data.get_joint_qpos(joint_name)).reshape(-1)[0]
            )
    except Exception:
        extra_joint_qpos = {}
    if extra_joint_qpos:
        state["extra_joint_qpos"] = extra_joint_qpos

    watched = set(str(name) for name in object_names or [])
    if watched:
        contacts = []
        try:
            for index in range(int(sim.data.ncon)):
                contact = sim.data.contact[index]
                geom1 = sim.model.geom_id2name(int(contact.geom1)) or ""
                geom2 = sim.model.geom_id2name(int(contact.geom2)) or ""
                if any(name in geom1 or name in geom2 for name in watched):
                    contacts.append(
                        {
                            "geom1": geom1,
                            "geom2": geom2,
                            "distance_m": float(contact.dist),
                        }
                    )
        except Exception:
            contacts = []
        state["contacts"] = contacts
    return state


class NullEpisodeRecorder:
    enabled = False
    episode_id = ""
    object_names: list[str] = []

    def record_replan(self, *args: Any, **kwargs: Any) -> None:
        return None

    def record_realized(self, *args: Any, **kwargs: Any) -> None:
        return None

    def update_replan_trace(self, *args: Any, **kwargs: Any) -> None:
        return None

    def record_control(self, *args: Any, **kwargs: Any) -> None:
        return None

    def record_event(self, *args: Any, **kwargs: Any) -> None:
        return None

    def finish(self, *args: Any, **kwargs: Any) -> None:
        return None


class EpisodeRecorder:
    enabled = True

    def __init__(
        self,
        root: Path,
        metadata: Mapping[str, Any],
        *,
        env_edit_spec: Mapping[str, Any] | None = None,
    ) -> None:
        self.metadata = dict(_jsonable(metadata))
        requested_profile = os.getenv("WAM_RECORD_PROFILE", "").strip().lower()
        self.compact_profile = requested_profile in {
            "compact",
            "a4_compact",
        }
        self.omit_observation_arrays = (
            self.compact_profile or requested_profile == "c1_compact"
        )
        if self.compact_profile or self.omit_observation_arrays:
            self.metadata["recording_profile"] = requested_profile
        self.object_names = _object_names_from_spec(env_edit_spec)
        requested_id = os.getenv("WAM_EPISODE_ID", "").strip()
        default_id = "-".join(
            [
                _slug(self.metadata.get("model_name")),
                _slug(self.metadata.get("suite")),
                f"task{int(self.metadata.get('task_id', -1)):02d}",
                f"trial{int(self.metadata.get('repeat_index', 0)):03d}",
                _slug(self.metadata.get("condition", "unknown")),
                uuid.uuid4().hex[:8],
            ]
        )
        self.episode_id = _slug(requested_id or default_id, maximum=180)
        self.directory = root / self.episode_id
        self.replan_directory = self.directory / "replans"
        self.directory.mkdir(parents=True, exist_ok=False)
        self.replan_directory.mkdir(parents=True, exist_ok=True)
        self._control_records: list[dict[str, Any]] = []
        self._events: list[dict[str, Any]] = []
        self._replan_by_index: dict[int, dict[str, Any]] = {}
        self._finished = False
        self.manifest: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "recorder_version": RECORDER_VERSION,
            "episode_id": self.episode_id,
            "status": "recording",
            "metadata": self.metadata,
            "replan_steps": [],
            "control_steps": {
                "count": 0,
                "artifact": None,
            },
            "events": [],
            "summary": {},
        }
        self._checkpoint_manifest()

    def _checkpoint_manifest(self) -> None:
        _atomic_json(self.directory / "episode.json", self.manifest)

    def _write_arrays(
        self,
        relative_path: str,
        arrays: Mapping[str, Any],
    ) -> dict[str, Any]:
        prepared: dict[str, np.ndarray] = {}
        for key, value in arrays.items():
            array = _array(value)
            if array is not None:
                prepared[str(key)] = array
        if not prepared:
            return {}
        path = self.directory / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        if self.compact_profile or _as_bool(os.getenv("WAM_RECORD_COMPRESS", "0")):
            np.savez_compressed(path, **prepared)
        else:
            np.savez(path, **prepared)
        return {
            key: _artifact_ref(relative_path, key, array)
            for key, array in prepared.items()
        }

    def record_replan(
        self,
        replan_index: int,
        *,
        control_step_start: int,
        phase: str | None = None,
        clean_observation: Any = None,
        model_observation: Any = None,
        predicted_frames: Any = None,
        latent_visual: Any = None,
        latent_action: Any = None,
        proposed_action_chunk: Any = None,
        executed_action_chunk: Any = None,
        state: Mapping[str, Any] | None = None,
        perturbation: Mapping[str, Any] | None = None,
        attention: Mapping[str, Any] | None = None,
        extra_artifacts: Mapping[str, Any] | None = None,
        frame_control_offsets: Any = None,
        extra: Mapping[str, Any] | None = None,
    ) -> None:
        index = int(replan_index)
        arrays: dict[str, Any] = {}
        if not self.omit_observation_arrays:
            arrays.update(_view_arrays("clean_obs", clean_observation))
            arrays.update(_view_arrays("model_obs", model_observation))
        if isinstance(predicted_frames, Mapping):
            arrays.update(_view_arrays("predicted", predicted_frames))
        elif predicted_frames is not None:
            arrays["predicted_frames"] = predicted_frames
        if latent_visual is not None:
            arrays["latent_visual"] = latent_visual
        if latent_action is not None:
            arrays["latent_action"] = latent_action
        if proposed_action_chunk is not None:
            arrays["proposed_action_chunk"] = proposed_action_chunk
        if executed_action_chunk is not None and not self.compact_profile:
            arrays["executed_action_chunk"] = executed_action_chunk
        if frame_control_offsets is not None:
            arrays["prediction_frame_control_offsets"] = frame_control_offsets
        for key, value in (attention or {}).items():
            arrays[f"attention_{key}"] = value
        for key, value in (extra_artifacts or {}).items():
            arrays[f"extra_{key}"] = value

        relative = f"replans/replan_{index:05d}.npz"
        entry = {
            "replan_index": index,
            "control_step_start": int(control_step_start),
            "control_step_end": None,
            "phase": str(phase or "unknown"),
            "perturbation": _jsonable(perturbation or {}),
            "state": {} if self.compact_profile else _jsonable(state or {}),
            "artifacts": self._write_arrays(relative, arrays),
            "realized_artifacts": {},
            "extra": _jsonable(extra or {}),
        }
        if index in self._replan_by_index:
            previous = self._replan_by_index[index]
            previous.update(entry)
            entry = previous
        else:
            self.manifest["replan_steps"].append(entry)
            self._replan_by_index[index] = entry
        self._checkpoint_manifest()

    def record_realized(
        self,
        replan_index: int,
        *,
        control_step_end: int,
        frames: Any = None,
        clean_frames: Any = None,
        model_frames: Any = None,
        executed_action_chunk: Any = None,
    ) -> None:
        index = int(replan_index)
        entry = self._replan_by_index.get(index)
        if entry is None:
            return
        arrays: dict[str, Any] = {}
        if not self.omit_observation_arrays:
            if isinstance(frames, Mapping):
                arrays.update(_view_arrays("realized", frames))
            elif frames is not None:
                arrays["realized_frames"] = frames
            arrays.update(_view_arrays("realized_clean", clean_frames))
            arrays.update(_view_arrays("realized_model", model_frames))
        if executed_action_chunk is not None and not self.compact_profile:
            arrays["executed_action_chunk"] = executed_action_chunk
        relative = f"replans/replan_{index:05d}_realized.npz"
        entry["realized_artifacts"] = self._write_arrays(relative, arrays)
        entry["control_step_end"] = int(control_step_end)
        self._checkpoint_manifest()

    def update_replan_trace(
        self,
        replan_index: int,
        *,
        perturbation: Mapping[str, Any] | None = None,
        extra: Mapping[str, Any] | None = None,
    ) -> None:
        entry = self._replan_by_index.get(int(replan_index))
        if entry is None:
            return
        if perturbation is not None:
            entry["perturbation"] = _jsonable(perturbation)
        if extra:
            entry["extra"].update(dict(_jsonable(extra)))
        self._checkpoint_manifest()

    def record_control(
        self,
        control_step_index: int,
        *,
        replan_index: int,
        chunk_step_index: int,
        phase: str | None,
        action: Any,
        state: Mapping[str, Any] | None = None,
        perturbation_active: bool | None = None,
        reward: float | None = None,
        done: bool | None = None,
        extra: Mapping[str, Any] | None = None,
    ) -> None:
        self._control_records.append(
            {
                "control_step_index": int(control_step_index),
                "replan_index": int(replan_index),
                "chunk_step_index": int(chunk_step_index),
                "phase": str(phase or "unknown"),
                "action": _array(action),
                "state": _jsonable(state or {}),
                "perturbation_active": perturbation_active,
                "reward": reward,
                "done": done,
                "extra": _jsonable(extra or {}),
            }
        )

    def record_event(
        self,
        event_type: str,
        *,
        control_step_index: int | None = None,
        replan_index: int | None = None,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        self._events.append(
            {
                "type": str(event_type),
                "control_step_index": control_step_index,
                "replan_index": replan_index,
                "details": _jsonable(details or {}),
            }
        )

    def _write_controls(self) -> dict[str, Any] | None:
        if not self._control_records:
            return None
        arrays: dict[str, Any] = {
            "control_step_index": np.asarray(
                [item["control_step_index"] for item in self._control_records],
                dtype=np.int64,
            ),
            "replan_index": np.asarray(
                [item["replan_index"] for item in self._control_records],
                dtype=np.int64,
            ),
            "chunk_step_index": np.asarray(
                [item["chunk_step_index"] for item in self._control_records],
                dtype=np.int64,
            ),
            "phase": np.asarray(
                [item["phase"] for item in self._control_records],
                dtype=np.str_,
            ),
            "perturbation_active": np.asarray(
                [
                    -1
                    if item["perturbation_active"] is None
                    else int(bool(item["perturbation_active"]))
                    for item in self._control_records
                ],
                dtype=np.int8,
            ),
            "done": np.asarray(
                [
                    -1 if item["done"] is None else int(bool(item["done"]))
                    for item in self._control_records
                ],
                dtype=np.int8,
            ),
        }
        action_values = [item["action"] for item in self._control_records]
        if all(value is not None for value in action_values):
            try:
                arrays["action"] = np.stack(action_values)
            except ValueError:
                pass
        for key in (
            "ee_pos",
            "ee_quat_xyzw",
            "gripper_qpos",
            "robot_joint_positions",
        ):
            values = [
                _array(item["state"].get(key))
                for item in self._control_records
            ]
            if all(value is not None for value in values):
                try:
                    arrays[key] = np.stack(values)
                except ValueError:
                    pass
        if self.compact_profile:
            compact_keys = {
                "control_step_index",
                "replan_index",
                "chunk_step_index",
                "phase",
                "perturbation_active",
                "done",
                "action",
                "ee_pos",
                "gripper_qpos",
            }
            arrays = {key: value for key, value in arrays.items() if key in compact_keys}
        relative = "control_steps.npz"
        refs = self._write_arrays(relative, arrays)
        sidecar = [
            {
                "control_step_index": item["control_step_index"],
                "state": (
                    {
                        key: item["state"][key]
                        for key in (
                            "sim_time",
                            "object_poses",
                            "contacts",
                            "extra_joint_qpos",
                        )
                        if key in item["state"]
                    }
                    if self.compact_profile
                    else item["state"]
                ),
                "reward": item["reward"],
                **({} if self.compact_profile else {"extra": item["extra"]}),
            }
            for item in self._control_records
        ]
        _atomic_json(self.directory / "control_steps.json", sidecar)
        return {
            "path": relative,
            "arrays": refs,
            "sidecar": "control_steps.json",
        }

    def finish(
        self,
        *,
        success: bool,
        final_phase: str | None = None,
        summary: Mapping[str, Any] | None = None,
        status: str = "complete",
    ) -> None:
        if self._finished:
            return
        self.manifest["control_steps"] = {
            "count": len(self._control_records),
            "artifact": self._write_controls(),
        }
        self.manifest["events"] = self._events
        self.manifest["summary"] = {
            "success": bool(success),
            "final_phase": str(final_phase or "unknown"),
            "total_replan_steps": len(self.manifest["replan_steps"]),
            "total_control_steps": len(self._control_records),
            **dict(_jsonable(summary or {})),
        }
        self.manifest["status"] = str(status)
        self.manifest["completed_at"] = dt.datetime.now(
            dt.timezone.utc
        ).isoformat()
        self._checkpoint_manifest()
        self._finished = True


def make_episode_recorder(
    *,
    model_name: str,
    model_type: str,
    suite: str,
    task_id: int,
    task_name: str,
    instruction: str,
    repeat_index: int,
    init_state_index: int | None = None,
    checkpoint: str | None = None,
    environment_seed: int | None = None,
    model_seed: int | None = None,
    config: Mapping[str, Any] | None = None,
    env_edit_spec: Mapping[str, Any] | None = None,
    perturbation_spec: Mapping[str, Any] | None = None,
    action_config: Mapping[str, Any] | None = None,
    extra_metadata: Mapping[str, Any] | None = None,
) -> EpisodeRecorder | NullEpisodeRecorder:
    if not recording_enabled():
        return NullEpisodeRecorder()

    root = Path(
        os.path.expandvars(
            os.path.expanduser(
                os.getenv(
                    "WAM_EPISODE_RECORD_ROOT",
                    str(PROJECT_ROOT / "Outputs" / "wam_syncbench_episode_records"),
                )
            )
        )
    ).resolve()
    condition = os.getenv("WAM_CONDITION", "").strip() or (
        "perturbed"
        if _spec_has_effect(perturbation_spec) or _spec_has_effect(env_edit_spec)
        else "clean"
    )
    pair_id = os.getenv("WAM_PAIR_ID", "").strip()
    metadata = {
        "model_name": model_name,
        "model_type": model_type,
        "suite": suite,
        "task_id": int(task_id),
        "task_name": task_name,
        "instruction": instruction,
        "repeat_index": int(repeat_index),
        "init_state_index": init_state_index,
        "checkpoint": checkpoint,
        "environment_seed": environment_seed,
        "model_seed": model_seed,
        "track": os.getenv("WAM_TRACK", ""),
        "subtask": os.getenv("WAM_SUBTASK", ""),
        "case_id": os.getenv("WAM_CASE_ID", ""),
        "condition": condition,
        "pair_id": pair_id,
        "clean_episode_id": os.getenv("WAM_CLEAN_EPISODE_ID", ""),
        "perturbed_episode_id": os.getenv("WAM_PERTURBED_EPISODE_ID", ""),
        "agent_reference_episode_id": os.getenv(
            "WAM_AGENT_REFERENCE_EPISODE_ID", ""
        ),
        "wrist_reference_episode_id": os.getenv(
            "WAM_WRIST_REFERENCE_EPISODE_ID", ""
        ),
        "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
        "config_hash": _hash_json(config or {}),
        "config": _jsonable(config or {}),
        "env_edit_spec": _jsonable(env_edit_spec or {}),
        "perturbation": _jsonable(perturbation_spec or {}),
        "action_config": _jsonable(action_config or {}),
        "observation_storage": {
            "orientation": "as_received_by_recorder",
            "clean": "unmodified simulator/policy observation when available",
            "model": "actual observation supplied to the model",
        },
        **dict(_jsonable(extra_metadata or {})),
    }
    return EpisodeRecorder(root, metadata, env_edit_spec=env_edit_spec)
