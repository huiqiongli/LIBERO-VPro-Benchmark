"""Shared LIBERO perturbation helpers for evaluation scripts."""

from __future__ import annotations

import contextlib
import copy
import itertools
import json
import math
import os
from pathlib import Path
from typing import Any, Iterator

import cv2
import numpy as np


LIBERO_OBJECT_XML_OVERRIDE_ENV = "LIBERO_OBJECT_XML_OVERRIDE_MAP"
LIBERO_USE_SEGMENTATION_ENV = "LIBERO_USE_SEGMENTATION_ENV"
PRECOMPILE_XML_SCALE_MODES = {"precompile_xml", "xml_physical", "physical_xml"}
LIBERO_STATE_LAYOUT_CACHE: dict[str, dict[str, Any]] = {}

VIEW_CONFIGS = {
    "agentview": {
        "image_key": "agentview_image",
        "segmentation_key": "agentview_segmentation_instance",
    },
    "robot0_eye_in_hand": {
        "image_key": "robot0_eye_in_hand_image",
        "segmentation_key": "robot0_eye_in_hand_segmentation_instance",
    },
}


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def load_json_or_inline(value: Any) -> tuple[dict[str, Any], str]:
    if not value:
        return {}, ""
    expanded = os.path.expanduser(os.path.expandvars(str(value)))
    path = Path(expanded)
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            return json.load(f), str(path)
    return json.loads(str(value)), "inline"


def load_spec_from_env(env_keys: list[str], *, default_enabled: bool = False) -> tuple[dict[str, Any], str]:
    for env_key in env_keys:
        value = os.environ.get(env_key, "").strip()
        if value:
            spec, source = load_json_or_inline(value)
            if default_enabled:
                spec.setdefault("enabled", True)
            return spec, source
    return {"enabled": False}, ""


def normalize_view_names(views: Any) -> list[str]:
    if not views:
        return ["agentview", "robot0_eye_in_hand"]
    aliases = {
        "image": "agentview",
        "wrist": "robot0_eye_in_hand",
        "wrist_image": "robot0_eye_in_hand",
        "eye_in_hand": "robot0_eye_in_hand",
        "observation.images.agentview_rgb": "agentview",
        "observation.images.eye_in_hand_rgb": "robot0_eye_in_hand",
    }
    normalized = []
    for view in views:
        view_name = aliases.get(str(view), str(view))
        if view_name not in VIEW_CONFIGS:
            raise ValueError(f"Unsupported LIBERO view: {view}")
        normalized.append(view_name)
    return normalized


def obs_perturbation_enabled(spec: dict[str, Any] | None) -> bool:
    return as_bool((spec or {}).get("enabled"))


def obs_perturbation_needs_segmentation(spec: dict[str, Any] | None) -> bool:
    return obs_perturbation_enabled(spec) and str((spec or {}).get("mask_source", "")).lower() in {
        "segmentation",
        "target_noise",
        "non_target_noise",
        "segmentation_noise",
        "object_label_overlay",
        "wrong_category_label",
        "category_label_overlay",
        "target_photo_replacement",
        "object_photo_replacement",
        "photo_replacement",
        "target_shape_warp",
    }


class DecisionPerturbationController:
    """Model-decision-level gate and per-view camera history.

    At each policy-decision boundary, call :meth:`apply_camera` once to record
    the current clean frames and select camera inputs, and :meth:`active` once
    to evaluate the observation gate. Low-level actions do not advance this
    decision-level history.
    """

    def __init__(self, spec: dict[str, Any] | None):
        self.spec = dict(spec or {})
        self.trigger = str(self.spec.get("trigger", "always")).lower()
        indices = (
            self.spec.get("decision_indices")
            or (self.spec.get("random_replans") or {}).get("indices")
            or []
        )
        self.indices = {int(v) for v in indices}
        self.window = self.spec.get("decision_window") or self.spec.get("replan_window") or {}
        self.time_window = self.spec.get("time_window") or {}
        self.phases = set(self.spec.get("phases") or [])
        self.invert = as_bool(self.spec.get("invert", False))
        self.grasp_pulse = self.spec.get("grasp_pulse") or {}
        self.place_pulse = self.spec.get("place_pulse") or {}
        self.repeat_place_events = as_bool(self.spec.get("repeat_place_events", False))
        self._grasp_seen = False
        self._place_seen = False
        self._was_grasped = False
        self._frames_after_grasp = 0
        self._frames_after_place = 0
        self._verified_pre_place_seen = False
        self._verified_place_started = False
        self._frames_after_verified_place = 0
        self._phase_pulse_start: int | None = None
        self.camera = self.spec.get("camera") or {}
        self.history: dict[str, list[np.ndarray]] = {}
        self.log: list[dict[str, Any]] = []

    def observe_phase(self, grasped: bool, phase: str | None = None) -> None:
        """Advance grasp/place pulse state once per simulator control step."""
        grasped = bool(grasped)
        if grasped and not self._grasp_seen:
            self._grasp_seen = True
            self._frames_after_grasp = 0
        elif self._grasp_seen:
            self._frames_after_grasp += 1

        if self._grasp_seen and self._was_grasped and not grasped and not self._place_seen:
            self._place_seen = True
            self._frames_after_place = 0
        elif self._place_seen:
            self._frames_after_place += 1
        self._was_grasped = grasped

        # A2 post-place must not fire on an arbitrary failed release.  Arm it
        # only after the transported object has actually entered pre-place,
        # then start the exact control-step counter on the following release.
        if phase == "pre_place" and not self._verified_place_started:
            self._verified_pre_place_seen = True
        just_started = False
        if (
            phase == "release"
            and self._verified_pre_place_seen
            and not self._verified_place_started
        ):
            self._verified_place_started = True
            self._frames_after_verified_place = 0
            just_started = True
        if self._verified_place_started and not just_started:
            self._frames_after_verified_place += 1

    def active(
        self,
        decision_idx: int,
        *,
        timestep: int | None = None,
        phase: str | None = None,
    ) -> bool:
        if not self.spec or not as_bool(self.spec.get("enabled", True)):
            return False
        if self.trigger in {"always", "camera"}:
            active = True
        elif self.trigger in {"decision_indices", "random_replans"}:
            active = int(decision_idx) in self.indices
        elif self.trigger in {"decision_window", "replan_window"}:
            start = int(self.window.get("start", self.window.get("start_replan", 0)))
            count = int(self.window.get("count", self.window.get("num_replans", 10**9)))
            end = int(self.window.get("end", start + count))
            active = start <= int(decision_idx) < end
        elif self.trigger == "time_window":
            step = int(decision_idx if timestep is None else timestep)
            start = int(self.time_window.get("start", 0))
            end = int(self.time_window.get("end", 10**9))
            active = start <= step < end
        elif self.trigger == "phase":
            active = phase in self.phases
        elif self.trigger in {"phase_pulse", "phase_once"}:
            # Trigger once on the first decision observed in the target phase,
            # then keep the corruption active for exactly N replans.
            count = int(self.spec.get("replan_count", self.spec.get("num_replans", 3)))
            count = max(1, count)
            if self._phase_pulse_start is None:
                if phase in self.phases:
                    self._phase_pulse_start = int(decision_idx)
                    active = True
                else:
                    active = False
            else:
                active = self._phase_pulse_start <= int(decision_idx) < self._phase_pulse_start + count
        elif self.trigger == "grasp_pulse":
            side = str(self.grasp_pulse.get("side", "after")).lower()
            n_frames = int(self.grasp_pulse.get("n_frames", 20))
            if side == "before":
                active = phase == "pre_grasp" and not self._grasp_seen
            else:
                active = self._grasp_seen and self._frames_after_grasp < n_frames
        elif self.trigger == "place_pulse":
            side = str(self.place_pulse.get("side", "after")).lower()
            n_frames = int(self.place_pulse.get("n_frames", 40))
            if side == "before":
                active = phase == "pre_place" and not self._place_seen
            else:
                active = self._place_seen and self._frames_after_place < n_frames
        elif self.trigger == "post_place_verified":
            n_frames = int(self.place_pulse.get("n_frames", 40))
            active = (
                self._verified_place_started
                and self._frames_after_verified_place < n_frames
            )
            # A2 placement timing applies around every valid placement in a
            # multi-object task.  Once the exact post-place dose has elapsed,
            # re-arm the verified pre-place -> release detector for the next
            # object.  The default remains one-shot for completed experiments.
            if (
                self.repeat_place_events
                and self._verified_place_started
                and self._frames_after_verified_place >= n_frames
            ):
                self._verified_pre_place_seen = False
                self._verified_place_started = False
                self._frames_after_verified_place = 0
        else:
            raise ValueError(f"Unsupported decision trigger: {self.trigger}")
        return not active if self.invert else active

    def apply_camera(self, raw_obs: dict[str, Any], decision_idx: int) -> dict[str, Any]:
        if not self.camera:
            return raw_obs
        out = dict(raw_obs)
        for view_name, camera_spec in self.camera.items():
            normalized = normalize_view_names([view_name])
            if not normalized:
                continue
            view = normalized[0]
            image_key = VIEW_CONFIGS[view]["image_key"]
            if image_key not in raw_obs:
                continue
            hist = self.history.setdefault(view, [])
            hist.append(np.array(raw_obs[image_key], copy=True))
            mode = str((camera_spec or {}).get("mode", "delay")).lower()
            cur = len(hist) - 1
            if mode == "delay":
                source = max(0, cur - int((camera_spec or {}).get("lag", 1)))
            elif mode == "freeze":
                source = min(cur, int((camera_spec or {}).get("freeze_at", 2)))
            elif mode == "replay":
                start = int((camera_spec or {}).get("replay_from", 2))
                length = max(1, int((camera_spec or {}).get("replay_length", 8)))
                source = cur if cur < start else min(cur, start + ((cur - start) % length))
            else:
                raise ValueError(f"Unsupported camera staleness mode: {mode}")
            out[image_key] = np.array(hist[source], copy=True)
            self.log.append(
                {
                    "decision": int(decision_idx),
                    "view": view,
                    "fed_decision": int(source),
                    "staleness_decisions": int(cur - source),
                    "mode": mode,
                }
            )
        return out


def _coverage_crop(mask: np.ndarray, coverage: Any) -> np.ndarray:
    coverage_f = max(0.0, min(1.0, float(coverage)))
    if coverage_f >= 1.0 or not mask.any():
        return mask
    ys, xs = np.where(mask)
    target = int(round(len(ys) * coverage_f))
    if target <= 0:
        return np.zeros_like(mask, dtype=bool)
    cy, cx = float(ys.mean()), float(xs.mean())
    order = np.argsort((ys - cy) ** 2 + (xs - cx) ** 2, kind="stable")[:target]
    cropped = np.zeros_like(mask, dtype=bool)
    cropped[ys[order], xs[order]] = True
    return cropped


def cross_task_view_swap_enabled(spec: dict[str, Any] | None) -> bool:
    return obs_perturbation_enabled(spec) and str((spec or {}).get("mask_source", "cross_task_view_swap")) in {
        "cross_task_view_swap",
        "cross_task_view_conflict",
    }


def resolve_cross_task_donor(
    spec: dict[str, Any],
    source_benchmark: str,
    source_task_idx: int,
    num_tasks: int,
) -> tuple[str, int]:
    donor_benchmark = str(spec.get("donor_benchmark") or source_benchmark)
    task_key = f"{source_benchmark}:{int(source_task_idx)}"
    donor_by_task = spec.get("donor_task_by_task", {})
    if task_key in donor_by_task:
        donor_task_idx = int(donor_by_task[task_key])
    elif str(source_task_idx) in donor_by_task:
        donor_task_idx = int(donor_by_task[str(source_task_idx)])
    else:
        donor_task_idx = (int(source_task_idx) + int(spec.get("donor_task_offset", 1))) % int(num_tasks)
    if donor_benchmark == source_benchmark and donor_task_idx == int(source_task_idx):
        raise ValueError(f"Cross-task donor must differ from source task: {task_key}")
    return donor_benchmark, donor_task_idx


def apply_cross_task_view_swap_obs(source_obs: dict[str, Any], donor_obs: dict[str, Any], views: Any) -> dict[str, Any]:
    swapped_views = normalize_view_names(views)
    mixed_obs = dict(source_obs)
    for view_name in swapped_views:
        image_key = VIEW_CONFIGS[view_name]["image_key"]
        if image_key not in donor_obs:
            raise KeyError(f"Donor observation is missing {image_key!r} for cross-task view swap")
        mixed_obs[image_key] = np.array(donor_obs[image_key], copy=True)
    return mixed_obs


def _task_matches_env_edit(edit: dict[str, Any], libero_benchmark: str, task_idx: int | None) -> bool:
    edit_benchmark = edit.get("benchmark", edit.get("suite", ""))
    if edit_benchmark and str(edit_benchmark) != str(libero_benchmark):
        return False
    if task_idx is not None and "task_id" in edit and int(edit["task_id"]) != int(task_idx):
        return False
    return True


def resolve_env_edit_spec(
    spec: dict[str, Any] | None,
    libero_benchmark: str = "",
    task_idx: int | None = None,
) -> dict[str, Any] | None:
    if not spec:
        return spec
    task_edits = spec.get("task_env_edits") or spec.get("per_task_env_edits") or []
    if not task_edits:
        return spec

    base = {
        key: copy.deepcopy(value)
        for key, value in spec.items()
        if key not in {"task_env_edits", "per_task_env_edits"}
    }
    list_merge_keys = {
        "object_pose_swaps",
        "object_scale_overrides",
        "object_pose_overrides",
        "object_color_overrides",
        "object_surface_black_fraction_overrides",
        "geom_color_overrides",
        "body_color_overrides",
        "body_mass_overrides",
    }
    for edit in task_edits:
        if task_idx is not None and _task_matches_env_edit(edit, libero_benchmark, task_idx):
            merged = copy.deepcopy(base)
            for key, value in edit.items():
                if key in list_merge_keys:
                    merged[key] = copy.deepcopy(base.get(key, [])) + copy.deepcopy(value)
                else:
                    merged[key] = copy.deepcopy(value)
            merged["_matched_task_env_edit"] = {
                "benchmark": libero_benchmark,
                "task_id": int(task_idx),
            }
            return merged

    if as_bool(spec.get("require_task_match", True)):
        raise ValueError(
            f"No per-task LIBERO env edit matched benchmark={libero_benchmark!r}, task_id={task_idx!r}"
        )
    return base


def is_precompile_xml_scale_override(override: dict[str, Any]) -> bool:
    mode = str(override.get("scale_mode") or override.get("apply_stage") or "").strip().lower()
    return mode in PRECOMPILE_XML_SCALE_MODES or bool(override.get("xml_path"))


def object_xml_override_map(spec: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    mapping: dict[str, dict[str, Any]] = {}
    for override in (spec or {}).get("object_scale_overrides", []):
        if not is_precompile_xml_scale_override(override):
            continue
        object_name = str(override["object"])
        xml_path = override.get("xml_path")
        if not xml_path:
            raise ValueError(f"Precompile XML scale override for {object_name!r} needs xml_path")
        mapping[object_name] = {
            "xml_path": str(xml_path),
            "scale": float(override["scale"]),
            "category": override.get("category"),
        }
    return mapping


def bddl_file_name_override(spec: dict[str, Any] | None) -> str:
    if not spec:
        return ""
    value = (
        spec.get("bddl_file_name_override")
        or spec.get("bddl_file_override")
        or spec.get("bddl_override")
    )
    if not value:
        return ""
    path = Path(os.path.expanduser(os.path.expandvars(str(value)))).resolve()
    if not path.exists():
        raise FileNotFoundError(f"LIBERO BDDL override does not exist: {path}")
    return str(path)


def base_bddl_file_name(spec: dict[str, Any] | None) -> str:
    if not spec:
        return ""
    value = (
        spec.get("base_bddl_file_name")
        or spec.get("base_bddl_file")
        or spec.get("init_state_bddl_file_name")
    )
    if not value:
        return ""
    path = Path(os.path.expanduser(os.path.expandvars(str(value)))).resolve()
    if not path.exists():
        raise FileNotFoundError(f"LIBERO base BDDL does not exist: {path}")
    return str(path)


def _addr_slice(addr: Any) -> slice:
    if isinstance(addr, (tuple, list)):
        return slice(int(addr[0]), int(addr[1]))
    return slice(int(addr), int(addr) + 1)


def _model_joint_state_layout(model: Any) -> dict[str, Any]:
    return {
        "nq": int(model.nq),
        "nv": int(model.nv),
        "na": int(getattr(model, "na", 0)),
        "qpos_by_joint": {
            name: _addr_slice(model.get_joint_qpos_addr(name))
            for name in list(model.joint_names)
        },
        "qvel_by_joint": {
            name: _addr_slice(model.get_joint_qvel_addr(name))
            for name in list(model.joint_names)
        },
    }


def _state_layout_for_bddl(bddl_file_name: str) -> dict[str, Any]:
    bddl_file_name = str(Path(bddl_file_name).resolve())
    cached = LIBERO_STATE_LAYOUT_CACHE.get(bddl_file_name)
    if cached is not None:
        return cached
    from libero.libero.envs.env_wrapper import ControlEnv

    old_xml_override = os.environ.pop(LIBERO_OBJECT_XML_OVERRIDE_ENV, None)
    try:
        env = ControlEnv(
            bddl_file_name=bddl_file_name,
            use_camera_obs=False,
            has_offscreen_renderer=False,
        )
        try:
            layout = _model_joint_state_layout(env.sim.model)
        finally:
            env.close()
    finally:
        if old_xml_override is not None:
            os.environ[LIBERO_OBJECT_XML_OVERRIDE_ENV] = old_xml_override
    LIBERO_STATE_LAYOUT_CACHE[bddl_file_name] = layout
    return layout


def set_init_state_with_bddl_remap(
    env_in: Any,
    init_state: Any,
    base_bddl_file_name_value: str = "",
) -> Any:
    init_state = np.asarray(init_state, dtype=float)
    current_state = np.asarray(env_in.get_sim_state(), dtype=float)
    if init_state.shape == current_state.shape:
        return env_in.set_init_state(init_state)
    if not base_bddl_file_name_value:
        raise ValueError(
            "The edited LIBERO model has a different state size, but no "
            "base_bddl_file_name was supplied for joint-name remapping."
        )

    base_layout = _state_layout_for_bddl(base_bddl_file_name_value)
    base_env = getattr(env_in, "env", env_in)
    current_layout = _model_joint_state_layout(base_env.sim.model)
    expected_len = 1 + base_layout["nq"] + base_layout["nv"] + base_layout["na"]
    if len(init_state) != expected_len:
        raise ValueError(
            f"LIBERO init state length {len(init_state)} does not match base model "
            f"length {expected_len} for {base_bddl_file_name_value}"
        )

    old_qpos = init_state[1 : 1 + base_layout["nq"]]
    old_qvel = init_state[1 + base_layout["nq"] : 1 + base_layout["nq"] + base_layout["nv"]]
    sim = base_env.sim
    sim.data.time = float(init_state[0])
    copied = 0
    for name, old_slice in base_layout["qpos_by_joint"].items():
        new_slice = current_layout["qpos_by_joint"].get(name)
        if new_slice is None:
            continue
        sim.data.qpos[new_slice] = old_qpos[old_slice]
        copied += 1
    for name, old_slice in base_layout["qvel_by_joint"].items():
        new_slice = current_layout["qvel_by_joint"].get(name)
        if new_slice is not None:
            sim.data.qvel[new_slice] = old_qvel[old_slice]
    if copied == 0:
        raise ValueError(
            f"No overlapping joints while remapping LIBERO state from {base_bddl_file_name_value}"
        )
    _forward_sim(sim)
    env_in.check_success()
    env_in._post_process()
    env_in._update_observables(force=True)
    return env_in.env._get_observations()


@contextlib.contextmanager
def libero_env_override_context(
    *,
    object_xml_override_map_value: dict[str, Any] | None = None,
    use_segmentation: bool | None = None,
) -> Iterator[None]:
    old_xml_override = os.environ.get(LIBERO_OBJECT_XML_OVERRIDE_ENV)
    old_use_segmentation = os.environ.get(LIBERO_USE_SEGMENTATION_ENV)
    try:
        if object_xml_override_map_value is not None:
            if object_xml_override_map_value:
                os.environ[LIBERO_OBJECT_XML_OVERRIDE_ENV] = json.dumps(object_xml_override_map_value)
            else:
                os.environ.pop(LIBERO_OBJECT_XML_OVERRIDE_ENV, None)
        if use_segmentation is not None:
            os.environ[LIBERO_USE_SEGMENTATION_ENV] = "1" if use_segmentation else "0"
        yield
    finally:
        if object_xml_override_map_value is not None:
            if old_xml_override is None:
                os.environ.pop(LIBERO_OBJECT_XML_OVERRIDE_ENV, None)
            else:
                os.environ[LIBERO_OBJECT_XML_OVERRIDE_ENV] = old_xml_override
        if use_segmentation is not None:
            if old_use_segmentation is None:
                os.environ.pop(LIBERO_USE_SEGMENTATION_ENV, None)
            else:
                os.environ[LIBERO_USE_SEGMENTATION_ENV] = old_use_segmentation


def _get_joint_qpos(sim: Any, joint_name: str) -> np.ndarray:
    try:
        return np.array(sim.data.get_joint_qpos(joint_name), dtype=float)
    except AttributeError:
        qpos_addr = sim.model.get_joint_qpos_addr(joint_name)
        return np.array(sim.data.qpos[qpos_addr], dtype=float)


def _set_joint_qpos(sim: Any, joint_name: str, qpos: np.ndarray) -> None:
    sim.data.set_joint_qpos(joint_name, np.array(qpos, dtype=float))


def _free_joint_name(base_env: Any, object_name: str) -> str | None:
    """Return an object's 7-DoF root joint, never an articulated fixture joint."""
    obj = base_env.get_object(object_name)
    if obj is None:
        return None
    for joint_name in reversed(list(getattr(obj, "joints", None) or [])):
        try:
            if np.asarray(_get_joint_qpos(base_env.sim, joint_name)).size == 7:
                return str(joint_name)
        except (KeyError, ValueError):
            continue
    return None


def _forward_sim(sim: Any) -> None:
    try:
        sim.forward()
    except AttributeError:
        pass


def _descendant_body_ids(sim: Any, root_body_id: int) -> list[int]:
    body_ids = [int(root_body_id)]
    idx = 0
    while idx < len(body_ids):
        current = body_ids[idx]
        for body_id in range(sim.model.nbody):
            if body_id != current and int(sim.model.body_parentid[body_id]) == current:
                body_ids.append(body_id)
        idx += 1
    return body_ids


def _surface_texture_id(sim: Any, geom_id: int) -> int | None:
    """Return the color texture used by a visual geom, if it has one."""
    material_id = int(sim.model.geom_matid[geom_id])
    if material_id < 0:
        return None
    texture_ids = [
        int(value)
        for value in np.asarray(sim.model.mat_texid[material_id]).reshape(-1)
        if int(value) >= 0
    ]
    return texture_ids[0] if texture_ids else None


def _upload_runtime_texture(sim: Any, texture_id: int) -> bool:
    """Upload a modified compiled-model texture to an existing offscreen renderer."""
    context = getattr(sim, "_render_context_offscreen", None)
    if context is None:
        return False
    try:
        context.upload_texture(int(texture_id))
        return True
    except TypeError:
        # robosuite 1.4 wraps ``mujoco.MjModel``.  Its upload_texture helper in
        # this environment passes the wrapper instead of the underlying model.
        import mujoco

        context.gl_ctx.make_current()
        mujoco.mjr_uploadTexture(context.model._model, context.con, int(texture_id))
        return True


def _mesh_surface_description(base_env: Any, geom_id: int) -> dict[str, Any] | None:
    """Return world-space triangles and UV indices for one textured mesh geom."""
    sim = base_env.sim
    model = sim.model
    if int(model.geom_type[geom_id]) != 7:  # mjGEOM_MESH
        return None
    mesh_id = int(model.geom_dataid[geom_id])
    if mesh_id < 0:
        return None
    uv_count = int(model.mesh_texcoordnum[mesh_id])
    texture_id = _surface_texture_id(sim, geom_id)
    if uv_count <= 0 or texture_id is None:
        return None

    vertex_address = int(model.mesh_vertadr[mesh_id])
    vertex_count = int(model.mesh_vertnum[mesh_id])
    face_address = int(model.mesh_faceadr[mesh_id])
    face_count = int(model.mesh_facenum[mesh_id])
    uv_address = int(model.mesh_texcoordadr[mesh_id])
    vertices = np.asarray(
        model.mesh_vert[vertex_address : vertex_address + vertex_count],
        dtype=float,
    )
    rotation = np.asarray(sim.data.geom_xmat[geom_id], dtype=float).reshape(3, 3)
    translation = np.asarray(sim.data.geom_xpos[geom_id], dtype=float)
    vertices_world = vertices @ rotation.T + translation
    faces = np.asarray(
        model.mesh_face[face_address : face_address + face_count],
        dtype=np.int64,
    )
    triangles = vertices_world[faces]
    areas = 0.5 * np.linalg.norm(
        np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]),
        axis=1,
    )
    face_uv_indices = np.asarray(
        model.mesh_facetexcoord[face_address : face_address + face_count],
        dtype=np.int64,
    )
    uv = np.asarray(
        model.mesh_texcoord[uv_address : uv_address + uv_count],
        dtype=float,
    )
    if np.any(face_uv_indices < 0) or np.any(face_uv_indices >= uv_count):
        return None
    return {
        "geom_id": int(geom_id),
        "geom_name": model.geom_id2name(int(geom_id)) or f"geom_{geom_id}",
        "texture_id": int(texture_id),
        "areas": areas,
        "centroids": triangles.mean(axis=1),
        "uv_triangles": uv[face_uv_indices],
    }


def _render_geom_surface_area(base_env: Any, geom_id: int) -> float | None:
    """Return the compiled render-geom surface area, including UV-less geoms."""
    model = base_env.sim.model
    geom_type = int(model.geom_type[geom_id])
    if geom_type == 7:  # mjGEOM_MESH
        mesh_id = int(model.geom_dataid[geom_id])
        if mesh_id < 0:
            return None
        vertex_address = int(model.mesh_vertadr[mesh_id])
        vertex_count = int(model.mesh_vertnum[mesh_id])
        face_address = int(model.mesh_faceadr[mesh_id])
        face_count = int(model.mesh_facenum[mesh_id])
        vertices = np.asarray(
            model.mesh_vert[vertex_address : vertex_address + vertex_count],
            dtype=float,
        )
        faces = np.asarray(
            model.mesh_face[face_address : face_address + face_count],
            dtype=np.int64,
        )
        triangles = vertices[faces]
        return float(
            (
                0.5
                * np.linalg.norm(
                    np.cross(
                        triangles[:, 1] - triangles[:, 0],
                        triangles[:, 2] - triangles[:, 0],
                    ),
                    axis=1,
                )
            ).sum()
        )

    a, b, c = [float(value) for value in model.geom_size[geom_id]]
    if geom_type == 2:  # sphere
        return 4.0 * math.pi * a * a
    if geom_type == 3:  # capsule: radius and cylinder half-length
        return 4.0 * math.pi * a * a + 4.0 * math.pi * a * b
    if geom_type == 4:  # ellipsoid, Knud Thomsen approximation
        p = 1.6075
        return 4.0 * math.pi * (
            ((a * b) ** p + (a * c) ** p + (b * c) ** p) / 3.0
        ) ** (1.0 / p)
    if geom_type == 5:  # closed cylinder: radius and half-height
        return 4.0 * math.pi * a * b + 2.0 * math.pi * a * a
    if geom_type == 6:  # box half-sizes
        return 8.0 * (a * b + a * c + b * c)
    return None


def _blacken_texture_triangles(
    sim: Any,
    texture_id: int,
    uv_triangles: np.ndarray,
    rgb: Any,
) -> dict[str, Any]:
    model = sim.model
    width = int(model.tex_width[texture_id])
    height = int(model.tex_height[texture_id])
    channels = (
        int(model.tex_nchannel[texture_id])
        if hasattr(model, "tex_nchannel")
        else 3
    )
    address = int(model.tex_adr[texture_id])
    texture_storage = (
        model.tex_data if hasattr(model, "tex_data") else model.tex_rgb
    )
    texture = texture_storage[
        address : address + width * height * channels
    ].reshape(height, width, channels)

    polygons = np.asarray(uv_triangles, dtype=float).copy()
    polygons[:, :, 0] *= float(width - 1)
    # MuJoCo UV origin is bottom-left; image rows start at the top.
    polygons[:, :, 1] = (1.0 - polygons[:, :, 1]) * float(height - 1)
    polygons = np.rint(polygons).astype(np.int32)
    texture_mask = np.zeros((height, width), dtype=np.uint8)
    for start in range(0, len(polygons), 2048):
        cv2.fillPoly(
            texture_mask,
            [polygon for polygon in polygons[start : start + 2048]],
            255,
            lineType=cv2.LINE_8,
        )
    color = np.clip(np.asarray(rgb, dtype=float).reshape(-1)[:3], 0.0, 1.0)
    color_u8 = np.rint(color * 255.0).astype(np.uint8)
    write_channels = min(3, channels)
    texture[texture_mask > 0, :write_channels] = color_u8[:write_channels]
    uploaded = _upload_runtime_texture(sim, texture_id)
    return {
        "texture_id": int(texture_id),
        "texture_size": [height, width, channels],
        "changed_texture_pixels": int(np.count_nonzero(texture_mask)),
        "renderer_upload_completed": bool(uploaded),
    }


def apply_object_surface_black_overrides(
    base_env: Any,
    overrides: Any,
) -> list[dict[str, Any]]:
    """Blacken a coherent fraction of a target object's physical render surface.

    Partial coverage is defined by three-dimensional triangle area, not by pixels
    in either camera.  Faces are ordered along a fixed world axis and their UV
    texels are recolored.  Both cameras therefore observe the same modified
    surface.  Collision geoms, poses, dynamics and task predicates are untouched.
    """
    sim = base_env.sim
    model = sim.model
    _forward_sim(sim)
    audits: list[dict[str, Any]] = []
    for raw_override in overrides or []:
        override = dict(raw_override)
        object_name = str(override["object"])
        if object_name not in base_env.obj_body_id:
            raise ValueError(
                f"LIBERO surface-color target {object_name!r} was not found in obj_body_id"
            )
        coverage = float(override.get("coverage", 1.0))
        if not 0.0 < coverage <= 1.0:
            raise ValueError(
                f"Surface black coverage for {object_name!r} must be in (0, 1], got {coverage}"
            )
        rgba = np.asarray(override.get("rgba", [0.0, 0.0, 0.0, 1.0]), dtype=float)
        if rgba.size != 4:
            raise ValueError("Surface-color rgba must contain four values")
        axis_name = str(override.get("selection_axis", "world_x")).lower()
        axis_index = {
            "world_x": 0,
            "world_y": 1,
            "world_z": 2,
            # Agentview-visible variants are precompiled and do not use this
            # runtime axis.  Keep a harmless value for the common validation
            # path and for the 100% case, where no split is needed.
            "agentview_visible_region": 0,
            "uniform_per_source_triangle": 0,
            "agentview_aligned_complete_surface_slice": 0,
        }.get(axis_name)
        if axis_index is None:
            raise ValueError(
                f"Unsupported physical surface selection_axis={axis_name!r}"
            )

        root_body_id = int(base_env.obj_body_id[object_name])
        body_ids = set(_descendant_body_ids(sim, root_body_id))
        visual_geom_ids = [
            geom_id
            for geom_id in range(int(model.ngeom))
            if int(model.geom_bodyid[geom_id]) in body_ids
            and int(model.geom_group[geom_id]) != 0
            and float(model.geom_rgba[geom_id][3]) > 0.0
        ]
        if not visual_geom_ids:
            raise ValueError(f"No visual geoms found for LIBERO target {object_name!r}")

        precompiled_audit = override.get("precompiled_surface_audit")
        if precompiled_audit:
            # Keep the historical A2 prefix as the default so existing A2
            # configs remain byte-for-byte compatible, while allowing other
            # tracks to use isolated precompiled overlay names.
            overlay_prefix = str(
                override.get("surface_overlay_prefix", "a2_")
            )
            overlay_geom_ids = [
                geom_id
                for geom_id in visual_geom_ids
                if overlay_prefix in (model.geom_id2name(geom_id) or "")
            ]
            black_geom_ids = [
                geom_id
                for geom_id in overlay_geom_ids
                if "_black" in (model.geom_id2name(geom_id) or "")
            ]
            original_geom_ids = [
                geom_id for geom_id in visual_geom_ids if geom_id not in overlay_geom_ids
            ]
            if not black_geom_ids or not original_geom_ids:
                visible_names = [
                    model.geom_id2name(geom_id) or f"geom_{geom_id}"
                    for geom_id in visual_geom_ids
                ]
                raise ValueError(
                    f"Precompiled surface partition for {object_name!r} was not loaded "
                    f"or contains unexpected visible geoms: {visible_names}"
                )
            original_areas = {
                geom_id: _render_geom_surface_area(base_env, geom_id)
                for geom_id in original_geom_ids
            }
            overlay_areas = {
                geom_id: _render_geom_surface_area(base_env, geom_id)
                for geom_id in black_geom_ids
            }
            if any(area is None for area in original_areas.values()) or any(
                area is None for area in overlay_areas.values()
            ):
                raise ValueError(
                    f"Cannot audit precompiled surface partition for {object_name!r}"
                )
            measured_total = float(sum(float(area) for area in original_areas.values()))
            measured_black = float(
                sum(float(overlay_areas[geom_id]) for geom_id in black_geom_ids)
            )
            measured_coverage = measured_black / measured_total
            coverage_definition = str(
                precompiled_audit.get(
                    "coverage_definition", "three_dimensional_triangle_surface_area"
                )
            )
            if (
                coverage_definition
                in {
                    "three_dimensional_triangle_surface_area",
                    "complete_3d_render_surface_area",
                }
                and as_bool(override.get("require_exact", True))
                and abs(measured_coverage - coverage) > 0.002
            ):
                raise ValueError(
                    f"Precompiled surface coverage mismatch for {object_name!r}: "
                    f"requested={coverage}, measured={measured_coverage}"
                )
            audit = copy.deepcopy(dict(precompiled_audit))
            audit.update(
                {
                    "object": object_name,
                    "mode": "precompiled_visual_surface_partition",
                    "requested_surface_coverage": coverage,
                    "actual_surface_coverage": measured_coverage,
                    "coverage_definition": coverage_definition,
                    "requested_visible_coverage": coverage
                    if coverage_definition == "agentview_visible_target_pixels"
                    else None,
                    "surface_area_total": measured_total,
                    "surface_area_black": measured_black,
                    "visual_geoms": [
                        model.geom_id2name(geom_id) or f"geom_{geom_id}"
                        for geom_id in original_geom_ids
                    ],
                    "black_visual_geoms": [
                        model.geom_id2name(geom_id) or f"geom_{geom_id}"
                        for geom_id in black_geom_ids
                    ],
                    "collision_unchanged": True,
                    "pose_unchanged": True,
                }
            )
            audits.append(audit)
            continue

        if coverage >= 1.0:
            touched = []
            for geom_id in visual_geom_ids:
                model.geom_matid[geom_id] = -1
                model.geom_rgba[geom_id] = rgba
                touched.append(model.geom_id2name(geom_id) or f"geom_{geom_id}")
            audits.append(
                {
                    "object": object_name,
                    "mode": "physical_render_surface",
                    "coverage_definition": str(
                        override.get(
                            "coverage_definition",
                            "three_dimensional_triangle_surface_area",
                        )
                    ),
                    "requested_surface_coverage": coverage,
                    "actual_surface_coverage": 1.0,
                    "selection_axis": axis_name,
                    "visual_geoms": touched,
                    "selected_face_count": None,
                    "total_face_count": None,
                    "collision_unchanged": True,
                    "pose_unchanged": True,
                }
            )
            continue

        descriptions = []
        unsupported_components = []
        for geom_id in visual_geom_ids:
            description = _mesh_surface_description(base_env, geom_id)
            if description is None:
                unsupported_components.append(
                    {
                        "geom_id": int(geom_id),
                        "geom_name": model.geom_id2name(geom_id) or f"geom_{geom_id}",
                        "area": _render_geom_surface_area(base_env, geom_id),
                        "centroid": np.asarray(sim.data.geom_xpos[geom_id], dtype=float),
                    }
                )
            else:
                descriptions.append(description)
        unknown_area = [
            item["geom_name"]
            for item in unsupported_components
            if item["area"] is None or not np.isfinite(float(item["area"]))
        ]
        if unknown_area and as_bool(override.get("require_exact", True)):
            raise ValueError(
                f"Exact partial physical blackening for {object_name!r} is unsupported; "
                f"surface area is unknown for visual geoms: {unknown_area}"
            )
        if not descriptions and not unsupported_components:
            raise ValueError(
                f"No measurable render surface available for partial blackening of {object_name!r}"
            )

        if descriptions:
            areas = np.concatenate([item["areas"] for item in descriptions])
            coordinates = np.concatenate(
                [item["centroids"][:, axis_index] for item in descriptions]
            )
            geom_slots = np.concatenate(
                [
                    np.full(len(item["areas"]), index, dtype=np.int64)
                    for index, item in enumerate(descriptions)
                ]
            )
            face_slots = np.concatenate(
                [
                    np.arange(len(item["areas"]), dtype=np.int64)
                    for item in descriptions
                ]
            )
            valid = np.isfinite(areas) & (areas > 0.0)
            areas = areas[valid]
            coordinates = coordinates[valid]
            geom_slots = geom_slots[valid]
            face_slots = face_slots[valid]
        else:
            areas = np.empty(0, dtype=float)
            coordinates = np.empty(0, dtype=float)
            geom_slots = np.empty(0, dtype=np.int64)
            face_slots = np.empty(0, dtype=np.int64)

        editable_area = float(areas.sum())
        atomic_components = [
            item for item in unsupported_components if item["area"] is not None
        ]
        atomic_area = float(sum(float(item["area"]) for item in atomic_components))
        total_area = editable_area + atomic_area
        target_area = total_area * coverage
        tolerance = max(1e-10, total_area * 1e-8)

        # UV-less meshes and primitives can still be recolored as whole visual
        # components.  Select a subset whose remaining target area can be filled
        # exactly by UV-addressable triangles. Both component types contribute
        # to the complete-surface-area denominator.
        atomic_choice: tuple[int, ...] | None = None
        candidate_key: tuple[float, int, float] | None = None
        for count in range(len(atomic_components) + 1):
            for choice in itertools.combinations(range(len(atomic_components)), count):
                whole_area = float(
                    sum(float(atomic_components[index]["area"]) for index in choice)
                )
                residual = target_area - whole_area
                if residual < -tolerance or residual > editable_area + tolerance:
                    continue
                # Prefer the smallest atomic contribution; when it is needed,
                # prefer components closest to the selected side of the axis.
                centroid_score = float(
                    sum(
                        atomic_components[index]["centroid"][axis_index]
                        for index in choice
                    )
                )
                key = (whole_area, len(choice), centroid_score)
                if candidate_key is None or key < candidate_key:
                    atomic_choice = choice
                    candidate_key = key
        if atomic_choice is None:
            names = [item["geom_name"] for item in atomic_components]
            raise ValueError(
                f"Exact partial physical blackening for {object_name!r} cannot be "
                f"represented by its compiled visual components; UV-less geoms: {names}"
            )

        selected_atomic = [atomic_components[index] for index in atomic_choice]
        selected_atomic_area = float(
            sum(float(item["area"]) for item in selected_atomic)
        )
        editable_target_area = float(
            np.clip(target_area - selected_atomic_area, 0.0, editable_area)
        )
        for item in selected_atomic:
            geom_id = int(item["geom_id"])
            model.geom_matid[geom_id] = -1
            model.geom_rgba[geom_id] = rgba

        texture_triangles: dict[int, list[np.ndarray]] = {}
        selected_geoms: set[str] = set()
        selected = np.empty(0, dtype=np.int64)
        boundary_fraction = 0.0
        if editable_target_area > tolerance:
            order = np.argsort(coordinates, kind="stable")
            cumulative = np.cumsum(areas[order])
            boundary_position = min(
                int(np.searchsorted(cumulative, editable_target_area, side="left")),
                len(order) - 1,
            )
            selected = order[:boundary_position]
            full_area = float(areas[selected].sum())
            boundary_face = int(order[boundary_position])
            boundary_fraction = float(
                np.clip(
                    (editable_target_area - full_area) / float(areas[boundary_face]),
                    0.0,
                    1.0,
                )
            )

            for description_index, description in enumerate(descriptions):
                local_faces = face_slots[
                    selected[geom_slots[selected] == description_index]
                ]
                if not len(local_faces):
                    local_uv = np.empty((0, 3, 2), dtype=float)
                else:
                    local_uv = description["uv_triangles"][local_faces]
                if len(local_uv):
                    texture_id = int(description["texture_id"])
                    texture_triangles.setdefault(texture_id, []).append(local_uv)
                    selected_geoms.add(str(description["geom_name"]))

            # Split the single boundary triangle in barycentric / UV space.  The
            # texture mapping is affine inside a triangle, so scaling both edges by
            # sqrt(f) colors exactly fraction f of that triangle's 3-D surface area.
            boundary_description_index = int(geom_slots[boundary_face])
            boundary_description = descriptions[boundary_description_index]
            boundary_local_face = int(face_slots[boundary_face])
            boundary_uv = np.asarray(
                boundary_description["uv_triangles"][boundary_local_face],
                dtype=float,
            )
            scale = float(np.sqrt(boundary_fraction))
            boundary_patch = np.stack(
                [
                    boundary_uv[0],
                    boundary_uv[0] + scale * (boundary_uv[1] - boundary_uv[0]),
                    boundary_uv[0] + scale * (boundary_uv[2] - boundary_uv[0]),
                ],
                axis=0,
            )[None, ...]
            boundary_texture_id = int(boundary_description["texture_id"])
            texture_triangles.setdefault(boundary_texture_id, []).append(boundary_patch)
            selected_geoms.add(str(boundary_description["geom_name"]))
        texture_audits = []
        for texture_id, triangle_groups in texture_triangles.items():
            texture_audits.append(
                _blacken_texture_triangles(
                    sim,
                    texture_id,
                    np.concatenate(triangle_groups, axis=0),
                    rgba[:3],
                )
            )
        audits.append(
            {
                "object": object_name,
                "mode": "physical_render_surface_uv",
                "requested_surface_coverage": coverage,
                "actual_surface_coverage": target_area / total_area,
                "surface_area_total": total_area,
                "surface_area_black": target_area,
                "editable_surface_area": editable_area,
                "whole_component_surface_area_black": selected_atomic_area,
                "selection_axis": axis_name,
                "visual_geoms": sorted(selected_geoms),
                "whole_colored_visual_geoms": [
                    item["geom_name"] for item in selected_atomic
                ],
                "unsupported_visual_geoms": [
                    item["geom_name"] for item in unsupported_components
                ],
                "selected_face_count": int(
                    len(selected) + (1 if editable_target_area > tolerance else 0)
                ),
                "total_face_count": int(len(areas)),
                "boundary_face_fraction": boundary_fraction,
                "texture_updates": texture_audits,
                "collision_unchanged": True,
                "pose_unchanged": True,
            }
        )

    _forward_sim(sim)
    try:
        base_env._update_observables(force=True)
    except AttributeError:
        pass
    return audits


def _scale_mesh_vertices(sim: Any, mesh_id: int, scale: float) -> None:
    vert_adr = int(sim.model.mesh_vertadr[mesh_id])
    vert_num = int(sim.model.mesh_vertnum[mesh_id])
    sim.model.mesh_vert[vert_adr : vert_adr + vert_num] *= float(scale)


def _object_lower_bound_z(sim: Any, body_ids: set[int]) -> float | None:
    _forward_sim(sim)
    lows = []
    for geom_id in range(sim.model.ngeom):
        if int(sim.model.geom_bodyid[geom_id]) not in body_ids:
            continue
        geom_z = float(sim.data.geom_xpos[geom_id][2])
        if hasattr(sim.model, "geom_rbound"):
            radius = float(sim.model.geom_rbound[geom_id])
        else:
            radius = float(np.max(sim.model.geom_size[geom_id]))
        lows.append(geom_z - radius)
    return min(lows) if lows else None


def _offset_object_instance(base_env: Any, object_name: str, pos_offset: Any) -> None:
    free_joint = _free_joint_name(base_env, object_name)
    if free_joint is None:
        raise ValueError(f"LIBERO object {object_name!r} has no movable free joint for pos_offset")
    current_qpos = _get_joint_qpos(base_env.sim, free_joint)
    current_qpos[:3] = current_qpos[:3] + np.array(pos_offset, dtype=float)
    _set_joint_qpos(base_env.sim, free_joint, current_qpos)


def _object_bottom_reference_offset_z(base_env: Any, object_name: str) -> float | None:
    if object_name not in base_env.obj_body_id:
        raise ValueError(f"LIBERO object {object_name!r} was not found in obj_body_id")
    sim = base_env.sim
    root_body_id = int(base_env.obj_body_id[object_name])
    body_ids = set(_descendant_body_ids(sim, root_body_id))
    _forward_sim(sim)
    root_z = float(sim.data.body_xpos[root_body_id][2])

    bottom_site_offsets = []
    if hasattr(sim.model, "site_bodyid"):
        for site_id in range(sim.model.nsite):
            if int(sim.model.site_bodyid[site_id]) not in body_ids:
                continue
            site_name = ""
            try:
                site_name = sim.model.id2name(int(site_id), "site") or ""
            except Exception:
                pass
            if "bottom_site" in site_name:
                bottom_site_offsets.append(float(sim.data.site_xpos[site_id][2]) - root_z)
    if bottom_site_offsets:
        return min(bottom_site_offsets)

    lower_bound = _object_lower_bound_z(sim, body_ids)
    if lower_bound is None:
        return None
    return float(lower_bound - root_z)


def _object_bottom_world_z(base_env: Any, object_name: str) -> float | None:
    offset = _object_bottom_reference_offset_z(base_env, object_name)
    if offset is None:
        return None
    _forward_sim(base_env.sim)
    root_body_id = int(base_env.obj_body_id[object_name])
    return float(base_env.sim.data.body_xpos[root_body_id][2]) + float(offset)


def _object_named_bottom_site_world_z(
    base_env: Any, object_name: str
) -> float | None:
    sim = base_env.sim
    _forward_sim(sim)
    matches = []
    for site_id in range(int(sim.model.nsite)):
        site_name = ""
        try:
            site_name = sim.model.id2name(int(site_id), "site") or ""
        except Exception:
            pass
        if "bottom_site" in site_name and str(object_name) in site_name:
            matches.append(float(sim.data.site_xpos[site_id][2]))
    return min(matches) if matches else _object_bottom_world_z(
        base_env, object_name
    )


def _object_collision_lower_world_z(
    base_env: Any, object_name: str
) -> float | None:
    sim = base_env.sim
    root_body_id = int(base_env.obj_body_id[object_name])
    body_ids = set(_descendant_body_ids(sim, root_body_id))
    _forward_sim(sim)
    lows = []
    for geom_id in range(int(sim.model.ngeom)):
        if (
            int(sim.model.geom_bodyid[geom_id]) not in body_ids
            or int(sim.model.geom_group[geom_id]) != 0
        ):
            continue
        geom_type = int(sim.model.geom_type[geom_id])
        size = np.asarray(sim.model.geom_size[geom_id], dtype=float)
        if geom_type == 2:
            local = np.array([size[0], size[0], size[0]])
        elif geom_type == 3:
            local = np.array([size[0], size[0], size[1] + size[0]])
        elif geom_type == 4:
            local = size[:3]
        elif geom_type == 5:
            local = np.array([size[0], size[0], size[1]])
        elif geom_type == 6:
            local = size[:3]
        else:
            radius = float(sim.model.geom_rbound[geom_id])
            local = np.array([radius, radius, radius])
        rotation = np.asarray(
            sim.data.geom_xmat[geom_id], dtype=float
        ).reshape(3, 3)
        half_z = float((np.abs(rotation) @ local)[2])
        lows.append(float(sim.data.geom_xpos[geom_id][2]) - half_z)
    return min(lows) if lows else None


def _object_collision_aabb_bounds(
    base_env: Any, object_name: str
) -> tuple[np.ndarray, np.ndarray]:
    """Return world-space collision AABB bounds for an object or fixture."""
    sim = base_env.sim
    if object_name not in getattr(base_env, "obj_body_id", {}):
        raise ValueError(
            f"LIBERO object or fixture {object_name!r} has no body id"
        )
    root_body_id = int(base_env.obj_body_id[object_name])
    body_ids = set(_descendant_body_ids(sim, root_body_id))
    lows: list[np.ndarray] = []
    highs: list[np.ndarray] = []
    _forward_sim(sim)
    for geom_id in range(int(sim.model.ngeom)):
        if (
            int(sim.model.geom_bodyid[geom_id]) not in body_ids
            or int(sim.model.geom_group[geom_id]) != 0
        ):
            continue
        geom_type = int(sim.model.geom_type[geom_id])
        size = np.asarray(sim.model.geom_size[geom_id], dtype=float)
        if geom_type == 2:
            local_half = np.repeat(size[0], 3)
        elif geom_type == 3:
            local_half = np.array([size[0], size[0], size[1] + size[0]])
        elif geom_type == 4:
            local_half = size[:3]
        elif geom_type == 5:
            local_half = np.array([size[0], size[0], size[1]])
        elif geom_type == 6:
            local_half = size[:3]
        else:
            local_half = np.repeat(float(sim.model.geom_rbound[geom_id]), 3)
        rotation = np.asarray(sim.data.geom_xmat[geom_id]).reshape(3, 3)
        world_half = np.abs(rotation) @ local_half
        center = np.asarray(sim.data.geom_xpos[geom_id], dtype=float)
        lows.append(center - world_half)
        highs.append(center + world_half)
    if not lows:
        raise ValueError(
            f"No collision geoms found for LIBERO object {object_name!r}"
        )
    return np.min(lows, axis=0), np.max(highs, axis=0)


def _keep_precompiled_scaled_object_bottom(base_env: Any, object_name: str, scale: float) -> None:
    if scale <= 0:
        raise ValueError(f"LIBERO scale override for {object_name!r} must be positive, got {scale}")
    bottom_offset_z = _object_bottom_reference_offset_z(base_env, object_name)
    if bottom_offset_z is None:
        return
    z_delta = (1.0 / float(scale) - 1.0) * float(bottom_offset_z)
    _offset_object_instance(base_env, object_name, [0.0, 0.0, z_delta])


def _named_anchor_position(base_env: Any, name: str) -> np.ndarray:
    sim = base_env.sim
    if name in getattr(base_env, "obj_body_id", {}):
        return np.array(sim.data.body_xpos[base_env.obj_body_id[name]], dtype=float)
    site_aliases = {
        "robot0_eef": ["gripper0_grip_site", "robot0_grip_site"],
        "eef": ["gripper0_grip_site", "robot0_grip_site"],
    }
    for site_name in site_aliases.get(name, [name]):
        try:
            site_id = int(sim.model.site_name2id(site_name))
            return np.array(sim.data.site_xpos[site_id], dtype=float)
        except Exception:
            pass
    for body_name in (name, f"{name}_main"):
        try:
            body_id = int(sim.model.body_name2id(body_name))
            return np.array(sim.data.body_xpos[body_id], dtype=float)
        except Exception:
            pass
    raise ValueError(f"Could not resolve LIBERO pose anchor {name!r} as an object, site, or body")


def _select_candidate_position(
    base_env: Any,
    object_name: str,
    base_pos: np.ndarray,
    override: dict[str, Any],
) -> np.ndarray:
    candidate_offsets = override.get("candidate_offsets") or []
    if not candidate_offsets:
        return base_pos + np.array(override.get("offset", [0.0, 0.0, 0.0]), dtype=float)
    avoid_positions = []
    relative_to = str(override.get("relative_to", ""))
    for name in override.get("avoid_objects", []):
        if str(name) in {str(object_name), relative_to}:
            continue
        try:
            avoid_positions.append(_named_anchor_position(base_env, str(name)))
        except ValueError:
            continue
    bounds = override.get("workspace_bounds_xy")
    min_distance = float(override.get("min_distance_from_others", 0.0))
    require_min_distance = as_bool(override.get("require_min_distance", False))
    candidate_selection = str(
        override.get("candidate_selection", "max_clearance")
    ).strip().lower()
    best_pos = None
    best_score = -np.inf
    for offset in candidate_offsets:
        pos = base_pos + np.array(offset, dtype=float)
        if bounds and not (
            float(bounds[0][0]) <= float(pos[0]) <= float(bounds[0][1])
            and float(bounds[1][0]) <= float(pos[1]) <= float(bounds[1][1])
        ):
            continue
        nearest = min(
            (float(np.linalg.norm(pos[:2] - other[:2])) for other in avoid_positions),
            default=1.0,
        )
        if require_min_distance and nearest < min_distance:
            continue
        if candidate_selection == "first_clear":
            return pos
        score = nearest - max(0.0, min_distance - nearest) * 10.0
        if score > best_score:
            best_score = score
            best_pos = pos
    if best_pos is None:
        if require_min_distance:
            raise ValueError(
                f"No collision-clear candidate for LIBERO object {object_name!r} "
                f"at minimum XY clearance {min_distance:.3f} m"
            )
        best_pos = base_pos + np.array(candidate_offsets[0], dtype=float)
    return best_pos


def _zero_joint_velocity(sim: Any, joint_name: str) -> None:
    """Discard dynamic state accumulated in an added object's parking region."""
    address = sim.model.get_joint_qvel_addr(joint_name)
    if isinstance(address, tuple):
        dof_slice = slice(int(address[0]), int(address[1]))
    else:
        dof_slice = slice(int(address), int(address) + 1)
    sim.data.qvel[dof_slice] = 0.0
    if hasattr(sim.data, "qacc"):
        sim.data.qacc[dof_slice] = 0.0
    if hasattr(sim.data, "qacc_warmstart"):
        sim.data.qacc_warmstart[dof_slice] = 0.0
    if hasattr(sim.data, "qfrc_applied"):
        sim.data.qfrc_applied[dof_slice] = 0.0


def _body_chain_names(sim: Any, body_id: int) -> list[str]:
    names = []
    while int(body_id) > 0:
        name = sim.model.body_id2name(int(body_id))
        if name:
            names.append(str(name))
        parent = int(sim.model.body_parentid[int(body_id)])
        if parent == int(body_id):
            break
        body_id = parent
    return names


def _contact_free_candidate_position(
    base_env: Any,
    object_name: str,
    free_joint: str,
    base_pos: np.ndarray,
    override: dict[str, Any],
    quat: np.ndarray,
    current_z: float,
) -> np.ndarray:
    """Try candidate poses in MuJoCo and return the first non-penetrating pose.

    Static centre-distance checks are insufficient in crowded LIBERO scenes:
    cabinets, racks, and irregular meshes can overlap a candidate even when body
    centres are far apart.  This routine poses the object at its final support
    height, calls ``forward()``, and evaluates the actual collision contacts.
    """
    sim = base_env.sim
    candidate_offsets = override.get("candidate_offsets") or []
    if not candidate_offsets:
        return base_pos
    bounds = override.get("workspace_bounds_xy")
    target_name = str(override.get("relative_to", ""))
    forbidden_names = {
        str(name)
        for name in override.get("avoid_objects", [])
        if str(name) != str(object_name)
    }
    if target_name:
        forbidden_names.add(target_name)
    allowed_names = {
        str(name) for name in override.get("allowed_contact_objects", [])
    }
    forbidden_names.difference_update(allowed_names)
    allowed_body_ids = {
        int(base_env.obj_body_id[name])
        for name in allowed_names
        if name in getattr(base_env, "obj_body_id", {})
    }
    forbidden_body_ids = {
        int(base_env.obj_body_id[name])
        for name in forbidden_names
        if name in getattr(base_env, "obj_body_id", {})
    }
    focus_root = int(base_env.obj_body_id[object_name])
    focus_body_ids = set(_descendant_body_ids(sim, focus_root))
    focus_geom_ids = {
        geom_id
        for geom_id in range(int(sim.model.ngeom))
        if int(sim.model.geom_bodyid[geom_id]) in focus_body_ids
        and int(sim.model.geom_group[geom_id]) == 0
    }
    support_tokens = ("floor", "table", "counter", "workspace")
    attempted = []
    fallback = None
    fallback_count = np.inf
    snapshot = {
        "time": float(sim.data.time),
        "qpos": np.array(sim.data.qpos, copy=True),
        "qvel": np.array(sim.data.qvel, copy=True),
        "act": np.array(sim.data.act, copy=True),
        "ctrl": np.array(sim.data.ctrl, copy=True),
        "qacc_warmstart": np.array(sim.data.qacc_warmstart, copy=True),
        "mocap_pos": np.array(sim.data.mocap_pos, copy=True),
        "mocap_quat": np.array(sim.data.mocap_quat, copy=True),
    }

    def restore_snapshot() -> None:
        sim.data.time = snapshot["time"]
        sim.data.qpos[:] = snapshot["qpos"]
        sim.data.qvel[:] = snapshot["qvel"]
        if sim.data.act.size:
            sim.data.act[:] = snapshot["act"]
        sim.data.ctrl[:] = snapshot["ctrl"]
        sim.data.qacc_warmstart[:] = snapshot["qacc_warmstart"]
        if sim.data.mocap_pos.size:
            sim.data.mocap_pos[:] = snapshot["mocap_pos"]
            sim.data.mocap_quat[:] = snapshot["mocap_quat"]
        _forward_sim(sim)

    for offset in candidate_offsets:
        restore_snapshot()
        pos = np.asarray(base_pos, dtype=float) + np.asarray(offset, dtype=float)
        if bounds and not (
            float(bounds[0][0]) <= float(pos[0]) <= float(bounds[0][1])
            and float(bounds[1][0]) <= float(pos[1]) <= float(bounds[1][1])
        ):
            continue
        provisional = np.concatenate(
            [np.array([pos[0], pos[1], current_z], dtype=float), quat]
        )
        _set_joint_qpos(sim, free_joint, provisional)
        _zero_joint_velocity(sim, free_joint)
        _forward_sim(sim)
        if override.get("support_world_z") is not None:
            lower = (
                _object_collision_lower_world_z(base_env, object_name)
                if as_bool(override.get("align_collision_bottom", False))
                else _object_named_bottom_site_world_z(base_env, object_name)
            )
            if lower is None:
                raise ValueError(
                    f"Could not resolve support bottom for LIBERO object "
                    f"{object_name!r}"
                )
            pos[2] = (
                float(current_z)
                + float(override["support_world_z"])
                - float(lower)
            )
        _set_joint_qpos(sim, free_joint, np.concatenate([pos, quat]))
        _zero_joint_velocity(sim, free_joint)
        _forward_sim(sim)

        disallowed = []
        for contact_index in range(int(sim.data.ncon)):
            contact = sim.data.contact[contact_index]
            geom1, geom2 = int(contact.geom1), int(contact.geom2)
            if geom1 in focus_geom_ids:
                other_geom = geom2
            elif geom2 in focus_geom_ids:
                other_geom = geom1
            else:
                continue
            other_body = int(sim.model.geom_bodyid[other_geom])
            if other_body in focus_body_ids:
                continue
            chain_ids = []
            cursor = other_body
            while cursor > 0:
                chain_ids.append(cursor)
                parent = int(sim.model.body_parentid[cursor])
                if parent == cursor:
                    break
                cursor = parent
            if any(body_id in allowed_body_ids for body_id in chain_ids):
                continue
            chain_names = _body_chain_names(sim, other_body)
            geom_name = sim.model.geom_id2name(other_geom) or ""
            joined = "/".join(chain_names + [str(geom_name)]).lower()
            if any(token in joined for token in support_tokens):
                continue
            explicitly_forbidden = any(
                body_id in forbidden_body_ids for body_id in chain_ids
            )
            # Match validator tolerance for incidental numerical contacts, but
            # never permit contact with the target or another named scene entity.
            if explicitly_forbidden or float(contact.dist) < -1e-5:
                disallowed.append(
                    {
                        "other": joined,
                        "distance_m": float(contact.dist),
                    }
                )
        trial_drift = 0.0
        if not disallowed:
            start_xyz = np.asarray(
                sim.data.body_xpos[focus_root], dtype=float
            ).copy()
            for _ in range(int(override.get("candidate_trial_sim_steps", 0))):
                sim.step()
            _forward_sim(sim)
            trial_drift = float(
                np.linalg.norm(
                    np.asarray(sim.data.body_xpos[focus_root], dtype=float)
                    - start_xyz
                )
            )
            if trial_drift > float(
                override.get("candidate_trial_max_drift", .012)
            ):
                disallowed.append(
                    {"other": "dynamic_drift", "distance_m": trial_drift}
                )
        attempted.append(
            {
                "xyz": np.asarray(pos, dtype=float).tolist(),
                "disallowed_contacts": disallowed,
                "trial_drift_m": trial_drift,
            }
        )
        if len(disallowed) < fallback_count:
            fallback = np.asarray(pos, dtype=float).copy()
            fallback_count = len(disallowed)
        if not disallowed:
            audit = getattr(base_env, "_a4_candidate_selection_audit", {})
            audit[str(object_name)] = {
                "selected_xyz": np.asarray(pos, dtype=float).tolist(),
                "candidate_index": len(attempted) - 1,
                "attempted": attempted,
            }
            base_env._a4_candidate_selection_audit = audit
            restore_snapshot()
            return np.asarray(pos, dtype=float)

    restore_snapshot()
    audit = getattr(base_env, "_a4_candidate_selection_audit", {})
    audit[str(object_name)] = {
        "selected_xyz": (
            np.asarray(fallback, dtype=float).tolist()
            if fallback is not None
            else None
        ),
        "candidate_index": None,
        "attempted": attempted,
        "fallback": True,
    }
    base_env._a4_candidate_selection_audit = audit
    if fallback is None:
        raise ValueError(
            f"No in-bounds MuJoCo candidate for LIBERO object {object_name!r}"
        )
    return fallback


def _set_libero_object_or_fixture_position(
    base_env: Any,
    object_name: str,
    position: Any,
) -> None:
    sim = base_env.sim
    position = np.array(position, dtype=float)
    free_joint = _free_joint_name(base_env, object_name)
    if free_joint is not None:
        qpos = _get_joint_qpos(sim, free_joint)
        _set_joint_qpos(sim, free_joint, np.concatenate([position, qpos[3:]]))
        return
    if object_name not in base_env.obj_body_id:
        raise ValueError(f"LIBERO object or fixture {object_name!r} has no body id")
    body_id = int(base_env.obj_body_id[object_name])
    _forward_sim(sim)
    current_world = np.array(sim.data.body_xpos[body_id], dtype=float)
    sim.model.body_pos[body_id, :3] += position - current_world
    _forward_sim(sim)


def _apply_same_position_replacement(
    base_env: Any,
    rule: dict[str, Any],
) -> None:
    sim = base_env.sim
    target = str(rule["target"])
    replacement = str(rule["replacement"])
    original_target_pos = _named_anchor_position(base_env, target).copy()
    target_joint = _free_joint_name(base_env, target)
    target_qpos = (
        _get_joint_qpos(sim, target_joint).copy()
        if target_joint is not None
        else None
    )
    target_bottom_world = _object_bottom_world_z(base_env, target)
    aside_rule = dict(rule)
    aside_rule["object"] = target
    aside_rule["relative_to"] = target
    aside_pos = _select_candidate_position(
        base_env,
        target,
        original_target_pos,
        aside_rule,
    )
    replacement_joint = _free_joint_name(base_env, replacement)
    replacement_fixed = (
        replacement_joint is None
        and as_bool(rule.get("replacement_fixed_fixture", False))
        and replacement in base_env.obj_body_id
    )
    if replacement_joint is None and not replacement_fixed:
        raise ValueError(
            f"LIBERO same-position replacement {replacement!r} needs a movable free joint"
        )
    if replacement_joint is not None:
        replacement_qpos = _get_joint_qpos(sim, replacement_joint)
        replacement_pos = replacement_qpos[:3].copy()
    else:
        _forward_sim(sim)
        replacement_body_id = int(base_env.obj_body_id[replacement])
        replacement_pos = np.asarray(
            sim.data.body_xpos[replacement_body_id], dtype=float
        ).copy()
    replacement_support_z = float(replacement_pos[2])
    replacement_pos[:2] = original_target_pos[:2]
    if rule.get("replacement_support_world_z") is not None:
        replacement_bottom = _object_named_bottom_site_world_z(
            base_env, replacement
        )
        if replacement_bottom is None:
            raise ValueError(
                f"Could not resolve bottom site for LIBERO replacement "
                f"{replacement!r}"
            )
        replacement_pos[2] += (
            float(rule["replacement_support_world_z"])
            - float(replacement_bottom)
        )
    elif rule.get("replacement_support_z_offset") is not None:
        replacement_pos[2] += float(rule["replacement_support_z_offset"])
    elif (
        as_bool(rule.get("replacement_support_surface_from_target", False))
        and target_bottom_world is not None
    ):
        replacement_bottom_world = _object_bottom_world_z(base_env, replacement)
        if replacement_bottom_world is not None:
            replacement_pos[2] += (
                float(target_bottom_world) - float(replacement_bottom_world)
            )
    elif not as_bool(rule.get("replacement_preserve_support_z", True)):
        replacement_pos[2] = original_target_pos[2]
    if target_qpos is not None:
        aside_pos[2] = float(target_qpos[2])
    elif as_bool(rule.get("target_support_z_from_replacement", False)):
        aside_pos[2] = replacement_support_z
    _set_libero_object_or_fixture_position(base_env, target, aside_pos)
    if replacement_joint is not None:
        _set_joint_qpos(
            sim,
            replacement_joint,
            np.concatenate([replacement_pos, replacement_qpos[3:]]),
        )
    else:
        _set_libero_object_or_fixture_position(base_env, replacement, replacement_pos)
    _forward_sim(sim)
    audit = getattr(base_env, "_a4_same_position_audit", {})
    audit[target] = {
        "original_target_xyz": original_target_pos.tolist(),
        "aside_target_xyz": np.asarray(aside_pos, dtype=float).tolist(),
        "replacement": replacement,
        "replacement_xyz": np.asarray(replacement_pos, dtype=float).tolist(),
    }
    base_env._a4_same_position_audit = audit


def _scale_object_instance(
    base_env: Any,
    object_name: str,
    scale: float,
    *,
    pos_offset: Any = None,
    mass_scale: Any = None,
    keep_bottom: bool = False,
) -> None:
    if object_name not in base_env.obj_body_id:
        raise ValueError(f"LIBERO object {object_name!r} was not found in obj_body_id")
    if scale <= 0:
        raise ValueError(f"LIBERO scale override for {object_name!r} must be positive, got {scale}")

    sim = base_env.sim
    root_body_id = int(base_env.obj_body_id[object_name])
    body_ids = set(_descendant_body_ids(sim, root_body_id))
    scaled_mesh_ids: set[int] = set()
    lower_before = _object_lower_bound_z(sim, body_ids) if keep_bottom else None

    for body_id in body_ids:
        if body_id != root_body_id:
            sim.model.body_pos[body_id] *= float(scale)

    for geom_id in range(sim.model.ngeom):
        if int(sim.model.geom_bodyid[geom_id]) not in body_ids:
            continue
        sim.model.geom_pos[geom_id] *= float(scale)
        sim.model.geom_size[geom_id] *= float(scale)
        mesh_id = int(sim.model.geom_dataid[geom_id])
        if mesh_id >= 0 and mesh_id not in scaled_mesh_ids:
            _scale_mesh_vertices(sim, mesh_id, scale)
            scaled_mesh_ids.add(mesh_id)

    if hasattr(sim.model, "site_bodyid"):
        for site_id in range(sim.model.nsite):
            if int(sim.model.site_bodyid[site_id]) in body_ids:
                sim.model.site_pos[site_id] *= float(scale)
                sim.model.site_size[site_id] *= float(scale)

    if mass_scale is not None:
        mass_scale_f = float(mass_scale)
        for body_id in body_ids:
            sim.model.body_mass[body_id] *= mass_scale_f
            if hasattr(sim.model, "body_inertia"):
                sim.model.body_inertia[body_id] *= mass_scale_f

    if keep_bottom and lower_before is not None:
        lower_after = _object_lower_bound_z(sim, body_ids)
        if lower_after is not None:
            _offset_object_instance(base_env, object_name, [0.0, 0.0, float(lower_before - lower_after)])

    if pos_offset is not None:
        _offset_object_instance(base_env, object_name, pos_offset)


def apply_libero_env_edit(env_in: Any, spec: dict[str, Any] | None) -> None:
    if not spec or spec.get("status") != "implemented":
        return

    base_env = getattr(env_in, "env", env_in)
    sim = base_env.sim
    runtime_audit = {"applied": True, "articulations": [], "object_poses": []}

    for override in spec.get("articulation_qpos_overrides", []):
        joint_name = str(override["joint"])
        _set_joint_qpos(
            sim,
            joint_name,
            np.array([float(override["qpos"])], dtype=float),
        )
        runtime_audit["articulations"].append(
            {
                "joint": joint_name,
                "requested_qpos": float(override["qpos"]),
                "actual_qpos": float(
                    np.asarray(_get_joint_qpos(sim, joint_name)).reshape(-1)[0]
                ),
            }
        )

    for override in spec.get("object_scale_overrides", []):
        pos_offset = override.get("pos_offset")
        if pos_offset is None and "z_offset" in override:
            pos_offset = [0.0, 0.0, float(override["z_offset"])]
        if is_precompile_xml_scale_override(override):
            if as_bool(override.get("keep_bottom", False)):
                _keep_precompiled_scaled_object_bottom(
                    base_env,
                    str(override["object"]),
                    float(override["scale"]),
                )
            if pos_offset is not None:
                _offset_object_instance(base_env, str(override["object"]), pos_offset)
            continue
        _scale_object_instance(
            base_env,
            str(override["object"]),
            float(override["scale"]),
            pos_offset=pos_offset,
            mass_scale=override.get("mass_scale"),
            keep_bottom=as_bool(override.get("keep_bottom", False)),
        )

    for rule in spec.get("same_position_replacements", []):
        _apply_same_position_replacement(base_env, rule)

    for override in spec.get("object_pose_overrides", []):
        object_name = str(override["object"])
        is_fixed_fixture = as_bool(override.get("fixed_fixture", False))
        free_joint = None if is_fixed_fixture else _free_joint_name(base_env, object_name)
        is_movable = free_joint is not None
        if not is_movable and not is_fixed_fixture:
            raise ValueError(f"LIBERO object {object_name!r} has no movable free joint")
        if is_movable:
            current_qpos = _get_joint_qpos(sim, free_joint)
            current_z = float(current_qpos[2])
            current_quat = current_qpos[3:]
        else:
            if object_name not in base_env.obj_body_id:
                raise ValueError(f"LIBERO fixed fixture {object_name!r} has no body id")
            body_id = int(base_env.obj_body_id[object_name])
            _forward_sim(sim)
            current_z = float(sim.data.body_xpos[body_id, 2])
            current_quat = np.array(sim.data.body_xquat[body_id], dtype=float)
        candidate_base_pos = None
        if "pos" in override:
            pos = np.array(override["pos"], dtype=float)
        elif "on_top_of" in override:
            support_name = str(override["on_top_of"])
            support_low, support_high = _object_collision_aabb_bounds(
                base_env, support_name
            )
            object_low, object_high = _object_collision_aabb_bounds(
                base_env, object_name
            )
            object_half_z = 0.5 * float(object_high[2] - object_low[2])
            pos = np.array(
                [
                    0.5 * float(support_low[0] + support_high[0]),
                    0.5 * float(support_low[1] + support_high[1]),
                    float(support_high[2])
                    + object_half_z
                    + float(override.get("support_gap_m", 0.001)),
                ],
                dtype=float,
            )
        elif "between" in override:
            anchors = list(override["between"])
            if len(anchors) != 2:
                raise ValueError(
                    f"LIBERO pose override for {object_name!r} needs exactly two between anchors"
                )
            fraction = float(override.get("fraction", 0.5))
            start = _named_anchor_position(base_env, str(anchors[0]))
            end = _named_anchor_position(base_env, str(anchors[1]))
            candidate_base_pos = start + fraction * (end - start)
            pos = _select_candidate_position(
                base_env,
                object_name,
                candidate_base_pos,
                override,
            )
        elif "relative_to" in override:
            anchor = str(override["relative_to"])
            candidate_base_pos = _named_anchor_position(base_env, anchor)
            pos = _select_candidate_position(
                base_env,
                object_name,
                candidate_base_pos,
                override,
            )
        else:
            raise ValueError(
                f"LIBERO pose override for {object_name!r} needs pos, between, or relative_to"
            )
        if override.get("z_from"):
            pos[2] = _named_anchor_position(base_env, str(override["z_from"]))[2]
        if override.get("support_world_z") is not None:
            object_bottom = _object_named_bottom_site_world_z(
                base_env, object_name
            )
            if object_bottom is None:
                raise ValueError(
                    f"Could not resolve bottom site for LIBERO object "
                    f"{object_name!r}"
                )
            pos[2] = (
                float(current_z)
                + float(override["support_world_z"])
                - float(object_bottom)
            )
        elif override.get("support_z_offset") is not None:
            pos[2] = float(current_z) + float(override["support_z_offset"])
        elif override.get("support_surface_from"):
            target_bottom = _object_bottom_world_z(
                base_env, str(override["support_surface_from"])
            )
            object_bottom = _object_bottom_world_z(base_env, object_name)
            if target_bottom is None or object_bottom is None:
                raise ValueError(
                    f"Could not align support surfaces for LIBERO object "
                    f"{object_name!r}"
                )
            pos[2] = float(current_z) + float(target_bottom) - float(object_bottom)
        elif as_bool(override.get("preserve_z", False)):
            pos[2] = current_z
        if "quat_from" in override:
            quat_joint = _free_joint_name(base_env, str(override["quat_from"]))
            if quat_joint is None:
                raise ValueError(
                    f"LIBERO quat_from object {override['quat_from']!r} has no movable free joint"
                )
            quat = _get_joint_qpos(sim, quat_joint)[3:]
        else:
            quat = np.array(override.get("quat", current_quat), dtype=float)
        if (
            is_movable
            and candidate_base_pos is not None
            and override.get("candidate_offsets")
            and str(override.get("candidate_selection", "")).strip().lower()
            == "first_contact_free"
        ):
            pos = _contact_free_candidate_position(
                base_env,
                object_name,
                free_joint,
                np.asarray(candidate_base_pos, dtype=float),
                override,
                np.asarray(quat, dtype=float),
                current_z,
            )
        if (
            is_movable
            and override.get("support_world_z") is not None
            and as_bool(override.get("align_collision_bottom", False))
            and str(override.get("candidate_selection", "")).strip().lower()
            != "first_contact_free"
        ):
            provisional = np.concatenate([
                np.array([pos[0], pos[1], current_z], dtype=float),
                np.asarray(quat, dtype=float),
            ])
            _set_joint_qpos(sim, free_joint, provisional)
            collision_lower = _object_collision_lower_world_z(
                base_env, object_name
            )
            if collision_lower is None:
                raise ValueError(
                    f"Could not resolve collision bottom for LIBERO object "
                    f"{object_name!r}"
                )
            pos[2] = (
                float(current_z)
                + float(override["support_world_z"])
                - float(collision_lower)
            )
        pose_audit = getattr(base_env, "_a4_pose_override_audit", {})
        pose_audit[str(object_name)] = {
            "requested_xyz": np.asarray(pos, dtype=float).tolist(),
            "current_z_before": float(current_z),
            "support_surface_from": override.get("support_surface_from"),
        }
        base_env._a4_pose_override_audit = pose_audit
        if is_movable:
            _set_joint_qpos(sim, free_joint, np.concatenate([pos, quat]))
            _zero_joint_velocity(sim, free_joint)
            runtime_audit["object_poses"].append(
                {
                    "object": object_name,
                    "joint": free_joint,
                    "actual_qpos": _get_joint_qpos(sim, free_joint).tolist(),
                }
            )
        else:
            _set_libero_object_or_fixture_position(base_env, object_name, pos)
            body_id = int(base_env.obj_body_id[object_name])
            sim.model.body_quat[body_id, :4] = quat
            _forward_sim(sim)

    runtime_audit["surface_blackening"] = apply_object_surface_black_overrides(
        base_env,
        spec.get("object_surface_black_fraction_overrides", []),
    )
    if runtime_audit["surface_blackening"]:
        spec["_applied_surface_black_overrides"] = copy.deepcopy(
            runtime_audit["surface_blackening"]
        )

    _forward_sim(sim)
    base_env._wam_syncbench_env_edit_audit = runtime_audit


def _resolve_occlusion_targets(env_in: Any, spec: dict[str, Any], libero_benchmark: str, task_idx: int) -> list[str]:
    policy = spec.get("target_policy", "obj_of_interest")
    if policy == "obj_of_interest":
        return list(getattr(env_in, "obj_of_interest", []))
    if policy == "object_names":
        keyed_targets = spec.get("target_objects_by_task", {})
        key = f"{libero_benchmark}:{task_idx}"
        targets = keyed_targets.get(key, spec.get("object_names", []))
        return [str(target) for target in targets]
    raise ValueError(f"Unsupported occlusion target_policy: {policy}")


def _target_mask_from_segmentation(env_in: Any, segmentation_image: Any, target_objects: list[str]) -> tuple[np.ndarray, list[str]]:
    seg = np.asarray(segmentation_image)
    if seg.ndim == 3 and seg.shape[-1] == 1:
        seg = seg[..., 0]

    instance_to_id = getattr(env_in, "instance_to_id", {})
    mask = np.zeros(seg.shape, dtype=bool)
    missing_targets = []
    for obj_name in target_objects:
        seg_id = instance_to_id.get(obj_name)
        if seg_id is None:
            missing_targets.append(obj_name)
            continue
        mask |= seg == seg_id
    return mask, missing_targets


def _expand_mask_geometry(mask: np.ndarray, geometry: Any) -> np.ndarray:
    geometry = str(geometry or "segmentation")
    if geometry == "segmentation":
        return mask
    if geometry != "bbox":
        raise ValueError(f"Unsupported occlusion mask_geometry: {geometry}")
    ys, xs = np.where(mask)
    bbox_mask = np.zeros_like(mask, dtype=bool)
    if len(xs) == 0 or len(ys) == 0:
        return bbox_mask
    bbox_mask[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1] = True
    return bbox_mask


def _bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _dilate_mask(mask: np.ndarray, dilate_px: Any) -> np.ndarray:
    dilate_px_i = int(dilate_px or 0)
    if dilate_px_i <= 0 or not mask.any():
        return mask
    kernel_size = dilate_px_i * 2 + 1
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    return cv2.dilate(mask.astype(np.uint8), kernel, iterations=1).astype(bool)


def _apply_target_shape_warp(image: np.ndarray, mask: np.ndarray, spec: dict[str, Any]) -> dict[str, Any]:
    """Warp only the visible target silhouette while simulator physics stays fixed."""
    mask = np.asarray(mask, dtype=bool)
    ys, xs = np.where(mask)
    if len(xs) < 4 or len(ys) < 4:
        return {"changed_pixels": 0, "target_mask_pixels": int(mask.sum()), "visible": False}

    family = str(spec.get("warp_family", "horizontal_expand"))
    amount = float(spec.get("warp_amount", 0.25))
    height, width = image.shape[:2]
    cx = 0.5 * float(xs.min() + xs.max())
    cy = 0.5 * float(ys.min() + ys.max())
    half_w = max(1.0, 0.5 * float(xs.max() - xs.min() + 1))
    half_h = max(1.0, 0.5 * float(ys.max() - ys.min() + 1))
    grid_y, grid_x = np.indices((height, width), dtype=np.float32)
    nx = (grid_x - cx) / half_w
    ny = (grid_y - cy) / half_h
    source_x = grid_x.copy()
    source_y = grid_y.copy()

    if family == "horizontal_expand":
        source_x = cx + (grid_x - cx) / (1.0 + amount)
    elif family == "horizontal_contract":
        source_x = cx + (grid_x - cx) / max(0.20, 1.0 - amount)
    elif family == "vertical_expand":
        source_y = cy + (grid_y - cy) / (1.0 + amount)
    elif family == "vertical_contract":
        source_y = cy + (grid_y - cy) / max(0.20, 1.0 - amount)
    elif family == "shear":
        source_x = grid_x - amount * half_w * ny
    elif family == "upper_offset":
        upper_weight = np.clip((1.0 - ny) * 0.5, 0.0, 1.0)
        source_x = grid_x - amount * half_w * upper_weight
    elif family == "hourglass":
        scale = 1.0 - amount * np.exp(-((ny / 0.55) ** 2))
        source_x = cx + (grid_x - cx) / np.maximum(scale, 0.20)
    elif family == "double_lobe":
        lobes = np.exp(-(((ny - 0.55) / 0.34) ** 2)) + np.exp(-(((ny + 0.55) / 0.34) ** 2))
        waist = np.exp(-((ny / 0.30) ** 2))
        scale = 1.0 + amount * (0.70 * lobes - 0.35 * waist)
        source_x = cx + (grid_x - cx) / np.maximum(scale, 0.25)
    else:
        raise ValueError(f"Unsupported target shape warp family: {family}")

    before = np.array(image, copy=True)
    mask_u8 = mask.astype(np.uint8) * 255
    warped_mask = cv2.remap(mask_u8, source_x, source_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    warped_image = cv2.remap(before, source_x, source_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    background = cv2.inpaint(before, mask_u8, int(spec.get("inpaint_radius", 3)), cv2.INPAINT_TELEA)
    destination = warped_mask >= 96
    image[:] = background
    image[destination] = warped_image[destination]
    changed = np.max(np.abs(image.astype(np.int16) - before.astype(np.int16)), axis=2) >= 8
    return {
        "changed_pixels": int(changed.sum()),
        "target_mask_pixels": int(mask.sum()),
        "warped_mask_pixels": int(destination.sum()),
        "visible": True,
        "warp_family": family,
        "warp_amount": amount,
    }


def _draw_target_photo_replacement(
    image: np.ndarray,
    source_image: np.ndarray,
    mask: np.ndarray,
    object_name: str,
    spec: dict[str, Any],
) -> dict[str, Any] | None:
    bbox = _bbox_from_mask(mask)
    if bbox is None:
        return None

    x0, y0, x1, y1 = bbox
    crop = source_image[y0:y1, x0:x1].copy()
    crop_mask = mask[y0:y1, x0:x1]
    if crop.size == 0 or not crop_mask.any():
        return None

    background_rgb = np.array(spec.get("photo_background_rgb", [245, 245, 245]), dtype=np.uint8)
    border_rgb = tuple(int(v) for v in spec.get("photo_border_rgb", [20, 20, 20]))
    border_px = max(0, int(spec.get("photo_border_px", 1) or 0))
    padding_px = max(0, int(spec.get("photo_padding_px", 2) or 0))
    alpha = max(0.0, min(1.0, float(spec.get("photo_alpha", 1.0))))
    desaturate = as_bool(spec.get("photo_desaturate", False))

    card = np.empty_like(crop)
    card[:, :] = background_rgb

    ys, xs = np.where(crop_mask)
    subject = crop[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1].copy()
    subject_mask = crop_mask[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
    subject_bg = np.empty_like(subject)
    subject_bg[:, :] = background_rgb
    subject_bg[subject_mask] = subject[subject_mask]
    if desaturate:
        gray = cv2.cvtColor(subject_bg, cv2.COLOR_RGB2GRAY)
        subject_bg = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

    inner_x0 = min(card.shape[1], border_px + padding_px)
    inner_y0 = min(card.shape[0], border_px + padding_px)
    inner_x1 = max(inner_x0, card.shape[1] - border_px - padding_px)
    inner_y1 = max(inner_y0, card.shape[0] - border_px - padding_px)
    inner_w = inner_x1 - inner_x0
    inner_h = inner_y1 - inner_y0
    if inner_w > 0 and inner_h > 0:
        resized = cv2.resize(subject_bg, (inner_w, inner_h), interpolation=cv2.INTER_AREA)
        card[inner_y0:inner_y1, inner_x0:inner_x1] = resized

    if border_px > 0:
        for offset in range(border_px):
            cv2.rectangle(
                card,
                (offset, offset),
                (max(offset, card.shape[1] - 1 - offset), max(offset, card.shape[0] - 1 - offset)),
                border_rgb,
                1,
            )

    region = image[y0:y1, x0:x1]
    if alpha < 1.0:
        image[y0:y1, x0:x1] = cv2.addWeighted(card, alpha, region, 1.0 - alpha, 0)
    else:
        image[y0:y1, x0:x1] = card

    changed_area = int((y1 - y0) * (x1 - x0))
    return {
        "object": str(object_name),
        "object_bbox": [int(x0), int(y0), int(x1), int(y1)],
        "photo_rect": [int(x0), int(y0), int(x1), int(y1)],
        "changed_pixels": changed_area,
        "photo_source": "current_frame_target_crop",
    }


def _stable_text_seed(text: Any) -> int:
    seed = 2166136261
    for char in str(text):
        seed ^= ord(char)
        seed = (seed * 16777619) & 0xFFFFFFFF
    return seed


def _apply_noise_to_mask(
    image: np.ndarray,
    mask: np.ndarray,
    spec: dict[str, Any],
    libero_benchmark: str,
    task_idx: int,
    view_name: str,
) -> dict[str, Any]:
    masked_pixels = int(mask.sum())
    if masked_pixels <= 0:
        return {
            "masked_pixels": 0,
            "mask_fraction": 0.0,
            "noise_type": spec.get("noise_type", "gaussian"),
            "noise_std": float(spec.get("noise_std", 45.0)),
        }

    seed = spec.get("noise_seed")
    if seed is None:
        rng = np.random.default_rng()
    else:
        image_signature = int(np.asarray(image, dtype=np.uint32).sum() % 1000003)
        seed_value = (
            int(seed)
            + int(task_idx) * 1009
            + _stable_text_seed(libero_benchmark)
            + _stable_text_seed(view_name)
            + image_signature
        ) & 0xFFFFFFFF
        rng = np.random.default_rng(seed_value)

    noise_type = str(spec.get("noise_type", "gaussian")).lower()
    if noise_type == "gaussian":
        noise_std = float(spec.get("noise_std", 45.0))
        noise_mean = float(spec.get("noise_mean", 0.0))
        region = image[mask].astype(np.float32)
        noise = rng.normal(noise_mean, noise_std, size=region.shape)
        image[mask] = np.clip(region + noise, 0, 255).astype(np.uint8)
        return {
            "masked_pixels": masked_pixels,
            "mask_fraction": float(mask.mean()),
            "noise_type": noise_type,
            "noise_std": noise_std,
            "noise_mean": noise_mean,
        }

    if noise_type in {"salt_pepper", "salt-and-pepper"}:
        probability = max(0.0, min(1.0, float(spec.get("noise_probability", 0.2))))
        region = image[mask].copy()
        selector = rng.random(region.shape[:1]) < probability
        salt = rng.random(selector.shape) < 0.5
        region[selector & salt] = 255
        region[selector & ~salt] = 0
        image[mask] = region
        return {
            "masked_pixels": masked_pixels,
            "mask_fraction": float(mask.mean()),
            "noise_type": noise_type,
            "noise_probability": probability,
        }

    raise ValueError(f"Unsupported noise_type: {noise_type}")


def _interaction_world_to_pixel(base_env, camera_name, points_world, height, width):
    """Project world points into a LIBERO camera without using segmentation."""
    try:
        camera_id = base_env.sim.model.camera_name2id(camera_name)
        camera_position = np.asarray(base_env.sim.data.cam_xpos[camera_id])
        camera_rotation = np.asarray(
            base_env.sim.data.cam_xmat[camera_id]
        ).reshape(3, 3)
        camera_rotation = camera_rotation @ np.diag([1.0, -1.0, -1.0])
        focal = 0.5 * height / np.tan(
            float(base_env.sim.model.cam_fovy[camera_id]) * np.pi / 360.0
        )
    except Exception:
        return [None] * len(points_world)
    margin = int(max(2, 0.03 * max(height, width)))
    projected = []
    for point in points_world:
        try:
            camera_point = camera_rotation.T @ (
                np.asarray(point, dtype=float) - camera_position
            )
            if camera_point[2] <= 0:
                projected.append(None)
                continue
            column = int(round(focal * camera_point[0] / camera_point[2] + width / 2))
            row = int(round(focal * camera_point[1] / camera_point[2] + height / 2))
            inside = (
                margin <= row < height - margin
                and margin <= column < width - margin
            )
            projected.append((row, column) if inside else None)
        except Exception:
            projected.append(None)
    return projected


def _interaction_project_raw(base_env, camera_name, point_world, height, width):
    try:
        camera_id = base_env.sim.model.camera_name2id(camera_name)
        camera_position = np.asarray(base_env.sim.data.cam_xpos[camera_id])
        camera_rotation = np.asarray(
            base_env.sim.data.cam_xmat[camera_id]
        ).reshape(3, 3)
        camera_rotation = camera_rotation @ np.diag([1.0, -1.0, -1.0])
        camera_point = camera_rotation.T @ (
            np.asarray(point_world, dtype=float) - camera_position
        )
        if camera_point[2] <= 0:
            return None
        focal = 0.5 * height / np.tan(
            float(base_env.sim.model.cam_fovy[camera_id]) * np.pi / 360.0
        )
        return (
            float(focal * camera_point[1] / camera_point[2] + height / 2),
            float(focal * camera_point[0] / camera_point[2] + width / 2),
        )
    except Exception:
        return None


def _interaction_object_radius(base_env, object_name):
    body_id = getattr(base_env, "obj_body_id", {}).get(object_name)
    if body_id is None:
        return None
    radius = 0.0
    for geom_id in range(base_env.sim.model.ngeom):
        if int(base_env.sim.model.geom_bodyid[geom_id]) == int(body_id):
            radius = max(radius, float(base_env.sim.model.geom_rbound[geom_id]))
    return radius if radius > 0 else None


def _interaction_points(base_env, spec, target_objects):
    parts = spec.get(
        "interaction_parts",
        ["gripper", "object_contact", "receptacle_contact"],
    )
    containers = (
        "basket",
        "plate",
        "cabinet",
        "microwave",
        "stove",
        "caddy",
        "drawer",
    )
    body_ids = getattr(base_env, "obj_body_id", {})
    points = {}
    grip = None
    for site_name in ("gripper0_grip_site", "robot0_grip_site", "robot0_eef_site"):
        if grip is not None:
            break
        try:
            grip = np.asarray(base_env.sim.data.site_xpos[
                base_env.sim.model.site_name2id(site_name)], dtype=float).copy()
        except Exception:
            continue
    if "gripper" in parts and grip is not None:
        points["gripper"] = grip
    manipulated = [
        name
        for name in target_objects
        if not any(token in name for token in containers)
    ]
    receptacles = [
        name
        for name in target_objects
        if any(token in name for token in containers)
    ]
    def body_geoms(body_id):
        if body_id is None:
            return []
        return [gid for gid in range(int(base_env.sim.model.ngeom))
                if int(base_env.sim.model.geom_bodyid[gid]) == int(body_id)]

    def contact_point(body_a, body_b=None, near=None):
        if body_a is None:
            return None
        best, best_distance = None, float("inf")
        for index in range(int(getattr(base_env.sim.data, "ncon", 0))):
            contact = base_env.sim.data.contact[index]
            g1, g2 = int(contact.geom1), int(contact.geom2)
            b1 = int(base_env.sim.model.geom_bodyid[g1])
            b2 = int(base_env.sim.model.geom_bodyid[g2])
            hit = ((b1 == int(body_a) and (body_b is None or b2 == int(body_b))) or
                   (b2 == int(body_a) and (body_b is None or b1 == int(body_b))))
            if not hit:
                continue
            point = np.asarray(contact.pos, dtype=float).copy()
            distance = float(np.linalg.norm(point - near)) if near is not None else 0.0
            if distance < best_distance:
                best, best_distance = point, distance
        return best

    def nearest_surface(body_id, toward):
        if body_id is None or toward is None:
            return None
        best, best_distance = None, float("inf")
        toward = np.asarray(toward, dtype=float)
        for geom_id in body_geoms(body_id):
            center = np.asarray(base_env.sim.data.geom_xpos[geom_id], dtype=float)
            vector = toward - center
            norm = float(np.linalg.norm(vector))
            if norm < 1e-8:
                continue
            radius = float(base_env.sim.model.geom_rbound[geom_id])
            point = center + vector / norm * max(0.0, 0.85 * radius)
            distance = norm - radius
            if distance < best_distance:
                best, best_distance = point, distance
        return best

    selected_object, selected_object_id, selected_position = None, None, None
    for name in manipulated:
        body_id = body_ids.get(name)
        if body_id is None:
            continue
        position = np.asarray(base_env.sim.data.body_xpos[int(body_id)], dtype=float)
        distance = float(np.linalg.norm(position - grip)) if grip is not None else 0.0
        if selected_position is None or distance < selected_position[0]:
            selected_object, selected_object_id, selected_position = name, body_id, (distance, position)
    if "object_contact" in parts and selected_object_id is not None:
        point = contact_point(selected_object_id, near=grip)
        point = point if point is not None else nearest_surface(selected_object_id, grip)
        if point is not None:
            points["object_contact"] = point
    if "receptacle_contact" in parts and receptacles:
        receptacle_id = body_ids.get(receptacles[0])
        if receptacle_id is not None:
            point = contact_point(receptacle_id, selected_object_id, near=selected_position[1] if selected_position else None)
            point = point if point is not None else nearest_surface(receptacle_id, selected_position[1] if selected_position else grip)
            if point is not None:
                points["receptacle_contact"] = point
    part_objects = {
        "gripper": None,
        "object_contact": manipulated[0] if manipulated else None,
        "receptacle_contact": receptacles[0] if receptacles else None,
    }
    return points, part_objects


def _apply_interaction_part_mask(raw_obs, env_in, spec, libero_benchmark, task_idx):
    base_env = getattr(env_in, "env", env_in)
    target_objects = _resolve_occlusion_targets(
        env_in, spec, libero_benchmark, int(task_idx)
    )
    if not target_objects:
        raise ValueError(
            f"No interaction targets resolved for {libero_benchmark}:{task_idx}"
        )
    points, part_objects = _interaction_points(base_env, spec, target_objects)
    views = normalize_view_names(spec.get("views"))
    output = dict(raw_obs)
    view_stats = {}
    fixed_radius = int(spec.get("part_radius_px", 8))
    radius_fraction = float(spec.get("part_radius_frac", 0.5))
    minimum_radius = int(spec.get("part_radius_min", 4))
    maximum_radius = int(spec.get("part_radius_max", 24))
    fill_rgb = np.asarray(spec.get("fill_rgb", [0, 0, 0]), dtype=np.uint8)
    for view in views:
        image_key = VIEW_CONFIGS[view]["image_key"]
        image = np.array(raw_obs[image_key], copy=True)
        height, width = image.shape[:2]
        names = list(points)
        coordinates = _interaction_world_to_pixel(
            base_env, view, [points[name] for name in names], height, width
        )
        drawn = []
        combined_mask = np.zeros((height, width), dtype=bool)
        for name, coordinate in zip(names, coordinates):
            if coordinate is None:
                continue
            row, column = coordinate
            radius = fixed_radius
            object_name = part_objects.get(name)
            world_radius = (
                _interaction_object_radius(base_env, object_name)
                if object_name
                else None
            )
            if world_radius is not None:
                edge = _interaction_project_raw(
                    base_env,
                    view,
                    points[name] + np.array([world_radius, 0.0, 0.0]),
                    height,
                    width,
                )
                if edge is not None:
                    size_pixels = float(
                        np.hypot(edge[0] - row, edge[1] - column)
                    )
                    if size_pixels > 0:
                        radius = int(round(size_pixels * radius_fraction))
            radius = max(minimum_radius, min(maximum_radius, radius))
            yy, xx = np.ogrid[:height, :width]
            disk = (yy - row) ** 2 + (xx - column) ** 2 <= radius**2
            combined_mask |= disk
            drawn.append(
                {
                    "part": name,
                    "center": [int(row), int(column)],
                    "radius": int(radius),
                }
            )
        image[combined_mask] = fill_rgb
        output[image_key] = image
        masked_pixels = int(combined_mask.sum())
        view_stats[view] = {
            "masked_pixels": masked_pixels,
            "mask_fraction": masked_pixels / float(height * width),
            "parts_drawn": drawn,
            "missing_targets": [],
        }
    return output, {
        "enabled": True,
        "target_policy": "interaction_part",
        "target_objects": target_objects,
        "mask_source": "interaction_part",
        "mask_geometry": "interaction_disk_adaptive",
        "part_radius_frac": radius_fraction,
        "views": views,
        "view_stats": view_stats,
    }


def _apply_cross_view_object_edit(raw_obs, env_in, spec):
    """Use the shared, exactly restored C1 counterfactual rendering path."""
    from source._shared_core.cross_view_render import apply_cross_view_object_edit

    return apply_cross_view_object_edit(
        raw_obs, env_in, spec,
        normalize_views=normalize_view_names,
        view_configs=VIEW_CONFIGS,
        free_joint_name=_free_joint_name,
        get_joint_qpos=_get_joint_qpos,
        set_joint_qpos=_set_joint_qpos,
    )


def apply_obs_noise_to_raw_obs(
    raw_obs: dict[str, Any],
    env_in: Any,
    spec: dict[str, Any] | None,
    libero_benchmark: str,
    task_idx: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not obs_perturbation_enabled(spec):
        return raw_obs, {"enabled": False}

    spec = dict(spec or {})
    mask_source = str(spec.get("mask_source", "segmentation"))
    if mask_source in {"interaction_part", "interaction"}:
        return _apply_interaction_part_mask(
            raw_obs, env_in, spec, libero_benchmark, task_idx
        )
    if mask_source == "object_edit":
        return _apply_cross_view_object_edit(raw_obs, env_in, spec)
    if mask_source not in {
        "segmentation",
        "center_rect",
        "target_noise",
        "non_target_noise",
        "segmentation_noise",
        "target_photo_replacement",
        "object_photo_replacement",
        "photo_replacement",
        "target_shape_warp",
    }:
        return raw_obs, {"enabled": False, "mask_source": mask_source}

    if mask_source == "center_rect":
        views = normalize_view_names(spec.get("views"))
        model_obs = dict(raw_obs)
        stats = {}
        fraction = max(0.0, min(1.0, float(spec.get("mask_fraction", 1.0))))
        for view_name in views:
            image_key = VIEW_CONFIGS[view_name]["image_key"]
            image = np.array(raw_obs[image_key], copy=True)
            h, w = image.shape[:2]
            rh, rw = int(round(h * fraction**0.5)), int(round(w * fraction**0.5))
            y0, x0 = (h - rh) // 2, (w - rw) // 2
            region = (slice(y0, y0 + rh), slice(x0, x0 + rw))
            if str(spec.get("fill_mode", "solid")).lower() in {"noise", "gaussian"}:
                seed = int(spec.get("noise_seed", 0)) + int(task_idx) * 1009
                rng = np.random.default_rng(seed)
                image[region] = np.clip(
                    rng.normal(127.0, float(spec.get("noise_std", 80.0)), image[region].shape),
                    0,
                    255,
                ).astype(np.uint8)
            else:
                image[region] = np.asarray(spec.get("fill_rgb", [0, 0, 0]), dtype=np.uint8)
            model_obs[image_key] = image
            stats[view_name] = {"masked_pixels": int(rh * rw), "mask_fraction": fraction}
        return model_obs, {
            "enabled": True,
            "mask_source": mask_source,
            "views": views,
            "view_stats": stats,
        }

    target_objects = _resolve_occlusion_targets(env_in, spec, libero_benchmark, int(task_idx))
    if not target_objects:
        raise ValueError(f"No target objects resolved for observation perturbation on {libero_benchmark}:{task_idx}")

    views = normalize_view_names(spec.get("views"))
    model_obs = dict(raw_obs)

    if mask_source == "target_shape_warp":
        view_stats = {}
        for view_name in views:
            cfg = VIEW_CONFIGS[view_name]
            image_key = cfg["image_key"]
            segmentation_key = cfg["segmentation_key"]
            if segmentation_key not in raw_obs:
                raise KeyError(f"Raw observation is missing {segmentation_key!r} for target shape warp")
            image = np.array(raw_obs[image_key], copy=True)
            warps = []
            missing_targets = []
            for object_name in target_objects:
                mask, missing = _target_mask_from_segmentation(env_in, raw_obs[segmentation_key], [object_name])
                missing_targets.extend(missing)
                warp = _apply_target_shape_warp(image, mask, spec)
                warp["object"] = object_name
                warps.append(warp)
            model_obs[image_key] = image
            changed_pixels = sum(int(warp.get("changed_pixels", 0)) for warp in warps)
            view_stats[view_name] = {
                "changed_pixels": changed_pixels,
                "mask_fraction": changed_pixels / float(image.shape[0] * image.shape[1]),
                "missing_targets": missing_targets,
                "shape_warps": warps,
            }
        return model_obs, {
            "enabled": True,
            "target_policy": spec.get("target_policy", "object_names"),
            "target_objects": target_objects,
            "mask_source": mask_source,
            "warp_family": spec.get("warp_family"),
            "warp_amount": float(spec.get("warp_amount", 0.0)),
            "views": views,
            "view_stats": view_stats,
        }

    if mask_source in {"target_photo_replacement", "object_photo_replacement", "photo_replacement"}:
        view_stats = {}
        for view_name in views:
            cfg = VIEW_CONFIGS[view_name]
            image_key = cfg["image_key"]
            segmentation_key = cfg["segmentation_key"]
            if image_key not in raw_obs:
                raise KeyError(f"Raw observation is missing {image_key!r}")
            if segmentation_key not in raw_obs:
                raise KeyError(
                    f"Raw observation is missing {segmentation_key!r}. "
                    "Enable SegmentationRenderEnv for target photo replacement."
                )

            image = np.array(raw_obs[image_key], copy=True)
            source_image = np.array(raw_obs[image_key], copy=True)
            replacements = []
            missing_targets = []
            changed_pixels = 0
            for object_name in target_objects:
                mask, missing = _target_mask_from_segmentation(
                    env_in,
                    raw_obs[segmentation_key],
                    [object_name],
                )
                missing_targets.extend(missing)
                mask = _dilate_mask(mask, spec.get("dilate_px", 0))
                replacement = _draw_target_photo_replacement(
                    image,
                    source_image,
                    mask,
                    object_name,
                    spec,
                )
                if replacement is None:
                    if object_name not in missing_targets:
                        missing_targets.append(object_name)
                    continue
                replacements.append(replacement)
                changed_pixels += int(replacement.get("changed_pixels", 0))

            view_stats[view_name] = {
                "masked_pixels": changed_pixels,
                "mask_fraction": changed_pixels / float(image.shape[0] * image.shape[1]),
                "missing_targets": missing_targets,
                "photo_replacements": replacements,
            }
            model_obs[image_key] = image

        return model_obs, {
            "enabled": True,
            "target_policy": spec.get("target_policy", "obj_of_interest"),
            "target_objects": target_objects,
            "mask_source": mask_source,
            "mask_geometry": "photo_bbox",
            "photo_source": "current_frame_target_crop",
            "photo_border_px": int(spec.get("photo_border_px", 1) or 0),
            "photo_padding_px": int(spec.get("photo_padding_px", 2) or 0),
            "dilate_px": int(spec.get("dilate_px", 0) or 0),
            "views": views,
            "view_stats": view_stats,
        }

    noise_region = str(spec.get("noise_region", "target")).lower()
    if mask_source == "target_noise":
        noise_region = "target"
    elif mask_source == "non_target_noise":
        noise_region = "non_target"

    view_stats = {}
    for view_name in views:
        cfg = VIEW_CONFIGS[view_name]
        image_key = cfg["image_key"]
        segmentation_key = cfg["segmentation_key"]
        if image_key not in raw_obs:
            raise KeyError(f"Raw observation is missing {image_key!r}")
        if segmentation_key not in raw_obs:
            raise KeyError(
                f"Raw observation is missing {segmentation_key!r}. "
                "Enable SegmentationRenderEnv for segmentation noise."
            )
        image = np.array(raw_obs[image_key], copy=True)
        mask, missing_targets = _target_mask_from_segmentation(
            env_in,
            raw_obs[segmentation_key],
            target_objects,
        )
        mask = _expand_mask_geometry(mask, spec.get("mask_geometry", "segmentation"))
        target_pixels = int(mask.sum())
        coverage = float(spec.get("coverage", 1.0))
        if coverage < 1.0:
            mask = _coverage_crop(mask, coverage)
        else:
            mask = _dilate_mask(mask, spec.get("dilate_px", 0))
        if mask_source == "segmentation":
            image[mask] = np.asarray(spec.get("fill_rgb", [0, 0, 0]), dtype=np.uint8)
            masked_pixels = int(mask.sum())
            view_stats[view_name] = {
                "target_mask_pixels": target_pixels,
                "masked_pixels": masked_pixels,
                "actual_coverage": (
                    float(masked_pixels) / float(target_pixels) if target_pixels else 0.0
                ),
                "requested_coverage": coverage,
                "missing_targets": missing_targets,
            }
            model_obs[image_key] = image
            continue
        if noise_region in {"non_target", "inverse", "background"}:
            mask = ~mask
        elif noise_region != "target":
            raise ValueError(f"Unsupported noise_region: {noise_region}")
        noise_stats = _apply_noise_to_mask(image, mask, spec, libero_benchmark, int(task_idx), view_name)
        noise_stats["missing_targets"] = missing_targets
        view_stats[view_name] = noise_stats
        model_obs[image_key] = image

    return model_obs, {
        "enabled": True,
        "target_policy": spec.get("target_policy", "obj_of_interest"),
        "target_objects": target_objects,
        "mask_source": mask_source,
        "mask_geometry": spec.get("mask_geometry", "segmentation"),
        "fill_mode": "solid" if mask_source == "segmentation" else None,
        "fill_rgb": (
            list(spec.get("fill_rgb", [0, 0, 0]))
            if mask_source == "segmentation"
            else None
        ),
        "noise_region": noise_region if mask_source != "segmentation" else None,
        "noise_type": spec.get("noise_type", "gaussian") if mask_source != "segmentation" else None,
        "noise_std": (
            float(spec.get("noise_std", 45.0))
            if mask_source != "segmentation"
            else None
        ),
        "dilate_px": int(spec.get("dilate_px", 0) or 0),
        "views": views,
        "view_stats": view_stats,
    }
