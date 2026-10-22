import json
import inspect
import logging
import multiprocessing as mp
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Optional

import hydra
import numpy as np
import torch
from accelerate import PartialState
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from tqdm import tqdm

RELEASE = next(parent for parent in Path(__file__).resolve().parents
               if (parent / "libero_vpro/runtime.py").is_file())
sys.path.insert(0, str(RELEASE))
from libero_vpro.runtime import configure_libero, prepare_env, resolve_runtime
runtime = resolve_runtime(require=False)
configure_libero()
for directory in reversed(prepare_env()["PYTHONPATH"].split(os.pathsep)):
    if directory not in sys.path:
        sys.path.insert(0, directory)
from source._shared_core.libero_hooks import install_object_xml_overrides
install_object_xml_overrides()

from source.fastwam_reference.libero_utils import (
    LIBERO_ENV_RESOLUTION,
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    invert_gripper_action,
    quat2axisangle,
    save_prediction_video,
    save_rollout_video,
)
from fastwam.datasets.lerobot.processors.fastwam_processor import FastWAMProcessor
from fastwam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json
from fastwam.utils.pytorch_utils import set_global_seed
from fastwam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT
from libero.libero import benchmark
from source.fastwam_reference.action_ensembler import ActionEnsembler
import robust_libero_perturbations as libero_perturb
try:
    import perturbation_schedule as phase_schedule
except ModuleNotFoundError:
    class _UnavailablePhaseSchedule:
        """Fail closed if a future phase experiment omits its isolated module."""

        LAST_DIAG = {}

        @staticmethod
        def reset_phase_state():
            _UnavailablePhaseSchedule.LAST_DIAG = {}

        @staticmethod
        def infer_phase(*_args, **_kwargs):
            raise RuntimeError(
                "phase tracking requested without an isolated "
                "perturbation_schedule.py dependency"
            )

    phase_schedule = _UnavailablePhaseSchedule()
from source.D2_D3_state_20260823.isolated_sources.lingbot_va.evaluation.libero import client as state_libero_client
from source._shared_core.episode_recorder import (
    capture_libero_state,
    make_episode_recorder,
)

from source.D2_D3_state_20260823.experiment_env import prepare_environment as prepare_state_environment

OmegaConf.register_new_resolver("eval", eval)
OmegaConf.register_new_resolver("max", lambda x: max(x))
OmegaConf.register_new_resolver("split", lambda s, idx: s.split("/")[int(idx)])

os.environ["TOKENIZERS_PARALLELISM"] = "false"


_SINGLE_VIEW_ALIASES = {
    "agent": "agentview",
    "agentview": "agentview",
    "wrist": "wrist",
    "eye_in_hand": "wrist",
    "robot0_eye_in_hand": "wrist",
}


def _single_view_name(cfg: DictConfig) -> str:
    raw = str(cfg.EVALUATION.get("single_view", "") or "").strip().lower()
    if not raw:
        raw = os.environ.get("WAM_SINGLE_VIEW", "").strip().lower()
    if not raw:
        return ""
    if raw not in _SINGLE_VIEW_ALIASES:
        raise ValueError(
            f"Unsupported single-view input {raw!r}; "
            f"expected one of {sorted(_SINGLE_VIEW_ALIASES)}"
        )
    return _SINGLE_VIEW_ALIASES[raw]


def _view_copy_name(cfg: DictConfig) -> str:
    raw = str(cfg.EVALUATION.get("view_copy", "") or "").strip().lower()
    if not raw:
        raw = os.environ.get("WAM_VIEW_COPY", "").strip().lower()
    if not raw:
        return ""
    if raw not in _SINGLE_VIEW_ALIASES:
        raise ValueError(
            f"Unsupported A5 view-copy source {raw!r}; "
            f"expected one of {sorted(_SINGLE_VIEW_ALIASES)}"
        )
    return _SINGLE_VIEW_ALIASES[raw]


def _c5_view_name(cfg: DictConfig) -> str:
    return _view_copy_name(cfg) or _single_view_name(cfg)


def _select_model_view_images(imgs: dict, cfg: DictConfig) -> dict:
    copy_view = _view_copy_name(cfg)
    if copy_view:
        source_key = "image" if copy_view == "agentview" else "wrist_image"
        if source_key not in imgs:
            raise KeyError(f"A5 view-copy input requires image key {source_key!r}")
        source = np.asarray(imgs[source_key])
        return {"image": source.copy(), "wrist_image": source.copy()}
    view_name = _single_view_name(cfg)
    if not view_name:
        return imgs
    source_key = "image" if view_name == "agentview" else "wrist_image"
    if source_key not in imgs:
        raise KeyError(f"A5 single-view input requires image key {source_key!r}")
    return {source_key: imgs[source_key]}


def _model_visual_input_keys(cfg: DictConfig) -> list[str]:
    if _view_copy_name(cfg):
        return ["image", "wrist_image"]
    view_name = _single_view_name(cfg)
    if view_name == "agentview":
        return ["image"]
    if view_name == "wrist":
        return ["wrist_image"]
    return ["image", "wrist_image"]


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def _normalize_mixed_precision(mixed_precision: str) -> str:
    key = str(mixed_precision).strip().lower()
    if key not in {"no", "fp16", "bf16"}:
        raise ValueError(
            f"Unsupported mixed_precision: {mixed_precision}. "
            "Expected one of: ['no', 'fp16', 'bf16']."
        )
    return key


def _mixed_precision_to_model_dtype(mixed_precision: str) -> torch.dtype:
    precision = _normalize_mixed_precision(mixed_precision)
    if precision == "no":
        return torch.float32
    if precision == "fp16":
        return torch.float16
    return torch.bfloat16


def _resolve_eval_device(cfg: DictConfig) -> str:
    eval_device = cfg.EVALUATION.get("device")
    if eval_device is not None:
        return str(eval_device)
    return "cuda" if torch.cuda.is_available() else "cpu"


def _resolve_dataset_stats_path(cfg: DictConfig) -> Path:
    explicit = cfg.EVALUATION.get("dataset_stats_path")
    candidates: list[Path] = []

    if explicit is not None:
        candidates.append(Path(os.path.expanduser(os.path.expandvars(str(explicit)))))

    ckpt = Path(os.path.expanduser(os.path.expandvars(str(cfg.ckpt))))
    for parent in list(ckpt.parents)[:4]:
        candidates.append(parent / "dataset_stats.json")

    seen = set()
    for path in candidates:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.exists():
            return resolved

    msg = (
        "Failed to locate dataset_stats.json. Tried explicit "
        "EVALUATION.dataset_stats_path and checkpoint parent directories. "
        "Please pass EVALUATION.dataset_stats_path=/path/to/dataset_stats.json."
    )
    raise FileNotFoundError(msg)


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _to_plain_container(value: Any) -> Any:
    if isinstance(value, DictConfig):
        return OmegaConf.to_container(value, resolve=True)
    return value


def _get_optional_cfg_or_env(cfg: DictConfig, key: str, env_keys: list[str], default: Any = "") -> Any:
    for env_key in env_keys:
        value = os.environ.get(env_key, "").strip()
        if value:
            return value
    return cfg.EVALUATION.get(key, default)


def _load_json_or_inline(value: Any) -> tuple[dict, str]:
    if not value:
        return {}, ""
    value = str(value)
    if value.lstrip().startswith("{"):
        return json.loads(value), "inline"
    path = Path(os.path.expanduser(os.path.expandvars(value)))
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f), str(path)
    return json.loads(value), "inline"


def _load_modified_instruction(
    task_descriptions_file: Any,
    task_suite_name: str,
    task_id: int,
    enabled: bool = True,
) -> tuple[str, str]:
    if not enabled or not task_descriptions_file:
        return "", ""
    path = Path(os.path.expanduser(os.path.expandvars(str(task_descriptions_file))))
    if not path.exists():
        raise FileNotFoundError(f"TaskDescriptions file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for task in data.get("tasks", []):
        if task.get("benchmark") == task_suite_name and int(task.get("task_id")) == int(task_id):
            return (task.get("modified_instruction") or "").strip(), str(path)
    logging.warning("TaskDescriptions entry not found for %s:%s in %s", task_suite_name, task_id, path)
    return "", str(path)


def _get_joint_qpos(sim: Any, joint_name: str) -> np.ndarray:
    try:
        return np.array(sim.data.get_joint_qpos(joint_name), dtype=float)
    except AttributeError:
        qpos_addr = sim.model.get_joint_qpos_addr(joint_name)
        return np.array(sim.data.qpos[qpos_addr], dtype=float)


def _set_joint_qpos(sim: Any, joint_name: str, qpos: np.ndarray) -> None:
    sim.data.set_joint_qpos(joint_name, np.array(qpos, dtype=float))


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


def _scale_mesh_vertices(sim: Any, mesh_id: int, scale: float) -> None:
    vert_adr = int(sim.model.mesh_vertadr[mesh_id])
    vert_num = int(sim.model.mesh_vertnum[mesh_id])
    sim.model.mesh_vert[vert_adr: vert_adr + vert_num] *= float(scale)


def _scale_object_instance(base_env: Any, object_name: str, scale: float) -> None:
    if object_name not in base_env.obj_body_id:
        raise ValueError(f"LIBERO object '{object_name}' was not found in obj_body_id")
    if scale <= 0:
        raise ValueError(f"LIBERO scale override for '{object_name}' must be positive, got {scale}")

    sim = base_env.sim
    root_body_id = int(base_env.obj_body_id[object_name])
    body_ids = set(_descendant_body_ids(sim, root_body_id))
    scaled_mesh_ids: set[int] = set()

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


def _apply_libero_env_edit(env_in: Any, spec: Optional[dict]) -> None:
    libero_perturb.apply_libero_env_edit(env_in, spec)


def _load_cross_task_view_swap_spec(cfg: DictConfig) -> dict:
    spec_path = (
        os.environ.get("FASTWAM_CROSS_TASK_VIEW_SWAP_SPEC", "").strip()
        or os.environ.get("LIBERO_CROSS_TASK_VIEW_SWAP_SPEC", "").strip()
    )
    if spec_path:
        with open(os.path.expanduser(os.path.expandvars(spec_path)), "r", encoding="utf-8") as f:
            spec = json.load(f)
    else:
        spec = _to_plain_container(cfg.EVALUATION.get("cross_task_view_swap", {})) or {}

    if not isinstance(spec, dict):
        raise ValueError(f"cross_task_view_swap spec must be a dict, got {type(spec)}")
    spec.setdefault("enabled", False)
    return spec


def _load_obs_occlusion_spec(cfg: DictConfig) -> tuple[dict, str]:
    value = _get_optional_cfg_or_env(
        cfg,
        "obs_occlusion_spec",
        ["FASTWAM_OBS_OCCLUSION_SPEC", "LIBERO_OBS_OCCLUSION_SPEC"],
        "",
    )
    spec, source = _load_json_or_inline(value)
    if not spec:
        return {"enabled": False}, ""
    spec.setdefault("enabled", True)
    spec.setdefault("mask_source", "segmentation")
    spec.setdefault("target_policy", "obj_of_interest")
    spec.setdefault("views", ["agentview", "robot0_eye_in_hand"])
    spec.setdefault("mask_geometry", "segmentation")
    spec.setdefault("dilate_px", 0)
    spec.setdefault("save_masked_video", False)
    return spec, source


def _load_decision_schedule_spec(cfg: DictConfig) -> tuple[dict, str]:
    value = _get_optional_cfg_or_env(
        cfg,
        "decision_schedule_spec",
        ["FASTWAM_DECISION_SCHEDULE_SPEC", "LIBERO_SCHEDULE_SPEC"],
        "",
    )
    spec, source = _load_json_or_inline(value)
    if not spec:
        return {}, ""
    spec.setdefault("enabled", True)
    return spec, source


def _cross_task_view_swap_enabled(spec: dict) -> bool:
    return _as_bool(spec.get("enabled")) and str(spec.get("mask_source", "cross_task_view_swap")) in {
        "cross_task_view_swap",
        "cross_task_view_conflict",
    }


def _normalize_cross_task_views(views: Any) -> list[str]:
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
        if view_name not in {"agentview", "robot0_eye_in_hand"}:
            raise ValueError(f"Unsupported FastWAM LIBERO view for cross-task swap: {view}")
        normalized.append(view_name)
    return normalized


def _resolve_cross_task_donor(spec: dict, source_task_idx: int, num_tasks: int) -> int:
    task_key = f"libero_10:{int(source_task_idx)}"
    donor_by_task = spec.get("donor_task_by_task", {})
    if task_key in donor_by_task:
        donor_task_idx = int(donor_by_task[task_key])
    elif str(source_task_idx) in donor_by_task:
        donor_task_idx = int(donor_by_task[str(source_task_idx)])
    else:
        donor_task_idx = (int(source_task_idx) + int(spec.get("donor_task_offset", 1))) % int(num_tasks)
    if donor_task_idx == int(source_task_idx):
        raise ValueError(f"Cross-task donor must differ from source task: {source_task_idx}")
    return donor_task_idx


def _apply_cross_task_view_swap_obs(source_obs: dict, donor_obs: dict, swapped_views: list[str]) -> dict:
    mixed_obs = dict(source_obs)
    if "agentview" in swapped_views:
        mixed_obs["agentview_image"] = np.array(donor_obs["agentview_image"], copy=True)
    if "robot0_eye_in_hand" in swapped_views:
        mixed_obs["robot0_eye_in_hand_image"] = np.array(donor_obs["robot0_eye_in_hand_image"], copy=True)
    return mixed_obs


def _donor_env_worker(conn, task_suite_name: str, donor_task_idx: int, episode_idx: int, seed: Optional[int]):
    env = None
    try:
        benchmark_dict = benchmark.get_benchmark_dict()
        task_suite = benchmark_dict[task_suite_name]()
        donor_task = task_suite.get_task(int(donor_task_idx))
        donor_states = task_suite.get_task_init_states(int(donor_task_idx))
        env, donor_description = get_libero_env(donor_task, LIBERO_ENV_RESOLUTION, seed)
        env.reset()
        obs = env.set_init_state(donor_states[int(episode_idx) % len(donor_states)])
        conn.send({"ok": True, "obs": obs, "task_description": donor_description})
        while True:
            msg = conn.recv()
            cmd = msg.get("cmd")
            if cmd == "close":
                conn.send({"ok": True})
                break
            if cmd != "step":
                raise ValueError(f"Unsupported donor worker command: {cmd}")
            obs, _, done, _ = env.step(np.asarray(msg["action"], dtype=np.float32))
            conn.send({"ok": True, "obs": obs, "done": bool(done)})
    except BaseException as exc:
        try:
            conn.send({"ok": False, "error": repr(exc)})
        except Exception:
            pass
    finally:
        if env is not None:
            env.close()
        conn.close()


class CrossTaskDonorEnvWorker:
    def __init__(self, task_suite_name: str, donor_task_idx: int, episode_idx: int, seed: Optional[int]):
        ctx = mp.get_context("spawn")
        self._parent_conn, child_conn = ctx.Pipe()
        self._process = ctx.Process(
            target=_donor_env_worker,
            args=(child_conn, task_suite_name, int(donor_task_idx), int(episode_idx), seed),
            daemon=True,
        )
        self._process.start()
        child_conn.close()
        first = self._recv("initialize donor env")
        self.obs = first["obs"]
        self.task_description = first["task_description"]

    def _recv(self, context: str) -> dict:
        msg = self._parent_conn.recv()
        if not msg.get("ok"):
            raise RuntimeError(f"Donor env worker failed to {context}: {msg.get('error')}")
        return msg

    def step(self, action: Any) -> dict:
        self._parent_conn.send({"cmd": "step", "action": np.asarray(action, dtype=np.float32)})
        msg = self._recv("step donor env")
        self.obs = msg["obs"]
        return self.obs

    def close(self):
        if self._parent_conn is not None:
            try:
                if self._process.is_alive():
                    self._parent_conn.send({"cmd": "close"})
                    self._recv("close donor env")
            except (BrokenPipeError, EOFError, RuntimeError):
                pass
            self._parent_conn.close()
            self._parent_conn = None
        if self._process is not None:
            self._process.join(timeout=5)
            if self._process.is_alive():
                self._process.terminate()
                self._process.join(timeout=5)
            self._process = None


def _load_model_checkpoint(model: torch.nn.Module, ckpt: str) -> None:
    model.load_checkpoint(ckpt)
    logging.info("Loaded checkpoint via model.load_checkpoint: %s", ckpt)
    return

    # deprecated legacy checkpoint loading
    payload = torch.load(ckpt, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"Legacy checkpoint payload must be dict, got: {type(payload)}")

    if "mot" in payload and hasattr(model, "mot"):
        missing, unexpected = model.mot.load_state_dict(payload["mot"], strict=False)
        logging.warning(
            "Loaded fallback `mot` state_dict with strict=False. Missing=%d Unexpected=%d",
            len(missing),
            len(unexpected),
        )
        return

    state_dict = None
    for key in ("model_state_dict", "state_dict", "model"):
        value = payload.get(key)
        if isinstance(value, dict):
            state_dict = value
            break
    if state_dict is None and all(torch.is_tensor(v) for v in payload.values()):
        state_dict = payload
    if state_dict is None:
        raise ValueError(f"Cannot parse legacy checkpoint keys from: {ckpt}")

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    logging.warning(
        "Loaded fallback model state_dict with strict=False. Missing=%d Unexpected=%d",
        len(missing),
        len(unexpected),
    )


def _center_crop_resize(image: np.ndarray, width: int, height: int) -> np.ndarray:
    pil_image = Image.fromarray(image)
    src_w, src_h = pil_image.size
    scale = max(width / src_w, height / src_h)
    resized = pil_image.resize((round(src_w * scale), round(src_h * scale)), resample=Image.BILINEAR)
    rw, rh = resized.size
    left = max((rw - width) // 2, 0)
    top = max((rh - height) // 2, 0)
    cropped = resized.crop((left, top, left + width, top + height))
    return np.asarray(cropped, dtype=np.uint8)


def _normalize_proprio(
    proprio: np.ndarray,
    processor: FastWAMProcessor,
) -> torch.Tensor:
    state_meta = processor.shape_meta["state"]
    if len(state_meta) != 1:
        raise ValueError(
            "LIBERO eval currently expects a single merged state key in shape_meta['state']."
        )
    state_key = state_meta[0]["key"]

    state_batch = {"state": {state_key: torch.as_tensor(proprio, dtype=torch.float32).unsqueeze(0)}}
    state_batch = processor.action_state_transform(state_batch)
    state_batch = processor.normalizer.forward(state_batch)
    return state_batch["state"][state_key]


def _obs_to_model_input(
    obs: dict,
    cfg: DictConfig,
    processor: FastWAMProcessor,
    width: int,
    height: int,
    device: str,
    dtype: torch.dtype,
):
    imgs = get_libero_image(obs)
    image_meta = processor.shape_meta["images"]
    if len(image_meta) < int(processor.num_output_cameras):
        raise ValueError(
            f"shape_meta.images has {len(image_meta)} entries, "
            f"but num_output_cameras={processor.num_output_cameras}."
        )

    def _meta_to_hw(meta: dict, camera_idx: int) -> tuple[int, int]:
        shape = meta["shape"]
        if len(shape) != 3:
            raise ValueError(f"shape_meta.images[{camera_idx}].shape must be [C,H,W], got {shape}")
        return int(shape[1]), int(shape[2])

    concatenation = cfg.data.train.get("concat_multi_camera", "horizontal")
    num_cameras = processor.num_output_cameras
    single_view = _single_view_name(cfg)
    copy_view = _view_copy_name(cfg)
    if single_view and copy_view:
        raise ValueError("A5 direct-single and view-copy modes are mutually exclusive.")
    if single_view and num_cameras != 1:
        raise ValueError(
            "A5 direct single-view evaluation requires "
            f"data.train.processor.num_output_cameras=1, got {num_cameras}."
        )
    if num_cameras == 1:
        meta_idx = 1 if single_view == "wrist" and len(image_meta) > 1 else 0
        primary_h, primary_w = _meta_to_hw(image_meta[meta_idx], camera_idx=meta_idx)
        source_key = "wrist_image" if single_view == "wrist" else "image"
        rgb = _center_crop_resize(imgs[source_key], width=primary_w, height=primary_h)
    elif num_cameras == 2:
        primary_h, primary_w = _meta_to_hw(image_meta[0], camera_idx=0)
        wrist_h, wrist_w = _meta_to_hw(image_meta[1], camera_idx=1)
        if copy_view:
            source_key = "image" if copy_view == "agentview" else "wrist_image"
            primary = _center_crop_resize(imgs[source_key], width=primary_w, height=primary_h)
            wrist = _center_crop_resize(imgs[source_key], width=wrist_w, height=wrist_h)
        else:
            primary = _center_crop_resize(imgs["image"], width=primary_w, height=primary_h)
            wrist = _center_crop_resize(imgs["wrist_image"], width=wrist_w, height=wrist_h)
        if concatenation == "horizontal":
            rgb = np.concatenate([primary, wrist], axis=1)
        elif concatenation == "vertical":
            rgb = np.concatenate([primary, wrist], axis=0)
        else:
            raise ValueError(f"Invalid concat_multi_camera: {concatenation}")
    else:
        raise ValueError(f"LIBERO eval currently supports num_output_cameras in [1, 2], got {num_cameras}.")

    actual_h, actual_w = int(rgb.shape[0]), int(rgb.shape[1])
    expected_h, expected_w = int(height), int(width)
    image_shapes = [meta["shape"] for meta in image_meta]
    assert actual_h == expected_h and actual_w == expected_w, (
        "Input image size mismatch after per-camera resize + concat: "
        f"got (H,W)=({actual_h},{actual_w}), expected (H,W)=({expected_h},{expected_w}) "
        f"from data.train.video_size={[expected_h, expected_w]}; "
        f"shape_meta.images={image_shapes}, concat_multi_camera={concatenation}."
    )

    x = torch.tensor(rgb).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=dtype)
    x = x * (2.0 / 255.0) - 1.0

    proprio = _normalize_proprio(_extract_sim_state(obs), processor)

    return x, proprio, _select_model_view_images(imgs, cfg)


def _extract_sim_state(obs: dict) -> np.ndarray:
    """Build simulator state from current observation.

    This is used as proprio input for model inference.
    """
    state = np.concatenate(
        (
            obs["robot0_eef_pos"],
            quat2axisangle(obs["robot0_eef_quat"]),
            obs["robot0_gripper_qpos"],
        )
    ).astype(np.float32)
    return state


def _denormalize_action(action: torch.Tensor, processor: FastWAMProcessor) -> np.ndarray:
    if action.ndim == 2:
        action = action.unsqueeze(0)
    if action.ndim != 3:
        raise ValueError(f"Expected action tensor [B, T, D], got {tuple(action.shape)}")

    action_meta = processor.shape_meta["action"]
    if len(action_meta) != 1:
        raise ValueError(
            "LIBERO eval currently expects a single merged action key in shape_meta['action']."
        )

    action_key = action_meta[0]["key"]
    normalizer = processor.normalizer.normalizers["action"][action_key]
    action = action.to(dtype=torch.float32, device="cpu")
    denorm = normalizer.backward(action)
    return denorm.numpy()


def _get_num_video_frames(cfg: DictConfig) -> int:
    return (int(cfg.data.train.num_frames) - 1) // int(cfg.data.train.action_video_freq_ratio) + 1


def _validate_visualize_future_video_cfg(cfg: DictConfig) -> None:
    if not bool(cfg.EVALUATION.get("visualize_future_video", False)):
        return

    action_conditioned = cfg.model.video_dit_config.get("action_conditioned", None)
    if action_conditioned is not False:
        raise ValueError(
            "EVALUATION.visualize_future_video=true requires "
            "model.video_dit_config.action_conditioned=false."
        )


def _select_predicted_future_frames(pred_video: list[Image.Image], cfg: DictConfig) -> list[Image.Image]:
    if len(pred_video) == 0:
        raise ValueError("`infer_joint` returned an empty predicted video.")

    replan_steps = int(cfg.EVALUATION.get("replan_steps", 5))
    action_video_freq_ratio = int(cfg.data.train.action_video_freq_ratio)
    num_future_frames = replan_steps // action_video_freq_ratio
    keep_frames = 1 + num_future_frames
    return list(pred_video[:keep_frames])


def _get_future_frame_capture_steps(cfg: DictConfig) -> list[int]:
    replan_steps = int(cfg.EVALUATION.get("replan_steps", 5))
    action_video_freq_ratio = int(cfg.data.train.action_video_freq_ratio)
    num_future_frames = replan_steps // action_video_freq_ratio
    return [step_idx * action_video_freq_ratio for step_idx in range(num_future_frames + 1)]


def _frame_to_rgb_array(frame: Any) -> np.ndarray:
    if isinstance(frame, dict):
        images = []
        for value in frame.values():
            value_array = np.array(value) if isinstance(value, Image.Image) else np.array(value, copy=True)
            images.append(value_array)
        return np.concatenate(images, axis=1)
    if isinstance(frame, Image.Image):
        return np.array(frame.convert("RGB"))
    return np.array(frame, copy=True)


def _compute_clip_mean_psnr(
    gt_frames: list[Any],
    pred_frames: list[Any],
    eps: float = 1e-8,
) -> Optional[float]:
    if len(gt_frames) == 0 or len(pred_frames) == 0:
        return None
    assert len(gt_frames) == len(pred_frames), (
        "GT/pred frame count mismatch for PSNR: "
        f"len(gt_frames)={len(gt_frames)} len(pred_frames)={len(pred_frames)}. "
        "This indicates temporal misalignment in future-video capture."
    )
    num_frames = len(gt_frames)

    frame_psnr_values = []
    for gt_frame, pred_frame in zip(gt_frames[:num_frames], pred_frames[:num_frames]):
        gt_image = _frame_to_rgb_array(gt_frame)
        pred_image = _frame_to_rgb_array(pred_frame)
        target_h, target_w = pred_image.shape[:2]
        if gt_image.shape[:2] != (target_h, target_w):
            gt_image = np.array(
                Image.fromarray(gt_image).resize((target_w, target_h), resample=Image.BILINEAR)
            )

        gt_f32 = gt_image.astype(np.float32)
        pred_f32 = pred_image.astype(np.float32)
        mse = float(np.mean((pred_f32 - gt_f32) ** 2))
        psnr = 10.0 * np.log10((255.0 * 255.0) / max(mse, eps))
        frame_psnr_values.append(float(psnr))

    if len(frame_psnr_values) == 0:
        return None
    return float(np.mean(frame_psnr_values))


def _predict_action_chunk(
    obs: dict,
    task_description: str,
    model: torch.nn.Module,
    processor: FastWAMProcessor,
    cfg: DictConfig,
    *,
    action_horizon: int,
    input_w: int,
    input_h: int,
    model_device: str,
) -> tuple[np.ndarray, dict, Optional[list[Image.Image]]]:
    num_inference_steps_cfg = cfg.EVALUATION.get("num_inference_steps", None)
    if num_inference_steps_cfg is None:
        num_inference_steps = int(cfg.get("eval_num_inference_steps", 20))
    else:
        num_inference_steps = int(num_inference_steps_cfg)
    prompt_template = DEFAULT_PROMPT
    prompt = prompt_template.format(task=task_description)

    image, proprio, imgs = _obs_to_model_input(
        obs,
        cfg=cfg,
        processor=processor,
        width=input_w,
        height=input_h,
        device=model_device,
        dtype=model.torch_dtype,
    )

    infer_kwargs = {
        "prompt": prompt,
        "input_image": image,
        "action_horizon": action_horizon,
        "negative_prompt": str(cfg.EVALUATION.get("negative_prompt", "")),
        "text_cfg_scale": float(cfg.EVALUATION.get("text_cfg_scale", 1.0)),
        "num_inference_steps": num_inference_steps,
        "proprio": proprio,
        "sigma_shift": (
            None
            if cfg.EVALUATION.get("sigma_shift") is None
            else float(cfg.EVALUATION.get("sigma_shift"))
        ),
        "seed": None if cfg.get("seed") is None else int(cfg.seed),
        "rand_device": str(cfg.EVALUATION.get("rand_device", "cpu")),
        "tiled": bool(cfg.EVALUATION.get("tiled", False)),
    }
    visualize_future_video = bool(cfg.EVALUATION.get("visualize_future_video", False))
    predicted_future_frames = None
    if visualize_future_video:
        infer_kwargs["num_video_frames"] = _get_num_video_frames(cfg)
    elif "num_video_frames" in inspect.signature(model.infer_action).parameters:
        infer_kwargs["num_video_frames"] = _get_num_video_frames(cfg)

    with torch.no_grad():
        if visualize_future_video:
            pred = model.infer_joint(**infer_kwargs)
            predicted_future_frames = _select_predicted_future_frames(pred["video"], cfg)
        else:
            pred = model.infer_action(**infer_kwargs)
    action = pred["action"]  # [T, D]

    action = _denormalize_action(action, processor)[0]  # [T, D]

    # The dataloader flips the sign of the gripper action to align with other datasets
    # (0 = close, 1 = open), so flip it back (-1 = open, +1 = close) before executing the action
    action[..., -1] = action[..., -1] * 2 - 1
    action = invert_gripper_action(action)
    if bool(cfg.EVALUATION.get("binarize_gripper", False)):
        action[..., -1] = np.sign(action[..., -1])
    return action, imgs, predicted_future_frames


def _get_max_steps(task_suite_name: str) -> int:
    suite_steps = {
        "libero_spatial": 400,
        "libero_object": 400,
        "libero_goal": 400,
        "libero_10": 700,
        "libero_90": 700,
    }
    if task_suite_name not in suite_steps:
        raise ValueError(f"Unknown task suite: {task_suite_name}")
    return suite_steps[task_suite_name]


def run_single_episode(
    env,
    initial_state,
    task_description: str,
    model: torch.nn.Module,
    processor: FastWAMProcessor,
    cfg: DictConfig,
    episode_idx: int,
    *,
    action_horizon: int,
    input_w: int,
    input_h: int,
    model_device: str,
    cross_task_spec: Optional[dict] = None,
    donor_task_idx: Optional[int] = None,
    env_edit_spec: Optional[dict] = None,
    obs_occlusion_spec: Optional[dict] = None,
    decision_schedule_spec: Optional[dict] = None,
    source_task_idx: Optional[int] = None,
) -> tuple[bool, list, list, list[dict[str, Any]], Optional[float], dict[str, Any]]:
    max_steps = _get_max_steps(cfg.EVALUATION.task_suite_name)
    replan_steps = int(cfg.EVALUATION.get("replan_steps", 5))
    num_steps_wait = int(cfg.EVALUATION.get("num_steps_wait", 5))
    use_action_ensembler = bool(cfg.EVALUATION.get("use_action_ensembler", False))
    visualize_future_video = bool(cfg.EVALUATION.get("visualize_future_video", False))
    capture_steps = set(_get_future_frame_capture_steps(cfg)[1:])
    cross_task_spec = cross_task_spec or {}
    obs_occlusion_spec = obs_occlusion_spec or {"enabled": False}
    task_idx_for_stats = int(source_task_idx if source_task_idx is not None else cfg.EVALUATION.task_id)
    decision_schedule_spec = dict(decision_schedule_spec or {})
    state_prepared = None
    state_case = (
        dict(env_edit_spec)
        if isinstance(env_edit_spec, dict)
        and str(env_edit_spec.get("track", "")) in {"D2", "D3"}
        else None
    )
    task_key = f"{cfg.EVALUATION.task_suite_name}:{task_idx_for_stats}"
    if task_key in {str(value) for value in decision_schedule_spec.get("not_applicable_tasks", [])}:
        decision_schedule_spec["enabled"] = False
        decision_schedule_spec["not_applicable"] = True
        decision_schedule_spec["not_applicable_reason"] = "task_has_no_placement_event"
    decision_controller = libero_perturb.DecisionPerturbationController(decision_schedule_spec)
    decision_trace: list[dict[str, Any]] = []
    phase_trace: list[dict[str, Any]] = []
    phase_state = {"phase": "initial", "latch": False}
    phase_schedule.reset_phase_state()
    phase_targets_by_task = (decision_schedule_spec or {}).get("phase_targets_by_task", {}) or {}
    phase_targets = list(
        phase_targets_by_task.get(
            f"{cfg.EVALUATION.task_suite_name}:{task_idx_for_stats}",
            (decision_schedule_spec or {}).get("phase_targets", []),
        )
        or []
    )
    if not phase_targets:
        phase_targets = list((obs_occlusion_spec or {}).get("target_objects", []) or [])
    if not phase_targets:
        obs_targets_by_task = (obs_occlusion_spec or {}).get("target_objects_by_task", {}) or {}
        phase_targets = list(
            obs_targets_by_task.get(
                f"{cfg.EVALUATION.task_suite_name}:{task_idx_for_stats}",
                obs_targets_by_task.get(str(task_idx_for_stats), []),
            )
            or []
        )
    if not phase_targets:
        phase_targets = list(getattr(env, "obj_of_interest", []) or [])
    if not phase_targets:
        phase_targets = list(getattr(getattr(env, "env", env), "obj_of_interest", []) or [])
    phase_diagnostics = _as_bool((decision_schedule_spec or {}).get("phase_diagnostics", False))
    phase_tracking = (
        str((decision_schedule_spec or {}).get("trigger", "always")).lower()
        in {
            "phase", "phase_pulse", "phase_once", "grasp_pulse", "place_pulse",
            "post_place_verified",
        }
        or phase_diagnostics
    )
    cross_task_swap = _cross_task_view_swap_enabled(cross_task_spec)
    swapped_views = (
        _normalize_cross_task_views(cross_task_spec.get("swapped_views", cross_task_spec.get("views")))
        if cross_task_swap
        else []
    )
    if cross_task_swap and donor_task_idx is None:
        raise ValueError("cross_task_view_swap is enabled but donor_task_idx is None.")
    seed = None if cfg.get("seed") is None else int(cfg.get("seed"))
    donor_worker = (
        CrossTaskDonorEnvWorker(
            str(cfg.EVALUATION.task_suite_name),
            int(donor_task_idx),
            int(episode_idx),
            seed,
        )
        if cross_task_swap
        else None
    )
    donor_obs = donor_worker.obs if donor_worker is not None else None
    episode_recorder = make_episode_recorder(
        model_name="FastWAM",
        model_type="jointly_trained",
        suite=str(cfg.EVALUATION.task_suite_name),
        task_id=task_idx_for_stats,
        task_name=str(task_description),
        instruction=str(task_description),
        repeat_index=int(episode_idx),
        init_state_index=int(episode_idx),
        checkpoint=str(cfg.ckpt),
        environment_seed=seed,
        model_seed=seed,
        config=OmegaConf.to_container(cfg, resolve=True),
        env_edit_spec=env_edit_spec,
        perturbation_spec={
            "observation": obs_occlusion_spec,
            "decision_schedule": decision_schedule_spec or {},
            "cross_task": cross_task_spec,
        },
        action_config={
            "action_horizon": int(action_horizon),
            "replan_steps": int(replan_steps),
            "action_ensembler": bool(use_action_ensembler),
        },
        extra_metadata={
            "source_task_id": task_idx_for_stats,
            "donor_task_id": donor_task_idx,
            "recorder_inference_invariant": "no_additional_model_calls",
            "single_view": _c5_view_name(cfg),
            "view_copy": _view_copy_name(cfg),
            "visual_input_keys": _model_visual_input_keys(cfg),
            "visual_input_contract": (
                "duplicated_single_camera_to_dual_slots"
                if _view_copy_name(cfg)
                else ("direct_single_camera" if _single_view_name(cfg) else "official_dual_camera")
            ),
            "visual_input_size": [int(input_h), int(input_w)],
        },
    )

    if state_case is not None:
        # The caller owns this reusable task environment. This episode owns
        # only the prepared D2/D3 environment, which its finally block closes.
        # Closing the borrowed environment here also breaks the next trial.
        state_suite = benchmark.get_benchmark_dict()[
            str(cfg.EVALUATION.task_suite_name)
        ]()
        state_prepared = prepare_state_environment(
            state_libero_client,
            state_suite,
            state_case,
            initial_state,
            camera_size=LIBERO_ENV_RESOLUTION,
        )
        env = state_prepared.env
        obs = state_prepared.initial_observation
        env_edit_audit = state_prepared.initialization_audit
    else:
        with libero_perturb.libero_env_override_context(
            object_xml_override_map_value=getattr(
                env, "_libero_object_xml_override_map", None
            )
        ):
            env.reset()
        obs = libero_perturb.set_init_state_with_bddl_remap(
            env,
            initial_state,
            libero_perturb.base_bddl_file_name(env_edit_spec),
        )
        pre_edit_settle_steps = int(
            (env_edit_spec or {}).get("pre_edit_settle_steps", 0)
        )
        for _ in range(pre_edit_settle_steps):
            obs, _, _, _ = env.step(get_libero_dummy_action())
        if pre_edit_settle_steps and libero_perturb.base_bddl_file_name(env_edit_spec):
            # Preserve settled new-object joints while restoring the official
            # original task and robot joints before applying the configured scene edit.
            obs = libero_perturb.set_init_state_with_bddl_remap(
                env,
                initial_state,
                libero_perturb.base_bddl_file_name(env_edit_spec),
            )
        _apply_libero_env_edit(env, env_edit_spec)
        env_edit_audit = getattr(
            getattr(env, "env", env),
            "_wam_syncbench_env_edit_audit",
            {},
        )
    if episode_recorder.enabled and env_edit_audit:
        episode_recorder.metadata["environment_edit_audit"] = env_edit_audit
        episode_recorder.metadata["initialization_audit"] = env_edit_audit
    if state_prepared is None:
        settle_steps = int((env_edit_spec or {}).get("settle_steps", 0))
        for _ in range(settle_steps):
            obs, _, _, _ = env.step(get_libero_dummy_action())
    if use_action_ensembler:
        ensembler = ActionEnsembler()
        ensembler.reset()

    # Keep video capture strictly out of the inference path.  ``replay_images``
    # is the clean simulator rollout at every control step, while
    # ``model_input_images`` is a control-step-aligned, zero-order-held view of
    # the exact two-view observation last passed to FastWAM.  FastWAM only gets
    # a new image at a policy decision/replan; between replans the most recent
    # model input remains visible so this evidence video stays aligned one-to-one
    # with the clean rollout without inventing unseen intermediate observations.
    replay_images = []
    model_input_images = []
    predicted_future_video_clips: list[dict[str, Any]] = []
    episode_future_clip_psnr: list[float] = []
    pending_actions: list[list[float]] = []
    current_predicted_future_clip: Optional[dict[str, Any]] = None
    current_replan_step = 0
    current_replan_idx = -1
    policy_decision_idx = -1
    recorded_realized_frames: list[dict[str, Any]] = []
    recorded_executed_actions: list[Any] = []
    current_perturb_active = False

    t = 0
    done = False
    pbar = tqdm(total=max_steps + num_steps_wait, desc=f"Episode {episode_idx + 1}")
    try:
        while t < max_steps + num_steps_wait:
            pbar.update(1)
            if t < num_steps_wait:
                dummy_action = get_libero_dummy_action()
                obs, _, done, _ = env.step(dummy_action)
                if donor_worker is not None:
                    donor_obs = donor_worker.step(dummy_action)
                t += 1
                continue

            model_obs = (
                _apply_cross_task_view_swap_obs(obs, donor_obs, swapped_views)
                if cross_task_swap
                else obs
            )
            state_observation_stats = {"enabled": False}
            if state_prepared is not None:
                model_obs, state_observation_stats = (
                    state_prepared.observation_for_model(
                        state_libero_client, obs
                    )
                )
            replay_images.append(get_libero_image(obs))
            if len(pending_actions) == 0:
                policy_decision_idx += 1
                current_decision_idx = policy_decision_idx
                model_obs = decision_controller.apply_camera(model_obs, current_decision_idx)
                perturb_active = (
                    decision_controller.active(
                        current_decision_idx,
                        timestep=t,
                        phase=phase_state["phase"],
                    )
                    if decision_schedule_spec
                    else True
                )
                current_perturb_active = bool(perturb_active)
                perturb_stats = {"enabled": False}
                if perturb_active:
                    model_obs, perturb_stats = libero_perturb.apply_obs_noise_to_raw_obs(
                        model_obs,
                        env,
                        obs_occlusion_spec,
                        str(cfg.EVALUATION.task_suite_name),
                        task_idx_for_stats,
                    )
                if state_prepared is not None:
                    perturb_stats = state_observation_stats
                decision_trace.append(
                    {
                        "decision": int(current_decision_idx),
                        "perturb_active": bool(perturb_active),
                        "perturbation": perturb_stats,
                    }
                )
                action_chunk, imgs, predicted_future_frames = _predict_action_chunk(
                    obs=model_obs,
                    task_description=task_description,
                    model=model,
                    processor=processor,
                    cfg=cfg,
                    action_horizon=action_horizon,
                    input_w=input_w,
                    input_h=input_h,
                    model_device=model_device,
                )
                if episode_recorder.enabled:
                    episode_recorder.record_replan(
                        current_decision_idx,
                        control_step_start=int(t),
                        phase=phase_state["phase"],
                        clean_observation=get_libero_image(obs),
                        model_observation=imgs,
                        predicted_frames=predicted_future_frames,
                        proposed_action_chunk=action_chunk,
                        state=capture_libero_state(
                            env,
                            obs,
                            episode_recorder.object_names,
                            include_sim_state=True,
                        ),
                        perturbation={
                            "active": bool(perturb_active),
                            "actual": perturb_stats,
                            "decision_schedule": decision_schedule_spec or {},
                        },
                        frame_control_offsets=sorted(capture_steps),
                        extra={
                            "cross_task_view_swap": bool(cross_task_swap),
                            "swapped_views": swapped_views,
                        },
                    )
                    recorded_realized_frames = [imgs.copy()]
                    recorded_executed_actions = []
                if predicted_future_frames is not None:
                    current_replan_idx += 1
                    current_predicted_future_clip = {
                        "replan_idx": current_replan_idx,
                        "gt_frames": [imgs.copy()],
                        "pred_frames": predicted_future_frames,
                    }
                else:
                    current_predicted_future_clip = None
                current_replan_step = 0
                if use_action_ensembler:
                    ensembler.add_actions(action_chunk, t)
                    pending_actions = [ensembler.get_action(ts).tolist() for ts in range(t, t + replan_steps)]
                else:
                    pending_actions = action_chunk[:replan_steps].tolist()

            model_input_images.append(imgs.copy())
            action_to_execute = pending_actions.pop(0)
            obs, reward, done, _ = env.step(action_to_execute)
            if phase_tracking and phase_targets:
                phase, latch = phase_schedule.infer_phase(
                    env,
                    phase_targets,
                    prev_phase=phase_state["phase"],
                    grasped_latch=phase_state["latch"],
                    gripper_cmd=float(np.asarray(action_to_execute).reshape(-1)[-1]),
                    pre_grasp_band=float((decision_schedule_spec or {}).get("pre_grasp_band", 0.20)),
                    grasp_hold_frames=int((decision_schedule_spec or {}).get("grasp_hold_frames", 24)),
                    initial_hold_frames=int((decision_schedule_spec or {}).get("initial_hold_frames", 24)),
                    release_hold_frames=int((decision_schedule_spec or {}).get("release_hold_frames", 24)),
                    pre_place_progress=float((decision_schedule_spec or {}).get("pre_place_progress", 0.70)),
                )
                phase_state = {"phase": phase, "latch": latch}
                decision_controller.observe_phase(latch, phase=phase)
                phase_trace.append(
                    {
                        "t": int(t),
                        "decision": int(policy_decision_idx),
                        "phase": phase,
                        "latch": bool(latch),
                        "targets": phase_targets,
                        **dict(phase_schedule.LAST_DIAG or {}),
                    }
                )
            if donor_worker is not None:
                donor_obs = donor_worker.step(action_to_execute)
            if episode_recorder.enabled:
                recorded_executed_actions.append(np.asarray(action_to_execute).copy())
                recorded_chunk_step = len(recorded_executed_actions)
                if (
                    current_predicted_future_clip is None
                    or recorded_chunk_step in capture_steps
                ):
                    recorded_realized_frames.append(
                        _select_model_view_images(get_libero_image(obs), cfg)
                    )
                episode_recorder.record_control(
                    int(t),
                    replan_index=int(policy_decision_idx),
                    chunk_step_index=int(recorded_chunk_step - 1),
                    phase=phase_state["phase"],
                    action=action_to_execute,
                    state=capture_libero_state(
                        env,
                        obs,
                        episode_recorder.object_names,
                    ),
                    perturbation_active=current_perturb_active,
                    reward=float(reward),
                    done=bool(done),
                )

            if visualize_future_video and current_predicted_future_clip is not None:
                current_replan_step += 1
                if current_replan_step in capture_steps:
                    future_obs = (
                        _apply_cross_task_view_swap_obs(obs, donor_obs, swapped_views)
                        if cross_task_swap
                        else obs
                    )
                    future_obs, _ = libero_perturb.apply_obs_noise_to_raw_obs(
                        future_obs,
                        env,
                        obs_occlusion_spec,
                        str(cfg.EVALUATION.task_suite_name),
                        task_idx_for_stats,
                    )
                    current_predicted_future_clip["gt_frames"].append(
                        _select_model_view_images(get_libero_image(future_obs), cfg)
                    )
                if done or len(pending_actions) == 0:
                    expected_frame_count = 1 + sum(
                        1 for capture_step in capture_steps if capture_step <= current_replan_step
                    )
                    gt_len = len(current_predicted_future_clip["gt_frames"])
                    pred_len = len(current_predicted_future_clip["pred_frames"])
                    assert gt_len == expected_frame_count, (
                        "GT future frames do not match expected capture count: "
                        f"gt_len={gt_len} expected={expected_frame_count} "
                        f"episode={episode_idx} replan={current_predicted_future_clip['replan_idx']} "
                        f"current_replan_step={current_replan_step} capture_steps={sorted(capture_steps)}."
                    )
                    assert pred_len >= expected_frame_count, (
                        "Predicted future frames shorter than expected capture count: "
                        f"pred_len={pred_len} expected={expected_frame_count} "
                        f"episode={episode_idx} replan={current_predicted_future_clip['replan_idx']}."
                    )
                    if pred_len != expected_frame_count:
                        logging.info(
                            "Align predicted clip length to executed steps: "
                            "episode=%s replan=%s done=%s expected=%s pred_full=%s",
                            episode_idx,
                            current_predicted_future_clip["replan_idx"],
                            done,
                            expected_frame_count,
                            pred_len,
                        )
                    current_predicted_future_clip["pred_frames"] = current_predicted_future_clip["pred_frames"][
                        :expected_frame_count
                    ]
                    assert len(current_predicted_future_clip["gt_frames"]) == len(
                        current_predicted_future_clip["pred_frames"]
                    ), (
                        "GT/pred frame count mismatch after alignment: "
                        f"len(gt_frames)={len(current_predicted_future_clip['gt_frames'])} "
                        f"len(pred_frames)={len(current_predicted_future_clip['pred_frames'])} "
                        f"episode={episode_idx} replan={current_predicted_future_clip['replan_idx']}."
                    )
                    clip_psnr = _compute_clip_mean_psnr(
                        current_predicted_future_clip["gt_frames"],
                        current_predicted_future_clip["pred_frames"],
                    )
                    if clip_psnr is not None:
                        episode_future_clip_psnr.append(clip_psnr)
                    predicted_future_video_clips.append(current_predicted_future_clip)
                    current_predicted_future_clip = None
            if episode_recorder.enabled and (done or len(pending_actions) == 0):
                realized_by_view = {
                    key: np.stack([frame[key] for frame in recorded_realized_frames])
                    for key in ("image", "wrist_image")
                    if recorded_realized_frames
                    and all(key in frame for frame in recorded_realized_frames)
                }
                episode_recorder.record_realized(
                    policy_decision_idx,
                    control_step_end=int(t),
                    frames=np.stack(
                        [
                            _frame_to_rgb_array(frame)
                            for frame in recorded_realized_frames
                        ]
                    ),
                    clean_frames=realized_by_view,
                    executed_action_chunk=recorded_executed_actions,
                )
            if done:
                break
            t += 1
    except BaseException as exc:
        episode_recorder.finish(
            success=False,
            final_phase=phase_state["phase"],
            status="failed",
            summary={
                "error_type": type(exc).__name__,
                "error": str(exc),
                "decision_trace": decision_trace,
                "phase_trace": phase_trace,
            },
        )
        raise
    finally:
        pbar.close()
        if donor_worker is not None:
            donor_worker.close()
        if state_prepared is not None:
            state_prepared.env.close()

    episode_mean_psnr = (
        float(np.mean(episode_future_clip_psnr)) if len(episode_future_clip_psnr) > 0 else None
    )
    episode_recorder.finish(
        success=bool(done),
        final_phase=phase_state["phase"],
        summary={
            "decision_trace": decision_trace,
            "phase_trace": phase_trace,
            "future_clip_mean_psnr": episode_mean_psnr,
            "environment_edit_audit": env_edit_audit,
        },
    )
    return (
        bool(done),
        replay_images,
        model_input_images,
        predicted_future_video_clips,
        episode_mean_psnr,
        {
            "decisions": decision_trace,
            "camera_staleness": decision_controller.log,
            "phase": phase_trace,
            "environment_edit": env_edit_audit,
        },
    )


def run_single_task(
    task,
    initial_states,
    model: torch.nn.Module,
    processor: FastWAMProcessor,
    cfg: DictConfig,
    video_dir: Path,
    predicted_video_dir: Path,
    *,
    action_horizon: int,
    input_w: int,
    input_h: int,
    model_device: str,
    cross_task_spec: Optional[dict] = None,
    source_task_idx: Optional[int] = None,
    donor_task_idx: Optional[int] = None,
    donor_task_description: str = "",
    task_description_override: str = "",
    task_descriptions_source: str = "",
    env_edit_spec: Optional[dict] = None,
    env_edit_source: str = "",
    obs_occlusion_spec: Optional[dict] = None,
    obs_occlusion_source: str = "",
    decision_schedule_spec: Optional[dict] = None,
    decision_schedule_source: str = "",
) -> dict:
    task_idx_for_env = int(source_task_idx if source_task_idx is not None else cfg.EVALUATION.task_id)
    resolved_env_edit_spec = libero_perturb.resolve_env_edit_spec(
        env_edit_spec,
        str(cfg.EVALUATION.task_suite_name),
        task_idx_for_env,
    )
    object_xml_override_map = libero_perturb.object_xml_override_map(resolved_env_edit_spec)
    bddl_override = libero_perturb.bddl_file_name_override(resolved_env_edit_spec)
    use_segmentation = libero_perturb.obs_perturbation_needs_segmentation(obs_occlusion_spec)
    env, task_description = get_libero_env(
        task,
        LIBERO_ENV_RESOLUTION,
        cfg.get("seed"),
        use_segmentation=use_segmentation,
        object_xml_override_map=object_xml_override_map,
        bddl_file_name_override=bddl_override or None,
    )
    original_task_description = task_description
    if task_description_override:
        task_description = task_description_override
    visualize_future_video = bool(cfg.EVALUATION.get("visualize_future_video", False))
    cross_task_spec = cross_task_spec or {}
    cross_task_swap = _cross_task_view_swap_enabled(cross_task_spec)
    swapped_views = (
        _normalize_cross_task_views(cross_task_spec.get("swapped_views", cross_task_spec.get("views")))
        if cross_task_swap
        else []
    )
    results = {
        "successes": 0,
        "failure_episodes": [],
        "success_episodes": [],
        "task_description": task_description,
        "original_task_description": original_task_description,
        "modified_instruction": task_description_override,
        "task_descriptions_source": task_descriptions_source,
        "env_edit_source": env_edit_source,
        "obs_occlusion_source": obs_occlusion_source,
        "decision_schedule_source": decision_schedule_source,
        "episode_decision_diagnostics": [],
        "episode_video_paths": [],
        "single_view": _c5_view_name(cfg),
        "view_copy": _view_copy_name(cfg),
        "visual_input_contract": (
            "duplicated_single_camera_to_dual_slots"
            if _view_copy_name(cfg)
            else ("direct_single_camera" if _single_view_name(cfg) else "official_dual_camera")
        ),
        "visual_input_size": [int(input_h), int(input_w)],
    }
    if libero_perturb.obs_perturbation_enabled(obs_occlusion_spec):
        results["obs_occlusion"] = {
            "enabled": True,
            "source": obs_occlusion_source,
            "mask_source": (obs_occlusion_spec or {}).get("mask_source"),
            "views": libero_perturb.normalize_view_names((obs_occlusion_spec or {}).get("views")),
        }
    if cross_task_swap:
        results["cross_task_view_swap"] = {
            "enabled": True,
            "schema_version": cross_task_spec.get("schema_version", ""),
            "mask_source": cross_task_spec.get("mask_source", "cross_task_view_swap"),
            "swap_mode": cross_task_spec.get("swap_mode", "dynamic_same_action"),
            "swapped_views": swapped_views,
            "source_task_suite": str(cfg.EVALUATION.task_suite_name),
            "source_task_idx": None if source_task_idx is None else int(source_task_idx),
            "source_task_description": task_description,
            "donor_task_suite": str(cfg.EVALUATION.task_suite_name),
            "donor_task_idx": None if donor_task_idx is None else int(donor_task_idx),
            "donor_task_description": donor_task_description,
            "donor_task_offset": int(cross_task_spec.get("donor_task_offset", 1)),
        }
    if visualize_future_video:
        results["episode_future_video_psnr"] = []
        results["future_video_psnr_mean"] = None

    trial_start = int(cfg.EVALUATION.get("trial_start", 0))
    num_trials = int(cfg.EVALUATION.num_trials)
    for trial_idx in range(trial_start, trial_start + num_trials):
        (
            success,
            replay_images,
            model_input_images,
            predicted_future_video_clips,
            episode_mean_psnr,
            decision_diag,
        ) = run_single_episode(
            env=env,
            initial_state=initial_states[trial_idx],
            task_description=task_description,
            model=model,
            processor=processor,
            cfg=cfg,
            episode_idx=trial_idx,
            action_horizon=action_horizon,
            input_w=input_w,
            input_h=input_h,
            model_device=model_device,
            cross_task_spec=cross_task_spec,
            donor_task_idx=donor_task_idx,
            env_edit_spec=resolved_env_edit_spec,
            obs_occlusion_spec=obs_occlusion_spec,
            decision_schedule_spec=decision_schedule_spec,
            source_task_idx=task_idx_for_env,
        )
        results["episode_decision_diagnostics"].append(decision_diag)
        if success:
            results["successes"] += 1
            results["success_episodes"].append(trial_idx)
        else:
            results["failure_episodes"].append(trial_idx)
        if visualize_future_video:
            results["episode_future_video_psnr"].append(episode_mean_psnr)

        rollout_video_path = save_rollout_video(
            video_dir,
            replay_images,
            f"task{cfg.EVALUATION.task_id}_trial{trial_idx}",
            success=success,
            task_description=task_description,
        )
        model_input_video_path = save_rollout_video(
            video_dir,
            model_input_images,
            f"task{cfg.EVALUATION.task_id}_trial{trial_idx}",
            success=success,
            task_description=task_description,
            fps=24,
            filename_suffix="--model-input",
        )
        results["episode_video_paths"].append(
            {
                "trial_index": int(trial_idx),
                "rollout": str(rollout_video_path),
                "model_input": str(model_input_video_path),
                "model_input_frame_unit": "control_step_zero_order_hold",
            }
        )
        if visualize_future_video:
            if len(predicted_future_video_clips) == 0:
                logging.warning(
                    "No predicted future frames collected for task %s trial %s.",
                    cfg.EVALUATION.task_id,
                    trial_idx,
                )
            else:
                all_gt_frames = []
                all_pred_frames = []
                for clip in predicted_future_video_clips:
                    all_gt_frames.extend(clip["gt_frames"])
                    all_pred_frames.extend(clip["pred_frames"])
                    save_prediction_video(
                        predicted_video_dir,
                        clip["gt_frames"],
                        clip["pred_frames"],
                        f"task{cfg.EVALUATION.task_id}_trial{trial_idx}",
                        clip["replan_idx"],
                        success=success,
                        task_description=task_description,
                    )
                save_prediction_video(
                    predicted_video_dir,
                    all_gt_frames,
                    all_pred_frames,
                    f"task{cfg.EVALUATION.task_id}_trial{trial_idx}",
                    "all",
                    success=success,
                    task_description=task_description,
                )

    if visualize_future_video:
        valid_episode_psnr = [x for x in results["episode_future_video_psnr"] if x is not None]
        if len(valid_episode_psnr) > 0:
            results["future_video_psnr_mean"] = float(np.mean(valid_episode_psnr))
    try:
        env.close()
    except Exception:
        logging.exception("Failed to close LIBERO environment cleanly.")
    return results


def _run_env_edit_manifest_batch(
    cfg: DictConfig,
    *,
    model: torch.nn.Module,
    processor: FastWAMProcessor,
    action_horizon: int,
    input_w: int,
    input_h: int,
    model_device: str,
) -> dict:
    manifest_path = Path(str(cfg.EVALUATION.get("batch_manifest", ""))).expanduser()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"FastWAM batch manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    # Load the optional batch-level temporal schedule. Without one, the
    # observation transform uses its default schedule.
    batch_schedule_spec, batch_schedule_source = _load_decision_schedule_spec(cfg)
    # A4 phase-targeted batches share one observation-corruption spec across
    # manifest rows. Load it once for this batch; an absent spec leaves the
    # observation inputs unchanged.
    batch_obs_occlusion_spec, batch_obs_occlusion_source = _load_obs_occlusion_spec(cfg)
    all_cases = manifest.get("cases", [])
    shard_id = int(cfg.EVALUATION.get("batch_shard_id", 0))
    num_shards = int(cfg.EVALUATION.get("batch_num_shards", 1))
    if num_shards <= 0 or not 0 <= shard_id < num_shards:
        raise ValueError(
            f"Invalid batch shard {shard_id}/{num_shards}; expected "
            f"0 <= shard_id < num_shards."
        )
    selected = [
        (index, case)
        for index, case in enumerate(all_cases)
        if index % num_shards == shard_id
    ]
    if not selected:
        raise RuntimeError(
            f"No manifest cases selected for shard {shard_id}/{num_shards}."
        )

    output_root = Path(str(cfg.EVALUATION.output_dir))
    case_root = output_root / "cases"
    case_root.mkdir(parents=True, exist_ok=True)
    summary_path = output_root / "batch_summary.json"
    suite_cache = {}
    completed_rows = []
    started_at = time.time()

    def write_summary(status: str, error: str = "") -> None:
        payload = {
            "schema_version": "a4_fastwam_manifest_batch.v1",
            "status": status,
            "manifest": str(manifest_path.resolve()),
            "manifest_case_count": len(all_cases),
            "shard_id": shard_id,
            "num_shards": num_shards,
            "selected_case_count": len(selected),
            "completed_case_count": len(completed_rows),
            "duration_seconds": time.time() - started_at,
            "error": error,
            "cases": completed_rows,
        }
        summary_path.write_text(
            json.dumps(payload, indent=2, cls=NumpyEncoder) + "\n",
            encoding="utf-8",
        )

    write_summary("running")
    benchmark_dict = benchmark.get_benchmark_dict()
    for local_index, (manifest_index, case) in enumerate(selected, start=1):
        case_id = str(case["case_id"])
        trial_index = int(case.get("trial_index", 0))
        # Trial-indexed manifests use one row per trial. Keep the semantic
        # case_id in recorder metadata and separate each trial's artifacts so
        # resume checks select the matching trial.
        case_dir_name = f"{case_id}__trial{trial_index:02d}" if "trial_index" in case else case_id
        suite_name = str(case["suite"])
        task_id = int(case["task_id"])
        spec_path = Path(str(case["spec"]))
        case_dir = case_root / case_dir_name
        result_path = case_dir / "result.json"
        case_dir.mkdir(parents=True, exist_ok=True)
        if result_path.is_file():
            prior = json.loads(result_path.read_text(encoding="utf-8"))
            if prior.get("status") == "complete":
                completed_rows.append(
                    {
                        "case_id": case_id,
                        "manifest_index": manifest_index,
                        "result": str(result_path),
                        "successes": int(prior.get("successes", 0)),
                        "resumed": True,
                    }
                )
                write_summary("running")
                print(
                    f"[resume {local_index}/{len(selected)}] {case_id}",
                    flush=True,
                )
                continue

        print(f"[run {local_index}/{len(selected)}] {case_id}", flush=True)
        case_started_at = time.time()
        try:
            # Batch cases share one loaded model process. Refresh the recorder
            # identity before each episode to retain its exact manifest case ID.
            os.environ["WAM_CASE_ID"] = case_id
            if suite_name not in suite_cache:
                suite_cache[suite_name] = benchmark_dict[suite_name]()
            task_suite = suite_cache[suite_name]
            task = task_suite.get_task(task_id)
            initial_states = list(task_suite.get_task_init_states(task_id))
            if not initial_states:
                raise RuntimeError(
                    f"Task {suite_name}:{task_id} returned no initial states."
                )
            env_edit_spec = json.loads(spec_path.read_text(encoding="utf-8"))
            os.environ["WAM_TRACK"] = str(env_edit_spec["track"])
            os.environ["WAM_SUBTASK"] = str(
                env_edit_spec.get(
                    "subtask", env_edit_spec.get("ambiguity_kind", "state_benchmark")
                )
            )
            os.environ["WAM_CONDITION"] = str(
                env_edit_spec.get("observation_mode", "perturbed")
            )
            case_cfg = OmegaConf.create(
                OmegaConf.to_container(cfg, resolve=True)
            )
            case_cfg.EVALUATION.task_suite_name = suite_name
            case_cfg.EVALUATION.task_id = task_id
            case_cfg.EVALUATION.trial_start = trial_index
            case_cfg.EVALUATION.num_trials = 1
            case_cfg.EVALUATION.output_dir = str(case_dir)
            video_dir = case_dir / "videos"
            predicted_video_dir = case_dir / "predicted_videos"
            video_dir.mkdir(parents=True, exist_ok=True)
            if bool(case_cfg.EVALUATION.get("visualize_future_video", False)):
                predicted_video_dir.mkdir(parents=True, exist_ok=True)

            task_results = run_single_task(
                task=task,
                initial_states=initial_states,
                model=model,
                processor=processor,
                cfg=case_cfg,
                video_dir=video_dir,
                predicted_video_dir=predicted_video_dir,
                action_horizon=action_horizon,
                input_w=input_w,
                input_h=input_h,
                model_device=model_device,
                cross_task_spec={},
                source_task_idx=task_id,
                donor_task_idx=None,
                donor_task_description="",
                task_description_override="",
                task_descriptions_source="",
                env_edit_spec=env_edit_spec,
                env_edit_source=str(spec_path.resolve()),
                obs_occlusion_spec=batch_obs_occlusion_spec,
                obs_occlusion_source=batch_obs_occlusion_source,
                decision_schedule_spec=batch_schedule_spec,
                decision_schedule_source=batch_schedule_source,
            )
            result = {
                "schema_version": "a4_fastwam_case_result.v1",
                "status": "complete",
                "case_id": case_id,
                "manifest_index": manifest_index,
                "suite": suite_name,
                "task_id": task_id,
                "condition": case.get("condition", _single_view_name(case_cfg)),
                "instruction": case.get("instruction", task_results["task_description"]),
                "spec": str(spec_path.resolve()),
                "gpu_id": int(cfg.gpu_id),
                "total_episodes": 1,
                "duration": time.time() - case_started_at,
                **task_results,
            }
            result_path.write_text(
                json.dumps(result, indent=2, cls=NumpyEncoder) + "\n",
                encoding="utf-8",
            )
            completed_rows.append(
                {
                    "case_id": case_id,
                    "manifest_index": manifest_index,
                    "result": str(result_path),
                    "successes": int(result["successes"]),
                    "resumed": False,
                }
            )
            write_summary("running")
            torch.cuda.empty_cache()
        except Exception as exc:
            error_payload = {
                "schema_version": "a4_fastwam_case_error.v1",
                "status": "error",
                "case_id": case_id,
                "manifest_index": manifest_index,
                "suite": suite_name,
                "task_id": task_id,
                "condition": case.get("condition", _c5_view_name(cfg)),
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
            (case_dir / "error.json").write_text(
                json.dumps(error_payload, indent=2) + "\n",
                encoding="utf-8",
            )
            write_summary("error", error_payload["error"])
            raise

    successes = sum(int(row["successes"]) for row in completed_rows)
    write_summary("complete")
    print(
        f"FastWAM manifest shard {shard_id}/{num_shards} complete: "
        f"{successes}/{len(completed_rows)} successes",
        flush=True,
    )
    return {
        "status": "complete",
        "shard_id": shard_id,
        "num_shards": num_shards,
        "case_count": len(completed_rows),
        "successes": successes,
        "summary": str(summary_path),
    }


@hydra.main(version_base="1.3", config_path=str(runtime.fastwam / "configs"), config_name="sim_libero.yaml")
def eval_single_process(cfg: DictConfig):
    start_time = time.time()
    partial_state = PartialState()
    partial_state.config = cfg

    if cfg.get("seed") is not None:
        set_global_seed(int(cfg.seed), get_worker_init_fn=False)

    if cfg.ckpt is None:
        raise ValueError("cfg.ckpt must not be None.")
    _validate_visualize_future_video_cfg(cfg)

    env_num = int(cfg.EVALUATION.get("env_num", 1))
    if env_num != 1:
        raise ValueError(
            "Only env_num=1 is supported in eval_libero_single.py. "
            "Use run_libero_manager/run_libero_parallel_test.sh for multi-GPU task parallelism."
        )

    model_device = _resolve_eval_device(cfg)
    model_dtype = _mixed_precision_to_model_dtype(cfg.get("mixed_precision", "bf16"))
    model = instantiate(cfg.model, model_dtype=model_dtype, device=model_device)
    _load_model_checkpoint(model, str(cfg.ckpt))
    model = model.to(model_device).eval()

    dataset_stats_path = _resolve_dataset_stats_path(cfg)
    dataset_stats = load_dataset_stats_from_json(str(dataset_stats_path))
    processor: FastWAMProcessor = instantiate(cfg.data.train.processor).eval()
    processor.set_normalizer_from_stats(dataset_stats)
    logging.info("Using dataset stats: %s", dataset_stats_path)

    action_horizon_cfg = cfg.EVALUATION.get("action_horizon", None)
    if action_horizon_cfg is None:
        action_horizon = int(cfg.data.train.num_frames) - 1
    else:
        action_horizon = int(action_horizon_cfg)
    if action_horizon <= 0:
        raise ValueError(f"EVALUATION.action_horizon must be positive, got {action_horizon}")

    video_size = cfg.data.train.get("video_size", [224, 224])
    if len(video_size) != 2:
        raise ValueError(f"data.train.video_size must be [H, W], got {video_size}")
    input_h = int(video_size[0])
    input_w = int(video_size[1])
    concat_multi_camera = cfg.data.train.get("concat_multi_camera", None)
    shape_meta_images = [meta["shape"] for meta in processor.shape_meta["images"]]

    if str(cfg.EVALUATION.get("batch_manifest", "")).strip():
        return _run_env_edit_manifest_batch(
            cfg,
            model=model,
            processor=processor,
            action_horizon=action_horizon,
            input_w=input_w,
            input_h=input_h,
            model_device=model_device,
        )

    local_log_dir = Path(cfg.EVALUATION.output_dir)
    local_log_dir.mkdir(parents=True, exist_ok=True)
    video_dir = local_log_dir / cfg.EVALUATION.task_suite_name / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)
    predicted_video_dir = local_log_dir / cfg.EVALUATION.task_suite_name / "predicted_videos"
    if bool(cfg.EVALUATION.get("visualize_future_video", False)):
        predicted_video_dir.mkdir(parents=True, exist_ok=True)

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.EVALUATION.task_suite_name]()
    task = task_suite.get_task(cfg.EVALUATION.task_id)
    initial_states = task_suite.get_task_init_states(cfg.EVALUATION.task_id)
    task_descriptions_file = _get_optional_cfg_or_env(
        cfg,
        "task_descriptions_file",
        ["FASTWAM_TASK_DESCRIPTIONS_FILE", "LIBERO_TASK_DESCRIPTIONS_FILE"],
        "",
    )
    use_task_descriptions = _as_bool(
        _get_optional_cfg_or_env(
            cfg,
            "use_task_descriptions",
            ["FASTWAM_USE_TASK_DESCRIPTIONS", "LIBERO_USE_TASK_DESCRIPTIONS"],
            True,
        )
    )
    task_description_override, task_descriptions_source = _load_modified_instruction(
        task_descriptions_file,
        str(cfg.EVALUATION.task_suite_name),
        int(cfg.EVALUATION.task_id),
        enabled=use_task_descriptions,
    )
    env_edit_spec, env_edit_source = _load_json_or_inline(
        _get_optional_cfg_or_env(
            cfg,
            "env_edit_spec",
            ["FASTWAM_ENV_EDIT_SPEC", "LIBERO_ENV_EDIT_SPEC"],
            "",
        )
    )
    obs_occlusion_spec, obs_occlusion_source = _load_obs_occlusion_spec(cfg)
    decision_schedule_spec, decision_schedule_source = _load_decision_schedule_spec(cfg)
    if task_description_override:
        logging.info("Modified instruction: %s", task_description_override)
        logging.info("TaskDescriptions source: %s", task_descriptions_source)
    if env_edit_source:
        logging.info("Environment edit source: %s", env_edit_source)
    if obs_occlusion_source:
        logging.info("Observation perturbation source: %s", obs_occlusion_source)
    if decision_schedule_source:
        logging.info("Decision schedule source: %s", decision_schedule_source)
    cross_task_spec = _load_cross_task_view_swap_spec(cfg)
    cross_task_swap = _cross_task_view_swap_enabled(cross_task_spec)
    donor_task_idx = None
    donor_task_description = ""
    if cross_task_swap:
        donor_task_idx = _resolve_cross_task_donor(
            cross_task_spec,
            int(cfg.EVALUATION.task_id),
            int(task_suite.get_num_tasks()),
        )
        donor_task = task_suite.get_task(donor_task_idx)
        donor_task_description = str(getattr(donor_task, "language", ""))
        logging.info(
            "FastWAM cross-task view swap: source=%s:%s donor=%s:%s swapped_views=%s",
            cfg.EVALUATION.task_suite_name,
            cfg.EVALUATION.task_id,
            cfg.EVALUATION.task_suite_name,
            donor_task_idx,
            _normalize_cross_task_views(cross_task_spec.get("swapped_views", cross_task_spec.get("views"))),
        )
        logging.info("Donor instruction: %s", donor_task_description)

    initial_states = list(initial_states)
    if len(initial_states) == 0:
        raise RuntimeError(f"Task {cfg.EVALUATION.task_id} returned empty initial states.")
    trial_start = int(cfg.EVALUATION.get("trial_start", 0))
    num_trials = int(cfg.EVALUATION.num_trials)
    required_initial_states = trial_start + num_trials
    while len(initial_states) < required_initial_states:
        initial_states.extend(initial_states[: (required_initial_states - len(initial_states))])

    results = {
        "task_suite": cfg.EVALUATION.task_suite_name,
        "task_id": cfg.EVALUATION.task_id,
        "task_description": None,
        "successes": 0,
        "trial_start": trial_start,
        "total_episodes": num_trials,
        "gpu_id": int(cfg.gpu_id),
        "success_episodes": [],
        "failure_episodes": [],
        "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "duration": 0,
    }

    logging.info("Running LIBERO evaluation with env_num=1")
    task_results = run_single_task(
        task=task,
        initial_states=initial_states,
        model=model,
        processor=processor,
        cfg=cfg,
        video_dir=video_dir,
        predicted_video_dir=predicted_video_dir,
        action_horizon=action_horizon,
        input_w=input_w,
        input_h=input_h,
        model_device=model_device,
        cross_task_spec=cross_task_spec,
        source_task_idx=int(cfg.EVALUATION.task_id),
        donor_task_idx=donor_task_idx,
        donor_task_description=donor_task_description,
        task_description_override=task_description_override,
        task_descriptions_source=task_descriptions_source,
        env_edit_spec=env_edit_spec,
        env_edit_source=env_edit_source,
        obs_occlusion_spec=obs_occlusion_spec,
        obs_occlusion_source=obs_occlusion_source,
        decision_schedule_spec=decision_schedule_spec,
        decision_schedule_source=decision_schedule_source,
    )
    results.update(task_results)

    results["duration"] = time.time() - start_time
    output_dir = Path(cfg.EVALUATION.output_dir) / cfg.EVALUATION.task_suite_name
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"gpu{cfg.gpu_id}_task{cfg.EVALUATION.task_id}_results.json"

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, cls=NumpyEncoder)

    print(
        f"Task {cfg.EVALUATION.task_id} completed: "
        f"{results['successes']}/{num_trials} successes"
    )
    if results.get("future_video_psnr_mean") is not None:
        print(f"Task {cfg.EVALUATION.task_id} future-video PSNR mean: {results['future_video_psnr_mean']:.4f}")
    print(f"Time taken: {results['duration']:.2f} seconds")
    return results


if __name__ == "__main__":
    eval_single_process()
