import numpy as np
try:
    from evaluation.robotwin.websocket_client_policy import WebsocketClientPolicy
except ModuleNotFoundError:
    class WebsocketClientPolicy:
        """Fail closed when this snapshot is imported only for env helpers."""

        def __init__(self, *_args, **_kwargs):
            raise RuntimeError(
                "LingBot websocket policy dependencies are unavailable; "
                "only the isolated LIBERO environment helpers may be used"
            )
import argparse
import copy
import json
from libero.libero import benchmark
import time
import multiprocessing as mp
from libero.libero.envs import OffScreenRenderEnv, SegmentationRenderEnv
from libero.libero.envs.env_wrapper import ControlEnv
from pathlib import Path
from tqdm import tqdm
import os as _os
import sys as _sys

_SIM_ROOT = Path(__file__).resolve().parents[4]
if str(_SIM_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_SIM_ROOT))

from source._shared_core.episode_recorder import (
    capture_libero_state,
    make_episode_recorder,
)

_STATE_BENCH_ROOT = Path(__file__).resolve().parents[4]
if str(_STATE_BENCH_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_STATE_BENCH_ROOT))
from source.D2_D3_state_20260823.experiment_env import prepare_environment as _prepare_state_environment

try:
    from perturbation_schedule import (
        PerturbationScheduler as _WSB_SCHEDULER_CLS,
        infer_phase as _wsb_infer_phase,
        PerViewFrameBuffer as _WSB_BUFFER_CLS,
        reset_phase_state as _wsb_reset_phase_state,
    )
except Exception:
    _WSB_SCHEDULER_CLS = None
    _wsb_infer_phase = None
    _WSB_BUFFER_CLS = None
    _wsb_reset_phase_state = None


def _wsb_load_schedule_spec():
    raw = _os.environ.get("LIBERO_SCHEDULE_SPEC", "").strip()
    if not raw:
        return {}
    try:
        if _os.path.exists(raw):
            with open(raw) as _f:
                return json.load(_f)
        return json.loads(raw)
    except Exception:
        return {}


try:
    from lerobot.datasets.utils import write_json
except ImportError:
    def write_json(data, fpath):
        Path(fpath).parent.mkdir(exist_ok=True, parents=True)
        with open(fpath, "w") as f:
            json.dump(data, f, indent=2)
import os
import imageio
import cv2
import re


LIBERO_OBJECT_XML_OVERRIDE_ENV = "LIBERO_OBJECT_XML_OVERRIDE_MAP"
PRECOMPILE_XML_SCALE_MODES = {"precompile_xml", "xml_physical", "physical_xml"}
LIBERO_STATE_LAYOUT_CACHE = {}


MODEL_VIEW_CONFIGS = {
    "agentview": {
        "image_key": "agentview_image",
        "segmentation_key": "agentview_segmentation_instance",
        "model_key": "observation.images.agentview_rgb",
    },
    "robot0_eye_in_hand": {
        "image_key": "robot0_eye_in_hand_image",
        "segmentation_key": "robot0_eye_in_hand_segmentation_instance",
        "model_key": "observation.images.eye_in_hand_rgb",
    },
}


_SINGLE_VIEW_ALIASES = {
    "agent": "agentview",
    "agentview": "agentview",
    "wrist": "robot0_eye_in_hand",
    "eye_in_hand": "robot0_eye_in_hand",
    "robot0_eye_in_hand": "robot0_eye_in_hand",
}


def _single_view_name():
    raw = _os.environ.get("WAM_SINGLE_VIEW", "").strip().lower()
    if not raw:
        return ""
    if raw not in _SINGLE_VIEW_ALIASES:
        raise ValueError(
            f"Unsupported WAM_SINGLE_VIEW={raw!r}; "
            f"expected one of {sorted(_SINGLE_VIEW_ALIASES)}"
        )
    return _SINGLE_VIEW_ALIASES[raw]


def _view_copy_name():
    """Return the retained A5 camera copied into both model camera slots."""
    raw = _os.environ.get("WAM_VIEW_COPY", "").strip().lower()
    if not raw:
        return ""
    if raw not in _SINGLE_VIEW_ALIASES:
        raise ValueError(
            f"Unsupported WAM_VIEW_COPY={raw!r}; "
            f"expected one of {sorted(_SINGLE_VIEW_ALIASES)}"
        )
    return _SINGLE_VIEW_ALIASES[raw]


def _c5_view_name():
    return _view_copy_name() or _single_view_name()


def _single_view_model_keys():
    if _view_copy_name():
        return [
            "observation.images.agentview_rgb",
            "observation.images.eye_in_hand_rgb",
        ]
    view_name = _single_view_name()
    if not view_name:
        return [
            "observation.images.agentview_rgb",
            "observation.images.eye_in_hand_rgb",
        ]
    return [MODEL_VIEW_CONFIGS[view_name]["model_key"]]


def _select_single_view_model_obs(model_obs):
    copy_view = _view_copy_name()
    if copy_view:
        source_key = MODEL_VIEW_CONFIGS[copy_view]["model_key"]
        if source_key not in model_obs:
            raise KeyError(
                f"A5 view-copy input {copy_view!r} requires model key {source_key!r}"
            )
        source = np.asarray(model_obs[source_key])
        return {
            "observation.images.agentview_rgb": source.copy(),
            "observation.images.eye_in_hand_rgb": source.copy(),
        }
    view_name = _single_view_name()
    if not view_name:
        return model_obs
    model_key = MODEL_VIEW_CONFIGS[view_name]["model_key"]
    if model_key not in model_obs:
        raise KeyError(
            f"A5 single-view input {view_name!r} requires model key {model_key!r}"
        )
    return {model_key: model_obs[model_key]}


def _as_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _load_libero_modified_instruction(task_descriptions_file, libero_benchmark, task_idx, enabled=True):
    if not _as_bool(enabled) or not task_descriptions_file:
        return "", ""

    task_descriptions_path = Path(task_descriptions_file)
    if not task_descriptions_path.exists():
        raise FileNotFoundError(f"TaskDescriptions file not found: {task_descriptions_path}")

    with open(task_descriptions_path, "r", encoding="utf-8") as f:
        task_descriptions = json.load(f)

    for task in task_descriptions.get("tasks", []):
        if task.get("benchmark") == libero_benchmark and int(task.get("task_id")) == int(task_idx):
            modified_instruction = (task.get("modified_instruction") or "").strip()
            if modified_instruction:
                return modified_instruction, str(task_descriptions_path)
            return "", str(task_descriptions_path)

    print(f"TaskDescriptions entry for LIBERO task '{libero_benchmark}:{task_idx}' was not found in {task_descriptions_path}")
    return "", str(task_descriptions_path)


def _existing_episode_results(out_dir, libero_benchmark, task_idx, test_num):
    benchmark_dir = Path(out_dir) / libero_benchmark
    if not benchmark_dir.exists():
        return {}

    results = {}
    for task_dir in benchmark_dir.glob(f"{task_idx}_*"):
        if not task_dir.is_dir():
            continue
        for video_path in task_dir.glob("*.mp4"):
            parts = video_path.stem.split("_")
            if len(parts) < 2 or not parts[0].isdigit():
                continue
            episode_idx = int(parts[0])
            if episode_idx >= test_num:
                continue
            success = parts[1] == "True"
            results[episode_idx] = success
    return results


def _load_env_edit_spec(env_edit_spec):
    if not env_edit_spec:
        return {}, ""

    spec_path = Path(env_edit_spec)
    if spec_path.exists():
        with open(spec_path, "r", encoding="utf-8") as f:
            return json.load(f), str(spec_path)

    return json.loads(env_edit_spec), "inline"


def _load_obs_occlusion_spec(obs_occlusion_spec):
    if not obs_occlusion_spec:
        return {"enabled": False}, ""

    spec_path = Path(obs_occlusion_spec)
    if spec_path.exists():
        with open(spec_path, "r", encoding="utf-8") as f:
            spec = json.load(f)
        source = str(spec_path)
    else:
        spec = json.loads(obs_occlusion_spec)
        source = "inline"

    spec.setdefault("enabled", True)
    spec.setdefault("mask_source", "segmentation")
    spec.setdefault("target_policy", "obj_of_interest")
    spec.setdefault("views", ["agentview", "robot0_eye_in_hand"])
    spec.setdefault("fill_rgb", [0, 0, 0])
    spec.setdefault("mask_geometry", "segmentation")
    spec.setdefault("dilate_px", 0)
    spec.setdefault("save_masked_video", False)
    return spec, source


def _obs_occlusion_enabled(spec):
    return _as_bool((spec or {}).get("enabled"))


def _obs_occlusion_needs_segmentation(spec):
    return _obs_occlusion_enabled(spec) and (spec or {}).get("mask_source", "segmentation") in {
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


def _is_cross_task_view_swap(spec):
    return _obs_occlusion_enabled(spec) and (spec or {}).get("mask_source") in {
        "cross_task_view_swap",
        "cross_task_view_conflict",
    }


def _normalize_occlusion_views(views):
    if not views:
        return ["agentview", "robot0_eye_in_hand"]

    normalized = []
    aliases = {
        "observation.images.agentview_rgb": "agentview",
        "observation.images.eye_in_hand_rgb": "robot0_eye_in_hand",
        "eye_in_hand": "robot0_eye_in_hand",
    }
    for view in views:
        view_name = aliases.get(str(view), str(view))
        if view_name not in MODEL_VIEW_CONFIGS:
            raise ValueError(f"Unsupported occlusion view: {view}")
        normalized.append(view_name)
    return normalized


def _resolve_cross_task_donor(spec, source_benchmark, source_task_idx, num_tasks):
    donor_benchmark = str(spec.get("donor_benchmark") or source_benchmark)
    task_key = f"{source_benchmark}:{source_task_idx}"
    donor_by_task = spec.get("donor_task_by_task", {})
    if task_key in donor_by_task:
        donor_task_idx = int(donor_by_task[task_key])
    elif str(source_task_idx) in donor_by_task:
        donor_task_idx = int(donor_by_task[str(source_task_idx)])
    else:
        donor_task_idx = (int(source_task_idx) + int(spec.get("donor_task_offset", 1))) % int(num_tasks)
    if donor_benchmark == source_benchmark and donor_task_idx == int(source_task_idx):
        raise ValueError(f"Cross-task view swap donor must differ from source task: {task_key}")
    return donor_benchmark, donor_task_idx


def _resolve_occlusion_targets(env_in, spec, libero_benchmark, task_idx):
    policy = spec.get("target_policy", "obj_of_interest")
    if policy == "obj_of_interest":
        return list(getattr(env_in, "obj_of_interest", []))
    if policy == "object_names":
        keyed_targets = spec.get("target_objects_by_task", {})
        key = f"{libero_benchmark}:{task_idx}"
        targets = keyed_targets.get(key, spec.get("object_names", []))
        return [str(target) for target in targets]
    raise ValueError(f"Unsupported occlusion target_policy: {policy}")


def _apply_env_visual_occluder(base_env, spec):
    """Apply a render-only A3 occluder using existing MuJoCo geoms.

    Adding a compiled geom after LIBERO construction is unsafe, so the isolated
    prototype uses the existing interaction geoms and changes only their render
    material. Collision masks, sizes, poses and dynamics remain untouched.
    """
    if not _as_bool((spec or {}).get("env_visual_occluder", False)):
        return {"enabled": False, "matched_geoms": []}
    sim = base_env.sim
    part = str((spec or {}).get("interaction_parts", ["gripper"])[0]).lower()
    tokens = {
        "gripper": ("gripper", "finger", "eef"),
        "object_contact": tuple(str(x).lower() for x in (spec or {}).get("object_geom_tokens", ())),
        "receptacle_contact": tuple(str(x).lower() for x in (spec or {}).get("receptacle_geom_tokens", ())),
    }.get(part, ())
    if not tokens:
        return {"enabled": True, "part": part, "matched_geoms": [], "reason": "no_tokens"}
    rgba = np.asarray((spec or {}).get("occluder_rgba", [0.0, 0.0, 0.0, 1.0]), dtype=float)
    if rgba.size != 4:
        raise ValueError("occluder_rgba must have 4 values")
    matched = []
    for gid in range(int(sim.model.ngeom)):
        name = sim.model.geom_id2name(gid) or ""
        if any(tok in name.lower() for tok in tokens):
            sim.model.geom_rgba[gid] = rgba
            matched.append({"geom_id": int(gid), "geom_name": name})
    return {
        "enabled": True,
        "mode": "existing_geom_render_material",
        "part": part,
        "matched_geoms": matched,
        "collision_unchanged": True,
        "rgba": rgba.tolist(),
    }


def _target_mask_from_segmentation(env_in, segmentation_image, target_objects):
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


def _expand_mask_geometry(mask, geometry):
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


def _dilate_mask(mask, dilate_px):
    dilate_px = int(dilate_px or 0)
    if dilate_px <= 0 or not mask.any():
        return mask
    kernel_size = dilate_px * 2 + 1
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    return cv2.dilate(mask.astype(np.uint8), kernel, iterations=1).astype(bool)


def _apply_target_shape_warp(image, mask, spec):
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


def _resolve_center_mask_fraction(spec):
    for key in ("mask_fraction", "occlusion_fraction", "area_fraction", "center_fraction"):
        if key in spec:
            fraction = float(spec[key])
            break
    else:
        fraction = 1.0
    if fraction > 1.0:
        fraction /= 100.0
    if not 0.0 <= fraction <= 1.0:
        raise ValueError(f"Center occlusion fraction must be in [0, 1] or [0, 100], got {fraction}")
    return fraction


def _center_rect_mask(image_shape, fraction):
    height, width = image_shape[:2]
    mask = np.zeros((height, width), dtype=bool)
    if fraction <= 0.0:
        return mask
    if fraction >= 1.0:
        mask[:, :] = True
        return mask

    scale = float(np.sqrt(fraction))
    rect_h = max(1, min(height, int(round(height * scale))))
    rect_w = max(1, min(width, int(round(width * scale))))
    y0 = (height - rect_h) // 2
    x0 = (width - rect_w) // 2
    mask[y0 : y0 + rect_h, x0 : x0 + rect_w] = True
    return mask


def _resolve_text_overlay(spec, libero_benchmark, task_idx):
    task_key = f"{libero_benchmark}:{task_idx}"
    text_by_task = spec.get("text_by_task", {})
    if task_key in text_by_task:
        return str(text_by_task[task_key])
    if str(task_idx) in text_by_task:
        return str(text_by_task[str(task_idx)])
    if "text" in spec:
        return str(spec.get("text") or "")
    raise ValueError(f"text_overlay spec needs text or text_by_task entry for {task_key}")


def _wrap_overlay_text(text, image_width, font, font_scale, thickness, margin, max_lines, allow_truncate=True):
    words = str(text).split()
    if not words:
        return [], False

    max_width = max(1, int(image_width - 2 * margin))
    lines = []
    current = ""
    truncated = False
    for word in words:
        candidate = word if not current else f"{current} {word}"
        (candidate_width, _), _ = cv2.getTextSize(candidate, font, font_scale, thickness)
        if candidate_width <= max_width:
            current = candidate
            continue
        if current:
            lines.append(current)
        current = word
        if len(lines) >= max_lines:
            truncated = True
            break

    if current and len(lines) < max_lines:
        lines.append(current)

    if truncated and allow_truncate and lines:
        while lines:
            line = lines[-1].rstrip(". ")
            candidate = line + "..."
            (candidate_width, _), _ = cv2.getTextSize(candidate, font, font_scale, thickness)
            if candidate_width <= max_width or len(line) <= 1:
                lines[-1] = candidate
                break
            lines[-1] = line[:-1]

    return lines, truncated


def _apply_text_overlay(image, text, spec):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = float(spec.get("font_scale", 0.34))
    min_font_scale = float(spec.get("min_font_scale", font_scale))
    thickness = int(spec.get("font_thickness", 1))
    margin = int(spec.get("margin_px", 3))
    line_gap = int(spec.get("line_gap_px", 2))
    max_lines = int(spec.get("max_lines", 3))
    allow_truncate = _as_bool(spec.get("allow_truncate", True))
    max_box_fraction = float(spec.get("max_box_fraction", 1.0))
    position = str(spec.get("position", "top")).lower()
    box_alpha = float(spec.get("box_alpha", 0.72))
    text_rgb = tuple(int(v) for v in spec.get("text_rgb", [255, 255, 255]))
    box_rgb = np.array(spec.get("box_rgb", [0, 0, 0]), dtype=image.dtype)

    chosen_scale = font_scale
    lines = []
    truncated = False
    scale = font_scale
    while True:
        lines, truncated = _wrap_overlay_text(
            text,
            image.shape[1],
            font,
            scale,
            thickness,
            margin,
            max_lines,
            allow_truncate=allow_truncate,
        )
        text_sizes = [cv2.getTextSize(line, font, scale, thickness)[0] for line in lines]
        line_height = max((size[1] for size in text_sizes), default=0)
        box_height = min(
            image.shape[0],
            2 * margin + len(lines) * line_height + max(0, len(lines) - 1) * line_gap,
        )
        fits_text = allow_truncate or not truncated
        fits_box = box_height <= int(round(image.shape[0] * max_box_fraction))
        if (fits_text and fits_box) or scale <= min_font_scale:
            chosen_scale = scale
            break
        scale = max(min_font_scale, scale - 0.02)

    if not lines:
        return {
            "masked_pixels": 0,
            "mask_fraction": 0.0,
            "missing_targets": [],
            "overlay_text": text,
            "overlay_lines": [],
            "overlay_rect": None,
            "overlay_font_scale": chosen_scale,
            "overlay_truncated": truncated,
        }

    text_sizes = [cv2.getTextSize(line, font, chosen_scale, thickness)[0] for line in lines]
    line_height = max(size[1] for size in text_sizes)
    box_height = min(image.shape[0], 2 * margin + len(lines) * line_height + (len(lines) - 1) * line_gap)
    if position == "bottom":
        y0 = image.shape[0] - box_height
    else:
        y0 = 0
    y1 = min(image.shape[0], y0 + box_height)
    x0, x1 = 0, image.shape[1]

    original_region = image[y0:y1, x0:x1].copy()
    overlay_region = np.empty_like(original_region)
    overlay_region[:, :] = box_rgb
    image[y0:y1, x0:x1] = cv2.addWeighted(
        overlay_region,
        box_alpha,
        original_region,
        1.0 - box_alpha,
        0,
    )

    baseline = y0 + margin + line_height
    for line in lines:
        cv2.putText(
            image,
            line,
            (margin, baseline),
            font,
            chosen_scale,
            text_rgb,
            thickness,
            lineType=cv2.LINE_AA,
        )
        baseline += line_height + line_gap

    rect_area = int((y1 - y0) * (x1 - x0))
    return {
        "masked_pixels": rect_area,
        "mask_fraction": rect_area / float(image.shape[0] * image.shape[1]),
        "missing_targets": [],
        "overlay_text": text,
        "overlay_lines": lines,
        "overlay_rect": [int(x0), int(y0), int(x1), int(y1)],
        "overlay_font_scale": chosen_scale,
        "overlay_truncated": truncated,
    }


def _normalize_object_label_targets(raw_targets, task_key):
    targets = []
    for item in raw_targets or []:
        if isinstance(item, dict):
            object_name = item.get("object") or item.get("object_name") or item.get("target")
            label = item.get("label") or item.get("wrong_label") or item.get("text")
        else:
            object_name, label = item
        if not object_name or not label:
            raise ValueError(f"Invalid object label entry for {task_key}: {item}")
        targets.append({"object": str(object_name), "label": str(label)})
    return targets


def _resolve_object_label_targets(env_in, spec, libero_benchmark, task_idx, view_name=None):
    task_key = f"{libero_benchmark}:{task_idx}"
    if view_name:
        targets_by_view = spec.get("label_targets_by_view", {})
        view_targets = targets_by_view.get(view_name)
        if view_targets is None and view_name in MODEL_VIEW_CONFIGS:
            view_targets = targets_by_view.get(MODEL_VIEW_CONFIGS[view_name]["model_key"])
        if isinstance(view_targets, dict):
            raw_targets = view_targets.get(task_key, view_targets.get(str(task_idx), view_targets.get("default")))
            if raw_targets is not None:
                return _normalize_object_label_targets(raw_targets, task_key)
        elif view_targets is not None:
            return _normalize_object_label_targets(view_targets, task_key)

    targets_by_task = spec.get("label_targets_by_task", {})
    raw_targets = targets_by_task.get(task_key, targets_by_task.get(str(task_idx)))
    if raw_targets:
        return _normalize_object_label_targets(raw_targets, task_key)

    object_names = _resolve_occlusion_targets(env_in, spec, libero_benchmark, task_idx)
    label_by_object = spec.get("label_by_object", {})
    default_label = spec.get("default_label")
    targets = []
    for object_name in object_names:
        label = label_by_object.get(object_name, default_label)
        if label:
            targets.append({"object": str(object_name), "label": str(label)})
    if not targets:
        raise ValueError(f"No object label targets resolved for {task_key}")
    return targets


def _bbox_from_mask(mask):
    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _draw_object_category_label(image, bbox, label, spec):
    x0, y0, x1, y1 = bbox
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = float(spec.get("label_font_scale", 0.36))
    thickness = int(spec.get("label_font_thickness", 1))
    margin = int(spec.get("label_margin_px", 2))
    offset = int(spec.get("label_offset_px", 3))
    text_rgb = tuple(int(v) for v in spec.get("label_text_rgb", [255, 255, 255]))
    box_rgb = tuple(int(v) for v in spec.get("label_box_rgb", [0, 0, 0]))
    border_rgb = tuple(int(v) for v in spec.get("label_border_rgb", [255, 64, 64]))
    draw_bbox = _as_bool(spec.get("draw_object_bbox", True))
    draw_leader = _as_bool(spec.get("draw_leader_line", True))

    h, w = image.shape[:2]
    (text_w, text_h), baseline = cv2.getTextSize(str(label), font, font_scale, thickness)
    label_w = min(w, text_w + 2 * margin)
    label_h = min(h, text_h + baseline + 2 * margin)

    if x1 + offset + label_w <= w:
        lx0 = x1 + offset
    elif x0 - offset - label_w >= 0:
        lx0 = x0 - offset - label_w
    else:
        lx0 = max(0, min(w - label_w, x0))

    if y0 - offset - label_h >= 0:
        ly0 = y0 - offset - label_h
    elif y1 + offset + label_h <= h:
        ly0 = y1 + offset
    else:
        ly0 = max(0, min(h - label_h, y0))

    lx1 = min(w, lx0 + label_w)
    ly1 = min(h, ly0 + label_h)

    if draw_bbox:
        cv2.rectangle(image, (x0, y0), (max(x0, x1 - 1), max(y0, y1 - 1)), border_rgb, 1)
    cv2.rectangle(image, (lx0, ly0), (max(lx0, lx1 - 1), max(ly0, ly1 - 1)), box_rgb, -1)
    cv2.rectangle(image, (lx0, ly0), (max(lx0, lx1 - 1), max(ly0, ly1 - 1)), border_rgb, 1)
    text_x = lx0 + margin
    text_y = min(ly1 - margin - baseline, ly0 + margin + text_h)
    cv2.putText(image, str(label), (text_x, text_y), font, font_scale, text_rgb, thickness, cv2.LINE_AA)

    if draw_leader:
        obj_cx = int(round((x0 + x1) / 2.0))
        obj_cy = int(round((y0 + y1) / 2.0))
        label_cx = int(round((lx0 + lx1) / 2.0))
        label_cy = int(round((ly0 + ly1) / 2.0))
        cv2.line(image, (obj_cx, obj_cy), (label_cx, label_cy), border_rgb, 1, cv2.LINE_AA)

    changed_area = (ly1 - ly0) * (lx1 - lx0)
    return {
        "label": str(label),
        "object_bbox": [x0, y0, x1, y1],
        "label_rect": [int(lx0), int(ly0), int(lx1), int(ly1)],
        "changed_pixels": int(changed_area),
    }


def _draw_target_photo_replacement(image, source_image, mask, object_name, spec):
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
    desaturate = _as_bool(spec.get("photo_desaturate", False))

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


def _stable_text_seed(text):
    seed = 2166136261
    for char in str(text):
        seed ^= ord(char)
        seed = (seed * 16777619) & 0xFFFFFFFF
    return seed


def _apply_noise_to_mask(image, mask, spec, libero_benchmark, task_idx, view_name):
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
        probability = float(spec.get("noise_probability", 0.2))
        probability = max(0.0, min(1.0, probability))
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


def _wsb_coverage_crop(mask, coverage, direction="centroid"):
    """A1 partial occlusion (25/50/75%): keep exactly `coverage` fraction of the
    object's mask pixels, as a COHERENT central blob (the pixels nearest the mask
    centroid). This is pose/scale-invariant: it is always `coverage` of the object's
    currently-visible pixels regardless of distance, orientation, or being grasped.

    NOTE: apply this to the RAW segmentation mask BEFORE any dilation. Dilation on a
    small (far/grasped) object inflates the effective coverage, so partial occlusion
    must not be dilated."""
    coverage = float(coverage)
    if coverage >= 1.0 or not mask.any():
        return mask
    ys, xs = np.where(mask)
    target = int(round(len(ys) * coverage))
    if target <= 0:
        return np.zeros_like(mask)
    if target >= len(ys):
        return mask
    cy, cx = ys.mean(), xs.mean()
    # coherent central blob: nearest `coverage` fraction of pixels to the centroid
    d2 = (ys - cy) ** 2 + (xs - cx) ** 2
    order = np.argsort(d2, kind="stable")[:target]
    out = np.zeros_like(mask)
    out[ys[order], xs[order]] = True
    return out


def _wsb_world_to_pixel(base_env, camera_name, points_world, h, w):
    """Project world-frame 3D points to (row, col) pixels for `camera_name`.
    Returns list of (r, c) or None per point (None if behind/out of frame)."""
    try:
        from robosuite.utils.camera_utils import (
            get_camera_transform_matrix, project_points_from_world_to_camera)
    except Exception:
        # Fallback that uses MuJoCo's camera pose directly.  Some evaluation
        # environments cannot import robosuite.camera_utils because an optional
        # h5py dependency is absent; silently returning None would disable A3.
        try:
            sim = base_env.sim
            cid = int(sim.model.camera_name2id(camera_name))
            pos = np.asarray(sim.data.cam_xpos[cid], dtype=float)
            R = np.asarray(sim.data.cam_xmat[cid], dtype=float).reshape(3, 3)
            fovy = float(sim.model.cam_fovy[cid])
            fy = 0.5 * float(h) / np.tan(np.deg2rad(fovy) * 0.5)
            fx = fy * float(w) / float(h)
            out = []
            for p in points_world:
                q = R.T @ (np.asarray(p, dtype=float) - pos)
                if not np.isfinite(q).all() or q[2] >= -1e-6:
                    out.append(None); continue
                out.append((int(round(0.5*h - fy*q[1]/(-q[2]))),
                            int(round(0.5*w + fx*q[0]/(-q[2])))))
            return out
        except Exception:
            return [None] * len(points_world)
    sim = base_env.sim
    try:
        T = get_camera_transform_matrix(sim, camera_name, h, w)
    except Exception:
        return [None] * len(points_world)
    # a point is "visible" only if it projects strictly inside the frame with a small
    # margin. Points outside the camera FOV (common for the wrist cam and far objects
    # like the receptacle) project to / get clamped near an edge -> treat as not visible
    # and skip (rather than drawing a disk on an irrelevant border pixel).
    margin = int(max(2, 0.03 * max(h, w)))  # ~4px on 128
    out = []
    for p in points_world:
        try:
            px = project_points_from_world_to_camera(np.asarray(p, dtype=float).reshape(1, 3), T, h, w)[0]
            r, c = int(round(px[0])), int(round(px[1]))
            # NOTE: robosuite's projection already returns final image-frame (row, col);
            # no extra vertical flip (an earlier flip put disks in the wrong place).
            inside = (margin <= r < h - margin) and (margin <= c < w - margin)
            out.append((r, c) if inside else None)
        except Exception:
            out.append(None)
    return out


def _wsb_interaction_points(base_env, spec, target_objects):
    """Resolve interaction points in *world coordinates* (A3).

    A3 must not choose pixels independently in the two cameras.  We therefore resolve
    one point from the MuJoCo state and project that same point into both cameras.  If
    a real MuJoCo contact exists we use its contact position; otherwise we use the
    nearest point on the relevant geom (rather than a body centre).  This keeps the
    perturbation tied to the physical gripper/object/receptacle interface.
    """
    sim = base_env.sim
    def _body_map(env):
        cur = env
        for _ in range(5):
            m = getattr(cur, "obj_body_id", None)
            if m:
                return m
            cur = getattr(cur, "env", None)
            if cur is None:
                break
        return {}
    pts = {}
    parts = spec.get("interaction_parts", ["gripper", "object_contact", "receptacle_contact"])
    containers = ("basket", "plate", "cabinet", "microwave", "stove", "caddy", "drawer")
    obj_body_id = _body_map(base_env)
    # gripper point = grip site (between fingers)
    grip = None
    for _site_name in (
        "gripper0_grip_site",
        "robot0_grip_site",
        "gripper0_grip_site_cylinder",
        "robot0_eef_site",
    ):
        if grip is not None:
            break
        try:
            _sid = int(sim.model.site_name2id(_site_name))
            grip = np.asarray(sim.data.site_xpos[_sid], dtype=float).copy()
        except Exception:
            try:
                _p = sim.data.get_site_xpos(_site_name)
                grip = np.asarray(_p, dtype=float).copy()
            except Exception:
                continue
    if "gripper" in parts and grip is not None:
        pts["gripper"] = grip
    manip_name = None
    manip_bid = None
    if "object_contact" in parts or "receptacle_contact" in parts:
        # The manipulated object = non-container target nearest the gripper.
        manip = [o for o in (target_objects or []) if not any(k in o for k in containers)]
        best, best_d = None, 1e9
        for o in manip:
            bid = obj_body_id.get(o)
            if bid is None:
                continue
            try:
                p = np.asarray(sim.data.body_xpos[int(bid)])
            except Exception:
                continue
            d = float(np.linalg.norm(p - grip)) if grip is not None else 0.0
            if d < best_d:
                best, best_d, manip_name, manip_bid = p, d, o, bid

    def _body_geoms(bid):
        if bid is None:
            return []
        return [gid for gid in range(int(sim.model.ngeom))
                if int(sim.model.geom_bodyid[gid]) == int(bid)]

    def _contact_for_pair(body_a, body_b=None, near=None):
        """Return the closest active contact point for body_a/body_b."""
        if body_a is None:
            return None
        best_c, best_d = None, 1e9
        for ci in range(int(getattr(sim.data, "ncon", 0))):
            c = sim.data.contact[ci]
            g1, g2 = int(c.geom1), int(c.geom2)
            b1, b2 = int(sim.model.geom_bodyid[g1]), int(sim.model.geom_bodyid[g2])
            hit = (b1 == int(body_a) and (body_b is None or b2 == int(body_b))) or \
                  (b2 == int(body_a) and (body_b is None or b1 == int(body_b)))
            if not hit:
                continue
            p = np.asarray(c.pos, dtype=float).copy()
            d = float(np.linalg.norm(p - near)) if near is not None else 0.0
            if d < best_d:
                best_c, best_d = p, d
        return best_c

    def _nearest_surface(bid, toward):
        """Approximate nearest point on the body's actual geom surface."""
        gids = _body_geoms(bid)
        if not gids:
            return None
        toward = np.asarray(toward, dtype=float)
        best = None
        best_d = 1e9
        for gid in gids:
            center = np.asarray(sim.data.geom_xpos[gid], dtype=float)
            v = toward - center
            norm = float(np.linalg.norm(v))
            if norm < 1e-8:
                continue
            # geom_rbound is conservative but gives a stable surface-directed point;
            # contact points are preferred whenever MuJoCo reports one.
            rb = float(sim.model.geom_rbound[gid])
            p = center + v / norm * max(0.0, 0.85 * rb)
            d = norm - rb
            if d < best_d:
                best, best_d = p, d
        return best

    if "object_contact" in parts and manip_bid is not None:
        # Prefer an actual gripper/object contact, otherwise nearest object surface
        # facing the grip site. Never use the object body centre as the mask point.
        p = _contact_for_pair(manip_bid, near=grip)
        if p is None and grip is not None:
            p = _nearest_surface(manip_bid, grip)
        if p is not None:
            pts["object_contact"] = p
    if "receptacle_contact" in parts:
        # receptacle = a container object named in interaction_receptacle_by_task, else
        # the target_objects that look like containers
        rname = None
        rbt = spec.get("interaction_receptacle_by_task", {})
        rname = rbt.get(f"{spec.get('_benchmark','')}:{spec.get('_task','')}") or rbt.get(str(spec.get('_task','')))
        if not rname:
            for o in (target_objects or []):
                if any(k in o for k in ("basket", "plate", "cabinet", "microwave", "stove", "caddy", "drawer")):
                    rname = o
                    break
        if rname is not None:
            obj_body_id = _body_map(base_env)
            bid = obj_body_id.get(rname)
            if bid is not None:
                try:
                    # Placement contact: use object/receptacle contact position when
                    # available. Before contact, use the receptacle surface nearest to
                    # the manipulated object, not the receptacle body centre.
                    p = _contact_for_pair(bid, manip_bid, near=best)
                    if p is None and manip_bid is not None:
                        p = _nearest_surface(bid, best)
                    if p is not None:
                        pts["receptacle_contact"] = p
                except Exception:
                    pass
    return pts


def _wsb_project_raw(base_env, camera_name, point_world, h, w):
    """Project a single world point to (row, col) with NO FOV-margin check — used as a
    size probe for the adaptive radius (returns coords even near/just outside frame)."""
    try:
        from robosuite.utils.camera_utils import (
            get_camera_transform_matrix, project_points_from_world_to_camera)
        T = get_camera_transform_matrix(base_env.sim, camera_name, h, w)
        px = project_points_from_world_to_camera(np.asarray(point_world, dtype=float).reshape(1, 3), T, h, w)[0]
        return (float(px[0]), float(px[1]))
    except Exception:
        try:
            sim = base_env.sim
            cid = int(sim.model.camera_name2id(camera_name))
            pos = np.asarray(sim.data.cam_xpos[cid], dtype=float)
            R = np.asarray(sim.data.cam_xmat[cid], dtype=float).reshape(3, 3)
            fovy = float(sim.model.cam_fovy[cid])
            fy = 0.5 * float(h) / np.tan(np.deg2rad(fovy) * 0.5)
            fx = fy * float(w) / float(h)
            q = R.T @ (np.asarray(point_world, dtype=float) - pos)
            if not np.isfinite(q).all() or q[2] >= -1e-6:
                return None
            return (float(0.5*h - fy*q[1]/(-q[2])),
                    float(0.5*w + fx*q[0]/(-q[2])))
        except Exception:
            return None


def _wsb_object_world_radius(base_env, obj_name):
    """Physical radius (metres) of an object from its geoms' bounding radii — no seg
    rendering needed. Returns max geom_rbound over the object's body, or None."""
    if not obj_name:
        return None

    sim = base_env.sim
    obj_body_id = getattr(base_env, "obj_body_id", None)
    if not obj_body_id:
        obj_body_id = getattr(getattr(base_env, "env", None), "obj_body_id", {})
    bid = obj_body_id.get(obj_name)
    if bid is None:
        return None
    try:
        rmax = 0.0
        for gid in range(sim.model.ngeom):
            if int(sim.model.geom_bodyid[gid]) == int(bid):
                rmax = max(rmax, float(sim.model.geom_rbound[gid]))
        return rmax if rmax > 0 else None
    except Exception:
        return None


def _wsb_render_3d_proxy_observation(base_env, real_obs, model_obs, spec,
                                     libero_benchmark, task_idx):
    """Render one shared, non-physical 3-D proxy in both cameras.

    Unlike a 2-D projected disk, the proxy is inserted into MuJoCo's render scene
    only (not the model/data), so it has no collision or dynamics effect.  Both
    camera images are rendered from the same world-space proxy poses; perspective,
    visibility and depth ordering are therefore handled by MuJoCo itself.
    """
    try:
        import mujoco
    except Exception as exc:
        raise RuntimeError("A3 3-D proxy requires the mujoco Python package") from exc
    sim = getattr(base_env, "sim", None)
    if sim is None:
        raise RuntimeError("A3 3-D proxy requires a MuJoCo-backed environment")
    targets = _resolve_occlusion_targets(base_env, spec, libero_benchmark, task_idx)
    spec2 = {**spec, "_benchmark": libero_benchmark, "_task": task_idx,
             "interaction_parts": ["gripper", "object_contact", "receptacle_contact"]}
    points = _wsb_interaction_points(base_env, spec2, targets)
    parts = [p for p in ("gripper", "object_contact", "receptacle_contact") if p in points]
    views = _normalize_occlusion_views(spec.get("views"))
    sizes = spec.get("proxy_size_m", {})
    default_size = float(spec.get("proxy_size_m_default", 0.018))
    rgba = np.asarray(spec.get("fill_rgb", [0, 0, 0]), dtype=np.float32) / 255.0
    rgba = np.concatenate([rgba[:3], np.asarray([1.0], dtype=np.float32)])
    out = {k: v.copy() for k, v in model_obs.items()}
    view_stats = {}
    # Build one contact-frame orientation per part.  The local z axis is the
    # surface normal (towards the opposing interaction element); the proxy is a
    # thin patch, rather than an axis-aligned cube around a body centre.
    def contact_frame(part):
        p = np.asarray(points[part], dtype=np.float64)
        if part == "gripper" and "object_contact" in points:
            n = np.asarray(points["object_contact"]) - p
        elif part == "object_contact" and "gripper" in points:
            n = np.asarray(points["gripper"]) - p
        elif part == "receptacle_contact" and "object_contact" in points:
            n = np.asarray(points["object_contact"]) - p
        else:
            n = np.asarray([0.0, 0.0, 1.0])
        n = n / max(float(np.linalg.norm(n)), 1e-8)
        ref = np.asarray([0.0, 0.0, 1.0]) if abs(n[2]) < 0.9 else np.asarray([1.0, 0.0, 0.0])
        x = np.cross(ref, n); x /= max(float(np.linalg.norm(x)), 1e-8)
        y = np.cross(n, x); y /= max(float(np.linalg.norm(y)), 1e-8)
        return np.column_stack([x, y, n]).reshape(-1), [float(v) for v in n]
    for view_name in views:
        cfg = MODEL_VIEW_CONFIGS[view_name]
        image = out[cfg["model_key"]]
        h, w = image.shape[:2]
        cam = "agentview" if view_name == "agentview" else "robot0_eye_in_hand"
        raw_model = getattr(sim.model, "_model", sim.model)
        raw_data = getattr(sim.data, "_data", sim.data)
        # Renderer creation allocates an EGL context and is very expensive. Keep
        # one renderer per camera for the whole episode; update_scene below resets
        # the scene before adding the current proxy geoms.
        cache = getattr(base_env, "_a3_proxy_renderers", None)
        if cache is None:
            cache = {}
            setattr(base_env, "_a3_proxy_renderers", cache)
        renderer = cache.get((cam, h, w))
        if renderer is None:
            renderer = mujoco.Renderer(raw_model, h, w)
            cache[(cam, h, w)] = renderer
        renderer.update_scene(raw_data, camera=cam)
        drawn = []
        for part in parts:
            if renderer.scene.ngeom >= renderer.scene.maxgeom:
                break
            geom = renderer.scene.geoms[renderer.scene.ngeom]
            size = float(sizes.get(part, default_size))
            thickness = float(spec.get("proxy_thickness_m", 0.003))
            mat, normal = contact_frame(part)
            mujoco.mjv_initGeom(
                geom, mujoco.mjtGeom.mjGEOM_BOX,
                np.asarray([size, size, thickness], dtype=np.float64),
                np.asarray(points[part], dtype=np.float64), mat, rgba)
            renderer.scene.ngeom += 1
            drawn.append({"part": part, "world_point_m": [float(x) for x in points[part]],
                          "proxy_size_m": size, "proxy_thickness_m": thickness,
                          "contact_normal_world": normal, "camera": cam})
        out[cfg["model_key"]] = np.asarray(renderer.render()).copy()
        # Renderer is intentionally retained in the episode cache.
        view_stats[view_name] = {"render": "mujoco_scene_3d_proxy", "parts_drawn": drawn,
                                 "proxy_depth_test": True}
    return out, {"enabled": True, "target_policy": "interaction_part",
                  "mask_source": "interaction_3d_proxy", "target_objects": targets,
                  "world_coordinate_source": "mujoco_interaction_points",
                  "cross_view_consistency": "same_world_proxy_rendered_per_camera",
                  "view_stats": view_stats}
def _wsb_interaction_part_objects(base_env, spec, target_objects):
    """Map each interaction part -> the object whose segmentation size sets its adaptive
    disk radius. gripper has no object seg (uses fixed radius, mapped to None)."""
    containers = ("basket", "plate", "cabinet", "microwave", "stove", "caddy", "drawer")
    manip = [o for o in (target_objects or []) if not any(k in o for k in containers)]
    recept = [o for o in (target_objects or []) if any(k in o for k in containers)]
    return {
        "gripper": None,                                  # no seg -> fixed small radius
        "object_contact": manip[0] if manip else None,    # scale to manipulated object
        "receptacle_contact": recept[0] if recept else None,  # scale to receptacle
    }


def _wsb_apply_imagined(model_observes, imagined_frames, kf_idx):
    """Replace the model-input views in `model_observes` with the model's own
    imagined frame at index kf_idx. imagined_frames is (F, H, W, C). The imagined image
    is resized to each view's shape. Env observations are untouched — only what the
    model *sees* on feedback is swapped for its imagination."""
    try:
        import cv2 as _cv2
    except Exception:
        return model_observes
    fr = np.asarray(imagined_frames)
    if fr.ndim != 4 or fr.shape[0] == 0:
        return model_observes
    idx = min(int(kf_idx), fr.shape[0] - 1)
    img = fr[idx]
    out = dict(model_observes)
    for _vcfg in MODEL_VIEW_CONFIGS.values():
        _mk = _vcfg["model_key"]
        if _mk in out:
            h, w = out[_mk].shape[:2]
            try:
                out[_mk] = _cv2.resize(img, (w, h)).astype(out[_mk].dtype)
            except Exception:
                pass
    return out


def _apply_obs_occlusion(real_obs, raw_obs, env_in, spec, libero_benchmark, task_idx):
    if not _obs_occlusion_enabled(spec):
        return real_obs, {"enabled": False}

    mask_source = spec.get("mask_source", "segmentation")

    fill_rgb = np.array(spec.get("fill_rgb", [0, 0, 0]), dtype=np.uint8)
    views = _normalize_occlusion_views(spec.get("views"))
    model_obs = {key: value.copy() for key, value in real_obs.items()}
    view_stats = {}

    if mask_source == "interaction_3d_proxy":
        return _wsb_render_3d_proxy_observation(
            getattr(env_in, "env", env_in), real_obs, model_obs, spec,
            libero_benchmark, task_idx
        )

    if mask_source in {"interaction_part", "interaction"}:
        # A3: precise SMALL-area occlusion of interaction parts only:
        #   gripper / object-gripper contact / object-receptacle contact.
        # Disk radius is ADAPTIVE: scaled to each part's object segmentation size, so
        # a small/far object gets a small disk and a large object a larger one
        # (固定像素半径对不同大小物体不通用). Falls back to a fixed radius if no seg.
        base_env = getattr(env_in, "env", env_in)
        r_fixed = int(spec.get("part_radius_px", 8))          # gripper / fallback
        r_frac = float(spec.get("part_radius_frac", 0.5))     # disk radius = frac * object equiv-radius
        r_min = int(spec.get("part_radius_min", 4))
        r_max = int(spec.get("part_radius_max", 24))
        target_objects = _resolve_occlusion_targets(env_in, spec, libero_benchmark, task_idx)
        spec = {**spec, "_benchmark": libero_benchmark, "_task": task_idx}
        pts = _wsb_interaction_points(base_env, spec, target_objects)
        # which object each part's radius scales to (gripper has no seg -> fixed)
        part_obj = _wsb_interaction_part_objects(base_env, spec, target_objects)
        for view_name in views:
            cfg = MODEL_VIEW_CONFIGS[view_name]
            image = model_obs[cfg["model_key"]]
            h, w = image.shape[:2]
            cam = "agentview" if view_name == "agentview" else "robot0_eye_in_hand"
            names = list(pts.keys())
            # project the interaction point AND a second point offset by the object's
            # physical radius, so the pixel distance between them = the object's size in
            # image px. This gives an adaptive radius WITHOUT segmentation rendering
            # (segmentation on two cams was triggering EGL failures).
            coords = _wsb_world_to_pixel(base_env, cam, [pts[k] for k in names], h, w)
            # Interaction points may legitimately be close to the image border (the
            # wrist camera often is).  The generic helper applies a safety margin for
            # segmentation targets; A3 instead accepts any finite in-frame projection.
            for _i, (_name, _p) in enumerate(zip(names, [pts[k] for k in names])):
                if coords[_i] is None:
                    _raw = _wsb_project_raw(base_env, cam, _p, h, w)
                    if _raw is not None and 0 <= _raw[0] < h and 0 <= _raw[1] < w:
                        coords[_i] = (int(round(_raw[0])), int(round(_raw[1])))
            drawn = []
            masked_px = 0
            for k, rc in zip(names, coords):
                if rc is None:
                    continue
                r, c = rc
                radius = r_fixed
                obj_name = part_obj.get(k)
                obj_r3d = _wsb_object_world_radius(base_env, obj_name) if obj_name else None
                if obj_r3d is not None:
                    # offset the point by obj_r3d in world x, project WITHOUT the FOV
                    # margin check (this is a size probe, not an occlusion target — an
                    # edge point may legitimately sit near the border).
                    edge = _wsb_project_raw(base_env, cam, pts[k] + np.array([obj_r3d, 0, 0]), h, w)
                    if edge is not None:
                        size_px = float(np.hypot(edge[0] - r, edge[1] - c))
                        if size_px > 0:
                            radius = int(round(size_px * r_frac))
                radius = max(r_min, min(r_max, radius))
                yy, xx = np.ogrid[:h, :w]
                disk = (yy - r) ** 2 + (xx - c) ** 2 <= radius ** 2
                image[disk] = fill_rgb
                masked_px += int(disk.sum())
                drawn.append({
                    "part": k,
                    "center": [int(r), int(c)],
                    "radius": int(radius),
                    # Exact simulator-space point used for both camera projections.
                    "world_point_m": [float(x) for x in np.asarray(pts[k], dtype=float)],
                    "camera": cam,
                })
            view_stats[view_name] = {
                "masked_pixels": masked_px,
                "mask_fraction": float(masked_px) / (h * w),
                "parts_drawn": drawn,
                "resolved_world_points": {
                    k: [float(x) for x in np.asarray(v, dtype=float)] for k, v in pts.items()
                },
                "projected_parts": [k for k, rc in zip(names, coords) if rc is not None],
                "missing_targets": [],
            }
        return model_obs, {
            "enabled": True,
            "target_policy": "interaction_part",
            "target_objects": target_objects,
            "mask_source": mask_source,
            "mask_geometry": "interaction_disk_adaptive",
            "part_radius_frac": r_frac,
            "world_coordinate_source": "mujoco_contact_or_nearest_geom_surface",
            "cross_view_consistency": "same_world_point_projected_per_camera",
            "views": views,
            "view_stats": view_stats,
        }

    if mask_source in {"center", "center_rect", "center_crop"}:
        requested_fraction = _resolve_center_mask_fraction(spec)
        # C1 needs the corrupted region filled with NOISE (sensor failure) as well as a
        # solid colour; fill_mode="noise" switches from flat fill to gaussian noise.
        _fill_mode = str(spec.get("fill_mode", "solid")).lower()
        _noise_std = float(spec.get("noise_std", 80.0))
        _rng = np.random.default_rng(int(spec.get("noise_seed", 0)) or None)
        for view_name in views:
            cfg = MODEL_VIEW_CONFIGS[view_name]
            image = model_obs[cfg["model_key"]]
            mask = _center_rect_mask(image.shape[:2], requested_fraction)
            if _fill_mode in {"noise", "gaussian", "gaussian_noise"}:
                n = int(mask.sum())
                noisy = _rng.normal(loc=127.0, scale=_noise_std, size=(n, image.shape[2]))
                image[mask] = np.clip(noisy, 0, 255).astype(image.dtype)
            else:
                image[mask] = fill_rgb
            view_stats[view_name] = {
                "masked_pixels": int(mask.sum()),
                "mask_fraction": float(mask.mean()),
                "fill_mode": _fill_mode,
                "missing_targets": [],
            }
        return model_obs, {
            "enabled": True,
            "target_policy": "center_rect",
            "target_objects": [],
            "mask_source": mask_source,
            "mask_geometry": "center_rect",
            "mask_fraction_requested": requested_fraction,
            "dilate_px": 0,
            "views": views,
            "view_stats": view_stats,
        }

    if mask_source in {"text_overlay", "visual_text", "fake_instruction_text"}:
        overlay_text = _resolve_text_overlay(spec, libero_benchmark, task_idx)
        for view_name in views:
            cfg = MODEL_VIEW_CONFIGS[view_name]
            image = model_obs[cfg["model_key"]]
            view_stats[view_name] = _apply_text_overlay(image, overlay_text, spec)
        return model_obs, {
            "enabled": True,
            "target_policy": "text_overlay",
            "target_objects": [],
            "mask_source": mask_source,
            "mask_geometry": "text_overlay",
            "dilate_px": 0,
            "views": views,
            "overlay_text": overlay_text,
            "view_stats": view_stats,
        }

    if mask_source in {"target_noise", "non_target_noise", "segmentation_noise"}:
        target_objects = _resolve_occlusion_targets(env_in, spec, libero_benchmark, task_idx)
        if not target_objects:
            raise ValueError(f"No target objects resolved for noise perturbation on {libero_benchmark}:{task_idx}")
        noise_region = str(spec.get("noise_region", "target")).lower()
        if mask_source == "target_noise":
            noise_region = "target"
        elif mask_source == "non_target_noise":
            noise_region = "non_target"

        for view_name in views:
            cfg = MODEL_VIEW_CONFIGS[view_name]
            if cfg["segmentation_key"] not in raw_obs:
                raise KeyError(
                    f"Raw observation is missing '{cfg['segmentation_key']}'. "
                    "Enable SegmentationRenderEnv for segmentation noise."
                )
            mask, missing_targets = _target_mask_from_segmentation(
                env_in, raw_obs[cfg["segmentation_key"]], target_objects
            )
            mask = np.ascontiguousarray(mask[::-1])
            mask = _expand_mask_geometry(mask, spec.get("mask_geometry", "segmentation"))
            mask = _dilate_mask(mask, spec.get("dilate_px", 0))
            if noise_region in {"non_target", "inverse", "background"}:
                mask = ~mask
            elif noise_region != "target":
                raise ValueError(f"Unsupported noise_region: {noise_region}")

            image = model_obs[cfg["model_key"]]
            noise_stats = _apply_noise_to_mask(image, mask, spec, libero_benchmark, task_idx, view_name)
            noise_stats["missing_targets"] = missing_targets
            view_stats[view_name] = noise_stats

        return model_obs, {
            "enabled": True,
            "target_policy": spec.get("target_policy", "obj_of_interest"),
            "target_objects": target_objects,
            "mask_source": mask_source,
            "mask_geometry": spec.get("mask_geometry", "segmentation"),
            "noise_region": noise_region,
            "noise_type": spec.get("noise_type", "gaussian"),
            "noise_std": float(spec.get("noise_std", 45.0)),
            "dilate_px": int(spec.get("dilate_px", 0) or 0),
            "views": views,
            "view_stats": view_stats,
        }

    if mask_source == "target_shape_warp":
        target_objects = _resolve_occlusion_targets(env_in, spec, libero_benchmark, task_idx)
        if not target_objects:
            raise ValueError(f"No target objects resolved for shape warp on {libero_benchmark}:{task_idx}")
        for view_name in views:
            cfg = MODEL_VIEW_CONFIGS[view_name]
            if cfg["segmentation_key"] not in raw_obs:
                raise KeyError(f"Raw observation is missing '{cfg['segmentation_key']}' for target shape warp")
            image = model_obs[cfg["model_key"]]
            warps = []
            missing_targets = []
            for object_name in target_objects:
                mask, missing = _target_mask_from_segmentation(
                    env_in, raw_obs[cfg["segmentation_key"]], [object_name]
                )
                missing_targets.extend(missing)
                mask = np.ascontiguousarray(mask[::-1])
                warp = _apply_target_shape_warp(image, mask, spec)
                warp["object"] = object_name
                warps.append(warp)
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
        target_objects = _resolve_occlusion_targets(env_in, spec, libero_benchmark, task_idx)
        if not target_objects:
            raise ValueError(f"No target objects resolved for photo replacement on {libero_benchmark}:{task_idx}")

        for view_name in views:
            cfg = MODEL_VIEW_CONFIGS[view_name]
            if cfg["segmentation_key"] not in raw_obs:
                raise KeyError(
                    f"Raw observation is missing '{cfg['segmentation_key']}'. "
                    "Enable SegmentationRenderEnv for target photo replacement."
                )
            image = model_obs[cfg["model_key"]]
            source_image = real_obs[cfg["model_key"]]
            replacements = []
            missing_targets = []
            changed_pixels = 0
            for object_name in target_objects:
                mask, missing = _target_mask_from_segmentation(
                    env_in, raw_obs[cfg["segmentation_key"]], [object_name]
                )
                missing_targets.extend(missing)
                mask = np.ascontiguousarray(mask[::-1])
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

    if mask_source in {"object_label_overlay", "wrong_category_label", "category_label_overlay"}:
        label_targets_by_view = {}
        all_target_objects = []
        for view_name in views:
            label_targets = _resolve_object_label_targets(env_in, spec, libero_benchmark, task_idx, view_name)
            label_targets_by_view[view_name] = label_targets
            all_target_objects.extend(item["object"] for item in label_targets)
            cfg = MODEL_VIEW_CONFIGS[view_name]
            if cfg["segmentation_key"] not in raw_obs:
                raise KeyError(
                    f"Raw observation is missing '{cfg['segmentation_key']}'. "
                    "Enable SegmentationRenderEnv for object label overlay."
                )
            image = model_obs[cfg["model_key"]]
            overlays = []
            missing_targets = []
            changed_pixels = 0
            for target in label_targets:
                object_name = target["object"]
                mask, missing = _target_mask_from_segmentation(
                    env_in, raw_obs[cfg["segmentation_key"]], [object_name]
                )
                missing_targets.extend(missing)
                mask = np.ascontiguousarray(mask[::-1])
                bbox = _bbox_from_mask(mask)
                if bbox is None:
                    if object_name not in missing_targets:
                        missing_targets.append(object_name)
                    continue
                overlay = _draw_object_category_label(image, bbox, target["label"], spec)
                overlay["object"] = object_name
                overlays.append(overlay)
                changed_pixels += int(overlay.get("changed_pixels", 0))
            view_stats[view_name] = {
                "masked_pixels": changed_pixels,
                "mask_fraction": changed_pixels / float(image.shape[0] * image.shape[1]),
                "missing_targets": missing_targets,
                "label_overlays": overlays,
            }
        return model_obs, {
            "enabled": True,
            "target_policy": "object_label_overlay",
            "target_objects": sorted(set(all_target_objects)),
            "label_targets": label_targets_by_view[views[0]] if len(views) == 1 else [],
            "label_targets_by_view": label_targets_by_view,
            "mask_source": mask_source,
            "mask_geometry": "object_label_overlay",
            "dilate_px": 0,
            "views": views,
            "view_stats": view_stats,
        }

    if mask_source != "segmentation":
        raise ValueError(f"Unsupported occlusion mask_source: {mask_source}")

    target_objects = _resolve_occlusion_targets(env_in, spec, libero_benchmark, task_idx)
    if not target_objects:
        raise ValueError(f"No target objects resolved for occlusion on {libero_benchmark}:{task_idx}")

    for view_name in views:
        cfg = MODEL_VIEW_CONFIGS[view_name]
        if cfg["segmentation_key"] not in raw_obs:
            raise KeyError(
                f"Raw observation is missing '{cfg['segmentation_key']}'. "
                "Enable SegmentationRenderEnv for input occlusion."
            )

        mask, missing_targets = _target_mask_from_segmentation(
            env_in, raw_obs[cfg["segmentation_key"]], target_objects
        )
        mask = np.ascontiguousarray(mask[::-1])
        mask = _expand_mask_geometry(mask, spec.get("mask_geometry", "segmentation"))
        _target_mask_px = int(mask.sum())  # object's full visible mask (denominator)
        _coverage = float(spec.get("coverage", 1.0))
        if _coverage < 1.0:
            # partial occlusion: crop coverage from the RAW mask, do NOT dilate
            # (dilation inflates the fraction on small/grasped objects)
            mask = _wsb_coverage_crop(mask, _coverage, spec.get("coverage_direction", "centroid"))
        else:
            mask = _dilate_mask(mask, spec.get("dilate_px", 0))

        image = model_obs[cfg["model_key"]]
        # fill_mode="inpaint" REMOVES the object (fills its region from the surrounding
        # background) instead of covering it with a colour patch. Occlusion says "something
        # is blocking my view"; inpaint says "there is nothing there" — the latter is what
        # C1's existence contradiction needs.
        if str(spec.get("fill_mode", "solid")).lower() in {"inpaint", "remove", "disappear"}:
            try:
                import cv2 as _cv2
                _m8 = (np.asarray(mask).astype(np.uint8) * 255)
                _rad = int(spec.get("inpaint_radius", 3))
                image[:] = _cv2.inpaint(np.ascontiguousarray(image), _m8, _rad, _cv2.INPAINT_TELEA)
            except Exception as _e:
                print(f"[wsb] inpaint failed, fell back to solid fill: {_e}")
                image[mask] = fill_rgb
        else:
            image[mask] = fill_rgb
        _masked_px = int(mask.sum())
        view_stats[view_name] = {
            "masked_pixels": _masked_px,
            "target_mask_pixels": _target_mask_px,
            "fill_mode": str(spec.get("fill_mode", "solid")),
            "actual_coverage": (float(_masked_px) / _target_mask_px) if _target_mask_px > 0 else 0.0,
            "requested_coverage": _coverage,
            "mask_fraction": float(mask.mean()),
            "missing_targets": missing_targets,
        }

    return model_obs, {
        "enabled": True,
        "target_policy": spec.get("target_policy", "obj_of_interest"),
        "target_objects": target_objects,
        "mask_source": spec.get("mask_source", "segmentation"),
        "mask_geometry": spec.get("mask_geometry", "segmentation"),
        "dilate_px": int(spec.get("dilate_px", 0) or 0),
        "views": views,
        "view_stats": view_stats,
    }


def _summarize_occlusion_stats(stats_list):
    enabled_stats = [stats for stats in stats_list if stats.get("enabled")]
    if not enabled_stats:
        return {"enabled": False}

    view_summary = {}
    for stats in enabled_stats:
        for view_name, view_stats in stats.get("view_stats", {}).items():
            summary = view_summary.setdefault(
                view_name,
                {
                    "frames": 0,
                    "frames_with_mask": 0,
                    "masked_pixels_total": 0,
                    "mask_fraction_sum": 0.0,
                    "missing_targets": [],
                },
            )
            summary["frames"] += 1
            masked_pixels = int(view_stats.get("masked_pixels", 0))
            summary["masked_pixels_total"] += masked_pixels
            summary["mask_fraction_sum"] += float(view_stats.get("mask_fraction", 0.0))
            if masked_pixels > 0:
                summary["frames_with_mask"] += 1
            for target in view_stats.get("missing_targets", []):
                if target not in summary["missing_targets"]:
                    summary["missing_targets"].append(target)
            if view_stats.get("label_overlays") and "first_label_overlays" not in summary:
                summary["first_label_overlays"] = view_stats["label_overlays"]
            if view_stats.get("photo_replacements") and "first_photo_replacements" not in summary:
                summary["first_photo_replacements"] = view_stats["photo_replacements"]

    for summary in view_summary.values():
        frames = max(summary["frames"], 1)
        summary["mask_fraction_mean"] = summary["mask_fraction_sum"] / frames
        del summary["mask_fraction_sum"]

    first = enabled_stats[0]
    return {
        "enabled": True,
        "frames": len(enabled_stats),
        "target_policy": first.get("target_policy"),
        "target_objects": first.get("target_objects", []),
        "mask_source": first.get("mask_source"),
        "mask_geometry": first.get("mask_geometry"),
        "mask_fraction_requested": first.get("mask_fraction_requested"),
        "overlay_text": first.get("overlay_text"),
        "label_targets": first.get("label_targets", []),
        "photo_source": first.get("photo_source"),
        "photo_border_px": first.get("photo_border_px"),
        "photo_padding_px": first.get("photo_padding_px"),
        "noise_region": first.get("noise_region"),
        "noise_type": first.get("noise_type"),
        "noise_std": first.get("noise_std"),
        "swapped_views": first.get("swapped_views", []),
        "source_benchmark": first.get("source_benchmark"),
        "source_task_idx": first.get("source_task_idx"),
        "donor_benchmark": first.get("donor_benchmark"),
        "donor_task_idx": first.get("donor_task_idx"),
        "donor_task_offset": first.get("donor_task_offset"),
        "swap_mode": first.get("swap_mode"),
        "dilate_px": first.get("dilate_px", 0),
        "by_view": view_summary,
    }


def _get_joint_qpos(sim, joint_name):
    try:
        return np.array(sim.data.get_joint_qpos(joint_name), dtype=float)
    except AttributeError:
        qpos_addr = sim.model.get_joint_qpos_addr(joint_name)
        return np.array(sim.data.qpos[qpos_addr], dtype=float)


def _set_joint_qpos(sim, joint_name, qpos):
    sim.data.set_joint_qpos(joint_name, np.array(qpos, dtype=float))


def _free_joint_name(base_env, object_name):
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


def _offset_object_instance(base_env, object_name, pos_offset):
    free_joint = _free_joint_name(base_env, object_name)
    if free_joint is None:
        raise ValueError(f"LIBERO object '{object_name}' has no movable free joint for pos_offset")
    current_qpos = _get_joint_qpos(base_env.sim, free_joint)
    current_qpos[:3] = current_qpos[:3] + np.array(pos_offset, dtype=float)
    _set_joint_qpos(base_env.sim, free_joint, current_qpos)


def _forward_sim(sim):
    try:
        sim.forward()
    except AttributeError:
        pass


def _descendant_body_ids(sim, root_body_id):
    body_ids = [int(root_body_id)]
    idx = 0
    while idx < len(body_ids):
        current = body_ids[idx]
        for body_id in range(sim.model.nbody):
            if body_id != current and int(sim.model.body_parentid[body_id]) == current:
                body_ids.append(body_id)
        idx += 1
    return body_ids


def _scale_mesh_vertices(sim, mesh_id, scale):
    vert_adr = int(sim.model.mesh_vertadr[mesh_id])
    vert_num = int(sim.model.mesh_vertnum[mesh_id])
    sim.model.mesh_vert[vert_adr: vert_adr + vert_num] *= float(scale)


def _object_lower_bound_z(sim, body_ids):
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


def _object_bottom_reference_offset_z(base_env, object_name):
    if object_name not in base_env.obj_body_id:
        raise ValueError(f"LIBERO object '{object_name}' was not found in obj_body_id")
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
            site_name = _mujoco_name(sim.model, "site", site_id)
            if "bottom_site" in site_name:
                bottom_site_offsets.append(float(sim.data.site_xpos[site_id][2]) - root_z)
    if bottom_site_offsets:
        return min(bottom_site_offsets)

    lower_bound = _object_lower_bound_z(sim, body_ids)
    if lower_bound is None:
        return None
    return float(lower_bound - root_z)


def _object_bottom_world_z(base_env, object_name):
    offset = _object_bottom_reference_offset_z(base_env, object_name)
    if offset is None:
        return None
    _forward_sim(base_env.sim)
    root_body_id = int(base_env.obj_body_id[object_name])
    return float(base_env.sim.data.body_xpos[root_body_id][2]) + float(offset)


def _object_named_bottom_site_world_z(base_env, object_name):
    sim = base_env.sim
    _forward_sim(sim)
    matches = []
    for site_id in range(int(sim.model.nsite)):
        site_name = _mujoco_name(sim.model, "site", site_id)
        if (
            "bottom_site" in site_name
            and str(object_name) in site_name
        ):
            matches.append(float(sim.data.site_xpos[site_id][2]))
    return min(matches) if matches else _object_bottom_world_z(
        base_env, object_name
    )


def _object_collision_lower_world_z(base_env, object_name):
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


def _object_collision_aabb_bounds(base_env, object_name):
    """Return world-space collision AABB bounds for an object or fixture."""
    sim = base_env.sim
    if object_name not in getattr(base_env, "obj_body_id", {}):
        raise ValueError(
            f"LIBERO object or fixture {object_name!r} has no body id"
        )
    root_body_id = int(base_env.obj_body_id[object_name])
    body_ids = set(_descendant_body_ids(sim, root_body_id))
    lows = []
    highs = []
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


def _keep_precompiled_scaled_object_bottom(base_env, object_name, scale):
    if scale <= 0:
        raise ValueError(f"LIBERO scale override for '{object_name}' must be positive, got {scale}")
    bottom_offset_z = _object_bottom_reference_offset_z(base_env, object_name)
    if bottom_offset_z is None:
        return
    z_delta = (1.0 / float(scale) - 1.0) * float(bottom_offset_z)
    _offset_object_instance(base_env, object_name, [0.0, 0.0, z_delta])


def _scale_object_instance(base_env, object_name, scale, pos_offset=None, mass_scale=None, keep_bottom=False):
    if object_name not in base_env.obj_body_id:
        raise ValueError(f"LIBERO object '{object_name}' was not found in obj_body_id")
    if scale <= 0:
        raise ValueError(f"LIBERO scale override for '{object_name}' must be positive, got {scale}")

    sim = base_env.sim
    root_body_id = int(base_env.obj_body_id[object_name])
    body_ids = set(_descendant_body_ids(sim, root_body_id))
    scaled_mesh_ids = set()
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
        mass_scale = float(mass_scale)
        for body_id in body_ids:
            sim.model.body_mass[body_id] *= mass_scale
            if hasattr(sim.model, "body_inertia"):
                sim.model.body_inertia[body_id] *= mass_scale

    if keep_bottom and lower_before is not None:
        lower_after = _object_lower_bound_z(sim, body_ids)
        if lower_after is not None:
            _offset_object_instance(base_env, object_name, [0.0, 0.0, float(lower_before - lower_after)])

    if pos_offset is not None:
        _offset_object_instance(base_env, object_name, pos_offset)


def _object_free_joint(base_env, object_name):
    obj = base_env.get_object(object_name)
    if obj is None or not getattr(obj, "joints", None):
        raise ValueError(f"LIBERO object '{object_name}' has no movable free joint")
    return obj.joints[-1]


def _swap_object_poses(base_env, object_a, object_b, pos_only=False, offset_a=None, offset_b=None):
    sim = base_env.sim
    joint_a = _object_free_joint(base_env, object_a)
    joint_b = _object_free_joint(base_env, object_b)
    qpos_a = _get_joint_qpos(sim, joint_a)
    qpos_b = _get_joint_qpos(sim, joint_b)
    next_a = qpos_a.copy()
    next_b = qpos_b.copy()
    next_a[:3] = qpos_b[:3] + np.array(offset_a or [0.0, 0.0, 0.0], dtype=float)
    next_b[:3] = qpos_a[:3] + np.array(offset_b or [0.0, 0.0, 0.0], dtype=float)
    if not pos_only:
        next_a[3:] = qpos_b[3:]
        next_b[3:] = qpos_a[3:]
    _set_joint_qpos(sim, joint_a, next_a)
    _set_joint_qpos(sim, joint_b, next_b)


def _set_object_visual_color(base_env, object_name, rgba=None, palette=None, mesh_only=True, update_material=True):
    if object_name not in base_env.obj_body_id:
        raise ValueError(f"LIBERO object '{object_name}' was not found in obj_body_id")
    sim = base_env.sim
    root_body_id = int(base_env.obj_body_id[object_name])
    body_ids = set(_descendant_body_ids(sim, root_body_id))
    colors = palette or ([rgba] if rgba is not None else None)
    if not colors:
        raise ValueError(f"Color override for '{object_name}' needs rgba or palette")
    colors = [np.array(color, dtype=float) for color in colors]
    for color in colors:
        if color.shape != (4,):
            raise ValueError(f"Color override for '{object_name}' must be RGBA with 4 values, got {color}")

    color_index = 0
    touched_materials = set()
    for geom_id in range(sim.model.ngeom):
        if int(sim.model.geom_bodyid[geom_id]) not in body_ids:
            continue
        mesh_id = int(sim.model.geom_dataid[geom_id])
        if mesh_only and mesh_id < 0:
            continue
        color = colors[color_index % len(colors)]
        color_index += 1
        sim.model.geom_rgba[geom_id] = color
        mat_id = int(sim.model.geom_matid[geom_id])
        if update_material and mat_id >= 0 and mat_id not in touched_materials:
            sim.model.mat_rgba[mat_id] = color
            touched_materials.add(mat_id)


def _mujoco_name(model, kind, idx):
    if idx is None or int(idx) < 0:
        return ""
    try:
        return model.id2name(int(idx), kind) or ""
    except Exception:
        pass
    method = getattr(model, f"{kind}_id2name", None)
    if callable(method):
        try:
            return method(int(idx)) or ""
        except Exception:
            pass
    names = getattr(model, f"{kind}_names", None)
    if names is not None and int(idx) < len(names):
        return names[int(idx)] or ""
    return ""


def _match_name(name, patterns):
    if not patterns:
        return False
    name = str(name or "")
    lower_name = name.lower()
    for pattern in patterns:
        pattern = str(pattern or "")
        if not pattern:
            continue
        if pattern.startswith("re:"):
            if re.search(pattern[3:], name):
                return True
        elif pattern.lower() in lower_name:
            return True
    return False


def _set_visual_color_by_names(base_env, override):
    sim = base_env.sim
    rgba = override.get("rgba")
    if rgba is None:
        raise ValueError("geom/body color override needs rgba")
    color = np.array(rgba, dtype=float)
    if color.shape != (4,):
        raise ValueError(f"geom/body color override must be RGBA with 4 values, got {color}")

    geom_patterns = override.get("geom_name_contains") or override.get("geom_patterns") or []
    body_patterns = override.get("body_name_contains") or override.get("body_patterns") or []
    material_patterns = override.get("material_name_contains") or override.get("material_patterns") or []
    mesh_only = _as_bool(override.get("mesh_only", False))
    include_descendants = _as_bool(override.get("include_descendants", True))

    matched_body_ids = set()
    for body_id in range(sim.model.nbody):
        body_name = _mujoco_name(sim.model, "body", body_id)
        if _match_name(body_name, body_patterns):
            if include_descendants:
                matched_body_ids.update(_descendant_body_ids(sim, body_id))
            else:
                matched_body_ids.add(int(body_id))

    touched_geoms = []
    touched_materials = set()
    for geom_id in range(sim.model.ngeom):
        if mesh_only and int(sim.model.geom_dataid[geom_id]) < 0:
            continue
        geom_name = _mujoco_name(sim.model, "geom", geom_id)
        body_id = int(sim.model.geom_bodyid[geom_id])
        body_name = _mujoco_name(sim.model, "body", body_id)
        mat_id = int(sim.model.geom_matid[geom_id])
        mat_name = _mujoco_name(sim.model, "material", mat_id) if mat_id >= 0 else ""
        if not (
            _match_name(geom_name, geom_patterns)
            or body_id in matched_body_ids
            or _match_name(body_name, body_patterns)
            or _match_name(mat_name, material_patterns)
        ):
            continue
        sim.model.geom_rgba[geom_id] = color
        if mat_id >= 0 and mat_id not in touched_materials:
            sim.model.mat_rgba[mat_id] = color
            touched_materials.add(mat_id)
        touched_geoms.append(
            {
                "geom_id": int(geom_id),
                "geom_name": geom_name,
                "body_id": int(body_id),
                "body_name": body_name,
                "material_id": int(mat_id),
                "material_name": mat_name,
            }
        )

    if _as_bool(override.get("require_match", True)) and not touched_geoms:
        raise ValueError(
            "No MuJoCo geoms matched visual color override "
            f"geom_patterns={geom_patterns}, body_patterns={body_patterns}, material_patterns={material_patterns}"
        )
    return touched_geoms


def _task_matches_libero_env_edit(edit, libero_benchmark, task_idx):
    edit_benchmark = edit.get("benchmark", edit.get("suite", ""))
    if edit_benchmark and str(edit_benchmark) != str(libero_benchmark):
        return False
    if "task_id" in edit and int(edit["task_id"]) != int(task_idx):
        return False
    return True


def _resolve_libero_env_edit_spec(spec, libero_benchmark="", task_idx=None):
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
    for edit in task_edits:
        if task_idx is not None and _task_matches_libero_env_edit(edit, libero_benchmark, task_idx):
            merged = copy.deepcopy(base)
            for key, value in edit.items():
                if key in {
                    "object_pose_swaps",
                    "object_scale_overrides",
                    "object_pose_overrides",
                    "object_color_overrides",
                    "object_surface_black_fraction_overrides",
                    "geom_color_overrides",
                    "body_color_overrides",
                    "body_mass_overrides",
                }:
                    merged[key] = copy.deepcopy(base.get(key, [])) + copy.deepcopy(value)
                else:
                    merged[key] = copy.deepcopy(value)
            merged["_matched_task_env_edit"] = {
                "benchmark": libero_benchmark,
                "task_id": int(task_idx),
            }
            return merged

    if _as_bool(spec.get("require_task_match", True)):
        raise ValueError(
            f"No per-task LIBERO env edit matched benchmark={libero_benchmark!r}, task_id={task_idx!r}"
        )
    return base


def _is_precompile_xml_scale_override(override):
    mode = str(override.get("scale_mode") or override.get("apply_stage") or "").strip().lower()
    return mode in PRECOMPILE_XML_SCALE_MODES or bool(override.get("xml_path"))


def _libero_object_xml_override_map(spec):
    mapping = {}
    for override in (spec or {}).get("object_scale_overrides", []):
        if not _is_precompile_xml_scale_override(override):
            continue
        object_name = str(override["object"])
        xml_path = override.get("xml_path")
        if not xml_path:
            raise ValueError(f"Precompile XML scale override for '{object_name}' needs xml_path")
        mapping[object_name] = {
            "xml_path": str(xml_path),
            "scale": float(override["scale"]),
            "category": override.get("category"),
        }
    return mapping


def _resolve_existing_path(path_value, *, base_dir=None):
    path = Path(os.path.expandvars(os.path.expanduser(str(path_value))))
    if not path.is_absolute() and base_dir is not None:
        path = Path(base_dir) / path
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(f"LIBERO path does not exist: {path}")
    return str(path)


def _libero_bddl_file_name(spec, default_bddl_file_name):
    if not spec:
        return str(default_bddl_file_name)
    override = (
        spec.get("bddl_file_name_override")
        or spec.get("bddl_file_override")
        or spec.get("bddl_override")
    )
    if not override:
        return str(default_bddl_file_name)
    return _resolve_existing_path(override)


def _libero_base_bddl_file_name(spec, default_bddl_file_name):
    if not spec:
        return ""
    base_bddl = (
        spec.get("base_bddl_file_name")
        or spec.get("base_bddl_file")
        or spec.get("init_state_bddl_file_name")
    )
    if base_bddl:
        return _resolve_existing_path(base_bddl)
    override_bddl = _libero_bddl_file_name(spec, default_bddl_file_name)
    if str(Path(override_bddl).resolve()) != str(Path(default_bddl_file_name).resolve()):
        return str(Path(default_bddl_file_name).resolve())
    return ""


def _addr_slice(addr):
    if isinstance(addr, (tuple, list)):
        return slice(int(addr[0]), int(addr[1]))
    return slice(int(addr), int(addr) + 1)


def _model_joint_state_layout(model):
    qpos_by_joint = {}
    qvel_by_joint = {}
    for joint_name in list(model.joint_names):
        qpos_by_joint[joint_name] = _addr_slice(model.get_joint_qpos_addr(joint_name))
        qvel_by_joint[joint_name] = _addr_slice(model.get_joint_qvel_addr(joint_name))
    return {
        "nq": int(model.nq),
        "nv": int(model.nv),
        "na": int(getattr(model, "na", 0)),
        "qpos_by_joint": qpos_by_joint,
        "qvel_by_joint": qvel_by_joint,
    }


def _state_layout_for_bddl(bddl_file_name):
    bddl_file_name = str(Path(bddl_file_name).resolve())
    cached = LIBERO_STATE_LAYOUT_CACHE.get(bddl_file_name)
    if cached is not None:
        return cached

    # This is the *base* BDDL layout used to remap an official init state into
    # an edited model.  Do not let the active object-XML override leak into the
    # base model, especially when the edit adds joints.
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


def _set_init_state_by_joint_names(env_in, init_state, base_bddl_file_name):
    init_state = np.asarray(init_state, dtype=float)
    current_state = np.asarray(env_in.get_sim_state(), dtype=float)
    if init_state.shape == current_state.shape:
        return env_in.set_init_state(init_state)

    base_layout = _state_layout_for_bddl(base_bddl_file_name)
    current_layout = _model_joint_state_layout(getattr(env_in, "env", env_in).sim.model)
    expected_old_len = 1 + base_layout["nq"] + base_layout["nv"] + base_layout["na"]
    if len(init_state) != expected_old_len:
        raise ValueError(
            f"Old LIBERO init_state length {len(init_state)} does not match base model length {expected_old_len} "
            f"for {base_bddl_file_name}"
        )

    old_qpos = init_state[1 : 1 + base_layout["nq"]]
    old_qvel = init_state[1 + base_layout["nq"] : 1 + base_layout["nq"] + base_layout["nv"]]
    sim = getattr(env_in, "env", env_in).sim
    sim.data.time = float(init_state[0])

    copied_joints = 0
    for joint_name, old_slice in base_layout["qpos_by_joint"].items():
        new_slice = current_layout["qpos_by_joint"].get(joint_name)
        if new_slice is None:
            continue
        sim.data.qpos[new_slice] = old_qpos[old_slice]
        copied_joints += 1
    for joint_name, old_slice in base_layout["qvel_by_joint"].items():
        new_slice = current_layout["qvel_by_joint"].get(joint_name)
        if new_slice is None:
            continue
        sim.data.qvel[new_slice] = old_qvel[old_slice]

    if copied_joints == 0:
        raise ValueError(f"No overlapping joints found while remapping LIBERO init state from {base_bddl_file_name}")

    sim.forward()
    env_in.check_success()
    env_in._post_process()
    env_in._update_observables(force=True)
    return env_in.env._get_observations()


def _object_body_position(base_env, object_name):
    sim = base_env.sim
    if object_name in getattr(base_env, "obj_body_id", {}):
        return np.array(sim.data.body_xpos[base_env.obj_body_id[object_name]], dtype=float)
    site_aliases = {
        "robot0_eef": ["gripper0_grip_site", "robot0_grip_site"],
        "eef": ["gripper0_grip_site", "robot0_grip_site"],
    }
    for site_name in site_aliases.get(object_name, [object_name]):
        try:
            return np.array(sim.data.site_xpos[sim.model.site_name2id(site_name)], dtype=float)
        except Exception:
            pass
    for body_name in (object_name, f"{object_name}_main"):
        try:
            return np.array(sim.data.body_xpos[sim.model.body_name2id(body_name)], dtype=float)
        except Exception:
            pass
    raise ValueError(f"Could not resolve LIBERO pose anchor {object_name!r} as an object, site, or body")


def _select_relative_pose_position(base_env, object_name, relative_to, override):
    sim = base_env.sim
    _forward_sim(sim)
    base_pos = _object_body_position(base_env, relative_to)
    candidate_offsets = override.get("candidate_offsets") or []
    if not candidate_offsets:
        return base_pos + np.array(override.get("offset", [0.0, 0.0, 0.0]), dtype=float)

    avoid_names = [
        str(name)
        for name in override.get("avoid_objects", [])
        if str(name) not in {str(object_name), str(relative_to)}
    ]
    avoid_positions = []
    for name in avoid_names:
        try:
            avoid_positions.append(_object_body_position(base_env, name))
        except ValueError:
            continue
    workspace_bounds = override.get("workspace_bounds_xy")
    min_distance = float(override.get("min_distance_from_others", 0.0))
    require_min_distance = _as_bool(override.get("require_min_distance", False))
    candidate_selection = str(
        override.get("candidate_selection", "max_clearance")
    ).strip().lower()

    best_pos = None
    best_score = -np.inf
    for offset in candidate_offsets:
        pos = base_pos + np.array(offset, dtype=float)
        if workspace_bounds:
            x_min, x_max = workspace_bounds[0]
            y_min, y_max = workspace_bounds[1]
            if not (float(x_min) <= float(pos[0]) <= float(x_max) and float(y_min) <= float(pos[1]) <= float(y_max)):
                continue

        nearest = min(
            (float(np.linalg.norm(pos[:2] - other_pos[:2])) for other_pos in avoid_positions),
            default=1.0,
        )
        if require_min_distance and nearest < min_distance:
            continue
        if candidate_selection == "first_clear":
            return pos
        penalty = max(0.0, min_distance - nearest) * 10.0
        score = nearest - penalty
        if score > best_score:
            best_score = score
            best_pos = pos

    if best_pos is None:
        if require_min_distance:
            raise ValueError(
                f"No collision-clear candidate for LIBERO object {object_name!r} "
                f"relative to {relative_to!r} at minimum XY clearance {min_distance:.3f} m"
            )
        best_pos = base_pos + np.array(candidate_offsets[0], dtype=float)
    return best_pos


def _zero_joint_velocity(sim, joint_name):
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


def _body_chain_names(sim, body_id):
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
    base_env,
    object_name,
    free_joint,
    base_pos,
    override,
    quat,
    current_z,
):
    """Try final MuJoCo poses and select the first non-penetrating candidate."""
    sim = base_env.sim
    candidate_offsets = override.get("candidate_offsets") or []
    if not candidate_offsets:
        return np.asarray(base_pos, dtype=float)
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

    def restore_snapshot():
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
                _object_collision_lower_world_z(base_env, str(object_name))
                if _as_bool(override.get("align_collision_bottom", False))
                else _object_named_bottom_site_world_z(base_env, str(object_name))
            )
            if lower is None:
                raise ValueError(
                    f"Could not resolve support bottom for LIBERO object "
                    f"'{object_name}'"
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
            f"No in-bounds MuJoCo candidate for LIBERO object '{object_name}'"
        )
    return fallback


def _set_libero_object_or_fixture_position(base_env, object_name, position):
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


def _select_between_pose_position(
    base_env, object_name, anchors, override, current_z
):
    anchor_pos = [
        _object_body_position(base_env, anchor) for anchor in anchors
    ]
    fractions = override.get("between_fractions") or [
        float(override.get("fraction", 0.5))
    ]
    avoid_positions = []
    for name in override.get("avoid_objects", []):
        if str(name) == str(object_name):
            continue
        try:
            avoid_positions.append(_object_body_position(base_env, str(name)))
        except ValueError:
            continue
    workspace_bounds = override.get("workspace_bounds_xy")
    min_distance = float(override.get("min_distance_from_others", 0.0))
    require_min_distance = _as_bool(
        override.get("require_min_distance", False)
    )
    for fraction in fractions:
        fraction = float(fraction)
        pos = (
            anchor_pos[0]
            + fraction * (anchor_pos[1] - anchor_pos[0])
            + np.array(override.get("offset", [0.0, 0.0, 0.0]), dtype=float)
        )
        if _as_bool(override.get("preserve_z", False)):
            pos[2] = current_z
        if workspace_bounds:
            x_min, x_max = workspace_bounds[0]
            y_min, y_max = workspace_bounds[1]
            if not (
                float(x_min) <= float(pos[0]) <= float(x_max)
                and float(y_min) <= float(pos[1]) <= float(y_max)
            ):
                continue
        nearest = min(
            (
                float(np.linalg.norm(pos[:2] - other[:2]))
                for other in avoid_positions
            ),
            default=1.0,
        )
        if nearest >= min_distance:
            return pos
    if require_min_distance:
        raise ValueError(
            f"No collision-clear on-path candidate for LIBERO object "
            f"{object_name!r} between {anchors!r}"
        )
    fraction = float(fractions[0])
    pos = anchor_pos[0] + fraction * (anchor_pos[1] - anchor_pos[0])
    pos += np.array(override.get("offset", [0.0, 0.0, 0.0]), dtype=float)
    if _as_bool(override.get("preserve_z", False)):
        pos[2] = current_z
    return pos


def _apply_same_position_replacement(base_env, rule):
    sim = base_env.sim
    target = str(rule["target"])
    replacement = str(rule["replacement"])
    original_target_pos = _object_body_position(base_env, target).copy()
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
    aside_pos = _select_relative_pose_position(
        base_env, target, target, aside_rule
    )

    replacement_joint = _free_joint_name(base_env, replacement)
    replacement_fixed = (
        replacement_joint is None
        and _as_bool(rule.get("replacement_fixed_fixture", False))
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
                f"'{replacement}'"
            )
        replacement_pos[2] += (
            float(rule["replacement_support_world_z"])
            - float(replacement_bottom)
        )
    elif rule.get("replacement_support_z_offset") is not None:
        replacement_pos[2] += float(rule["replacement_support_z_offset"])
    elif (
        _as_bool(rule.get("replacement_support_surface_from_target", False))
        and target_bottom_world is not None
    ):
        replacement_bottom_world = _object_bottom_world_z(base_env, replacement)
        if replacement_bottom_world is not None:
            replacement_pos[2] += (
                float(target_bottom_world) - float(replacement_bottom_world)
            )
    elif not _as_bool(rule.get("replacement_preserve_support_z", True)):
        replacement_pos[2] = original_target_pos[2]
    if target_qpos is not None:
        aside_pos[2] = float(target_qpos[2])
    elif _as_bool(rule.get("target_support_z_from_replacement", False)):
        aside_pos[2] = replacement_support_z

    _set_libero_object_or_fixture_position(base_env, target, aside_pos)
    if replacement_joint is not None:
        _set_joint_qpos(
            sim,
            replacement_joint,
            np.concatenate([replacement_pos, replacement_qpos[3:]]),
        )
    else:
        _set_libero_object_or_fixture_position(
            base_env, replacement, replacement_pos
        )
    _forward_sim(sim)
    audit = getattr(base_env, "_a4_same_position_audit", {})
    audit[target] = {
        "original_target_xyz": original_target_pos.tolist(),
        "aside_target_xyz": np.asarray(aside_pos, dtype=float).tolist(),
        "replacement": replacement,
        "replacement_xyz": np.asarray(replacement_pos, dtype=float).tolist(),
    }
    base_env._a4_same_position_audit = audit


def apply_libero_env_edit(env_in, spec, libero_benchmark="", task_idx=None):
    spec = _resolve_libero_env_edit_spec(spec, libero_benchmark, task_idx)
    if not spec or spec.get("status") != "implemented":
        return

    base_env = getattr(env_in, "env", env_in)
    sim = base_env.sim
    # Apply each configured articulation joint position (drawer/door/knob).
    # These are named articulated joints, not object free joints.
    for override in spec.get("articulation_qpos_overrides", []):
        joint_name = override["joint"]
        qpos_val = float(override["qpos"])
        _set_joint_qpos(sim, joint_name, np.array([qpos_val], dtype=float))
    for swap in spec.get("object_pose_swaps", []):
        _swap_object_poses(
            base_env,
            swap.get("object_a") or swap.get("a"),
            swap.get("object_b") or swap.get("b"),
            pos_only=_as_bool(swap.get("pos_only", False)),
            offset_a=swap.get("offset_a"),
            offset_b=swap.get("offset_b"),
        )

    for override in spec.get("object_scale_overrides", []):
        pos_offset = override.get("pos_offset")
        if pos_offset is None and "z_offset" in override:
            pos_offset = [0.0, 0.0, float(override["z_offset"])]
        if _is_precompile_xml_scale_override(override):
            if _as_bool(override.get("keep_bottom", False)):
                _keep_precompiled_scaled_object_bottom(
                    base_env,
                    override["object"],
                    float(override["scale"]),
                )
            if pos_offset is not None:
                _offset_object_instance(base_env, override["object"], pos_offset)
            continue
        _scale_object_instance(
            base_env,
            override["object"],
            float(override["scale"]),
            pos_offset=pos_offset,
            mass_scale=override.get("mass_scale"),
            keep_bottom=_as_bool(override.get("keep_bottom", False)),
        )

    for rule in spec.get("same_position_replacements", []):
        _apply_same_position_replacement(base_env, rule)

    for override in spec.get("object_pose_overrides", []):
        object_name = override["object"]
        is_fixed_fixture = _as_bool(override.get("fixed_fixture", False))
        free_joint = None if is_fixed_fixture else _free_joint_name(base_env, object_name)
        is_movable = free_joint is not None
        if not is_movable and not is_fixed_fixture:
            raise ValueError(f"LIBERO object '{object_name}' has no movable free joint")

        if is_movable:
            current_qpos = _get_joint_qpos(sim, free_joint)
            current_z = current_qpos[2]
            current_quat = current_qpos[3:]
        else:
            if object_name not in base_env.obj_body_id:
                raise ValueError(f"LIBERO fixed fixture '{object_name}' has no body id")
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
            anchors = override["between"]
            if len(anchors) != 2:
                raise ValueError(f"LIBERO pose override for '{object_name}' needs exactly two between anchors")
            pos = _select_between_pose_position(
                base_env,
                object_name,
                anchors,
                override,
                current_z,
            )
            fraction = float(override.get("fraction", 0.5))
            start = _object_body_position(base_env, anchors[0])
            end = _object_body_position(base_env, anchors[1])
            candidate_base_pos = start + fraction * (end - start)
        else:
            relative_to = override.get("relative_to")
            if not relative_to:
                raise ValueError(f"LIBERO pose override for '{object_name}' needs pos, between, or relative_to")
            candidate_base_pos = _object_body_position(base_env, relative_to)
            pos = _select_relative_pose_position(base_env, object_name, relative_to, override)
        if override.get("z_from"):
            pos[2] = _object_body_position(base_env, str(override["z_from"]))[2]
        if override.get("support_world_z") is not None:
            object_bottom = _object_named_bottom_site_world_z(
                base_env, str(object_name)
            )
            if object_bottom is None:
                raise ValueError(
                    f"Could not resolve bottom site for LIBERO object "
                    f"'{object_name}'"
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
            object_bottom = _object_bottom_world_z(base_env, str(object_name))
            if target_bottom is None or object_bottom is None:
                raise ValueError(
                    f"Could not align support surfaces for LIBERO object "
                    f"'{object_name}'"
                )
            pos[2] = float(current_z) + float(target_bottom) - float(object_bottom)
        elif _as_bool(override.get("preserve_z", False)):
            pos[2] = current_z

        if "quat_from" in override:
            quat_from = override["quat_from"]
            quat_joint = _free_joint_name(base_env, quat_from)
            if quat_joint is None:
                raise ValueError(f"LIBERO object '{quat_from}' has no movable free joint for quat_from")
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
                str(object_name),
                free_joint,
                np.asarray(candidate_base_pos, dtype=float),
                override,
                np.asarray(quat, dtype=float),
                float(current_z),
            )
        if (
            is_movable
            and override.get("support_world_z") is not None
            and _as_bool(override.get("align_collision_bottom", False))
            and str(override.get("candidate_selection", "")).strip().lower()
            != "first_contact_free"
        ):
            provisional = np.concatenate([
                np.array([pos[0], pos[1], current_z], dtype=float),
                np.asarray(quat, dtype=float),
            ])
            _set_joint_qpos(sim, free_joint, provisional)
            collision_lower = _object_collision_lower_world_z(
                base_env, str(object_name)
            )
            if collision_lower is None:
                raise ValueError(
                    f"Could not resolve collision bottom for LIBERO object "
                    f"'{object_name}'"
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
        else:
            _set_libero_object_or_fixture_position(base_env, object_name, pos)
            body_id = int(base_env.obj_body_id[object_name])
            sim.model.body_quat[body_id, :4] = quat
            _forward_sim(sim)

    if spec.get("object_surface_black_fraction_overrides"):
        from robust_libero_perturbations import apply_object_surface_black_overrides

        spec["_applied_surface_black_overrides"] = apply_object_surface_black_overrides(
            base_env,
            spec.get("object_surface_black_fraction_overrides", []),
        )

    for override in spec.get("object_color_overrides", []):
        _set_object_visual_color(
            base_env,
            override["object"],
            rgba=override.get("rgba"),
            palette=override.get("palette"),
            mesh_only=_as_bool(override.get("mesh_only", True)),
            update_material=_as_bool(override.get("update_material", True)),
        )

    visual_matches = []
    for override in spec.get("geom_color_overrides", []) + spec.get("body_color_overrides", []):
        visual_matches.extend(_set_visual_color_by_names(base_env, override))
    if visual_matches:
        spec["_applied_visual_color_overrides"] = visual_matches

    for override in spec.get("body_mass_overrides", []):
        object_name = override["object"]
        body_id = base_env.obj_body_id[object_name]
        mass = float(override.get("mass", 1000.0))
        sim.model.body_mass[body_id] = mass
        if hasattr(sim.model, "body_inertia"):
            sim.model.body_inertia[body_id] = np.maximum(sim.model.body_inertia[body_id], mass * 0.01)

    _forward_sim(sim)
    try:
        env_in.check_success()
        env_in._post_process()
        env_in._update_observables(force=True)
    except AttributeError:
        pass
    for override in spec.get("object_color_overrides", []):
        _set_object_visual_color(
            base_env,
            override["object"],
            rgba=override.get("rgba"),
            palette=override.get("palette"),
            mesh_only=_as_bool(override.get("mesh_only", True)),
            update_material=_as_bool(override.get("update_material", True)),
        )
    _forward_sim(sim)


def save_video(real_obs_list, save_path, fps=15, video_names=["observation.images.agentview_rgb", "observation.images.eye_in_hand_rgb"]):
    if not real_obs_list:
        print("❌ No real observation frames")
        return

    first_obs = real_obs_list[0]
    base_h, width_base = first_obs[video_names[0]].shape[:2]
    target_size = (width_base, base_h)
    
    print(f"Saving video: {len(real_obs_list)} frames...")

    final_frames = [
        np.hstack([cv2.resize(obs[name], target_size) for name in video_names]).astype(np.uint8)
        for obs in real_obs_list
    ]

    imageio.mimsave(save_path, final_frames, fps=fps)
    print(f"✅ Video saved to: {save_path}")


def construct_single_env(env_args, use_segmentation=False, object_xml_override_map=None):
    count = 0
    env = None
    env_creation = False
    env_cls = SegmentationRenderEnv if use_segmentation else OffScreenRenderEnv
    old_xml_override = os.environ.get(LIBERO_OBJECT_XML_OVERRIDE_ENV)
    should_override_xml = object_xml_override_map is not None
    if should_override_xml:
        if object_xml_override_map:
            os.environ[LIBERO_OBJECT_XML_OVERRIDE_ENV] = json.dumps(object_xml_override_map)
        else:
            os.environ.pop(LIBERO_OBJECT_XML_OVERRIDE_ENV, None)
    try:
        while not env_creation and count < 5:
            try:
                env = env_cls(**env_args)
                env._libero_object_xml_override_map = object_xml_override_map
                env_creation = True
            except Exception as e:
                print(f"Error!!!  construct env failed: {e}")
                time.sleep(5)
                count += 1
    finally:
        if should_override_xml:
            if old_xml_override is None:
                os.environ.pop(LIBERO_OBJECT_XML_OVERRIDE_ENV, None)
            else:
                os.environ[LIBERO_OBJECT_XML_OVERRIDE_ENV] = old_xml_override
    if count >= 5:
        return None
    return env


def _extract_obs(obs):
    """
    Extract agentview and eye_in_hand images from raw env obs dict.

    Avoids torch round-trip: the env already returns uint8 numpy arrays [H, W, C].
    We just flip the vertical axis ([::-1]) and make a contiguous copy once.
    """
    agentview = np.ascontiguousarray(obs["agentview_image"][::-1])
    eye_in_hand = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1])
    return {"observation.images.agentview_rgb": agentview, "observation.images.eye_in_hand_rgb": eye_in_hand}


def _render_image_stats(image):
    image = np.asarray(image, dtype=float)
    gray = image.mean(axis=2)
    dx = np.abs(np.diff(gray, axis=1))
    dy = np.abs(np.diff(gray, axis=0))
    row_diff = np.abs(np.diff(image, axis=0)).mean(axis=2)
    col_diff = np.abs(np.diff(image, axis=1)).mean(axis=2)
    channel_std = image.std(axis=2)
    channel_range = image.max(axis=2) - image.min(axis=2)
    return {
        "mean": float(gray.mean()),
        "std": float(gray.std()),
        "black": float((gray < 8).mean()),
        "dark": float((gray < 20).mean()),
        "bright": float((gray > 245).mean()),
        "dx_mean": float(dx.mean()),
        "dy_mean": float(dy.mean()),
        "row_high": float((row_diff > 45).mean()),
        "col_high": float((col_diff > 45).mean()),
        "chroma_mean": float(channel_std.mean()),
        "crange_mean": float(channel_range.mean()),
    }


def _agentview_render_stats(real_obs):
    agent_stats = _render_image_stats(real_obs["observation.images.agentview_rgb"])
    eye_stats = _render_image_stats(real_obs["observation.images.eye_in_hand_rgb"])
    stats = dict(agent_stats)
    stats.update({f"eye_{key}": value for key, value in eye_stats.items()})
    return stats


def _render_view_stats_is_bad(stats, prefix=""):
    mean = stats[f"{prefix}mean"]
    std = stats[f"{prefix}std"]
    black = stats[f"{prefix}black"]
    bright = stats[f"{prefix}bright"]
    dx_mean = stats[f"{prefix}dx_mean"]
    dy_mean = stats[f"{prefix}dy_mean"]
    row_high = stats[f"{prefix}row_high"]
    col_high = stats[f"{prefix}col_high"]
    chroma_mean = stats[f"{prefix}chroma_mean"]
    crange_mean = stats[f"{prefix}crange_mean"]

    high_frequency_artifact = (
        (dx_mean > 25.0 or dy_mean > 25.0)
        and (row_high > 0.15 or col_high > 0.15 or chroma_mean > 14.0 or crange_mean > 35.0)
    )
    color_stripe_artifact = chroma_mean > 30.0 and crange_mean > 70.0
    return (
        mean < 15.0
        or std < 8.0
        or black > 0.35
        or bright > 0.55
        or high_frequency_artifact
        or color_stripe_artifact
    )


def _initial_agentview_render_is_bad(real_obs):
    stats = _agentview_render_stats(real_obs)
    return (
        _render_view_stats_is_bad(stats)
        or _render_view_stats_is_bad(stats, "eye_")
    ), stats


def _extract_obs_for_policy(obs, env_in, obs_occlusion_spec, libero_benchmark, task_idx):
    real_obs = _extract_obs(obs)
    model_obs, occlusion_stats = _apply_obs_occlusion(
        real_obs,
        obs,
        env_in,
        obs_occlusion_spec,
        libero_benchmark,
        task_idx,
    )
    model_obs = _select_single_view_model_obs(model_obs)
    return real_obs, model_obs, occlusion_stats


def _apply_cross_task_view_swap(source_model_obs, donor_model_obs, spec, source_benchmark, source_task_idx, donor_benchmark, donor_task_idx):
    swapped_views = _normalize_occlusion_views(spec.get("swapped_views", spec.get("views")))
    model_obs = {key: value.copy() for key, value in source_model_obs.items()}
    view_stats = {}
    for view_name in swapped_views:
        cfg = MODEL_VIEW_CONFIGS[view_name]
        model_key = cfg["model_key"]
        model_obs[model_key] = donor_model_obs[model_key].copy()
        height, width = model_obs[model_key].shape[:2]
        view_stats[view_name] = {
            "masked_pixels": int(height * width),
            "mask_fraction": 1.0,
            "missing_targets": [],
            "source": f"{source_benchmark}:{source_task_idx}",
            "donor": f"{donor_benchmark}:{donor_task_idx}",
        }
    return model_obs, {
        "enabled": True,
        "target_policy": "cross_task_view_swap",
        "target_objects": [],
        "mask_source": spec.get("mask_source", "cross_task_view_swap"),
        "mask_geometry": "view_swap",
        "dilate_px": 0,
        "views": swapped_views,
        "swapped_views": swapped_views,
        "source_benchmark": source_benchmark,
        "source_task_idx": int(source_task_idx),
        "donor_benchmark": donor_benchmark,
        "donor_task_idx": int(donor_task_idx),
        "donor_task_offset": spec.get("donor_task_offset"),
        "swap_mode": spec.get("swap_mode", "dynamic_same_action"),
        "view_stats": view_stats,
    }


def _apply_c1_fullframe_to_keyframes(keyframes, spec, replan_idx):
    """Fully corrupt every frame of a LingBot replan input (both cameras).

    LingBot builds one inference input from several frames collected during the
    preceding action chunk.  Per-control gating can therefore create a mixed
    clean/corrupt input when a phase begins mid-chunk.  C1 is defined per model
    decision, so once the consumer replan is active its entire visual sequence
    must receive the declared full-frame corruption.
    """
    variant = str((spec or {}).get("variant", "")).lower()
    if variant not in {"fullscreen_noise", "fullscreen_black"}:
        return keyframes, None
    model_keys = (
        "observation.images.agentview_rgb",
        "observation.images.eye_in_hand_rgb",
    )
    output = []
    rng = np.random.default_rng(int((spec or {}).get("noise_seed", 20260723)) + int(replan_idx))
    for frame in keyframes:
        item = {key: value.copy() if hasattr(value, "copy") else value for key, value in frame.items()}
        for key in model_keys:
            if key not in item:
                continue
            array = np.asarray(item[key])
            if variant == "fullscreen_black":
                item[key] = np.zeros_like(array, dtype=np.uint8)
            else:
                std = float((spec or {}).get("noise_std", 80.0))
                item[key] = np.clip(rng.normal(127.0, std, array.shape), 0, 255).astype(np.uint8)
        output.append(item)
    stat = {
        "enabled": True,
        "variant": variant,
        "full_frame": True,
        "views": ["agentview", "robot0_eye_in_hand"],
        "consumer_replan_index": int(replan_idx),
        "sequence_frames": len(output),
    }
    return output, stat


def init_single_env(env_in, init_state, env_edit_spec=None, obs_occlusion_spec=None, libero_benchmark="", task_idx=0):
    xml_override_map = getattr(env_in, "_libero_object_xml_override_map", None)
    old_xml_override = os.environ.get(LIBERO_OBJECT_XML_OVERRIDE_ENV)
    if xml_override_map:
        os.environ[LIBERO_OBJECT_XML_OVERRIDE_ENV] = json.dumps(xml_override_map)
    try:
        env_in.reset()
        base_bddl_file_name = (env_edit_spec or {}).get("_base_bddl_file_name_for_init_state", "")
        if base_bddl_file_name:
            obs = _set_init_state_by_joint_names(env_in, init_state, base_bddl_file_name)
        else:
            obs = env_in.set_init_state(init_state)
        pre_edit_settle_steps = int(
            (env_edit_spec or {}).get("pre_edit_settle_steps", 0)
        )
        for _ in range(pre_edit_settle_steps):
            obs, _, _, _ = env_in.step([0.] * 7)
        if pre_edit_settle_steps and base_bddl_file_name:
            # Keep the newly added joints at their physically settled parking
            # heights, but restore every original task / robot joint exactly.
            obs = _set_init_state_by_joint_names(
                env_in, init_state, base_bddl_file_name
            )
        apply_libero_env_edit(env_in, env_edit_spec, libero_benchmark, task_idx)
        # A3 prototype: direct simulator-render material occlusion. This is applied
        # after initialization and before the first observation; it is isolated by
        # the spec and does not modify collision or dynamics.
        _a3_env_visual = _apply_env_visual_occluder(
            getattr(env_in, "env", env_in), obs_occlusion_spec or {}
        )
        obs = getattr(env_in, "env", env_in)._get_observations()
        settle_steps = int((env_edit_spec or {}).get("settle_steps", 5))
        for _ in range(settle_steps):
            obs, _, _, _ = env_in.step([0.] * 7)
        real_obs, model_obs, stats = _extract_obs_for_policy(
            obs, env_in, obs_occlusion_spec, libero_benchmark, task_idx
        )
        if _a3_env_visual.get("enabled"):
            stats = dict(stats or {})
            stats["env_visual_occluder"] = _a3_env_visual
        return real_obs, model_obs, stats
    finally:
        if xml_override_map:
            if old_xml_override is None:
                os.environ.pop(LIBERO_OBJECT_XML_OVERRIDE_ENV, None)
            else:
                os.environ[LIBERO_OBJECT_XML_OVERRIDE_ENV] = old_xml_override


def env_one_step(env_in, action, obs_occlusion_spec=None, libero_benchmark="", task_idx=0):
    obs, _, done, _ = env_in.step(action)
    real_obs, model_obs, occlusion_stats = _extract_obs_for_policy(
        obs,
        env_in,
        obs_occlusion_spec,
        libero_benchmark,
        task_idx,
    )
    return real_obs, model_obs, occlusion_stats, done


def _donor_env_worker(conn, donor_benchmark, donor_task_idx, episode_idx):
    env = None
    try:
        benchmark_dict = benchmark.get_benchmark_dict()
        donor_benchmark_instance = benchmark_dict[donor_benchmark]()
        donor_env_args = {
            "bddl_file_name": donor_benchmark_instance.get_task_bddl_file_path(donor_task_idx),
            "camera_heights": 128,
            "camera_widths": 128,
        }
        donor_init_states = donor_benchmark_instance.get_task_init_states(donor_task_idx)
        env = construct_single_env(donor_env_args, use_segmentation=False)
        if env is None:
            raise RuntimeError("Failed to construct donor env")
        _, first_obs, _ = init_single_env(
            env,
            donor_init_states[episode_idx % donor_init_states.shape[0]],
            None,
            None,
            donor_benchmark,
            donor_task_idx,
        )
        conn.send({"ok": True, "obs": first_obs})

        while True:
            msg = conn.recv()
            cmd = msg.get("cmd")
            if cmd == "close":
                conn.send({"ok": True})
                break
            if cmd != "step":
                raise ValueError(f"Unsupported donor worker command: {cmd}")
            _, model_obs, _, done = env_one_step(
                env,
                np.asarray(msg["action"]),
                None,
                donor_benchmark,
                donor_task_idx,
            )
            conn.send({"ok": True, "obs": model_obs, "done": bool(done)})
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
    def __init__(self, donor_benchmark, donor_task_idx, episode_idx):
        ctx = mp.get_context("spawn")
        self._parent_conn, child_conn = ctx.Pipe()
        self._process = ctx.Process(
            target=_donor_env_worker,
            args=(child_conn, donor_benchmark, int(donor_task_idx), int(episode_idx)),
            daemon=True,
        )
        self._process.start()
        child_conn.close()
        self.first_obs = self._recv("initialize donor env")["obs"]

    def _recv(self, context):
        msg = self._parent_conn.recv()
        if not msg.get("ok"):
            raise RuntimeError(f"Donor env worker failed to {context}: {msg.get('error')}")
        return msg

    def step(self, action):
        self._parent_conn.send({"cmd": "step", "action": np.asarray(action)})
        return self._recv("step donor env")["obs"]

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


def run_one(
    model,
    libero_benchmark,
    task_idx,
    out_dir,
    episode_idx,
    task_descriptions_file="",
    use_task_descriptions=True,
    env_edit_spec=None,
    env_edit_source="",
    flat_video_path="",
    prompt_override=None,
    obs_occlusion_spec=None,
    obs_occlusion_source="",
):
    benchmark_dict = benchmark.get_benchmark_dict()
    benchmark_instance = benchmark_dict[libero_benchmark]()
    _state_prepared = None
    _state_case_path = _os.environ.get("WAM_STATE_CASE_JSON", "").strip()
    _state_case = (
        json.loads(Path(_state_case_path).read_text(encoding="utf-8"))
        if _state_case_path
        else None
    )
    num_tasks = benchmark_instance.get_num_tasks()
    assert task_idx < num_tasks, f"Error: error id must smaller than {num_tasks}"
    original_prompt = benchmark_instance.get_task(task_idx).language
    modified_instruction, task_descriptions_source = _load_libero_modified_instruction(
        task_descriptions_file,
        libero_benchmark,
        task_idx,
        use_task_descriptions,
    )
    prompt = prompt_override if prompt_override is not None else (modified_instruction or original_prompt)
    if prompt_override is not None:
        print(f"Original instruction: {original_prompt}")
        print(f"Prompt override: {prompt!r}")
    elif modified_instruction:
        print(f"Original instruction: {original_prompt}")
        print(f"Modified instruction: {prompt}")
    if task_descriptions_source:
        print(f"TaskDescriptions source: {task_descriptions_source}")
    if env_edit_source:
        print(f"Environment edit source: {env_edit_source}")
    if obs_occlusion_source:
        print(f"Observation occlusion source: {obs_occlusion_source}")
    resolved_env_edit_spec = _resolve_libero_env_edit_spec(env_edit_spec, libero_benchmark, task_idx)
    original_bddl_file_name = benchmark_instance.get_task_bddl_file_path(task_idx)
    bddl_file_name = _libero_bddl_file_name(resolved_env_edit_spec, original_bddl_file_name)
    base_bddl_file_name = _libero_base_bddl_file_name(resolved_env_edit_spec, original_bddl_file_name)
    if base_bddl_file_name:
        resolved_env_edit_spec = copy.deepcopy(resolved_env_edit_spec)
        resolved_env_edit_spec["_base_bddl_file_name_for_init_state"] = base_bddl_file_name
        print(f"BDDL override: {bddl_file_name}")
        print(f"Init-state base BDDL: {base_bddl_file_name}")
    env_args = {
                "bddl_file_name": bddl_file_name,
                "camera_heights": 128,
                "camera_widths": 128,
            }
    init_states = benchmark_instance.get_task_init_states(task_idx)

    cross_task_view_swap = _is_cross_task_view_swap(obs_occlusion_spec)
    obs_spec_for_env = None if cross_task_view_swap else obs_occlusion_spec
    # --- LIBERO-VPro temporal observation schedule ---
    _wsb_schedule_spec = _wsb_load_schedule_spec()
    _wsb_task_key = f"{libero_benchmark}:{int(task_idx)}"
    if (
        _wsb_schedule_spec
        and _wsb_task_key
        in {str(value) for value in _wsb_schedule_spec.get("not_applicable_tasks", [])}
    ):
        _wsb_schedule_spec = dict(_wsb_schedule_spec)
        _wsb_schedule_spec["enabled"] = False
        _wsb_schedule_spec["not_applicable"] = True
        _wsb_schedule_spec["not_applicable_reason"] = "task_has_no_placement_event"
    _wsb_scheduler = _WSB_SCHEDULER_CLS(_wsb_schedule_spec) if (_WSB_SCHEDULER_CLS and _wsb_schedule_spec) else None
    # phase targets: prefer explicit target_objects; else per-task map (precise
    # configs use target_objects_by_task keyed "<benchmark>:<task_idx>"); else fall
    # back to obj_of_interest inside infer_phase. The per-task map was previously
    # missed, leaving phase targets empty -> phase detection used the wrong objects.
    # phase detection tracks the MANIPULATED object (to detect grasp/place actions),
    # which may differ from the occlusion target (e.g. A2 occludes the receptacle but
    # phase must follow the object being placed). Priority: schedule.phase_targets_by_task
    # -> obs_occlusion target_objects -> target_objects_by_task -> obj_of_interest.
    _wsb_ptbt = (_wsb_schedule_spec or {}).get("phase_targets_by_task", {}) or {}
    _wsb_phase_targets = list(
        _wsb_ptbt.get(f"{libero_benchmark}:{task_idx}")
        or _wsb_ptbt.get(str(task_idx)) or [])
    if not _wsb_phase_targets:
        _wsb_phase_targets = list((obs_occlusion_spec or {}).get("target_objects", []) or [])
    if not _wsb_phase_targets:
        _wsb_tbt = (obs_occlusion_spec or {}).get("target_objects_by_task", {}) or {}
        _wsb_phase_targets = list(
            _wsb_tbt.get(f"{libero_benchmark}:{task_idx}")
            or _wsb_tbt.get(str(task_idx)) or [])
    _wsb_phase_state = {"phase": "initial", "latch": False}
    if _wsb_reset_phase_state is not None:
        _wsb_reset_phase_state()
    if _wsb_scheduler is not None:
        print(f"LIBERO-VPro temporal schedule active: {_wsb_schedule_spec}")
    donor_worker = None
    donor_benchmark = ""
    donor_task_idx = None
    donor_prompt = ""

    use_segmentation = _obs_occlusion_needs_segmentation(obs_spec_for_env)
    object_xml_override_map = _libero_object_xml_override_map(resolved_env_edit_spec)
    cur_env = None
    first_real_obs = None
    first_obs = None
    first_occlusion_stats = None
    initial_render_retries = int(os.environ.get("LIBERO_INITIAL_RENDER_RETRIES", "6"))
    for render_attempt in range(1, max(1, initial_render_retries) + 1):
        if cur_env is not None:
            cur_env.close()
            cur_env = None
        cur_env = construct_single_env(
            env_args,
            use_segmentation=use_segmentation,
            object_xml_override_map=object_xml_override_map or None,
        )
        if cur_env is None:
            raise RuntimeError("Failed to construct LIBERO environment")
        first_real_obs, first_obs, first_occlusion_stats = init_single_env(
            cur_env,
            init_states[episode_idx % init_states.shape[0]],
            resolved_env_edit_spec,
            obs_spec_for_env,
            libero_benchmark,
            task_idx,
        )
        bad_render, render_stats = _initial_agentview_render_is_bad(first_real_obs)
        if not bad_render:
            if render_attempt > 1:
                print(f"Initial render accepted after retry {render_attempt}: {render_stats}")
            break
        print(f"Initial render looked invalid on attempt {render_attempt}; retrying env init: {render_stats}")
    else:
        print(f"Warning: using initial render after {initial_render_retries} failed quality checks")
    if _state_case is not None:
        if (
            str(_state_case["suite"]) != str(libero_benchmark)
            or int(_state_case["task_id"]) != int(task_idx)
        ):
            raise ValueError(
                f"State case/task mismatch: {_state_case['case_id']} vs "
                f"{libero_benchmark}:{task_idx}"
            )
        cur_env.close()
        _state_prepared = _prepare_state_environment(
            _sys.modules[__name__],
            benchmark_instance,
            _state_case,
            init_states[episode_idx % init_states.shape[0]],
            camera_size=128,
        )
        cur_env = _state_prepared.env
        _state_true_raw = _state_prepared.initial_observation
        _state_model_raw, first_occlusion_stats = (
            _state_prepared.observation_for_model(
                _sys.modules[__name__], _state_true_raw
            )
        )
        first_real_obs = _extract_obs(_state_true_raw)
        first_obs = _extract_obs(_state_model_raw)
        resolved_env_edit_spec = copy.deepcopy(_state_case)
    # A scheduled perturbation must also gate the very first model observation.
    # `init_single_env` builds both clean and perturbed variants eagerly; without
    # this override every phase-targeted case would receive an unintended corrupt
    # initial frame, even when its target phase is later in the episode.
    if _wsb_scheduler is not None:
        _wsb_initial_active = _wsb_scheduler.should_perturb(
            timestep=0,
            replan_idx=0,
            phase=_wsb_phase_state["phase"],
            grasped=_wsb_phase_state["latch"],
        )
        if not _wsb_initial_active:
            first_obs = {key: value.copy() for key, value in first_real_obs.items()}
            first_occlusion_stats = {
                "enabled": False,
                "scheduled_out": True,
                "target_phase": list((_wsb_schedule_spec or {}).get("phases", []) or []),
            }
    if cross_task_view_swap:
        donor_benchmark, donor_task_idx = _resolve_cross_task_donor(
            obs_occlusion_spec,
            libero_benchmark,
            task_idx,
            num_tasks,
        )
        donor_benchmark_instance = benchmark_dict[donor_benchmark]()
        donor_prompt = donor_benchmark_instance.get_task(donor_task_idx).language
        donor_worker = CrossTaskDonorEnvWorker(donor_benchmark, donor_task_idx, episode_idx)
        donor_first_obs = donor_worker.first_obs
        print(
            "Cross-task view swap: "
            f"source={libero_benchmark}:{task_idx}, donor={donor_benchmark}:{donor_task_idx}, "
            f"swapped_views={_normalize_occlusion_views(obs_occlusion_spec.get('swapped_views', obs_occlusion_spec.get('views')))}"
        )
        print(f"Donor instruction: {donor_prompt}")
        first_obs, first_occlusion_stats = _apply_cross_task_view_swap(
            first_obs,
            donor_first_obs,
            obs_occlusion_spec,
            libero_benchmark,
            task_idx,
            donor_benchmark,
            donor_task_idx,
        )

    ret = model.infer(dict(reset=True, prompt=prompt))
    server_model_seed_info = dict(ret.get("model_seed_info") or {})
    server_model_seed = server_model_seed_info.get("base_seed")
    if server_model_seed is None:
        # Backward compatibility for an older server that does not return seed
        # metadata. LINGBOT_MODEL_SEED is metadata-only and does not drive inference.
        legacy_model_seed = _os.environ.get("LINGBOT_MODEL_SEED", "").strip()
        server_model_seed = int(legacy_model_seed) if legacy_model_seed else None
    episode_recorder = make_episode_recorder(
        model_name="LingBot-VA",
        model_type="imagine_then_act",
        suite=str(libero_benchmark),
        task_id=int(task_idx),
        task_name=str(original_prompt),
        instruction=str(prompt),
        repeat_index=int(episode_idx),
        init_state_index=int(episode_idx % init_states.shape[0]),
        checkpoint=_os.environ.get("LINGBOT_CHECKPOINT", ""),
        environment_seed=(
            int(_os.environ["LIBERO_ENV_SEED"])
            if _os.environ.get("LIBERO_ENV_SEED", "").strip()
            else None
        ),
        model_seed=server_model_seed,
        config={
            "task_descriptions_file": task_descriptions_file,
            "use_task_descriptions": bool(use_task_descriptions),
            "schedule": _wsb_schedule_spec,
            "single_view": _c5_view_name(),
            "view_copy": _view_copy_name(),
        },
        env_edit_spec=resolved_env_edit_spec,
        perturbation_spec={
            "observation": obs_occlusion_spec or {},
            "schedule": _wsb_schedule_spec,
            "cross_task_view_swap": bool(cross_task_view_swap),
        },
        action_config={
            "action_tensor_layout": "[batch, temporal_block, control, action_dim]",
            "keyframes_per_chunk": 4,
        },
        extra_metadata={
            "donor_benchmark": donor_benchmark,
            "donor_task_id": donor_task_idx,
            "recorder_inference_invariant": "no_additional_model_calls_or_decodes",
            "model_seed_info": server_model_seed_info,
            "single_view": _c5_view_name(),
            "view_copy": _view_copy_name(),
            "visual_input_keys": _single_view_model_keys(),
            "visual_input_contract": (
                "duplicated_single_camera_to_dual_slots"
                if _view_copy_name()
                else ("direct_single_camera" if _single_view_name() else "official_dual_camera")
            ),
            "visual_input_size": [128, 128],
            "state_case_runtime": bool(_state_prepared is not None),
            "initialization_audit": (
                _state_prepared.initialization_audit
                if _state_prepared is not None else {}
            ),
        },
    )

    full_obs_list = []
    # Include the observation consumed by replan 0.  Previously the masked-input
    # video began with frames collected for replan 1, so an initial-phase pulse
    # was applied to the model but its first corrupted decision was absent from
    # the evidence video.
    masked_obs_list = [copy.deepcopy(first_obs)]
    occlusion_stats_list = [first_occlusion_stats]
    predicted_video_dirs = set()
    predicted_video_paths = []
    done = False
    first = True
    # Count replans (each model.infer = one replan / action chunk),
    # so schedules can gate perturbation by "N replans" rather than raw timesteps.
    _wsb_replan_idx = 0
    # B1 per-view camera staleness buffer (delay/freeze/replay by replan)
    _wsb_cam_spec = (_wsb_schedule_spec or {}).get("camera") or {}
    _wsb_cam_buf = _WSB_BUFFER_CLS(_wsb_cam_spec) if (_WSB_BUFFER_CLS and _wsb_cam_spec) else None
    if _wsb_cam_buf is not None:
        print(f"LIBERO-VPro B1 camera staleness active: {_wsb_cam_spec}")
    # Phases in which to use predicted frames as policy inputs
    _wsb_imag_phases = set((_wsb_schedule_spec or {}).get("imagine_feedback_phases", []) or [])
    _wsb_imagined = None
    _wsb_imag_stats = {"applied": 0, "keyframes_total": 0, "phases": set()}
    # fixed dose: how many key frames get replaced by imagination in this run
    _wsb_imag_budget = int((_wsb_schedule_spec or {}).get("imagine_budget_frames", 12))
    if _wsb_imag_phases:
        print(f"LIBERO-VPro prediction feedback in phases: {_wsb_imag_phases}")
    _record_clean_input = first_real_obs
    _record_model_input = first_obs
    _record_actual_stats = first_occlusion_stats or {"enabled": False}
    while cur_env.env.timestep < 800:
        ret = model.infer(dict(obs=first_obs, prompt=prompt,
                               return_imagined=bool(_wsb_imag_phases)))
        action = ret['action']
        if ret.get("predicted_video_dir"):
            predicted_video_dirs.add(ret["predicted_video_dir"])
        predicted_video_paths.extend(ret.get("predicted_video_paths") or [])
        # Keep this replan's predicted future frames so that, in the
        # targeted phase, they can be fed back to the model IN PLACE OF the real
        # observation (the model then acts on its own imagination, not on reality).
        _wsb_imagined = ret.get("imagined_frames")
        if _wsb_imag_phases:
            print(f"[prediction feedback] server returned imagined_frames: "
                  f"{None if _wsb_imagined is None else np.asarray(_wsb_imagined).shape}")
        if episode_recorder.enabled:
            episode_recorder.record_replan(
                _wsb_replan_idx,
                control_step_start=int(cur_env.env.timestep),
                phase=_wsb_phase_state["phase"],
                clean_observation=_record_clean_input,
                model_observation=_record_model_input,
                predicted_frames=_wsb_imagined,
                proposed_action_chunk=action,
                state=capture_libero_state(
                    cur_env,
                    object_names=episode_recorder.object_names,
                    include_sim_state=True,
                ),
                perturbation={
                    "active": bool((_record_actual_stats or {}).get("enabled", False)),
                    "actual": _record_actual_stats,
                    "schedule": _wsb_schedule_spec,
                },
                extra={
                    "predicted_video_dir": ret.get("predicted_video_dir", ""),
                    "predicted_video_paths": ret.get("predicted_video_paths") or [],
                    "attention_dump_jsonl": _os.environ.get(
                        "LINGBOT_ATTENTION_DUMP_JSONL", ""
                    ),
                },
            )

        key_frame_list = []
        clean_key_frame_list = []
        executed_action_chunk = []
        chunk_occlusion_stats = []
        chunk_control_step = 0
        assert action.shape[2] % 4 == 0
        action_per_frame = action.shape[2] // 4
        start_idx = 1 if first else 0
        for i in range(start_idx, action.shape[1]):
            for j in range(action.shape[2]):
                ee_action = action[:, i, j]
                _wsb_step_spec = obs_spec_for_env
                if _wsb_scheduler is not None:
                    _wsb_phase = None
                    if (
                        _wsb_scheduler.trigger in (
                            "phase", "phase_pulse", "phase_once", "grasp_pulse", "place_pulse",
                            "post_place_verified"
                        )
                        or _as_bool((_wsb_schedule_spec or {}).get("phase_diagnostics", False))
                        or _wsb_imag_phases
                    ) and _wsb_infer_phase is not None:
                        # gripper command = last action dim; <0 (or >0 depending on
                        # convention) marks the policy's intent to CLOSE = grasp intent.
                        try:
                            _wsb_grip_cmd = float(np.asarray(ee_action).reshape(-1)[-1])
                        except Exception:
                            _wsb_grip_cmd = None
                        _wsb_pgb = float((_wsb_schedule_spec.get("pre_grasp_band")
                                          or (_wsb_schedule_spec.get("grasp_pulse", {}) or {}).get("pre_grasp_band")
                                          or 0.20))
                        _wsb_phase, _wsb_phase_state["latch"] = _wsb_infer_phase(
                            cur_env, _wsb_phase_targets,
                            prev_phase=_wsb_phase_state["phase"],
                            grasped_latch=_wsb_phase_state["latch"],
                            gripper_cmd=_wsb_grip_cmd,
                            pre_grasp_band=_wsb_pgb,
                            grasp_hold_frames=int((_wsb_schedule_spec or {}).get("grasp_hold_frames", 24)),
                            initial_hold_frames=int((_wsb_schedule_spec or {}).get("initial_hold_frames", 24)),
                            release_hold_frames=int((_wsb_schedule_spec or {}).get("release_hold_frames", 24)),
                            pre_place_progress=float((_wsb_schedule_spec or {}).get("pre_place_progress", 0.70)),
                        )
                        _wsb_phase_state["phase"] = _wsb_phase
                        try:
                            import perturbation_schedule as _wsb_ps_mod
                            _wsb_diag = dict(getattr(_wsb_ps_mod, "LAST_DIAG", {}) or {})
                        except Exception:
                            _wsb_diag = {}
                        _wsb_phase_state.setdefault("log", []).append(
                            {"t": int(getattr(getattr(cur_env, "env", cur_env), "timestep", 0)),
                             # Observations collected while executing chunk r are
                             # consumed by inference r+1.  Record that consumer index
                             # so calibrated decision windows have exactly the same
                             # zero-based meaning as decision_indices below.
                             "decision": int(_wsb_replan_idx + 1),
                             "phase": _wsb_phase, "latch": bool(_wsb_phase_state["latch"]),
                             "targets": list(_wsb_phase_targets or []), **_wsb_diag})
                    _wsb_ts = int(getattr(getattr(cur_env, "env", cur_env), "timestep", 0))
                    # The frames collected while executing chunk r become the visual
                    # input for inference r+1. Gate by that consumer decision index so
                    # decision_indices has the same meaning as direct-observation
                    # policies such as FastWAM.
                    if not _wsb_scheduler.should_perturb(
                            timestep=_wsb_ts, replan_idx=_wsb_replan_idx + 1,
                            phase=_wsb_phase,
                            grasped=_wsb_phase_state["latch"]):
                        _wsb_step_spec = None
                observes, model_observes, occlusion_stats, done = env_one_step(
                    cur_env,
                    ee_action,
                    _wsb_step_spec,
                    libero_benchmark,
                    task_idx,
                )
                if _state_prepared is not None:
                    _state_true_raw = cur_env.env._get_observations()
                    _state_model_raw, occlusion_stats = (
                        _state_prepared.observation_for_model(
                            _sys.modules[__name__], _state_true_raw
                        )
                    )
                    observes = _extract_obs(_state_true_raw)
                    model_observes = _extract_obs(_state_model_raw)
                if cross_task_view_swap:
                    donor_model_observes = donor_worker.step(ee_action)
                    model_observes, occlusion_stats = _apply_cross_task_view_swap(
                        model_observes,
                        donor_model_observes,
                        obs_occlusion_spec,
                        libero_benchmark,
                        task_idx,
                        donor_benchmark,
                        donor_task_idx,
                    )
                if episode_recorder.enabled:
                    executed_action_chunk.append(np.asarray(ee_action).copy())
                    episode_recorder.record_control(
                        int(cur_env.env.timestep),
                        replan_index=int(_wsb_replan_idx),
                        chunk_step_index=int(chunk_control_step),
                        phase=_wsb_phase_state["phase"],
                        action=ee_action,
                        state=capture_libero_state(
                            cur_env,
                            object_names=episode_recorder.object_names,
                        ),
                        perturbation_active=bool((occlusion_stats or {}).get("enabled", False)),
                        done=bool(done),
                        extra={
                            "actual_perturbation": occlusion_stats or {},
                        },
                    )
                    chunk_control_step += 1
                    chunk_occlusion_stats.append(occlusion_stats or {})
                if done:
                    break
                if (j+1) % action_per_frame == 0:
                    full_obs_list.append(observes)
                    clean_key_frame_list.append(observes)
                    # Prediction feedback: in the targeted phase, replace the
                    # real observation fed back to the model with the model's OWN imagined
                    # frame for this key-frame index (so it closes the loop on imagination,
                    # not reality). The env still advances on the real action.
                    # Cap the substitution at a FIXED number of key frames per run so
                    # every phase gets the same perturbation dose. Without the cap a
                    # failing run keeps looping inside one phase, inflating its frame
                    # count — the dose then becomes a consequence of failure, not its
                    # cause, and phases are no longer comparable.
                    if _wsb_imag_phases and _wsb_imagined is not None \
                            and _wsb_phase in _wsb_imag_phases \
                            and _wsb_imag_stats["applied"] < _wsb_imag_budget:
                        _kf_idx = len(key_frame_list)
                        model_observes = _wsb_apply_imagined(
                            model_observes, _wsb_imagined, _kf_idx)
                        _wsb_imag_stats["applied"] += 1
                        _wsb_imag_stats["phases"].add(_wsb_phase)
                    _wsb_imag_stats["keyframes_total"] += 1
                    key_frame_list.append(model_observes)
                    occlusion_stats_list.append(occlusion_stats)

            if done:
                break

        # A4 phase pulses are defined over complete model decisions. If the
        # target phase began in the middle of this action chunk, normalize the
        # entire sequence that will be consumed by inference r+1 instead of
        # leaving a mixed clean/corrupt sequence.
        _wsb_consumer_replan = int(_wsb_replan_idx + 1)
        _wsb_fullframe_phase_active = False
        if (
            _wsb_scheduler is not None
            and _wsb_scheduler.trigger in ("phase_pulse", "phase_once")
            and key_frame_list
        ):
            _wsb_fullframe_phase_active = _wsb_scheduler.should_perturb(
                timestep=int(getattr(getattr(cur_env, "env", cur_env), "timestep", 0)),
                replan_idx=_wsb_consumer_replan,
                phase=_wsb_phase_state["phase"],
                grasped=_wsb_phase_state["latch"],
            )
        if _wsb_fullframe_phase_active:
            key_frame_list, _wsb_fullframe_stat = _apply_c1_fullframe_to_keyframes(
                key_frame_list,
                obs_spec_for_env,
                _wsb_consumer_replan,
            )
            if _wsb_fullframe_stat is not None:
                chunk_occlusion_stats = [
                    dict(_wsb_fullframe_stat) for _ in range(max(1, len(key_frame_list)))
                ]

        # B1 camera staleness — applied at the REPLAN boundary (staleness is counted in
        # replans, so the buffer must advance here, not per key frame). The whole
        # key_frame_list for this replan is swapped for an earlier replan's frames on
        # the stale view; the other view stays clean, and the env advances normally.
        if _wsb_cam_buf is not None and key_frame_list:
            for _vn, _vcfg in MODEL_VIEW_CONFIGS.items():
                _mk = _vcfg["model_key"]
                _cur = [kf[_mk] for kf in key_frame_list if _mk in kf]
                if not _cur:
                    continue
                _wsb_cam_buf.push_replan(_vn, _cur)
                _old = _wsb_cam_buf.get(_vn)
                if _old:
                    for _i, kf in enumerate([k for k in key_frame_list if _mk in k]):
                        kf[_mk] = _old[min(_i, len(_old) - 1)]
            _wsb_cam_buf.end_replan()
            masked_obs_list.extend(key_frame_list)   # what the model actually saw
        else:
            masked_obs_list.extend(key_frame_list)

        if episode_recorder.enabled:
            clean_realized = {
                key: [frame[key] for frame in clean_key_frame_list if key in frame]
                for key in (
                    "observation.images.agentview_rgb",
                    "observation.images.eye_in_hand_rgb",
                )
            }
            model_realized = {
                key: [frame[key] for frame in key_frame_list if key in frame]
                for key in (
                    "observation.images.agentview_rgb",
                    "observation.images.eye_in_hand_rgb",
                )
            }
            combined_realized = [
                np.concatenate(
                    [
                        frame["observation.images.agentview_rgb"],
                        frame["observation.images.eye_in_hand_rgb"],
                    ],
                    axis=1,
                )
                for frame in clean_key_frame_list
                if "observation.images.agentview_rgb" in frame
                and "observation.images.eye_in_hand_rgb" in frame
            ]
            episode_recorder.record_realized(
                _wsb_replan_idx,
                control_step_end=int(cur_env.env.timestep),
                frames=(
                    np.stack(combined_realized)
                    if combined_realized
                    else clean_realized
                ),
                clean_frames=clean_realized,
                model_frames=model_realized,
                executed_action_chunk=executed_action_chunk,
            )
            episode_recorder.update_replan_trace(
                _wsb_replan_idx,
                extra={
                    # These frames are collected while executing replan r and are
                    # consumed by inference r+1.  Do not overwrite replan r's own
                    # model-input perturbation metadata with this future input.
                    "next_replan_observation_perturbation": {
                        "consumer_replan_index": int(_wsb_replan_idx + 1),
                        "active": any(
                            bool((item or {}).get("enabled", False))
                            for item in chunk_occlusion_stats
                        ),
                        "actual_per_control": chunk_occlusion_stats,
                        "schedule": _wsb_schedule_spec,
                    },
                    "camera_staleness": (
                        list(getattr(_wsb_cam_buf, "log", [])[-1:])
                        if _wsb_cam_buf is not None
                        else []
                    ),
                    "imagination_feedback_applied_total": _wsb_imag_stats["applied"],
                },
            )
            _record_clean_input = clean_realized
            _record_model_input = model_realized
            _record_actual_stats = {
                "enabled": any(
                    bool((item or {}).get("enabled", False))
                    for item in chunk_occlusion_stats
                ),
                "per_control": chunk_occlusion_stats,
            }

        first = False
        _wsb_replan_idx += 1   # one more replan / action chunk consumed

        if done:
            break
        else:
            model.infer(dict(obs=key_frame_list, compute_kv_cache=True, imagine=False, state=action))

    if flat_video_path:
        out_file = Path(flat_video_path)
    else:
        out_file = Path(out_dir) / libero_benchmark / f"{task_idx}_{prompt.replace(' ', '_')}" / f"{episode_idx}_{done}.mp4"
    out_file.parent.mkdir(exist_ok=True, parents=True)

    save_video(
        real_obs_list=full_obs_list,
        save_path=out_file,
        fps=60,
        video_names=["observation.images.agentview_rgb", "observation.images.eye_in_hand_rgb"]
    )
    masked_video_path = ""
    # Record model-input views when masking, camera staleness or prediction
    # feedback is enabled so the video preserves the policy observations.
    if _as_bool((obs_occlusion_spec or {}).get("save_masked_video")) \
            or (_wsb_cam_buf is not None) or _wsb_imag_phases or _c5_view_name() \
            or (_state_prepared is not None):
        masked_video_file = out_file.with_name(f"{out_file.stem}_masked_input{out_file.suffix}")
        save_video(
            real_obs_list=masked_obs_list,
            save_path=masked_video_file,
            fps=60,
            video_names=_single_view_model_keys(),
        )
        masked_video_path = str(masked_video_file)

    # Record per-step occlusion coverage statistics
    try:
        _cov_records = []
        for _st in occlusion_stats_list:
            if not (_st or {}).get("enabled"):
                continue
            for _vn, _vs in (_st.get("view_stats") or {}).items():
                if "actual_coverage" in _vs:
                    _cov_records.append({
                        "view": _vn,
                        "target_mask_pixels": _vs.get("target_mask_pixels"),
                        "masked_pixels": _vs.get("masked_pixels"),
                        "actual_coverage": _vs.get("actual_coverage"),
                        "requested_coverage": _vs.get("requested_coverage"),
                    })
        # A3 interaction-part: dump the per-part pixel centres for objective verification
        try:
            _a3_records = []
            for _st in occlusion_stats_list:
                if not (_st or {}).get("enabled"):
                    continue
                for _vn, _vs in (_st.get("view_stats") or {}).items():
                    if _vs.get("parts_drawn"):
                        _a3_records.append({"view": _vn, "masked_pixels": _vs.get("masked_pixels"),
                                            "mask_fraction": _vs.get("mask_fraction"),
                                            "parts": _vs.get("parts_drawn")})
            if _a3_records:
                _a3_file = out_file.with_name(f"{out_file.stem}_a3parts.json")
                write_json({"records": _a3_records}, str(_a3_file))
        except Exception as _e:
            print(f"[wsb] a3 parts dump skipped: {_e}")
        if _cov_records:
            import statistics as _stt
            _by_view = {}
            for _r in _cov_records:
                _by_view.setdefault(_r["view"], []).append(_r["actual_coverage"])
            _summary = {v: {"mean_actual_coverage": _stt.mean(a), "n": len(a),
                            "min": min(a), "max": max(a)} for v, a in _by_view.items()}
            _cov_file = out_file.with_name(f"{out_file.stem}_coverage.json")
            write_json({"requested": (obs_occlusion_spec or {}).get("coverage", 1.0),
                        "per_view_summary": _summary, "records": _cov_records}, str(_cov_file))
    except Exception as _e:
        print(f"[wsb] coverage stats dump skipped: {_e}")
    # Record per-step phase and grasp-detector diagnostics
    try:
        _plog = _wsb_phase_state.get("log") if isinstance(_wsb_phase_state, dict) else None
        if _plog:
            _plog_file = out_file.with_name(f"{out_file.stem}_phaselog.json")
            write_json({"targets_used": _plog[0].get("targets"), "steps": _plog}, str(_plog_file))
    except Exception as _e:
        print(f"[wsb] phase log dump skipped: {_e}")
    if _wsb_imag_phases:
        print(f"[prediction feedback] TOTAL imagination substitutions: {_wsb_imag_stats['applied']}"
              f" / {_wsb_imag_stats['keyframes_total']} key frames"
              f" (phases hit: {sorted(_wsb_imag_stats['phases'])})")
    # B1: dump per-replan staleness (which replan's frame each view was fed) so the
    # delay/freeze/replay behaviour can be verified objectively, not by eyeballing.
    try:
        if _wsb_cam_buf is not None and getattr(_wsb_cam_buf, "log", None):
            _c2f = out_file.with_name(f"{out_file.stem}_c2staleness.json")
            write_json({"camera_spec": _wsb_cam_spec, "records": _wsb_cam_buf.log}, str(_c2f))
    except Exception as _e:
        print(f"[LIBERO-VPro] B1 staleness dump skipped: {_e}")

    episode_recorder.finish(
        success=bool(done),
        final_phase=_wsb_phase_state["phase"],
        summary={
            "video_path": str(out_file),
            "masked_video_path": masked_video_path,
            "predicted_video_paths": predicted_video_paths,
            "phase_trace": _wsb_phase_state.get("log", []),
            "imagination_feedback": {
                "applied": _wsb_imag_stats["applied"],
                "keyframes_total": _wsb_imag_stats["keyframes_total"],
                "phases": sorted(_wsb_imag_stats["phases"]),
            },
            "camera_staleness_trace": (
                list(getattr(_wsb_cam_buf, "log", []))
                if _wsb_cam_buf is not None
                else []
            ),
        },
    )

    # Release cached 3-D proxy renderers before destroying the MuJoCo context.
    proxy_renderers = getattr(getattr(cur_env, "env", cur_env), "_a3_proxy_renderers", None)
    if proxy_renderers is not None:
        proxy_renderers.clear()
    cur_env.close()
    if donor_worker is not None:
        donor_worker.close()
    return {
        "success": done,
        "prompt": prompt,
        "original_prompt": original_prompt,
        "modified_instruction": modified_instruction,
        "prompt_override": prompt_override,
        "task_descriptions_source": task_descriptions_source,
        "env_edit_source": env_edit_source,
        "obs_occlusion_source": obs_occlusion_source,
        "obs_occlusion": _summarize_occlusion_stats(occlusion_stats_list),
        "cross_task_donor_benchmark": donor_benchmark,
        "cross_task_donor_task_idx": donor_task_idx,
        "cross_task_donor_prompt": donor_prompt,
        "video_path": str(out_file),
        "masked_video_path": masked_video_path,
        "predicted_video_dirs": sorted(predicted_video_dirs),
        "predicted_video_paths": predicted_video_paths,
    }


def run(
    libero_benchmark,
    port,
    out_dir,
    test_num,
    task_range=None,
    task_descriptions_file="",
    use_task_descriptions=True,
    env_edit_spec="",
    flat_video_path="",
    prompt_override=None,
    obs_occlusion_spec="",
    resume_existing=False,
    shard_index=None,
    num_shards=1,
    trial_ids=None,
):
    '''
        task_range: [start, end) for splitting tasks
    '''
    if task_range is None:
        benchmark_dict = benchmark.get_benchmark_dict()
        benchmark_instance = benchmark_dict[libero_benchmark]()
        num_tasks = benchmark_instance.get_num_tasks()
        progress_bar = tqdm(range(num_tasks), total=num_tasks)
    else:
        assert len(task_range) == 2, f'task_range: [start, end) for splitting tasks, however, task_range: {task_range}'
        num_tasks = task_range[1] - task_range[0]
        progress_bar = tqdm(range(task_range[0], task_range[1]), total=num_tasks)

    print(f"#################### Use benchmark: {libero_benchmark}, num_tasks: {num_tasks} #############")
    model = WebsocketClientPolicy(port=port)
    env_edit, env_edit_source = _load_env_edit_spec(env_edit_spec)
    obs_occlusion, obs_occlusion_source = _load_obs_occlusion_spec(obs_occlusion_spec)
    requested_episodes = (
        list(range(int(test_num)))
        if trial_ids is None
        else [int(value) for value in trial_ids]
    )
    if (
        not requested_episodes
        or len(set(requested_episodes)) != len(requested_episodes)
        or any(value < 0 or value >= int(test_num) for value in requested_episodes)
    ):
        raise ValueError(
            f"Invalid trial_ids={trial_ids!r} for test_num={int(test_num)}."
        )

    for task_idx in progress_bar:
        existing_results = (
            _existing_episode_results(out_dir, libero_benchmark, task_idx, test_num)
            if _as_bool(resume_existing)
            else {}
        )
        succ_num = float(sum(1 for success in existing_results.values() if success))
        completed_num = len(existing_results)
        if shard_index is None:
            assigned_episodes = requested_episodes
        else:
            if int(num_shards) <= 0:
                raise ValueError(f"num_shards must be positive, got {num_shards}")
            if not 0 <= int(shard_index) < int(num_shards):
                raise ValueError(
                    f"shard_index must be in [0, {num_shards}), got {shard_index}"
                )
            # Globally shard the task x trial Cartesian product.  With 10 tasks
            # and 10 trials this yields 16--17 mutually exclusive episodes per
            # worker for six workers, unlike task-only splitting (2+2+2+2+1+1).
            assigned_episodes = [
                episode_idx
                for episode_idx in requested_episodes
                if (int(task_idx) * int(test_num) + int(episode_idx)) % int(num_shards)
                == int(shard_index)
            ]
        episode_list = [
            episode_idx
            for episode_idx in assigned_episodes
            if episode_idx not in existing_results
        ]
        if existing_results:
            print(
                f"Resume existing results for task {task_idx}: "
                f"{succ_num:g}/{completed_num} done, remaining {len(episode_list)}"
            )

        for episode_idx in tqdm(episode_list, total=len(episode_list)):
            res_i = run_one(
                model,
                libero_benchmark,
                task_idx,
                out_dir,
                episode_idx,
                task_descriptions_file,
                use_task_descriptions,
                env_edit,
                env_edit_source,
                flat_video_path,
                prompt_override,
                obs_occlusion,
                obs_occlusion_source,
            )
            succ_num += res_i["success"]
            completed_num += 1
            succ_rate = succ_num / completed_num
            print(f"Success rate: {succ_rate}, success num: {succ_num}, total num: {completed_num}")
            summary_suffix = (
                ""
                if shard_index is None
                else f"_shard{int(shard_index):02d}-of-{int(num_shards):02d}"
            )
            out_file = Path(out_dir) / f"{libero_benchmark}_{task_idx}{summary_suffix}.json"
            out_file.parent.mkdir(exist_ok=True, parents=True)
            write_json({
                "succ_num": succ_num,
                "total_num": float(completed_num),
                "succ_rate": succ_rate,
                "prompt": res_i["prompt"],
                "original_prompt": res_i["original_prompt"],
                "modified_instruction": res_i["modified_instruction"],
                "prompt_override": res_i["prompt_override"],
                "task_descriptions_source": res_i["task_descriptions_source"],
                "env_edit_source": res_i["env_edit_source"],
                "obs_occlusion_source": res_i["obs_occlusion_source"],
                "obs_occlusion": res_i["obs_occlusion"],
                "cross_task_donor_benchmark": res_i.get("cross_task_donor_benchmark", ""),
                "cross_task_donor_task_idx": res_i.get("cross_task_donor_task_idx"),
                "cross_task_donor_prompt": res_i.get("cross_task_donor_prompt", ""),
                "video_path": res_i["video_path"],
                "masked_video_path": res_i["masked_video_path"],
                "predicted_video_dirs": res_i.get("predicted_video_dirs", []),
                "predicted_video_paths": res_i.get("predicted_video_paths", []),
                }, out_file
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--libero-benchmark",
        type=str,
        default="libero_10",
        choices=["libero_10", "libero_goal", "libero_spatial", "libero_object", "libero_90"],
        help="Benchmark name",
    )
    parser.add_argument(
        "--task-range",
        type=int,
        nargs="+",
        default=[0, 10],
        help="Task range [start, end) for splitting tasks",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=23908,
        help="WebSocket port",
    )
    parser.add_argument(
        "--test-num",
        type=int,
        default=50,
        help="Number of test episodes",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="outputs/libero",
        help="Output directory for results",
    )
    parser.add_argument(
        "--task-descriptions-file",
        type=str,
        default="",
        help="Path to TaskDescriptions/libero_tasks.json. Non-empty modified_instruction values override benchmark language.",
    )
    parser.add_argument(
        "--use-task-descriptions",
        type=str,
        default="true",
        help="Whether to use modified_instruction from --task-descriptions-file.",
    )
    parser.add_argument(
        "--env-edit-spec",
        type=str,
        default="",
        help="Path to JSON env-edit spec or inline JSON. XML overrides are applied during env construction; other edits are applied after set_init_state.",
    )
    parser.add_argument(
        "--flat-video-path",
        type=str,
        default="",
        help="Optional exact mp4 path for the single rollout video.",
    )
    parser.add_argument(
        "--prompt-override",
        type=str,
        default=None,
        help="Use this exact prompt instead of the benchmark or TaskDescriptions instruction. Empty string is valid.",
    )
    parser.add_argument(
        "--obs-occlusion-spec",
        type=str,
        default="",
        help="Path to JSON observation-occlusion spec or inline JSON. Applies only to model input observations.",
    )
    parser.add_argument(
        "--resume-existing",
        type=str,
        default="false",
        help="Skip rollout videos already present in out-dir and include them in success counts.",
    )
    parser.add_argument(
        "--shard-index",
        type=int,
        default=None,
        help="Optional zero-based worker shard for the task x trial Cartesian product.",
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=1,
        help="Number of mutually exclusive task x trial shards.",
    )
    parser.add_argument(
        "--trial-ids",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Optional exact zero-based trials to evaluate. Values remain bounded "
            "by --test-num, so trial 7 uses official init_state_index 7 without "
            "executing trials 0-6."
        ),
    )
    args = parser.parse_args()
    run(**vars(args))
    print("Finish all process!!!!!!!!!!!!")


if __name__ == "__main__":
    main()
