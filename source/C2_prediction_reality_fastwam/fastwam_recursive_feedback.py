#!/usr/bin/env python3
"""FastWAM evaluator for C2 recursive RGB self-prediction feedback.

The bundled reference evaluator is adapted within this process. Hydra arguments
use the same interface as ``eval_libero_single.py``.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import hydra

RELEASE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RELEASE))
from libero_vpro.runtime import resolve_runtime
FASTWAM_ROOT = resolve_runtime(require=False).fastwam
from source.fastwam_reference import eval_libero_single as base  # noqa: E402

from source.C2_prediction_reality_fastwam.feedback_frames import (  # noqa: E402
    frame_sha256,
    resize_rgb,
    select_feedback_frame,
    split_stitched_views,
)


@dataclass
class FeedbackSession:
    episode_index: int
    pending_frame: Optional[Any] = None
    pending_alignment: Optional[dict[str, Any]] = None
    decision_index: int = 0
    trace: list[dict[str, Any]] = field(default_factory=list)


_SESSION: Optional[FeedbackSession] = None
_ORIGINAL_PREDICT = base._predict_action_chunk
_ORIGINAL_OBS_TO_INPUT = base._obs_to_model_input
_ORIGINAL_RUN_EPISODE = base.run_single_episode


def _camera_shapes(processor) -> list[tuple[int, int]]:
    count = int(processor.num_output_cameras)
    if count != 2:
        raise ValueError(
            "C2 FastWAM requires the two-camera checkpoint/config; "
            f"num_output_cameras={count}"
        )
    shapes = []
    for meta in processor.shape_meta["images"][:count]:
        shape = meta["shape"]
        shapes.append((int(shape[1]), int(shape[2])))
    return shapes


def _tensor_and_views(combined, *, cfg, processor, device, dtype):
    shapes = _camera_shapes(processor)
    concatenation = str(cfg.data.train.get("concat_multi_camera", "horizontal"))
    views = split_stitched_views(combined, shapes, concatenation)
    stitched = (
        np.concatenate(views, axis=1)
        if concatenation == "horizontal"
        else np.concatenate(views, axis=0)
    )
    tensor = torch.as_tensor(stitched).permute(2, 0, 1).unsqueeze(0).to(
        device=device, dtype=dtype
    )
    tensor = tensor * (2.0 / 255.0) - 1.0
    return tensor, {"image": views[0], "wrist_image": views[1]}


def _patched_obs_to_model_input(obs, cfg, processor, width, height, device, dtype):
    if _SESSION is None:
        return _ORIGINAL_OBS_TO_INPUT(obs, cfg, processor, width, height, device, dtype)

    if _SESSION.pending_frame is None:
        clean_tensor, proprio, record_views = _ORIGINAL_OBS_TO_INPUT(
            obs, cfg, processor, width, height, device, dtype
        )
        clean_rgb = (
            (clean_tensor[0].detach().float().cpu().permute(1, 2, 0).numpy() + 1.0)
            * 127.5
        ).round().clip(0, 255).astype(np.uint8)
        tensor = clean_tensor
        source = {
            "visual_source": "simulator",
            "source_replan_index": None,
            "feedback_frame_sha256": frame_sha256(clean_rgb),
        }
    else:
        tensor, model_resolution_views = _tensor_and_views(
            _SESSION.pending_frame,
            cfg=cfg,
            processor=processor,
            device=device,
            dtype=dtype,
        )
        proprio = base._normalize_proprio(base._extract_sim_state(obs), processor)
        # The base recorder stores model observations in LIBERO's camera
        # resolution (128x128) and later stacks them with realized simulator
        # frames. Keep that storage contract while the actual tensor above stays
        # at FastWAM's native 224x224-per-view resolution.
        simulator_views = base.get_libero_image(obs)
        record_views = {
            key: resize_rgb(
                model_resolution_views[key],
                int(simulator_views[key].shape[0]),
                int(simulator_views[key].shape[1]),
            )
            for key in ("image", "wrist_image")
        }
        source = {
            "visual_source": "self_prediction",
            "source_replan_index": int(_SESSION.decision_index - 1),
            "feedback_frame_sha256": frame_sha256(_SESSION.pending_frame),
            **dict(_SESSION.pending_alignment or {}),
        }
    _SESSION.trace.append({"replan_index": int(_SESSION.decision_index), **source})
    return tensor, proprio, record_views


def _patched_predict_action_chunk(*args, **kwargs):
    if _SESSION is None:
        return _ORIGINAL_PREDICT(*args, **kwargs)
    cfg = kwargs.get("cfg")
    if cfg is None:
        raise RuntimeError("C2 wrapper requires cfg as a keyword argument")
    action, images, predicted_frames = _ORIGINAL_PREDICT(*args, **kwargs)
    ratio = int(cfg.data.train.action_video_freq_ratio)
    executed_steps = int(cfg.EVALUATION.get("replan_steps", 10))
    selected, alignment = select_feedback_frame(predicted_frames, ratio, executed_steps)
    _SESSION.pending_frame = selected
    _SESSION.pending_alignment = alignment
    _SESSION.decision_index += 1
    return action, images, predicted_frames


def _trace_path(cfg, episode_idx: int) -> Path:
    suite = str(cfg.EVALUATION.task_suite_name)
    task = int(cfg.EVALUATION.task_id)
    root = Path(str(cfg.EVALUATION.output_dir)) / suite / "self_feedback_traces"
    root.mkdir(parents=True, exist_ok=True)
    return root / f"task{task}_trial{int(episode_idx)}.json"


def _patched_run_single_episode(*args, **kwargs):
    global _SESSION
    # Enable task-aware phase diagnostics for post-step state inspection and
    # recorder annotations. This does not alter actions or simulator observations.
    kwargs = dict(kwargs)
    phase_spec = dict(kwargs.get("decision_schedule_spec") or {})
    phase_spec.update(
        {
            "phase_diagnostics": True,
            "trigger": "always",
            "pre_grasp_band": 0.20,
            "grasp_hold_frames": 24,
            "initial_hold_frames": 24,
            "release_hold_frames": 24,
            "pre_place_progress": 0.70,
        }
    )
    kwargs["decision_schedule_spec"] = phase_spec
    cfg = kwargs.get("cfg")
    episode_idx = int(kwargs.get("episode_idx", -1))
    if cfg is None:
        raise RuntimeError("C2 wrapper requires cfg as a keyword argument")
    if not bool(cfg.EVALUATION.get("visualize_future_video", False)):
        raise ValueError(
            "C2 requires EVALUATION.visualize_future_video=true to provide predicted "
            "RGB frames at every replan."
        )
    _SESSION = FeedbackSession(episode_index=episode_idx)
    try:
        result = list(_ORIGINAL_RUN_EPISODE(*args, **kwargs))
        payload = {
            "schema_version": "wam_syncbench.c2_recursive_self_prediction_phase.v1",
            "model": "FastWAM",
            "condition": "C2-B2_recursive_self_prediction",
            "episode_index": episode_idx,
            "initial_visual_source": "simulator",
            "later_visual_source": "previous_replan_self_prediction",
            "proprioception_source": "simulator_current_state",
            "simulator_rgb_after_initial_passed_to_model": False,
            "prediction_selection": "latest_frame_not_after_next_replan_boundary",
            "phase_detector": {
                "name": "task_aware_v2",
                "source": "shared perturbation_schedule.infer_phase",
                "phase_diagnostics": True,
                "pre_grasp_band": 0.20,
                "grasp_hold_frames": 24,
                "initial_hold_frames": 24,
                "release_hold_frames": 24,
                "pre_place_progress": 0.70,
            },
            "trace": list(_SESSION.trace),
        }
        trace_path = _trace_path(cfg, episode_idx)
        trace_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        diagnostics = dict(result[-1])
        diagnostics["recursive_self_prediction"] = payload
        diagnostics["recursive_self_prediction_trace"] = str(trace_path)
        result[-1] = diagnostics
        return tuple(result)
    finally:
        _SESSION = None


base._obs_to_model_input = _patched_obs_to_model_input
base._predict_action_chunk = _patched_predict_action_chunk
base.run_single_episode = _patched_run_single_episode


@hydra.main(
    version_base="1.3",
    config_path=str(FASTWAM_ROOT / "configs"),
    config_name="sim_libero.yaml",
)
def _isolated_eval_entry(cfg):
    """Call the undecorated evaluator with a config rooted at FastWAM."""
    return base.eval_single_process.__wrapped__(cfg)


if __name__ == "__main__":
    os.environ.setdefault("WAM_TRACK", "C2")
    os.environ.setdefault("WAM_SUBTASK", "recursive_self_prediction")
    os.environ.setdefault("WAM_CONDITION", "recursive_self_prediction")
    _isolated_eval_entry()
