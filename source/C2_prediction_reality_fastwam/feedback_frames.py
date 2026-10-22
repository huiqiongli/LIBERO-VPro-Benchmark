"""Pure image helpers for C2 recursive self-prediction feedback.

This module intentionally has no model or simulator imports so its layout and
temporal alignment contracts can be checked without reserving a GPU.
"""

from __future__ import annotations

import hashlib
from typing import Mapping, Sequence

import numpy as np
from PIL import Image


def as_rgb_uint8(frame) -> np.ndarray:
    """Convert a predicted frame to contiguous HWC RGB uint8."""
    if isinstance(frame, Image.Image):
        array = np.asarray(frame.convert("RGB"))
    else:
        array = np.asarray(frame)
    if array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError(f"Expected an HWC RGB frame, got shape={array.shape}")
    if array.dtype != np.uint8:
        if np.issubdtype(array.dtype, np.floating) and array.size:
            max_value = float(np.nanmax(array))
            min_value = float(np.nanmin(array))
            if min_value >= -1.01 and max_value <= 1.01:
                array = (array + 1.0) * 127.5 if min_value < 0.0 else array * 255.0
        array = np.clip(array, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(array)


def resize_rgb(frame, height: int, width: int) -> np.ndarray:
    array = as_rgb_uint8(frame)
    if array.shape[:2] != (int(height), int(width)):
        array = np.asarray(
            Image.fromarray(array).resize((int(width), int(height)), Image.Resampling.BILINEAR)
        )
    return np.ascontiguousarray(array)


def split_stitched_views(
    frame,
    view_shapes: Sequence[tuple[int, int]],
    concatenation: str = "horizontal",
) -> list[np.ndarray]:
    """Split a model-space stitched frame into ordered camera views.

    ``view_shapes`` contains ``(height, width)`` for each camera. The stitched
    frame is first resized to the exact aggregate model input size. This keeps
    the feedback tensor on the same spatial contract as the clean inference.
    """
    if not view_shapes:
        raise ValueError("view_shapes must contain at least one camera")
    shapes = [(int(h), int(w)) for h, w in view_shapes]
    if any(h <= 0 or w <= 0 for h, w in shapes):
        raise ValueError(f"Invalid view_shapes={shapes}")

    mode = str(concatenation).lower()
    if mode == "horizontal":
        if len({h for h, _ in shapes}) != 1:
            raise ValueError(f"Horizontal views must share height, got {shapes}")
        total_h, total_w = shapes[0][0], sum(w for _, w in shapes)
        stitched = resize_rgb(frame, total_h, total_w)
        output = []
        left = 0
        for height, width in shapes:
            output.append(np.ascontiguousarray(stitched[:height, left:left + width]))
            left += width
        return output
    if mode == "vertical":
        if len({w for _, w in shapes}) != 1:
            raise ValueError(f"Vertical views must share width, got {shapes}")
        total_h, total_w = sum(h for h, _ in shapes), shapes[0][1]
        stitched = resize_rgb(frame, total_h, total_w)
        output = []
        top = 0
        for height, width in shapes:
            output.append(np.ascontiguousarray(stitched[top:top + height, :width]))
            top += height
        return output
    raise ValueError(f"Unsupported camera concatenation: {concatenation!r}")


def split_lingbot_views(
    frame,
    model_observation: Mapping[str, np.ndarray],
    ordered_model_keys: Sequence[str],
) -> dict[str, np.ndarray]:
    """Split LingBot's horizontal decoded canvas instead of duplicating it.

    LingBot encodes the two 128x128 views by concatenating their VAE latents
    along width; decoding therefore returns a 128x256 RGB canvas. This function
    separates that canvas into the two camera-view crops.
    """
    keys = [key for key in ordered_model_keys if key in model_observation]
    if not keys:
        raise KeyError("No expected LingBot visual keys are present")
    shapes = [tuple(map(int, model_observation[key].shape[:2])) for key in keys]
    crops = split_stitched_views(frame, shapes, concatenation="horizontal")
    output = dict(model_observation)
    for key, crop in zip(keys, crops):
        output[key] = crop.astype(model_observation[key].dtype, copy=False)
    return output


def frame_sha256(frame) -> str:
    return hashlib.sha256(as_rgb_uint8(frame).tobytes()).hexdigest()


def select_feedback_frame(predicted_frames, action_video_freq_ratio: int, executed_steps: int):
    """Select the latest prediction not later than the next replan boundary.

    Frame zero is the current/conditioning frame. Future frame ``i`` represents
    control offset ``i * action_video_freq_ratio``. Using the latest non-future
    offset avoids feeding a frame that depicts a state beyond the actions that
    the simulator has actually executed.
    """
    frames = list(predicted_frames or [])
    if not frames:
        raise RuntimeError("The model returned no predicted RGB frames for feedback")
    ratio = int(action_video_freq_ratio)
    steps = int(executed_steps)
    if ratio <= 0 or steps <= 0:
        raise ValueError(f"Invalid ratio/steps: ratio={ratio}, executed_steps={steps}")
    index = min(len(frames) - 1, steps // ratio)
    offset = index * ratio
    return frames[index], {
        "prediction_frame_index": int(index),
        "prediction_control_offset": int(offset),
        "next_replan_control_offset": int(steps),
        "alignment_error_steps": int(steps - offset),
        "fallback": "hold_latest_prediction" if offset != steps else "none",
    }
