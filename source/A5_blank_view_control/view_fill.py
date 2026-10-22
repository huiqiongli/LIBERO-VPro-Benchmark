"""Shared A5 full-frame blank/white camera controls."""

from __future__ import annotations

import os
from typing import Any, Mapping

import numpy as np


FILLS = {"blank": 0, "white": 255}
VIEWS = ("agentview", "wrist")


def contract(
    fill: str | None = None,
    kept_view: str | None = None,
) -> tuple[str, str, int] | None:
    fill = str(fill if fill is not None else os.getenv("WAM_A5_VIEW_FILL", "")).strip().lower()
    kept_view = str(
        kept_view if kept_view is not None else os.getenv("WAM_A5_KEEP_VIEW", "")
    ).strip().lower()
    if not fill and not kept_view:
        return None
    if fill not in FILLS:
        raise ValueError(f"WAM_A5_VIEW_FILL must be one of {tuple(FILLS)}, got {fill!r}")
    if kept_view not in VIEWS:
        raise ValueError(f"WAM_A5_KEEP_VIEW must be one of {VIEWS}, got {kept_view!r}")
    return fill, kept_view, FILLS[fill]


def case_id(fill: str, kept_view: str) -> str:
    missing = "wrist" if kept_view == "agentview" else "agent"
    kept = "agent" if kept_view == "agentview" else "wrist"
    return f"A5_{kept}_plus_{fill}_{missing}"


def apply_pair(
    agentview: Any,
    wrist: Any,
    *,
    fill: str | None = None,
    kept_view: str | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    agent = np.ascontiguousarray(np.asarray(agentview, dtype=np.uint8))
    wrist_array = np.ascontiguousarray(np.asarray(wrist, dtype=np.uint8))
    selected = contract(fill, kept_view)
    if selected is None:
        return agent, wrist_array, {"enabled": False}

    fill_name, kept, value = selected
    if kept == "agentview":
        wrist_array = np.full_like(wrist_array, value, dtype=np.uint8)
        filled_view = "wrist"
    else:
        agent = np.full_like(agent, value, dtype=np.uint8)
        filled_view = "agentview"
    metadata = {
        "enabled": True,
        "type": "full_frame_constant_fill",
        "fill": fill_name,
        "fill_value_rgb": [value, value, value],
        "kept_view": kept,
        "filled_view": filled_view,
        "mask_fraction": 1.0,
        "valid_camera_slots": 2,
    }
    return agent, wrist_array, metadata


def apply_mapping(
    images: Mapping[str, Any],
    *,
    agent_key: str,
    wrist_key: str,
    fill: str | None = None,
    kept_view: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    output = dict(images)
    agent, wrist, metadata = apply_pair(
        output[agent_key], output[wrist_key], fill=fill, kept_view=kept_view
    )
    output[agent_key] = agent
    output[wrist_key] = wrist
    return output, metadata


def assert_pair(agentview: Any, wrist: Any, *, fill: str, kept_view: str) -> None:
    agent = np.asarray(agentview)
    wrist_array = np.asarray(wrist)
    value = FILLS[fill]
    filled = wrist_array if kept_view == "agentview" else agent
    if filled.size == 0 or not np.all(filled == value):
        raise RuntimeError(
            f"A5 fill invariant failed: fill={fill}, kept_view={kept_view}, "
            f"range=({filled.min(initial=0)}, {filled.max(initial=0)})"
        )
