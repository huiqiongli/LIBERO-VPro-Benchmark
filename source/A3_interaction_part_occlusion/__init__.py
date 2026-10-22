"""A3 cross-view-consistent interaction-part perturbations."""

import json
import os
from pathlib import Path
from typing import Any

from .a3_blur import A3WorldBlurEnv
from .a3_physical import A3PhysicalOcclusionEnv


def maybe_wrap_env(
    env: Any,
    *,
    suite: str,
    task_idx: int,
    spec_path: str | os.PathLike[str] | None = None,
) -> Any:
    """Select the explicitly requested A3 implementation from its spec."""

    selected = str(spec_path or os.environ.get("WAM_A3_SPEC", "")).strip()
    if not selected:
        return env
    if os.environ.get("WAM_A5_VIEW_FILL", "").strip():
        raise RuntimeError("A3 and A5 view fill cannot be enabled together")
    selected_path = Path(selected).expanduser().resolve()
    payload = json.loads(selected_path.read_text(encoding="utf-8"))
    if payload.get("occlusion_space") == "world_anchored_camera_blur":
        if isinstance(env, A3WorldBlurEnv):
            raise RuntimeError("environment is already wrapped by A3WorldBlurEnv")
        return A3WorldBlurEnv(
            env, spec_path=selected_path, suite=suite, task_idx=task_idx
        )
    if isinstance(env, A3PhysicalOcclusionEnv):
        raise RuntimeError("environment is already wrapped by A3PhysicalOcclusionEnv")
    return A3PhysicalOcclusionEnv(
        env, spec_path=selected_path, suite=suite, task_idx=task_idx
    )


__all__ = ["A3PhysicalOcclusionEnv", "A3WorldBlurEnv", "maybe_wrap_env"]
