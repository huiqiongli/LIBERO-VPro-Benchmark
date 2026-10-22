"""Decision-boundary B1 camera staleness for LIBERO-VPro.

The buffer is deliberately model agnostic.  Call :meth:`apply_replan` exactly
once when a policy is about to request a new action chunk.  Clean frames are
always appended to history before the requested delay/freeze/replay transform
is selected, so stale inputs never contaminate future history.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np


VIEW_ALIASES = {
    "agentview": "agentview",
    "image": "agentview",
    "primary_image": "agentview",
    "third_person": "agentview",
    "robot0_eye_in_hand": "wrist",
    "robot0_eye_in_hand_image": "wrist",
    "wrist": "wrist",
    "wrist_image": "wrist",
    "eye_in_hand": "wrist",
}


class CameraStalenessError(ValueError):
    """Raised when a B1 camera schedule is malformed or misapplied."""


def _canonical_view(name: str) -> str:
    key = str(name).strip().lower()
    if key not in VIEW_ALIASES:
        raise CameraStalenessError(f"Unsupported B1 camera view: {name!r}")
    return VIEW_ALIASES[key]


def _validate_view_config(view: str, raw: Mapping[str, Any]) -> dict[str, Any]:
    cfg = dict(raw)
    mode = str(cfg.get("mode", "")).strip().lower()
    if mode not in {"delay", "freeze", "replay"}:
        raise CameraStalenessError(
            f"Unsupported B1 mode for {view}: {mode!r}; expected delay/freeze/replay."
        )
    cfg["mode"] = mode
    if mode == "delay":
        lag = int(cfg.get("lag", -1))
        if lag < 1:
            raise CameraStalenessError(f"B1 delay lag must be >=1, got {lag}.")
        cfg["lag"] = lag
    elif mode == "freeze":
        freeze_at = int(cfg.get("freeze_at", -1))
        if freeze_at < 0:
            raise CameraStalenessError(
                f"B1 freeze_at must be >=0, got {freeze_at}."
            )
        cfg["freeze_at"] = freeze_at
    else:
        replay_from = int(cfg.get("replay_from", -1))
        replay_length = int(cfg.get("replay_length", 0))
        if replay_from < 0 or replay_length < 1:
            raise CameraStalenessError(
                "B1 replay requires replay_from>=0 and replay_length>=1, "
                f"got replay_from={replay_from}, replay_length={replay_length}."
            )
        cfg["replay_from"] = replay_from
        cfg["replay_length"] = replay_length
    return cfg


class PerViewReplanBuffer:
    """Apply one-clean/one-stale B1 transforms at policy replans."""

    def __init__(self, schedule_spec: Mapping[str, Any] | None) -> None:
        spec = dict(schedule_spec or {})
        enabled_raw = spec.get("enabled", True)
        self.enabled = bool(enabled_raw) and isinstance(spec.get("camera"), Mapping)
        self.spec: dict[str, dict[str, Any]] = {}
        if self.enabled:
            for raw_view, raw_cfg in dict(spec["camera"]).items():
                if not isinstance(raw_cfg, Mapping):
                    raise CameraStalenessError(
                        f"B1 configuration for {raw_view!r} must be an object."
                    )
                view = _canonical_view(raw_view)
                if view in self.spec:
                    raise CameraStalenessError(f"Duplicate B1 view configuration: {view}")
                self.spec[view] = _validate_view_config(view, raw_cfg)
            if len(self.spec) != 1:
                raise CameraStalenessError(
                    "B1 requires exactly one stale camera and one clean camera; "
                    f"got configured views={sorted(self.spec)}."
                )
        self._history: dict[str, list[np.ndarray]] = {
            "agentview": [],
            "wrist": [],
        }
        self.trace: list[dict[str, Any]] = []

    @staticmethod
    def _source_index(current: int, cfg: Mapping[str, Any] | None) -> int:
        if not cfg:
            return current
        mode = str(cfg["mode"])
        if mode == "delay":
            return max(0, current - int(cfg["lag"]))
        if mode == "freeze":
            return min(current, int(cfg["freeze_at"]))
        start = int(cfg["replay_from"])
        length = int(cfg["replay_length"])
        if current < start + length:
            return current
        return start + ((current - start) % length)

    def apply_replan(
        self,
        replan_index: int,
        *,
        agentview: np.ndarray,
        wrist: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
        """Return model frames and an auditable trace for this replan."""
        replan = int(replan_index)
        expected = len(self._history["agentview"])
        if replan != expected:
            raise CameraStalenessError(
                f"Non-sequential B1 replan: got {replan}, expected {expected}."
            )
        self._history["agentview"].append(np.array(agentview, copy=True))
        self._history["wrist"].append(np.array(wrist, copy=True))

        output: dict[str, np.ndarray] = {}
        entries: list[dict[str, Any]] = []
        for view in ("agentview", "wrist"):
            cfg = self.spec.get(view) if self.enabled else None
            source = self._source_index(replan, cfg)
            output[view] = np.array(self._history[view][source], copy=True)
            entry = {
                "replan": replan,
                "view": view,
                "mode": "clean" if cfg is None else str(cfg["mode"]),
                "fed_replan": int(source),
                "staleness": int(replan - source),
                "perturbed": bool(cfg is not None and source != replan),
            }
            entries.append(entry)
            self.trace.append(dict(entry))
        return output["agentview"], output["wrist"], entries

