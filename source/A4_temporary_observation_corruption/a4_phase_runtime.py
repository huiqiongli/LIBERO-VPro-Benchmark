"""A4 phase-targeted, one-shot corruption gate.

The phase label is produced by the existing shared phase detector, but the gate is
A4-specific: first entry into a requested phase opens one contiguous window of
exactly three policy replans and can never retrigger in the episode.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
from typing import Any

from perturbation_schedule import infer_phase, reset_phase_state


PHASES = ("initial", "reach", "pre_grasp", "grasp", "transport", "pre_place", "release")


@dataclass
class C1PhasePulse:
    target_phase: str
    replan_count: int = 3
    phase: str = "initial"
    grasped_latch: bool = False
    start_replan: int | None = None
    target_pending: bool = False
    trace: list[dict[str, Any]] = field(default_factory=list)
    pre_grasp_band: float | None = None
    grasp_hold_frames: int | None = None
    initial_hold_frames: int | None = None
    release_hold_frames: int | None = None
    pre_place_progress: float | None = None

    def __post_init__(self) -> None:
        if self.target_phase not in PHASES:
            raise ValueError(f"Unsupported A4 target phase: {self.target_phase}")
        self.replan_count = max(1, int(self.replan_count))
        schedule = {}
        schedule_path = os.environ.get("LIBERO_SCHEDULE_SPEC", "").strip()
        if schedule_path:
            try:
                schedule = json.loads(Path(schedule_path).read_text())
            except Exception:
                schedule = {}
        self.pre_grasp_band = float(
            schedule.get("pre_grasp_band", 0.20)
            if self.pre_grasp_band is None else self.pre_grasp_band
        )
        self.grasp_hold_frames = int(
            schedule.get("grasp_hold_frames", 24)
            if self.grasp_hold_frames is None else self.grasp_hold_frames
        )
        self.initial_hold_frames = int(
            schedule.get("initial_hold_frames", 24)
            if self.initial_hold_frames is None else self.initial_hold_frames
        )
        self.release_hold_frames = int(
            schedule.get("release_hold_frames", 24)
            if self.release_hold_frames is None else self.release_hold_frames
        )
        self.pre_place_progress = float(
            schedule.get("pre_place_progress", 0.70)
            if self.pre_place_progress is None else self.pre_place_progress
        )
        reset_phase_state()

    def active(self, replan_index: int) -> bool:
        replan_index = int(replan_index)
        # ``update`` runs at every control step, while ``active`` is sampled only
        # at policy-replan boundaries.  Latch a target phase that occurs wholly
        # inside an action chunk so a short Transport/Grasp phase cannot vanish
        # before the next model decision.
        if self.start_replan is None and (
            self.phase == self.target_phase or self.target_pending
        ):
            self.start_replan = replan_index
            self.target_pending = False
        active = self.start_replan is not None and self.start_replan <= replan_index < self.start_replan + self.replan_count
        self.trace.append({"replan_index": replan_index, "phase": self.phase, "active": bool(active)})
        return bool(active)

    def update(self, env: Any, *, gripper_cmd: float | None = None, target_objects: list[str] | None = None) -> str:
        self.phase, self.grasped_latch = infer_phase(
            env,
            target_objects,
            prev_phase=self.phase,
            grasped_latch=self.grasped_latch,
            gripper_cmd=gripper_cmd,
            pre_grasp_band=self.pre_grasp_band,
            grasp_hold_frames=self.grasp_hold_frames,
            initial_hold_frames=self.initial_hold_frames,
            release_hold_frames=self.release_hold_frames,
            pre_place_progress=self.pre_place_progress,
        )
        if self.start_replan is None and self.phase == self.target_phase:
            self.target_pending = True
        return self.phase


def target_objects_from_env(env: Any) -> list[str]:
    names = list(getattr(env, "obj_of_interest", []) or [])
    if names:
        return names
    base = getattr(env, "env", env)
    return list(getattr(base, "obj_of_interest", []) or [])
