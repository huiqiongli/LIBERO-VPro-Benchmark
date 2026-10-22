"""Shared, CPU-only selection and manifest handling for reference runners."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import sys

RELEASE = Path(__file__).resolve().parents[1]
if str(RELEASE) not in sys.path:
    sys.path.insert(0, str(RELEASE))

SUITES = ("libero_object", "libero_spatial", "libero_goal", "libero_10")


def add_selection_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--suite", choices=SUITES, default=None)
    parser.add_argument("--task", type=int, choices=range(10), default=None)
    parser.add_argument("--trials", type=int, default=None)
    parser.add_argument("--trial-start", type=int, default=None)


@dataclass(frozen=True)
class Selection:
    suite: str | None = None
    task: int | None = None
    trials: int = 10
    trial_start: int = 0

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "Selection":
        def value(name: str, default=None):
            explicit = getattr(args, name, None)
            return explicit if explicit is not None else os.environ.get(
                "LIBERO_VPRO_" + name.upper(), default
            )
        selected = cls(
            suite=value("suite") or None,
            task=int(value("task")) if value("task") not in (None, "") else None,
            trials=int(value("trials", 10)),
            trial_start=int(value("trial_start", 0)),
        )
        if selected.suite is not None and selected.suite not in SUITES:
            raise ValueError(f"unsupported suite: {selected.suite}")
        if selected.task is not None and selected.task not in range(10):
            raise ValueError("--task must be between 0 and 9")
        if selected.trial_start < 0 or selected.trials < 1 or selected.trial_start + selected.trials > 10:
            raise ValueError("trial selection must satisfy 0 <= trial-start < trial-start + trials <= 10")
        return selected

    def suites(self):
        return (self.suite,) if self.suite is not None else SUITES

    def tasks(self):
        return (self.task,) if self.task is not None else range(10)

    def trial_indices(self):
        return range(self.trial_start, self.trial_start + self.trials)

    def as_env(self) -> dict[str, str]:
        return {
            "LIBERO_VPRO_SUITE": self.suite or "",
            "LIBERO_VPRO_TASK": "" if self.task is None else str(self.task),
            "LIBERO_VPRO_TRIALS": str(self.trials),
            "LIBERO_VPRO_TRIAL_START": str(self.trial_start),
        }

    def includes(self, row: dict) -> bool:
        return (
            (self.suite is None or row.get("suite") == self.suite)
            and (self.task is None or int(row.get("task_id", -1)) == self.task)
            and int(row.get("trial_index", 0)) in self.trial_indices()
        )


def selected_manifest(source: Path, output: Path, selection: Selection, *, dry_run: bool) -> Path:
    """Write an execution-only subset outside the release; never mutate its inventory."""
    original = json.loads(source.read_text(encoding="utf-8"))
    payload = dict(original)
    keys = [key for key in ("cases", "episodes") if isinstance(original.get(key), list)]
    if not keys:
        raise ValueError(f"manifest has no episode rows: {source}")
    changed = False
    for key in keys:
        rows = [row for row in original[key] if selection.includes(row)]
        if not rows:
            raise ValueError(f"selection matches no episodes in {source}")
        payload[key] = rows
        changed = changed or len(rows) != len(original[key])
    if not changed:
        return source
    payload["trials_per_case"] = selection.trials
    payload["trial_start"] = selection.trial_start
    payload["selection"] = selection.as_env()
    target = output / "_manifests" / source.name
    if not dry_run:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return target
