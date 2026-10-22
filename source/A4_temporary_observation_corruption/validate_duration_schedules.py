#!/usr/bin/env python3
"""Smoke-check the contiguous A4 duration schedules.

This check is deliberately model-independent: it verifies the schedule contract
that every backend consumes before a GPU rollout is started.  D1/D2/D3 must
activate consecutive replans beginning at replan 2, unlike the scattered N1/N2/N3
configurations.
"""
from __future__ import annotations

import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
OUT = HERE.parents[2] / "smoke_outputs" / "a4_duration_schedule_smoke.json"


def active_replans(spec: dict, horizon: int = 8) -> list[int]:
    assert spec.get("trigger") == "replan_window"
    window = spec["replan_window"]
    start = int(window["start_replan"])
    count = int(window["num_replans"])
    return [idx for idx in range(horizon) if start <= idx < start + count]


def main() -> None:
    rows = []
    for duration in (1, 2, 3):
        path = HERE / "configs" / f"schedule_D{duration}.json"
        spec = json.loads(path.read_text(encoding="utf-8"))
        observed = active_replans(spec)
        expected = list(range(2, 2 + duration))
        assert observed == expected, (path, observed, expected)
        rows.append({"schedule": path.name, "active_replans": observed,
                     "expected": expected, "contiguous": True})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"status": "pass", "checks": rows}, indent=2) + "\n",
                   encoding="utf-8")
    print(json.dumps({"status": "pass", "checks": rows}, indent=2))


if __name__ == "__main__":
    main()
