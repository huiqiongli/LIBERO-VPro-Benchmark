#!/usr/bin/env python3
"""Validate the six one-trial FastWAM duration smoke rollouts."""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2] / "smoke_outputs"
OUT = ROOT / "a4_duration_fastwam_smoke.json"


def main() -> None:
    rows = []
    for kind in ("black", "noise"):
        for duration in (2, 3):
            setting = f"{kind}_D{duration}"
            files = sorted((ROOT / f"fastwam_{setting}").glob("episode_records/*/episode.json"))
            assert len(files) == 1, f"expected one episode for {setting}, found {len(files)}"
            episode = json.loads(files[0].read_text(encoding="utf-8"))
            schedule = episode["metadata"]["perturbation"]["decision_schedule"]
            active = [row["replan_index"] for row in episode["replan_steps"]
                      if row["perturbation"].get("active")]
            expected = list(range(2, 2 + duration))
            variant = episode["metadata"]["perturbation"]["observation"]["variant"]
            assert episode["status"] == "complete"
            assert active == expected, (setting, active, expected)
            assert variant == ("fullscreen_black" if kind == "black" else "fullscreen_noise")
            rows.append({"setting": setting, "status": episode["status"],
                         "observation_variant": variant, "active_replans": active,
                         "expected_active_replans": expected,
                         "schedule": schedule["scenario"]})
    OUT.write_text(json.dumps({"status": "pass", "rollouts": rows}, indent=2) + "\n",
                   encoding="utf-8")
    print(json.dumps({"status": "pass", "rollouts": rows}, indent=2))


if __name__ == "__main__":
    main()
