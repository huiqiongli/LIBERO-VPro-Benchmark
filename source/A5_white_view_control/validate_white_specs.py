#!/usr/bin/env python3

import json
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path("/home/huiqiongli/Sim")
SPEC_ROOT = PROJECT_ROOT / "reviewed_script" / "A5_white_view_control" / "specs"
sys.path.insert(0, str(PROJECT_ROOT))

import robust_libero_perturbations as perturb  # noqa: E402


def validate(spec_name: str, white_key: str, preserved_key: str) -> None:
    spec = json.loads((SPEC_ROOT / spec_name).read_text(encoding="utf-8"))
    raw_obs = {
        "agentview_image": np.full((32, 48, 3), 37, dtype=np.uint8),
        "robot0_eye_in_hand_image": np.full((32, 48, 3), 193, dtype=np.uint8),
    }
    output, stats = perturb.apply_obs_noise_to_raw_obs(
        raw_obs,
        env_in=None,
        spec=spec,
        libero_benchmark="libero_10",
        task_idx=0,
    )
    if not stats.get("enabled"):
        raise AssertionError(f"{spec_name}: perturbation was not enabled")
    if not np.all(output[white_key] == 255):
        raise AssertionError(f"{spec_name}: filled view is not all-white")
    if not np.array_equal(output[preserved_key], raw_obs[preserved_key]):
        raise AssertionError(f"{spec_name}: preserved view was modified")
    if np.shares_memory(output[white_key], raw_obs[white_key]):
        raise AssertionError(f"{spec_name}: source observation was modified in place")
    print(f"PASS {spec_name}: {white_key}=all_white, {preserved_key}=unchanged")


def main() -> None:
    validate("white_wrist.json", "robot0_eye_in_hand_image", "agentview_image")
    validate("white_agent.json", "agentview_image", "robot0_eye_in_hand_image")


if __name__ == "__main__":
    main()
