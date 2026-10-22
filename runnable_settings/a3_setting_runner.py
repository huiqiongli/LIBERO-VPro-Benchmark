#!/usr/bin/env python3
"""Run one A3-v26 interaction-cue masking setting.

The bundled ``run_model.py`` uses the four task inventories (40/10/6/13 cases)
and shared world-anchored image blur. The wrapper leaves environment geometry
unchanged.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from runner_common import Selection, add_selection_arguments
from libero_vpro.runtime import prepare_setting_env, resolve_runtime

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "source/A3_interaction_part_occlusion"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--setting", type=Path, required=True)
    p.add_argument("--model", choices=("fastwam",), required=True)
    p.add_argument("--gpu", default="0")
    p.add_argument("--port", type=int, default=0)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--plan", action="store_true")
    add_selection_arguments(p)
    args = p.parse_args()
    selection = Selection.from_args(args)
    setting = _load(args.setting)
    condition = setting["setting_id"].split(".", 1)[1]
    if condition not in {"gripper", "target_contact", "fixture_contact", "bounded_receiver"}:
        raise SystemExit(f"unsupported A3 condition: {condition}")
    manifest = _load(args.setting.parent / setting["manifest"])
    episodes = [row for row in manifest["episodes"] if selection.includes(row)]
    if not episodes:
        raise SystemExit(f"selection matches no episodes in {setting['setting_id']}")
    if args.plan:
        print(json.dumps({
            "setting_id": setting["setting_id"], "model": args.model,
            "source": str(SOURCE / "run_model.py"),
            "cases": len({(row["suite"], row["task_id"]) for row in episodes}),
            "trials_per_case": selection.trials,
            "trial_start": selection.trial_start,
            "episodes": len(episodes),
            "condition": condition,
        }, indent=2))
        return 0

    resolve_runtime(require=True)
    args.output_root.mkdir(parents=True, exist_ok=True)
    env = prepare_setting_env()
    env.update(selection.as_env())
    env.update({
        "CUDA_VISIBLE_DEVICES": str(args.gpu), "MUJOCO_EGL_DEVICE_ID": str(args.gpu),
        "MUJOCO_GL": "egl", "PYOPENGL_PLATFORM": "egl", "PYTHONUNBUFFERED": "1",
        "LIBERO_VPRO_A3_SETTING": setting["setting_id"],
    })
    command = [sys.executable, str(SOURCE / "run_model.py"), "--model", args.model,
               "--gpu", str(args.gpu), "--output-root", str(args.output_root),
               "--conditions", condition, "--trials", str(selection.trials)]
    result = subprocess.run(command, cwd=SOURCE, env=env, check=False)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
