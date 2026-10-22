#!/usr/bin/env python3
"""FastWAM runner for A1/A2 observation-timing settings."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from runner_common import Selection, add_selection_arguments
from libero_vpro.runtime import prepare_setting_env, resolve_runtime
from libero_vpro.validation import validate_setting_inputs

HERE = Path(__file__).resolve().parent
RELEASE = HERE.parent


def episode_plan(output: Path, selection: Selection):
    for suite in selection.suites():
        for task in selection.tasks():
            for trial in selection.trial_indices():
                yield suite, task, trial, output / suite / f"task_{task:02d}" / f"trial_{trial:02d}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", choices=("A1", "A2"), required=True)
    parser.add_argument("--setting", required=True)
    parser.add_argument("--setting-json", type=Path, required=True)
    parser.add_argument("--model", choices=("fastwam",), required=True)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--plan", action="store_true")
    add_selection_arguments(parser)
    args = parser.parse_args()
    selection = Selection.from_args(args)

    setting = json.loads(args.setting_json.read_text(encoding="utf-8"))
    if setting.get("setting_id") != f"{args.category}.{args.setting}":
        raise SystemExit(f"setting JSON does not match {args.category}.{args.setting}")
    errors = validate_setting_inputs(args.setting_json, RELEASE)
    if errors:
        raise SystemExit("invalid timing inputs:\n" + "\n".join(errors))
    schedule = Path(setting["config_files"][0])
    if not schedule.is_absolute():
        schedule = RELEASE / schedule
    observation = Path(setting["observation_spec"])
    if not observation.is_absolute():
        observation = RELEASE / observation
    for path in (schedule, observation):
        if not path.is_file():
            raise SystemExit(f"missing timing input: {path}")

    runtime = resolve_runtime(require=not args.plan)
    fastwam_root, python, evaluator = runtime.fastwam, runtime.python, runtime.runner
    checkpoint, stats = runtime.checkpoint, runtime.dataset_stats
    env = prepare_setting_env()
    env.update(selection.as_env())
    env.update({
        "FASTWAM_OBS_OCCLUSION_SPEC": str(observation),
        "LIBERO_OBS_OCCLUSION_SPEC": str(observation),
        "FASTWAM_DECISION_SCHEDULE_SPEC": str(schedule),
        "LIBERO_SCHEDULE_SPEC": str(schedule),
        "WAM_TRACK": args.category,
        "WAM_SUBTASK": args.setting,
        "WAM_CONDITION": "perturbed",
        "WAM_RECORD_EPISODES": "1",
        "WAM_RECORD_PROFILE": "compact",
        "WAM_RECORD_COMPRESS": "1",
        "WAM_EPISODE_RECORD_ROOT": str(args.output_root / "episode_records"),
        "CUDA_VISIBLE_DEVICES": str(args.gpu),
        "MUJOCO_EGL_DEVICE_ID": str(args.gpu),
        "MUJOCO_GL": "egl",
        "PYOPENGL_PLATFORM": "egl",
    })

    commands = []
    for suite, task, trial, output in episode_plan(args.output_root, selection):
        command = [str(python), str(evaluator), "task=libero_uncond_2cam224_1e-4",
                   f"ckpt={checkpoint}", f"EVALUATION.dataset_stats_path={stats}",
                   f"EVALUATION.task_suite_name={suite}", f"EVALUATION.task_id={task}",
                   f"++EVALUATION.trial_start={trial}", "EVALUATION.num_trials=1",
                   f"EVALUATION.output_dir={output}", "EVALUATION.device=cuda",
                   "EVALUATION.num_inference_steps=10", "model.skip_dit_load_from_pretrain=true",
                   f"gpu_id={args.gpu}"]
        commands.append((command, output))
    if args.plan:
        print(" ".join(commands[0][0]))
        print(f"planned episodes={len(commands)} setting={setting['setting_id']}")
        return 0

    args.output_root.mkdir(parents=True, exist_ok=True)
    for command, output in commands:
        output.mkdir(parents=True, exist_ok=True)
        result = subprocess.run(command, cwd=fastwam_root, env=env, check=False)
        if result.returncode:
            return result.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
