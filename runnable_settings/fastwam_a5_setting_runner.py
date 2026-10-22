#!/usr/bin/env python3
"""FastWAM adapter for the eight paper-facing A5 camera-input settings."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys

from runner_common import Selection, add_selection_arguments
from libero_vpro.runtime import prepare_setting_env, resolve_runtime

HERE = Path(__file__).resolve().parent
RELEASE = HERE.parent


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--setting", required=True)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--plan", action="store_true")
    add_selection_arguments(parser)
    args = parser.parse_args()
    selection = Selection.from_args(args)

    setting = args.setting.split(".", 1)[-1]
    transforms: dict[str, tuple[str, str]] = {
        "agent_only_direct": ("single", "agentview"),
        "wrist_only_direct": ("single", "wrist"),
        "agentview_copy_to_dual": ("copy", "agentview"),
        "wrist_copy_to_dual": ("copy", "wrist"),
        "agent_plus_black_wrist": ("fill", "black_wrist"),
        "wrist_plus_black_agent": ("fill", "black_agent"),
        "agent_plus_white_wrist": ("fill", "white_wrist"),
        "wrist_plus_white_agent": ("fill", "white_agent"),
    }
    if setting not in transforms:
        raise SystemExit(f"unsupported A5 setting: {args.setting}")

    suites = selection.suites()
    kind, value = transforms[setting]
    runtime = resolve_runtime(require=not args.plan)
    evaluator, fastwam_root, python = runtime.runner, runtime.fastwam, runtime.python
    checkpoint, stats = runtime.checkpoint, runtime.dataset_stats
    env = prepare_setting_env()
    env.update(selection.as_env())
    env.update({
        "CUDA_VISIBLE_DEVICES": str(args.gpu),
        "MUJOCO_EGL_DEVICE_ID": str(args.gpu),
        "MUJOCO_GL": "egl",
        "PYOPENGL_PLATFORM": "egl",
        "WAM_TRACK": "A5",
        "WAM_SUBTASK": "A5",
        "WAM_CONDITION": "view_input_control",
        "WAM_CASE_ID": f"A5_{setting}",
        "WAM_RECORD_EPISODES": "1",
        "WAM_RECORD_PROFILE": "compact",
        "WAM_RECORD_COMPRESS": "1",
        "WAM_EPISODE_RECORD_ROOT": str(args.output_root / "episode_records"),
        "DIFFSYNTH_DOWNLOAD_SOURCE": os.environ.get("DIFFSYNTH_DOWNLOAD_SOURCE", "modelscope"),
        "DIFFSYNTH_MODEL_BASE_PATH": os.environ.get(
            "DIFFSYNTH_MODEL_BASE_PATH", str(fastwam_root / "checkpoints/diffsyn")
        ),
    })
    extra: list[str] = []
    if kind == "single":
        extra.extend([f"++EVALUATION.single_view={value}",
                      "data.train.processor.num_output_cameras=1",
                      "data.train.video_size=[224,224]"])
        env.pop("FASTWAM_OBS_OCCLUSION_SPEC", None)
        env.pop("LIBERO_OBS_OCCLUSION_SPEC", None)
    elif kind == "copy":
        extra.append(f"++EVALUATION.view_copy={value}")
        env.pop("FASTWAM_OBS_OCCLUSION_SPEC", None)
        env.pop("LIBERO_OBS_OCCLUSION_SPEC", None)
    else:
        fill, view = value.split("_")
        fill_name = "blank" if fill == "black" else "white"
        spec_name = f"{fill_name}_{'wrist' if view == 'wrist' else 'agent'}.json"
        spec_root = RELEASE / f"source/A5_{fill_name}_view_control/specs"
        spec = spec_root / spec_name
        if not spec.is_file():
            raise SystemExit(f"missing A5 view-fill spec: {spec}")
        env["FASTWAM_OBS_OCCLUSION_SPEC"] = str(spec)
        env["LIBERO_OBS_OCCLUSION_SPEC"] = str(spec)

    if not args.plan:
        args.output_root.mkdir(parents=True, exist_ok=True)
    count = 0
    for suite in suites:
        for task in selection.tasks():
            for trial in selection.trial_indices():
                output = args.output_root / suite / f"task_{task:02d}" / f"trial_{trial:02d}"
                command = [str(python), str(evaluator), "task=libero_uncond_2cam224_1e-4",
                           f"ckpt={checkpoint}", f"EVALUATION.dataset_stats_path={stats}",
                           f"EVALUATION.task_suite_name={suite}", f"EVALUATION.task_id={task}",
                           f"++EVALUATION.trial_start={trial}", "EVALUATION.num_trials=1",
                           f"EVALUATION.output_dir={output}", "EVALUATION.device=cuda",
                           "EVALUATION.num_inference_steps=10", "model.skip_dit_load_from_pretrain=true",
                           *extra, f"gpu_id={args.gpu}"]
                if args.plan:
                    if count == 0:
                        print(" ".join(command))
                    count += 1
                    continue
                output.mkdir(parents=True, exist_ok=True)
                result = subprocess.run(command, cwd=fastwam_root, env=env, check=False)
                if result.returncode:
                    return result.returncode
                count += 1
    if args.plan:
        print(f"planned episodes: {count}; setting={args.setting}; suites={','.join(suites)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
