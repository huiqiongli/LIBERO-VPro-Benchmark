#!/usr/bin/env python3
"""Run the C2 recursive-feedback setting over its complete 40-case matrix."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess

from runner_common import Selection, add_selection_arguments
from libero_vpro.runtime import prepare_setting_env, resolve_runtime

HERE = Path(__file__).resolve().parent
RELEASE = HERE.parent


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--setting", required=True)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--output-root", type=Path, required=True)
    add_selection_arguments(parser)
    parser.add_argument("--plan", action="store_true")
    args = parser.parse_args()
    selection = Selection.from_args(args)
    if args.setting != "C2.recursive_self_prediction.all_steps":
        raise SystemExit(f"unsupported FastWAM C2 setting: {args.setting}")
    suites = selection.suites()
    runner = RELEASE / "source/C2_prediction_reality_fastwam/run_fastwam_suite.sh"
    env = prepare_setting_env()
    env.update(selection.as_env())
    if args.plan:
        for suite in suites:
            for task in selection.tasks():
                print(f"bash {runner} {suite} {args.gpu} # task={task} trials={selection.trials} trial_start={selection.trial_start}")
        print(f"planned cases={len(suites) * len(selection.tasks())} trials_per_case={selection.trials} setting={args.setting}")
        return 0
    resolve_runtime(require=True)
    for suite in suites:
        for task in selection.tasks():
            task_output = args.output_root / suite / f"task_{task:02d}"
            task_output.mkdir(parents=True, exist_ok=True)
            env.update({
                "C2_B2_OUTPUT_DIR": str(task_output),
                "C2_B2_TASK_ID": str(task),
                "C2_B2_TRIAL_START": str(selection.trial_start),
                "C2_B2_TRIAL_COUNT": str(selection.trials),
            })
            result = subprocess.run(["bash", str(runner), suite, args.gpu], env=env, check=False)
            if result.returncode:
                return result.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
