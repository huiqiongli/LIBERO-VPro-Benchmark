#!/usr/bin/env python3
"""Run one D1 FastWAM setting over the four LIBERO suites."""

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
    category, condition = args.setting.split(".", 1)
    if category != "D1":
        raise SystemExit(f"expected a D1 setting, got {args.setting}")
    suites = selection.suites()
    runner = RELEASE / "source/D1_4_suites_full/run_fastwam_suite_condition.sh"
    if not runner.is_file():
        raise SystemExit(f"missing FastWAM D1 runner: {runner}")
    env = prepare_setting_env()
    env.update(selection.as_env())
    env.update({
        "D1_MODEL_EVAL_ROOT": str(args.output_root),
        "D1_TRIAL_START": str(selection.trial_start),
        "D1_TRIAL_COUNT": str(selection.trials),
        "D1_RUN_TAG": args.setting,
    })
    if args.plan:
        for suite in suites:
            print(f"bash {runner} {suite} {condition} {args.gpu}")
        print(f"planned suites={len(suites)} tasks_per_suite={len(selection.tasks())} "
              f"trials_per_case={selection.trials} trial_start={selection.trial_start} setting={args.setting}")
        return 0
    resolve_runtime(require=True)
    for suite in suites:
        result = subprocess.run(["bash", str(runner), suite, condition, str(args.gpu)],
                                env=env, check=False)
        if result.returncode:
            return result.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
