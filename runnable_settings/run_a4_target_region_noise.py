#!/usr/bin/env python3
"""Run target-region Gaussian noise at its accepted [2, 5, 8] schedule."""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

from runner_common import Selection, add_selection_arguments
from libero_vpro.runtime import prepare_setting_env, resolve_runtime

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--setting", required=True)
    parser.add_argument("--model", choices=("fastwam",), required=True)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--plan", action="store_true")
    add_selection_arguments(parser)
    args = parser.parse_args()
    selection = Selection.from_args(args)
    if args.setting != "A4.target_region_noise":
        raise SystemExit(f"unsupported target-noise setting: {args.setting}")
    root = Path(__file__).resolve().parent.parent / "source/A4_temporary_observation_corruption"
    output = Path(os.environ.get(
        "A4_TARGET_NOISE_OUTPUT_ROOT",
        str(Path(__file__).resolve().parents[1] / "outputs/A4_target_region_noise"),
    ))
    script = root / "run_fastwam_a4_target_region_noise_suite.sh"
    status = 0
    for n in (3,):
        for suite in selection.suites():
            env = prepare_setting_env()
            env.update(selection.as_env())
            env["A4_TARGET_NOISE_OUTPUT_ROOT"] = str(output)
            command = ["bash", str(script), args.gpu, suite, str(n), str(selection.trials)]
            if args.plan:
                print(" ".join(command))
                continue
            resolve_runtime(require=True)
            result = subprocess.run(
                command,
                env=env,
                check=False,
            )
            status |= int(result.returncode != 0)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
