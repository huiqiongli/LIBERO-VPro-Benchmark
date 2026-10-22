#!/usr/bin/env python3
"""LIBERO-PRO-style unified entry point for one LIBERO-VPro setting."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from libero_vpro.config import ReleaseConfig
from libero_vpro.manifest import load_json, setting_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--setting", required=True,
                        help="setting id such as A4.black_N1 or a setting.json path")
    parser.add_argument("--model", default=None)
    parser.add_argument("--suite", default=None)
    parser.add_argument("--task", type=int, default=None)
    parser.add_argument("--trials", type=int, default=None)
    parser.add_argument("--trial-start", type=int, default=None)
    parser.add_argument("--gpu", default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--config", type=Path,
                        default=ROOT / "evaluation_config.yaml")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--plan", action="store_true")
    args = parser.parse_args()
    config = ReleaseConfig.load(args.config)
    env = config.environment()
    setting = Path(args.setting)
    setting_json = setting if setting.is_file() else setting_path(config, args.setting)
    payload = load_json(setting_json)
    command = [sys.executable, str(config.root / "runnable_settings" / "run_setting.py"),
               "--setting", str(setting_json)]
    if args.model:
        command.extend(["--model", args.model])
    if args.suite:
        command.extend(["--suite", args.suite])
    command.extend(["--gpu", args.gpu or str(config.data.get("evaluation", {}).get("default_gpu", "0"))])
    for name, value in (("task", args.task), ("trials", args.trials), ("trial-start", args.trial_start)):
        if value is not None:
            command.extend([f"--{name}", str(value)])
    if args.trials is None and not env.get("LIBERO_VPRO_TRIALS"):
        env["LIBERO_VPRO_TRIALS"] = str(config.data.get("evaluation", {}).get("trials_per_case", 10))
    output_root = args.output_root
    if output_root is None:
        output_root = config.output_root / payload["category"] / payload["setting_id"].split(".", 1)[1]
    command.extend(["--output-root", str(output_root)])
    if args.check:
        command.append("--check")
    if args.plan:
        command.append("--plan")
    return subprocess.run(command, cwd=config.root, env=env, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
