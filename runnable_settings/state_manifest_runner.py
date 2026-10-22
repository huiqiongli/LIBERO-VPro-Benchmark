#!/usr/bin/env python3
"""Run one filtered D2/D3 setting through the FastWAM manifest worker."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from runner_common import Selection, add_selection_arguments, selected_manifest
from libero_vpro.runtime import prepare_setting_env, resolve_runtime

HERE = Path(__file__).resolve().parent
RELEASE = HERE.parent


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--setting", type=Path, required=True)
    parser.add_argument("--model", required=True, choices=("fastwam",))
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--plan", action="store_true")
    add_selection_arguments(parser)
    args = parser.parse_args()
    selection = Selection.from_args(args)
    setting_path = args.setting.resolve()
    payload = json.loads(setting_path.read_text(encoding="utf-8"))
    setting_dir = setting_path.parent
    manifests = payload.get("manifests_by_suite", {})
    if not manifests:
        raise SystemExit(f"setting has no suite manifests: {setting_path}")
    root = RELEASE / "source" / "D2_D3_state_20260823" / "runners"
    output = args.output_root.resolve()
    commands: list[tuple[list[str], Path]] = []
    for suite, manifest_name in sorted(manifests.items()):
        if selection.suite is not None and suite != selection.suite:
            continue
        manifest = setting_dir / manifest_name
        rows = json.loads(manifest.read_text(encoding="utf-8"))["cases"]
        if not any(selection.includes(row) for row in rows):
            continue
        suite_output = output / suite
        manifest = selected_manifest(manifest, output, selection, dry_run=args.plan)
        command = ["bash", str(root / "run_fastwam_manifest.sh"), str(manifest), args.gpu, str(suite_output)]
        commands.append((command, suite_output))
    if not commands:
        raise SystemExit(f"selection matches no episodes in {payload['setting_id']}")
    if args.plan:
        for command, _ in commands:
            print(" ".join(subprocess.list2cmdline([part]) for part in command))
        return 0
    resolve_runtime(require=True)
    env = prepare_setting_env()
    env.update(selection.as_env())
    env.update({"LIBERO_VPRO_SETTING_ID": payload["setting_id"], "LIBERO_VPRO_MODEL": args.model})
    output.mkdir(parents=True, exist_ok=True)
    for command, _ in commands:
        result = subprocess.run(command, env=env, cwd=RELEASE, check=False)
        if result.returncode:
            return result.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
