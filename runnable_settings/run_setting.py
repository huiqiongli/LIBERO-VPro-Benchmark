#!/usr/bin/env python3
"""FastWAM-only dispatcher for the 96 paper-facing LIBERO-VPro settings."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

from runner_common import Selection, add_selection_arguments, selected_manifest
from libero_vpro.runtime import prepare_setting_env, resolve_runtime

HERE = Path(__file__).resolve().parent
RELEASE = HERE.parent


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else RELEASE / path


def missing_inputs(setting: dict) -> list[str]:
    missing = []
    for value in (setting.get("source_roots", []) + setting.get("config_files", [])
                  + setting.get("asset_roots", [])):
        value = value.replace("{suite}", "libero_object").replace("{task}", "task_00")
        if not resolve(value).exists():
            missing.append(value)
    return missing


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--setting", type=Path, required=True)
    parser.add_argument("--model", choices=("fastwam",), default="fastwam")
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--port", type=int, default=0)
    add_selection_arguments(parser)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--plan", action="store_true")
    args = parser.parse_args()
    selection = Selection.from_args(args)
    dry_run = args.check or args.plan

    setting_path = args.setting.resolve()
    setting = load(setting_path)
    setting_dir = setting_path.parent
    sid = setting["setting_id"].split(".", 1)[1]
    category = setting["category"]
    missing = missing_inputs(setting)
    if missing:
        print("missing release inputs:", file=sys.stderr)
        for item in missing:
            print(f"  {item}", file=sys.stderr)
        return 2

    from libero_vpro.validation import validate_setting_inputs
    validation_errors = validate_setting_inputs(setting_path, RELEASE)
    if validation_errors:
        print("invalid release inputs:", file=sys.stderr)
        for error in validation_errors:
            print(f"  {error}", file=sys.stderr)
        return 2
    if category in {"A3", "D2", "D3"}:
        inventory = load(setting_dir / setting["manifest"])
        if not any(selection.includes(row) for row in inventory["episodes"]):
            print(f"selection matches no episodes in {setting['setting_id']}", file=sys.stderr)
            return 2

    sim_root = Path(os.environ.get("LIBERO_VPRO_SIM_ROOT", str(RELEASE.parent))).expanduser()
    output = (args.output_root or Path(os.environ.get(
        "LIBERO_VPRO_OUTPUT_ROOT", sim_root / "Outputs/libero_vpro"
    )) / category / sid).resolve()
    env = prepare_setting_env()
    env.update({
        "LIBERO_VPRO_SETTING_ID": setting["setting_id"],
        "LIBERO_VPRO_OUTPUT_ROOT": str(output),
    })
    env.update(selection.as_env())

    if category in {"A1", "A2"} and sid.startswith("ratio_"):
        source_root = resolve(setting["source_roots"][0])
        manifest = setting_dir / setting.get("batch_manifest", setting.get("manifest", "manifest.json"))
        manifest = selected_manifest(manifest, output, selection, dry_run=dry_run)
        env.update({
            "LIBERO_VPRO_GPU": args.gpu,
            "A1_RATIO_OUTPUT_ROOT" if category == "A1" else "A2_RATIO_OUTPUT_ROOT": str(output),
            "A1_SETTING_MANIFEST" if category == "A1" else "A2_SETTING_MANIFEST": str(manifest),
            "A1_SETTING_RATIO" if category == "A1" else "A2_SETTING_RATIO": sid.split("_")[1].replace("pct", ""),
        })
        command = ["bash", str(source_root / "run_fastwam_worker.sh"), args.gpu]
    elif category in {"A1", "A2"}:
        command = [sys.executable, str(HERE / "timing_setting_runner.py"),
                   "--category", category, "--setting", sid,
                   "--setting-json", str(setting_path), "--model", "fastwam",
                   "--gpu", args.gpu, "--output-root", str(output)]
    elif category == "A3":
        command = [sys.executable, str(HERE / "a3_setting_runner.py"),
                   "--setting", str(setting_path), "--model", "fastwam",
                   "--gpu", args.gpu, "--output-root", str(output)]
    elif category == "A4":
        root = resolve("source/A4_temporary_observation_corruption")
        if sid == "target_region_noise":
            env["A4_TARGET_NOISE_OUTPUT_ROOT"] = str(output)
            command = [sys.executable, str(HERE / "run_a4_target_region_noise.py"),
                       "--setting", setting["setting_id"], "--model", "fastwam", "--gpu", args.gpu]
        elif sid.startswith(("black_", "noise_")) and sid.endswith("_3replan_once"):
            phase_names = ("reach", "pre_grasp", "grasp", "transport", "pre_place")
            phase = next((name for name in phase_names if f"_{name}_" in sid), None)
            if not phase:
                raise SystemExit(f"unrecognised A4 phase setting: {sid}")
            env.update({"A4_PHASES": phase, "A4_PHASE_KIND": sid.split("_", 1)[0],
                        "A4_PHASE_FULL_ROOT": str(output)})
            command = ["bash", str(root / "run_fastwam_a4_phase_full.sh"), args.gpu]
        else:
            env.update({"A4_ONLY_SETTING": sid, "A4_OUTPUT_ROOT": str(output)})
            if sid.startswith(("black_D", "noise_D")):
                env["A4_KIND"] = sid.split("_", 1)[0]
            command = ["bash", str(root / "run_fastwam_a4_full.sh"), args.gpu]
    elif category == "A5":
        command = [sys.executable, str(HERE / "fastwam_a5_setting_runner.py"),
                   "--setting", setting["setting_id"], "--gpu", args.gpu,
                   "--output-root", str(output)]
    elif category == "B1":
        env.update({"B1_ONLY_CONDITION": f"B1_{sid}", "B1_OUTPUT_ROOT": str(output),
                    "B1_TRIALS": str(selection.trials)})
        command = ["bash", str(resolve("source/B1_camera_staleness/run_fastwam_b1_full_worker.sh")),
                   args.gpu, "0", "1"]
    elif category == "C1":
        source_root = resolve("source/C1_cross_view_20260825")
        manifest = selected_manifest(setting_dir / setting["manifest"], output, selection, dry_run=dry_run)
        command = ["bash", str(source_root / "runners/run_fastwam_manifest.sh"),
                   str(manifest), args.gpu, str(output)]
    elif category == "C2":
        command = [sys.executable, str(HERE / "fastwam_c2_setting_runner.py"),
                   "--setting", setting["setting_id"], "--gpu", args.gpu,
                   "--output-root", str(output)]
    elif category == "D1":
        command = [sys.executable, str(HERE / "fastwam_d1_setting_runner.py"),
                   "--setting", setting["setting_id"], "--gpu", args.gpu,
                   "--output-root", str(output)]
    elif category in {"D2", "D3"}:
        command = [sys.executable, str(HERE / "state_manifest_runner.py"),
                   "--setting", str(setting_path), "--model", "fastwam",
                   "--gpu", args.gpu, "--output-root", str(output)]
    elif category == "D4":
        source_root = resolve(setting["source_roots"][0])
        manifest = setting_dir / setting["batch_manifest"]
        manifest = selected_manifest(manifest, output, selection, dry_run=dry_run)
        env.update({
            "D4_OUTPUT_ROOT": str(output),
            "D4_SETTING_MANIFEST": str(manifest),
            "D4_ONLY_SETTING": sid.replace(".", "_"),
            "LIBERO_VPRO_GPU": args.gpu,
        })
        command = ["bash", str(source_root / "eval_6models/run_fastwam_worker.sh"), args.gpu]
    else:
        raise SystemExit(f"no FastWAM implementation for {setting['setting_id']}")

    # A successful preflight must mean that the concrete FastWAM entry point
    # is present, not merely that a command string could be constructed.
    if len(command) < 2 or not Path(command[1]).is_file():
        target = command[1] if len(command) >= 2 else "<missing command target>"
        print(f"missing FastWAM runner entry point: {target}", file=sys.stderr)
        return 2

    if args.check:
        print(f"PASS {setting['setting_id']} (FastWAM)")
        print("  " + " ".join(shlex.quote(part) for part in command))
        return 0
    if args.plan:
        print(" ".join(shlex.quote(part) for part in command))
        print(json.dumps({"setting_id": setting["setting_id"], "model": "fastwam",
                          "output_root": str(output),
                          "env": {key: value for key, value in env.items()
                                  if key.startswith(("A1_", "A2_", "A3_", "A4_", "A5_", "B1_", "C1_", "C2_", "D1_", "D2_", "D3_", "D4_", "LIBERO_VPRO_"))}},
                         indent=2))
        return 0
    resolve_runtime(require=True)
    output.mkdir(parents=True, exist_ok=True)
    return subprocess.run(command, env=env, cwd=RELEASE, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
