from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

from .config import ReleaseConfig
from .manifest import load_json, setting_rows


def _default_config() -> Path:
    return Path(__file__).resolve().parent.parent / "evaluation_config.yaml"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="libero-vpro")
    parser.add_argument("command", choices=("list", "check"))
    parser.add_argument("--config", type=Path, default=_default_config())
    parser.add_argument("--paper-root", type=Path, default=None)
    args = parser.parse_args(argv)
    config = ReleaseConfig.load(args.config)

    if args.command == "list":
        manifest = load_json(config.release_manifest)
        print(json.dumps({
            "schema_version": manifest.get("schema_version"),
            "categories": len(manifest.get("categories", [])),
            "settings": len(setting_rows(config)),
            "task_condition_cases": manifest.get("protocol", {}).get("paper_task_condition_case_count"),
        }, indent=2))
        return 0

    verifier = config.root / "verify_paper_alignment.py"
    command = [sys.executable, str(verifier)]
    if args.paper_root is not None:
        command.extend(["--paper-root", str(args.paper_root.resolve())])
    return subprocess.run(command, cwd=config.root, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
