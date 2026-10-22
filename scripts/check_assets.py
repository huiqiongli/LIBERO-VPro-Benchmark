#!/usr/bin/env python3
"""Check bundled perturbation assets and external LIBERO assets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from libero_vpro.config import ReleaseConfig
from libero_vpro.validation import validate_release_inputs


def bundled_asset_roots() -> list[Path]:
    """Read the same per-setting contract used by every launcher.

    Each setting declares the asset families included in this check.
    """
    index_path = ROOT / "runnable_settings/index.json"
    if not index_path.is_file():
        raise FileNotFoundError(index_path)
    index = json.loads(index_path.read_text(encoding="utf-8"))
    roots: set[str] = set()
    invalid: list[str] = []
    for row in index.get("settings", []):
        setting_path = ROOT / "runnable_settings" / row["path"] / "setting.json"
        payload = json.loads(setting_path.read_text(encoding="utf-8"))
        if (payload.get("asset_contract_schema") != "libero_vpro.asset_contract.v1"
                or not payload.get("asset_mode")
                or not isinstance(payload.get("asset_roots"), list)
                or not isinstance(payload.get("external_dependencies"), list)):
            invalid.append(str(setting_path))
        roots.update(str(item) for item in payload.get("asset_roots", []))
    if invalid:
        raise ValueError("settings without a complete asset contract: " + ", ".join(invalid))
    return [ROOT / item for item in sorted(roots)]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path,
                        default=ROOT / "evaluation_config.yaml")
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--libero-assets-root", type=Path, default=None)
    parser.add_argument("--max-errors", type=int, default=20)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    config = ReleaseConfig.load(args.config)
    environment = config.environment()
    data_value = environment.get("LIBERO_VPRO_DATA_ROOT")
    assets_value = environment.get("LIBERO_VPRO_LIBERO_ASSETS_ROOT")
    data_root = args.data_root or (Path(data_value) if data_value else None)
    if data_root is not None:
        data_root = data_root.expanduser().resolve()
    assets_root = args.libero_assets_root or (Path(assets_value) if assets_value else None)
    if assets_root is not None:
        assets_root = assets_root.expanduser().resolve()
    report = validate_release_inputs(config.root, data_root=data_root,
                                     libero_assets_root=assets_root)
    if data_root is None:
        report.unconfigured_external.add("data")
    else:
        for key in ("bddl_subdir", "init_subdir"):
            directory = data_root / config.data["external"][key]
            if not directory.is_dir():
                report.external_errors.append(f"Missing external LIBERO directory: {directory}")
    if assets_root is not None and not assets_root.is_dir():
        report.external_errors.append(f"Missing external LIBERO asset directory: {assets_root}")

    counts = report.counts
    print(f"Checked {counts['settings']} setting input closures; "
          f"{counts['json_syntax']} JSON files, {counts['xml']} XML files, "
          f"{counts['xml_file_references']} XML file references, "
          f"{counts['symlinks']} symlinks.")
    limit = max(1, args.max_errors)
    for label, errors in (("Bundled input errors", report.errors),
                          ("External LIBERO errors", report.external_errors)):
        if errors:
            print(f"{label}: {len(errors)}")
            for error in errors[:limit]:
                print(f"  {error}")
            if len(errors) > limit:
                print(f"  ... {len(errors) - limit} more (use --max-errors to display)")
    if not report.errors:
        print("Bundled setting contracts, nested inputs, XML dependencies, and symlinks: OK")
    if report.unconfigured_external:
        variables = {"data": "LIBERO_VPRO_DATA_ROOT / --data-root",
                     "assets": "LIBERO_VPRO_LIBERO_ASSETS_ROOT / --libero-assets-root"}
        print("External LIBERO prerequisites were not fully checked:")
        for kind in sorted(report.unconfigured_external):
            print(f"  Configure {variables[kind]}")
    elif not report.external_errors:
        print(f"External LIBERO inputs: {len(report.external_files)} referenced files OK")
    if report.errors or report.external_errors:
        return 1
    return 2 if args.strict and report.unconfigured_external else 0


if __name__ == "__main__":
    raise SystemExit(main())
