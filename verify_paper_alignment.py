#!/usr/bin/env python3
"""Verify the paper-facing LIBERO-VPro setting and case inventory.

This verifier treats PAPER_FACING_SETTINGS.json as the release contract. It
checks the 12 categories, 96 unique expanded settings, 3,296 task-condition cases,
source-root coverage, and (when available) the matching claims in root.tex.
Historical or provenance-only entries are reported separately.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


EXPECTED_CATEGORY_COUNT = 12
EXPECTED_SETTING_COUNT = 96
EXPECTED_CASE_COUNT = 3296
EXPECTED_CATEGORY_IDS = {
    "A1", "A2", "A3", "A4", "A5", "B1", "C1", "C2", "D1", "D2", "D3", "D4"
}


def load_json(path: Path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def resolve_release_path(release_root: Path, value: str) -> Path:
    return (release_root / value).resolve()


def runnable_dir_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")


def statistical_leaf_key(category_id: str, setting_id: str) -> str:
    """Map a compact release id to the hierarchy's subclass.parameter key."""
    if category_id in {"A1", "A2"}:
        subclass = "physical_target_object_ratio" if category_id == "A1" else "physical_target_receptacle_ratio"
        if not setting_id.startswith("ratio_"):
            subclass = "timing"
        return f"{subclass}.{setting_id}"
    if category_id == "A3":
        return f"interaction_part.{setting_id}"
    if category_id == "A4":
        phase_names = ("reach", "pre_grasp", "grasp", "transport", "pre_place")
        phase_ids = {
            f"{kind}_{name}_3replan_once"
            for kind in ("black", "noise")
            for name in phase_names
        }
        subclass = "phase_observation_corruption" if setting_id in phase_ids else "observation_corruption"
        return f"{subclass}.{setting_id}"
    if category_id == "A5":
        return f"view_condition.{setting_id}"
    if category_id == "B1":
        return f"staleness.{setting_id}"
    if category_id == "C1":
        return setting_id
    if category_id == "C2":
        return f"recursive_self_prediction.{setting_id.split('.', 1)[-1]}"
    if category_id == "D1":
        return f"semantic_distractor.{setting_id}"
    if category_id == "D2":
        subclass = "articulation_precondition" if setting_id.startswith("open_") else "object_support_state"
        return f"{subclass}.{setting_id}"
    if category_id in {"D3", "D4"}:
        return setting_id
    return setting_id


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release-root", type=Path,
                        default=Path(__file__).resolve().parent)
    parser.add_argument("--paper-root", type=Path, default=None)
    parser.add_argument("--statistics-json", type=Path, default=None,
                        help="override the statistical hierarchy JSON")
    args = parser.parse_args()

    release_root = args.release_root.resolve()
    manifest_path = release_root / "PAPER_FACING_SETTINGS.json"
    manifest = load_json(manifest_path)
    categories = manifest["categories"]
    errors: list[str] = []
    warnings: list[str] = []

    category_ids = [item["id"] for item in categories]
    if len(categories) != EXPECTED_CATEGORY_COUNT:
        errors.append(f"category count={len(categories)} != {EXPECTED_CATEGORY_COUNT}")
    if len(set(category_ids)) != len(category_ids):
        errors.append("duplicate category id")
    if set(category_ids) != EXPECTED_CATEGORY_IDS:
        errors.append(f"public category ids={sorted(category_ids)} != {sorted(EXPECTED_CATEGORY_IDS)}")

    setting_ids: list[str] = []
    total_cases = 0
    category_rows = []
    expected_runnable: dict[str, str] = {}
    for category in categories:
        settings = category["settings"]
        setting_count = len(settings)
        for item in settings:
            full_id = f"{category['id']}.{item['id']}"
            if full_id in setting_ids:
                errors.append(f"duplicate setting id: {full_id}")
            setting_ids.append(full_id)
            expected_runnable[full_id] = runnable_dir_name(item["id"])
            status = item.get("status", "implemented")
            if status == "historical_statistical_entry":
                warnings.append(f"historical/provenance-only setting: {full_id}")

        for source in category.get("source_roots", []):
            path = resolve_release_path(release_root, source)
            if not path.exists():
                errors.append(f"missing source root for {category['id']}: {source}")

    for category_id, entrypoints in manifest.get("entrypoints", {}).items():
        if isinstance(entrypoints, str):
            entrypoints = [entrypoints]
        for entrypoint in entrypoints:
            if "<" in entrypoint or ">" in entrypoint:
                continue  # descriptive path pattern, not a literal file path
            path = resolve_release_path(release_root, entrypoint)
            if not path.is_file():
                errors.append(f"missing entrypoint for {category_id}: {entrypoint}")

    if len(setting_ids) != EXPECTED_SETTING_COUNT:
        errors.append(f"setting count={len(setting_ids)} != {EXPECTED_SETTING_COUNT}")

    # A release setting is not considered reproducible merely because a
    # category-level adapter exists. Each expanded setting must have its own
    # model-independent contract and case inventory where applicable.
    runnable_root = resolve_release_path(
        release_root, manifest.get("runnable_settings_root", "runnable_settings")
    )
    runnable_index_path = resolve_release_path(
        release_root, manifest.get("runnable_settings_index", "runnable_settings/index.json")
    )
    if not runnable_index_path.is_file():
        errors.append(f"missing runnable settings index: {runnable_index_path}")
    else:
        try:
            runnable_index = load_json(runnable_index_path)
            rows = {row["id"]: row for row in runnable_index.get("settings", [])}
            if runnable_index.get("setting_count") != EXPECTED_SETTING_COUNT:
                errors.append(
                    f"runnable setting count={runnable_index.get('setting_count')} != {EXPECTED_SETTING_COUNT}"
                )
            for full_id, safe_id in expected_runnable.items():
                row = rows.get(full_id)
                if row is None:
                    errors.append(f"missing runnable setting entry: {full_id}")
                    continue
                setting_dir = runnable_root / row["path"]
                contract = setting_dir / "setting.json"
                if not contract.is_file():
                    errors.append(f"missing runnable contract: {full_id}")
                    continue
                try:
                    payload = load_json(contract)
                    if payload.get("setting_id") != full_id:
                        errors.append(f"runnable contract id mismatch: {full_id}")
                    for value in payload.get("source_roots", []) + payload.get("config_files", []):
                        value = value.replace("{suite}", "libero_object").replace("{task}", "task_00")
                        if not resolve_release_path(release_root, value).exists():
                            errors.append(f"runnable input missing for {full_id}: {value}")
                    forbidden = ("/snapshots/", "_quarantine", "/old/")
                    for value in payload.get("source_roots", []) + payload.get("config_files", []):
                        if any(token in value for token in forbidden):
                            errors.append(f"runnable setting points to stale source: {full_id}: {value}")
                except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
                    errors.append(f"invalid runnable contract {full_id}: {exc}")
        except (OSError, json.JSONDecodeError, TypeError, KeyError) as exc:
            errors.append(f"invalid runnable settings index: {exc}")

    statistics_path = args.statistics_json
    if statistics_path is None:
        statistics_path = resolve_release_path(
            release_root, manifest["statistical_source"])
        # The release tree is stored under /data on this machine while the
        # paper tree remains under /home.  When --paper-root is supplied,
        # resolve the statistical hierarchy beside that paper tree as a
        # portable fallback instead of hard-coding either mount point.
        if not statistics_path.exists() and args.paper_root is not None:
            statistics_path = (
                args.paper_root.resolve().parent
                / "LIBERO-VPro_old/data/dataset_hierarchy_from_full_results.json"
            )
    if statistics_path.exists():
        statistics = load_json(statistics_path)
        for key, expected in (("category_count", EXPECTED_CATEGORY_COUNT),
                              ("setting_count", EXPECTED_SETTING_COUNT),
                              ("case_count", EXPECTED_CASE_COUNT)):
            if statistics.get(key) != expected:
                errors.append(
                    f"statistical manifest {key}={statistics.get(key)!r} != {expected}"
                )
        stat_by_id = {item["code"]: item for item in statistics.get("categories", [])}
        for category in categories:
            stat = stat_by_id.get(category["id"])
            if stat is None:
                errors.append(f"category missing from statistical manifest: {category['id']}")
                continue
            setting_count = len(category["settings"])
            case_count = stat.get("case_count")
            if stat.get("setting_count") != setting_count:
                errors.append(
                    f"{category['id']} setting count {setting_count} != "
                    f"statistical {stat.get('setting_count')}"
                )
            if not isinstance(case_count, int):
                errors.append(f"{category['id']} category case total is missing")
                case_count = 0
            total_cases += case_count
            category_rows.append((category["id"], setting_count, case_count))
        if total_cases != EXPECTED_CASE_COUNT:
            errors.append(f"category case totals sum to {total_cases} != {EXPECTED_CASE_COUNT}")
    else:
        warnings.append(f"statistical manifest not found: {statistics_path}")

    claims = manifest["protocol"]
    for key, expected in (("paper_category_count", EXPECTED_CATEGORY_COUNT),
                          ("paper_setting_count", EXPECTED_SETTING_COUNT),
                          ("paper_task_condition_case_count", EXPECTED_CASE_COUNT)):
        if claims.get(key) != expected:
            errors.append(f"manifest protocol {key}={claims.get(key)!r} != {expected}")

    if args.paper_root is not None:
        tex_path = args.paper_root / "root.tex"
        if not tex_path.exists():
            errors.append(f"paper source not found: {tex_path}")
        else:
            tex = tex_path.read_text(encoding="utf-8", errors="replace")
            if not re.search(r"96\s+experimental settings", tex):
                errors.append("root.tex does not state 96 experimental settings")
            if not re.search(r"3[,{]?296\s+task[-~ ]condition cases", tex):
                errors.append("root.tex does not state 3,296 task-condition cases")
            normalized_tex = re.sub(r"[–—]", "-", tex).lower()
            for category in categories:
                name = re.sub(r"[–—]", "-", category["name"]).lower()
                if name not in normalized_tex:
                    errors.append(
                        f"root.tex does not contain paper category name: {category['name']}"
                    )

    print("Paper-facing LIBERO-VPro alignment")
    print(f"  categories: {len(categories)}")
    print(f"  expanded settings: {len(setting_ids)}")
    print(f"  task-condition cases: {total_cases}")
    print("  category breakdown:")
    for category_id, setting_count, case_count in category_rows:
        print(f"    {category_id}: {setting_count} settings, {case_count} cases")
    for warning in warnings:
        print(f"  WARNING: {warning}")
    if errors:
        print("FAIL")
        for error in errors:
            print(f"  ERROR: {error}")
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
