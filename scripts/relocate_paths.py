#!/usr/bin/env python3
"""Rewrite copied JSON manifests for a new LIBERO-VPro checkout.

The benchmark sources were collected from several machines and older manifests
therefore contain absolute checkout paths. This utility rebases release paths
and explicitly configured external LIBERO roots. It is safe to run repeatedly.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
from xml.sax.saxutils import escape, unescape


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "source"

ASSET_ALIASES = {
    "source/A1_physical_target_object_ratio_20260822/precompiled_variants_full_surface_v3":
        "assets/A1_target_object/precompiled_variants_full_surface_v3",
    "source/A2_physical_receptacle_ratio_20260822/precompiled_variants_full_surface_v3":
        "assets/A2_target_receptacle/precompiled_variants_full_surface_v3",
    "source/D1_4_suites_full/bddl_overrides":
        "assets/D1_task_relevant_distractors/bddl_overrides",
    "source/D1_4_suites_full/assets":
        "assets/D1_task_relevant_distractors/assets",
    "source/A4_4_suites_full/assets":
        "assets/D1_task_relevant_distractors/assets",
    "source/D2_D3_state_20260823/assets":
        "assets/D2_D3_state_variation/assets",
    "source/D2_D3_state_20260823/bddl_overrides":
        "assets/D2_D3_state_variation/bddl_overrides",
    "source/D4_contact4_physical_3severity_40task_20260815/assets":
        "assets/D4_target_geometry/assets",
}


LIBERO_ASSET_FAMILIES = (
    "stable_scanned_objects", "stable_hope_objects", "stable_ycb_objects",
    "scanned_objects", "hope_objects", "turbosquid_objects",
    "articulated_objects", "textures", "scenes", "robots",
)


def canonical_release_path(suffix: str, root: Path) -> str:
    for old, new in ASSET_ALIASES.items():
        if suffix == old or suffix.startswith(old + "/"):
            suffix = new + suffix[len(old):]
            break
    path = root / suffix
    if suffix.startswith("assets/D1_task_relevant_distractors/assets/"):
        if path.suffix == ".xml":
            path = path.with_name(path.name.replace("_a4_", "_d1_"))
    return str(path)


def rewrite_external_path(
    value: str, *, libero_assets_root: Path | None = None,
    data_root: Path | None = None,
) -> str:
    """Recognize LIBERO layout suffixes after any previous root relocation."""
    if not Path(value).is_absolute():
        return value
    if data_root is not None:
        for folder in ("bddl_files", "init_files"):
            marker = f"/{folder}/"
            if marker in value:
                return str(data_root / folder / value.rsplit(marker, 1)[1])
            if value.endswith("/" + folder):
                return str(data_root / folder)
    if libero_assets_root is not None:
        marker = "/libero/libero/assets/"
        if marker in value:
            return str(libero_assets_root / value.split(marker, 1)[1])
        # A previous invocation may have used an asset root with another name.
        for family in LIBERO_ASSET_FAMILIES:
            marker = f"/{family}/"
            if marker in value:
                return str(libero_assets_root / family / value.rsplit(marker, 1)[1])
    return value


def rewrite_string(
    value: str, root: Path = ROOT, *, libero_assets_root: Path | None = None,
    data_root: Path | None = None, old_root: Path | None = None,
) -> str:
    current_root = str(root)
    if value == current_root:
        return current_root
    if value.startswith(current_root + "/"):
        return canonical_release_path(value[len(current_root) + 1:], root)
    if old_root is not None:
        previous = str(old_root)
        if value == previous:
            return current_root
        if value.startswith(previous + "/"):
            return canonical_release_path(value[len(previous) + 1:], root)
    # Manifests generated on another checkout often embed that checkout's
    # absolute root (for example ``/home/user/project/release/source/...``).
    # Re-root any such path by its release-relative suffix, without knowing
    # the original username or workspace layout.
    marker = "/release/"
    if marker in value:
        suffix = value.rsplit(marker, 1)[1]
        return canonical_release_path(suffix, root)
    if value.endswith("/release"):
        return str(root)
    release_alias = "external/reviewed_script/release/"
    if value.startswith(release_alias):
        return canonical_release_path(value[len(release_alias):], root)
    if value == "external/reviewed_script/release":
        return str(root)
    markers = ("/reviewed_script/", "/Harness/Dataset/")
    for marker in markers:
        if marker in value:
            tail = value.split(marker, 1)[1]
            source_tail = tail if tail.startswith("source/") else "source/" + tail
            for old in ASSET_ALIASES:
                if source_tail == old or source_tail.startswith(old + "/"):
                    return canonical_release_path(source_tail, root)
            if tail.startswith("adapted/"):
                return str(root / "source" / tail)
            head, separator, rest = tail.partition("/")
            if (root / "source" / head).exists():
                return str(root / "source" / head / rest) if separator else str(root / "source" / head)
            return "external/source_snapshot/" + tail

    marker = "/Outputs/"
    if marker in value:
        return "external/results/" + value.split(marker, 1)[1]
    # Recognize another checkout even when its directory is not named release.
    # --old-root additionally handles root-valued fields and absent targets.
    if Path(value).is_absolute():
        for folder in ("source", "runnable_settings", "paper_facing_configs", "assets"):
            marker = f"/{folder}/"
            if marker in value:
                suffix = folder + "/" + value.split(marker, 1)[1]
                candidate = canonical_release_path(suffix, root)
                if Path(candidate).exists():
                    return candidate
    return rewrite_external_path(value, libero_assets_root=libero_assets_root,
                                 data_root=data_root)


def rewrite(value, root: Path = ROOT, **roots):
    if isinstance(value, str):
        return rewrite_string(value, root, **roots)
    if isinstance(value, list):
        return [rewrite(item, root, **roots) for item in value]
    if isinstance(value, dict):
        return {key: rewrite(item, root, **roots) for key, item in value.items()}
    return value


def rewrite_xml_text(
    text: str, path: Path, root: Path = ROOT, *,
    libero_assets_root: Path | None = None, old_root: Path | None = None,
) -> str:
    """Remove workstation paths from bundled MuJoCo object XML files.

    Custom perturbation meshes live beside their XML. Original LIBERO meshes
    and textures remain an external prerequisite and are rebased to the user's
    LIBERO asset checkout when supplied.
    """
    def replace(match: re.Match[str]) -> str:
        prefix, quote, raw = match.groups()
        value = unescape(raw, {"&quot;": '"', "&apos;": "'"})
        updated = rewrite_string(value, root, libero_assets_root=libero_assets_root,
                                 old_root=old_root)
        candidate = Path(updated)
        if candidate.is_absolute() and candidate.is_relative_to(root):
            updated = os.path.relpath(candidate, path.parent)
        elif "/reviewed_script/" in value:
            # Historical A1/A2 overlays were copied next to their XML.
            local = path.parent / Path(value).name
            if local.is_file():
                updated = local.name
        if updated == value:
            return match.group(0)
        return prefix + quote + escape(updated, {quote: "&quot;" if quote == '"' else "&apos;"}) + quote

    return re.sub(r'''(\bfile\s*=\s*)(["'])(.*?)\2''', replace, text)


def rewrite_xml(path: Path, libero_assets_root: Path | None, root: Path = ROOT,
                *, old_root: Path | None = None, write: bool = True) -> bool:
    original = path.read_text(encoding="utf-8")
    updated = rewrite_xml_text(original, path, root,
                               libero_assets_root=libero_assets_root, old_root=old_root)
    if updated == original:
        return False
    if write:
        path.write_text(updated, encoding="utf-8")
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--old-root", type=Path, default=None,
                        help="previous release root, including a custom directory name")
    parser.add_argument(
        "--data-root", type=Path,
        default=Path(os.environ["LIBERO_VPRO_DATA_ROOT"])
        if os.environ.get("LIBERO_VPRO_DATA_ROOT") else None,
        help="external LIBERO data directory containing bddl_files and init_files",
    )
    parser.add_argument(
        "--libero-assets-root", type=Path,
        default=Path(os.environ["LIBERO_VPRO_LIBERO_ASSETS_ROOT"])
        if os.environ.get("LIBERO_VPRO_LIBERO_ASSETS_ROOT") else None,
        help="external LIBERO package assets directory (containing stable_*_objects)",
    )
    parser.add_argument("--write", action="store_true",
                        help="rewrite JSON/XML files; without this flag only report files")
    args = parser.parse_args()
    root = args.root.resolve()
    data_root = args.data_root.expanduser().resolve() if args.data_root else None
    assets_root = (args.libero_assets_root.expanduser().resolve()
                   if args.libero_assets_root else None)
    old_root = args.old_root.expanduser().absolute() if args.old_root else None
    changed = 0
    scanned = 0
    for path in sorted(root.rglob("*.json")):
        if any(part in {"__pycache__", ".git"} for part in path.parts):
            continue
        scanned += 1
        try:
            original = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        updated = rewrite(original, root, libero_assets_root=assets_root,
                          data_root=data_root, old_root=old_root)
        if updated == original:
            continue
        changed += 1
        if args.write:
            path.write_text(json.dumps(updated, indent=2, ensure_ascii=False) + "\n",
                            encoding="utf-8")
        else:
            print(path)
    xml_changed = 0
    for path in sorted(root.rglob("*.xml")):
        if path.is_symlink():
            continue
        if rewrite_xml(path, assets_root, root, old_root=old_root, write=args.write):
            xml_changed += 1
            if not args.write:
                print(path)
    action = "rewrote" if args.write else "would rewrite"
    print(f"{action} {changed} of {scanned} JSON files")
    print(f"{action} {xml_changed} XML files")
    if args.write:
        if args.libero_assets_root is None:
            print("LIBERO XML base-asset paths were left unchanged; set LIBERO_VPRO_LIBERO_ASSETS_ROOT and rerun to rebase them.")
        if data_root is None:
            print("LIBERO BDDL/init paths were left unchanged; set LIBERO_VPRO_DATA_ROOT or pass --data-root to rebase them.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
