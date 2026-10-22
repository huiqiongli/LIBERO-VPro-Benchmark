"""Read-only validation of runnable inputs, independent of model runtimes."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
import glob
import json
import math
import os
from pathlib import Path
import re
from typing import Any
import xml.etree.ElementTree as ET

from .asset_contract import ASSET_CONTRACT_SCHEMA, contract_for


SUITES = {"libero_object", "libero_spatial", "libero_goal", "libero_10"}
INPUT_KEYS = {
    "spec", "spec_path", "case_json", "case_path", "xml_path",
    "trajectory_xml_path", "bddl_file", "base_bddl_file_name",
    "bddl_file_name_override", "bddl_file_override", "bddl_override",
    "obs_spec", "observation_spec", "schedule_spec", "env_edit_spec", "environment_spec",
    "phase_task_map", "task_region_manifest", "geometry_reference_spec",
    "runner", "worker", "manifest", "batch_manifest", "config_path",
    "config_file", "input_path", "init_file", "init_file_name",
    "init_states_path", "occupancy_spec_path",
}
PROVENANCE_KEYS = {
    "audit", "precompiled_surface_audit", "source_xml", "visual_source_obj",
    "source_manifest", "destination_source", "full_result_source", "source",
    "validation", "provenance", "external_dependencies",
}
OUTPUT_KEY = re.compile(r"(^|_)(output|outputs|result|results|archive|archives|report|video|log|record)(_|$)")
INPUT_SUFFIXES = {
    ".json", ".xml", ".bddl", ".py", ".sh", ".yaml", ".yml",
    ".obj", ".msh", ".stl", ".png", ".jpg", ".jpeg", ".npy", ".npz",
}
LIBERO_ASSET_FAMILIES = {
    "stable_scanned_objects", "stable_hope_objects", "stable_ycb_objects",
    "scanned_objects", "hope_objects", "turbosquid_objects",
    "articulated_objects", "textures", "scenes", "robots",
}


@dataclass
class ValidationReport:
    errors: list[str] = field(default_factory=list)
    external_errors: list[str] = field(default_factory=list)
    unconfigured_external: set[str] = field(default_factory=set)
    files: set[Path] = field(default_factory=set)
    external_files: set[Path] = field(default_factory=set)
    counts: Counter = field(default_factory=Counter)


def external_libero_reference(path: Path) -> tuple[str, Path] | None:
    """Return the portable layout suffix of a recognized external LIBERO input."""
    value = str(path)
    for folder in ("bddl_files", "init_files"):
        marker = f"/{folder}/"
        if marker in value:
            return "data", Path(folder) / value.rsplit(marker, 1)[1]
    marker = "/libero/libero/assets/"
    if marker in value:
        return "assets", Path(value.split(marker, 1)[1])
    for family in LIBERO_ASSET_FAMILIES:
        marker = f"/{family}/"
        if marker in value:
            return "assets", Path(family) / value.rsplit(marker, 1)[1]
    return None


class InputValidator:
    def __init__(self, release_root: Path, *, data_root: Path | None = None,
                 libero_assets_root: Path | None = None):
        self.root = Path(release_root).absolute()
        self.external_roots = {"data": data_root, "assets": libero_assets_root}
        self.report = ValidationReport()
        self._json: dict[Path, Any] = {}
        self._visited: set[Path] = set()
        self._directories: set[Path] = set()
        self._bddl_names: dict[Path, set[str]] = {}
        self._error_set: set[str] = set()

    def error(self, message: str) -> None:
        if message not in self._error_set:
            self.report.errors.append(message)
            self._error_set.add(message)

    def label(self, path: Path) -> str:
        return str(path.relative_to(self.root)) if path.is_relative_to(self.root) else str(path)

    def read_json(self, path: Path) -> Any:
        if path not in self._json:
            try:
                self._json[path] = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                self.error(f"{self.label(path)}: cannot read JSON: {exc}")
                self._json[path] = None
        return self._json[path]

    def resolve(self, value: str, owner: Path) -> Path:
        path = Path(os.path.expandvars(os.path.expanduser(value)))
        if not path.is_absolute():
            base = self.root if path.parts and path.parts[0] in {
                "source", "assets", "runnable_settings", "paper_facing_configs", "scripts",
            } else owner.parent
            path = base / path
        # Preserve XML symlink locations: adjacent meshes may differ by ratio.
        return Path(os.path.abspath(path))

    def check_path(self, path: Path, context: str, *, directory: bool = False) -> bool:
        if not path.is_relative_to(self.root):
            external = external_libero_reference(path)
            if external is None:
                self.error(f"{context}: input is outside the release and is not standard LIBERO data: {path}")
                return False
            kind, suffix = external
            configured = self.external_roots[kind]
            if configured is None:
                self.report.unconfigured_external.add(kind)
                return False
            expected = Path(configured).expanduser().absolute() / suffix
            self.report.external_files.add(expected)
            if not expected.is_file():
                self.report.external_errors.append(f"{context}: missing external LIBERO input: {expected}")
            elif path.resolve() != expected.resolve():
                self.report.external_errors.append(
                    f"{context}: external reference needs relocate_paths.py: {path} -> {expected}"
                )
            return False
        if not path.exists():
            self.error(f"{context}: missing bundled input: {self.label(path)}")
            return False
        if directory and not path.is_dir():
            self.error(f"{context}: expected directory: {self.label(path)}")
            return False
        if not path.resolve().is_relative_to(self.root.resolve()):
            self.error(f"{context}: bundled input resolves outside release: {self.label(path)}")
            return False
        self.report.files.add(path)
        return True

    def reference(self, value: str, owner: Path, key: str, *, recurse: bool = True) -> None:
        if not value or value.startswith(("external/results/", "external/full_results/")):
            return
        if value.startswith("external/"):
            self.error(f"{self.label(owner)} [{key}]: required input is only an archive marker: {value}")
            return
        pattern = re.sub(r"\{(?:suite|task)\}", "*", value)
        if "{" in pattern or "}" in pattern:
            self.error(f"{self.label(owner)} [{key}]: unresolved input template: {value}")
            return
        path = self.resolve(pattern, owner)
        if "*" in pattern:
            matches = [Path(item) for item in sorted(glob.glob(str(path)))]
            if not matches:
                self.error(f"{self.label(owner)} [{key}]: input template has no matches: {value}")
            for item in matches:
                self.visit(item, f"{self.label(owner)} [{key}]", recurse=recurse)
        else:
            self.visit(path, f"{self.label(owner)} [{key}]", recurse=recurse)

    def walk_inputs(self, value: Any, owner: Path, key: str = "") -> None:
        if isinstance(value, dict):
            for name, item in value.items():
                if name in PROVENANCE_KEYS or OUTPUT_KEY.search(name) or name.endswith("calibration_source"):
                    continue
                self.walk_inputs(item, owner, name)
        elif isinstance(value, list):
            for item in value:
                self.walk_inputs(item, owner, key)
        elif isinstance(value, str):
            looks_like_path = ("/" in value and Path(value).suffix in INPUT_SUFFIXES
                               and (value.startswith(("/", "../", "./", "source/", "assets/", "runnable_settings/"))))
            if key in INPUT_KEYS or looks_like_path:
                self.reference(value, owner, key)

    def visit(self, path: Path, context: str, *, recurse: bool = True) -> None:
        if not self.check_path(path, context):
            return
        if not recurse:
            return
        if path.is_dir():
            if path in self._directories:
                return
            self._directories.add(path)
            for item in sorted(path.rglob("*")):
                if item.suffix in {".json", ".xml", ".bddl"} and item.is_file():
                    self.visit(item, context)
            return
        if path in self._visited:
            return
        self._visited.add(path)
        if path.suffix == ".json":
            payload = self.read_json(path)
            self.walk_inputs(payload, path)
            self.validate_runtime_spec(payload, path)
        elif path.suffix == ".xml":
            self.validate_xml(path)

    def bddl_object_names(self, path: Path) -> set[str]:
        """Read declared instance names without requiring a simulator install."""
        if path in self._bddl_names:
            return self._bddl_names[path]
        names: set[str] = set()
        try:
            text = re.sub(r";[^\n]*", "", path.read_text(encoding="utf-8"))
            roots: list = []
            stack = [roots]
            for token in re.findall(r"\(|\)|[^\s()]+", text):
                if token == "(":
                    group: list = []
                    stack[-1].append(group)
                    stack.append(group)
                elif token == ")":
                    if len(stack) == 1:
                        raise ValueError("unmatched closing parenthesis")
                    stack.pop()
                else:
                    stack[-1].append(token)
            if len(stack) != 1:
                raise ValueError("unclosed parenthesis")

            def collect(groups):
                for group in groups:
                    if not isinstance(group, list):
                        continue
                    if group and group[0] in (":objects", ":fixtures"):
                        skip_type = False
                        for item in group[1:]:
                            if not isinstance(item, str):
                                raise ValueError("nested object declaration")
                            if item == "-":
                                skip_type = True
                            elif skip_type:
                                skip_type = False
                            else:
                                names.add(item)
                    else:
                        collect(group)

            collect(roots)
            if not names:
                raise ValueError("no object or fixture declarations")
        except (OSError, UnicodeError, ValueError) as exc:
            self.error(f"{self.label(path)}: cannot read BDDL instance names: {exc}")
        self._bddl_names[path] = names
        return names

    def validate_runtime_spec(self, spec: Any, path: Path) -> None:
        """Reject known schema/name mismatches before a policy is loaded."""
        if not isinstance(spec, dict):
            return
        if spec.get("track") == "C1" and spec.get("mask_source") == "object_edit":
            edits = spec.get("object_edits", spec.get("edits", []))
            if not isinstance(edits, list) or not edits:
                self.error(f"{self.label(path)}: C1 object_edit requires a nonempty edit list")
                return
            for index, edit in enumerate(edits):
                context = f"{self.label(path)} [object_edits {index}]"
                if not isinstance(edit, dict):
                    self.error(f"{context}: edit must be an object")
                    continue
                if edit.get("op") not in ("state", "set_joint"):
                    continue
                if not isinstance(edit.get("joint"), str) or not edit["joint"].strip():
                    self.error(f"{context}: state edit requires a joint name")
                mode = edit.get("value_mode", "explicit")
                if mode not in ("explicit", "joint_range_mirror", "farthest_joint_limit"):
                    self.error(f"{context}: unsupported state value_mode {mode!r}")
                if mode == "explicit":
                    value = edit.get("value")
                    if type(value) not in (int, float) or not math.isfinite(value):
                        self.error(f"{context}: explicit state edit requires a finite numeric value")

        if spec.get("schema_version") != "d1_4_suites_full.v1":
            return
        value = spec.get("bddl_file_name_override")
        if not isinstance(value, str) or not value:
            self.error(f"{self.label(path)}: D1 requires a BDDL override")
            return
        bddl = self.resolve(value, path)
        if not self.check_path(bddl, self.label(path)):
            return
        names = self.bddl_object_names(bddl)
        if not names:
            return
        references = []
        if isinstance(spec.get("definition"), dict):
            for key in ("target", "distractor"):
                references.append((f"definition.{key}", spec["definition"].get(key)))
        for key in ("object_scale_overrides", "object_pose_overrides", "same_position_replacements"):
            entries = spec.get(key, [])
            if not isinstance(entries, list):
                self.error(f"{self.label(path)} [{key}]: overrides must be a list")
                continue
            for index, entry in enumerate(entries):
                if not isinstance(entry, dict):
                    self.error(f"{self.label(path)} [{key} {index}]: entry must be an object")
                    continue
                for field in ("object", "target", "replacement", "relative_to", "on_top_of", "z_from", "quat_from"):
                    if field in entry:
                        references.append((f"{key}[{index}].{field}", entry[field]))
                avoid = entry.get("avoid_objects", [])
                if not isinstance(avoid, list):
                    self.error(f"{self.label(path)} [{key} {index}]: avoid_objects must be a list")
                    continue
                for name in avoid:
                    references.append((f"{key}[{index}].avoid_objects", name))
        for field, name in references:
            if not isinstance(name, str) or name not in names:
                self.error(
                    f"{self.label(path)} [{field}]: object {name!r} is not declared "
                    f"in {self.label(bddl)}"
                )

    def validate_xml(self, path: Path) -> None:
        try:
            tree = ET.parse(path)
        except (OSError, ET.ParseError) as exc:
            self.error(f"{self.label(path)}: invalid XML: {exc}")
            return
        self.report.counts["xml"] += 1
        compiler = tree.find("compiler")
        attrs = {} if compiler is None else compiler.attrib
        definitions = {kind: {item.get("name") for item in tree.findall("./asset/" + kind)}
                       for kind in ("mesh", "texture", "material")}
        for element in tree.iter():
            for kind, names in definitions.items():
                if element.get(kind) and element.get(kind) not in names:
                    self.error(f"{self.label(path)}: undefined {kind} {element.get(kind)!r}")
            value = element.get("file")
            if value is None:
                continue
            self.report.counts["xml_file_references"] += 1
            folder = attrs.get("meshdir" if element.tag == "mesh" else
                               "texturedir" if element.tag == "texture" else "", "")
            dependency = Path(value)
            if not dependency.is_absolute():
                dependency = path.parent / folder / dependency
            self.visit(Path(os.path.abspath(dependency)), f"{self.label(path)} <{element.tag}>")

    @staticmethod
    def rows(payload: Any) -> list:
        if not isinstance(payload, dict):
            return []
        rows = payload.get("episodes", payload.get("cases", []))
        return rows if isinstance(rows, list) else []

    def row_input(self, row: dict, owner: Path) -> Path | None:
        for key in ("spec", "case_json", "spec_path", "case_path"):
            value = row.get(key)
            if isinstance(value, str) and value:
                return self.resolve(value, owner)
        return None

    def validate_manifest(self, path: Path, setting: dict, *, batch: bool = False,
                          suite: str | None = None) -> list[dict]:
        payload = self.read_json(path)
        if not isinstance(payload, dict):
            self.error(f"{self.label(path)}: manifest must be a JSON object")
            return []
        if payload.get("setting_id") != setting["setting_id"]:
            self.error(f"{self.label(path)}: setting_id must be {setting['setting_id']!r}")
        if batch and not isinstance(payload.get("cases"), list):
            self.error(f"{self.label(path)}: FastWAM batch requires a cases list")
            return []
        rows = payload.get("cases", []) if batch else self.rows(payload)
        if not rows:
            self.error(f"{self.label(path)}: manifest has no cases/episodes")
        seen: set[tuple] = set()
        groups: dict[tuple, set[int]] = defaultdict(set)
        valid_rows = []
        trials = setting["trials_per_case"]
        for index, row in enumerate(rows):
            context = f"{self.label(path)} row {index}"
            if not isinstance(row, dict):
                self.error(f"{context}: case must be an object")
                continue
            valid_rows.append(row)
            task = row.get("task_id")
            case_suite = row.get("suite")
            case_id = row.get("case_id")
            if (not isinstance(case_suite, str) or case_suite not in SUITES
                    or (suite is not None and case_suite != suite)):
                self.error(f"{context}: invalid suite {case_suite!r}")
            if type(task) is not int or not 0 <= task < 10:
                self.error(f"{context}: task_id must be an integer in [0, 9]")
            if not isinstance(case_id, str) or not case_id:
                self.error(f"{context}: missing case_id")
            trial = row.get("trial_index")
            if batch or trial is not None:
                if type(trial) is not int or not 0 <= trial < trials:
                    self.error(f"{context}: trial_index must be an integer in [0, {trials - 1}]")
                else:
                    groups[(str(case_suite), str(task), str(case_id))].add(trial)
            identity = (str(case_id), str(trial)) if batch else (
                str(case_suite), str(task), str(case_id), str(trial))
            if identity in seen:
                self.error(f"{context}: duplicate {'(case_id, trial_index)' if batch else 'task/case/trial'} {identity}")
            seen.add(identity)
            if batch and not isinstance(row.get("spec"), str):
                self.error(f"{context}: FastWAM batch requires spec")
            reference = self.row_input(row, path)
            if reference is None:
                self.error(f"{context}: case has no input spec/case_json/spec_path")
                continue
            aliases = {self.resolve(row[key], path) for key in
                       ("spec", "case_json", "spec_path", "case_path")
                       if isinstance(row.get(key), str) and row[key]}
            if len(aliases) > 1:
                self.error(f"{context}: spec/case input aliases refer to different files")
            self.visit(reference, context)
            if reference.suffix == ".json" and reference.is_file():
                spec = self.read_json(reference)
                if isinstance(spec, dict):
                    spec_suite = spec.get("suite", spec.get("benchmark"))
                    spec_task = spec.get("task_id")
                    if spec_suite is not None and spec_suite != case_suite:
                        self.error(f"{context}: suite disagrees with {self.label(reference)}")
                    if spec_task is not None and spec_task != task:
                        self.error(f"{context}: task_id disagrees with {self.label(reference)}")
        for identity, indices in groups.items():
            if indices != set(range(trials)):
                self.error(f"{self.label(path)}: incomplete trials for {identity}: {sorted(indices)}")
        if isinstance(payload.get("episodes"), list) and isinstance(payload.get("cases"), list):
            def identities(items):
                return Counter(tuple(str(row.get(key)) for key in
                                     ("suite", "task_id", "case_id", "trial_index"))
                               + (str(self.row_input(row, path)),)
                               for row in items if isinstance(row, dict))
            if identities(payload["episodes"]) != identities(payload["cases"]):
                self.error(f"{self.label(path)}: cases and episodes identify different trials")
        return valid_rows

    def validate_timing_observation(self, setting: dict, setting_path: Path) -> None:
        """Require explicit masking targets for every task the timing runner visits."""
        context = f"{self.label(setting_path)} [observation_spec]"
        value = setting.get("observation_spec")
        if not isinstance(value, str) or not value.strip():
            self.error(f"{context}: timing settings require an observation spec path")
            return
        path = self.resolve(value, setting_path)
        if not self.check_path(path, context):
            return
        if not path.is_file():
            self.error(f"{context}: observation spec must be a JSON file")
            return
        # This is an active dependency even if config_files omits it.
        self.visit(path, context)
        observation = self.read_json(path)
        if not isinstance(observation, dict):
            self.error(f"{self.label(path)}: observation spec must be a JSON object")
            return
        if observation.get("target_policy") != "object_names":
            self.error(f"{self.label(path)}: timing masking requires target_policy='object_names'")
        targets = observation.get("target_objects_by_task")
        if not isinstance(targets, dict):
            self.error(f"{self.label(path)}: target_objects_by_task must be an object")
            return
        for suite in sorted(SUITES):
            for task in range(10):
                key = f"{suite}:{task}"
                names = targets.get(key)
                if not (isinstance(names, list) and names
                        and all(isinstance(name, str) and name.strip() for name in names)):
                    self.error(
                        f"{self.label(path)}: target_objects_by_task[{key!r}] "
                        "must be a nonempty list of nonempty object names"
                    )

    def validate_setting(self, setting_path: Path) -> None:
        setting_path = Path(setting_path).absolute()
        setting = self.read_json(setting_path)
        if not isinstance(setting, dict):
            self.error(f"{self.label(setting_path)}: setting must be a JSON object")
            return
        expected_id = f"{setting_path.parent.parent.name}.{setting_path.parent.name}"
        category = setting.get("category")
        if setting.get("setting_id") != expected_id:
            self.error(f"{self.label(setting_path)}: setting_id must match directory: {expected_id}")
        if not isinstance(setting.get("setting_id"), str):
            return
        if category != setting_path.parent.parent.name:
            self.error(f"{self.label(setting_path)}: category does not match directory")
        if type(setting.get("trials_per_case")) is not int or setting["trials_per_case"] <= 0:
            self.error(f"{self.label(setting_path)}: trials_per_case must be a positive integer")
            return
        if setting.get("asset_contract_schema") != ASSET_CONTRACT_SCHEMA:
            self.error(f"{self.label(setting_path)}: invalid asset contract schema")
        try:
            contract = contract_for(category)
            for key in ("asset_mode", "asset_roots", "external_dependencies"):
                if setting.get(key) != contract[key]:
                    self.error(f"{self.label(setting_path)}: {key} disagrees with canonical asset contract")
        except (KeyError, TypeError):
            self.error(f"{self.label(setting_path)}: unknown category {category!r}")
        for key in ("source_roots", "config_files", "asset_roots"):
            values = setting.get(key)
            if not isinstance(values, list) or not values:
                self.error(f"{self.label(setting_path)}: {key} must be a nonempty list")
                continue
            for value in values:
                if not isinstance(value, str):
                    self.error(f"{self.label(setting_path)}: {key} entries must be paths")
                    continue
                if key in {"source_roots", "asset_roots"}:
                    path = self.resolve(value, setting_path)
                    if not self.check_path(path, f"{self.label(setting_path)} [{key}]", directory=True):
                        continue
                self.reference(value, setting_path, key, recurse=key != "source_roots")
        if setting.get("worker"):
            self.reference(setting["worker"], setting_path, "worker")
        if category in {"A1", "A2"} and setting.get("backend") == "timing_client":
            self.validate_timing_observation(setting, setting_path)
        canonical_rows = []
        if setting.get("manifest"):
            manifest = self.resolve(setting["manifest"], setting_path)
            self.visit(manifest, self.label(setting_path))
            canonical_rows = self.validate_manifest(manifest, setting)
        suite_manifests = setting.get("manifests_by_suite", {})
        if isinstance(suite_manifests, dict):
            def inventory(rows, owner):
                return Counter((str(row.get("suite")), str(row.get("task_id")),
                                str(row.get("case_id")), str(row.get("trial_index")),
                                str(self.row_input(row, owner))) for row in rows)
            for suite, value in suite_manifests.items():
                if not isinstance(value, str):
                    self.error(f"{self.label(setting_path)}: suite manifest must be a path")
                    continue
                path = self.resolve(value, setting_path)
                self.visit(path, self.label(setting_path))
                rows = self.validate_manifest(path, setting, suite=suite)
                if canonical_rows:
                    expected = [row for row in canonical_rows if row.get("suite") == suite]
                    if inventory(rows, path) != inventory(expected, manifest):
                        self.error(f"{self.label(path)}: suite inventory differs from canonical manifest")
            if suite_manifests and canonical_rows:
                if set(suite_manifests) != {row.get("suite") for row in canonical_rows}:
                    self.error(f"{self.label(setting_path)}: suite manifest coverage differs from canonical manifest")
        if setting.get("batch_manifest"):
            batch = self.resolve(setting["batch_manifest"], setting_path)
            self.visit(batch, self.label(setting_path))
            batch_rows = self.validate_manifest(batch, setting, batch=True)
            def cases(rows, owner):
                return {(str(row.get("suite")), str(row.get("task_id")), str(self.row_input(row, owner)))
                        for row in rows}
            if canonical_rows and cases(canonical_rows, manifest) != cases(batch_rows, batch):
                self.error(f"{self.label(batch)}: batch task/spec inventory differs from canonical manifest")
        self.report.counts["settings"] += 1


def validate_setting_inputs(setting_path: Path, release_root: Path) -> list[str]:
    """Check one active input closure; external prerequisites are reported separately."""
    validator = InputValidator(release_root)
    validator.validate_setting(Path(setting_path))
    return validator.report.errors


def validate_release_inputs(release_root: Path, *, data_root: Path | None = None,
                            libero_assets_root: Path | None = None) -> ValidationReport:
    validator = InputValidator(release_root, data_root=data_root,
                               libero_assets_root=libero_assets_root)
    root = validator.root
    index_path = root / "runnable_settings/index.json"
    index = validator.read_json(index_path)
    rows = index.get("settings", []) if isinstance(index, dict) else []
    if not rows:
        validator.error(f"{validator.label(index_path)}: no settings")
    ids: set[str] = set()
    for row in rows:
        if (not isinstance(row, dict) or not isinstance(row.get("path"), str)
                or not isinstance(row.get("id"), str)):
            validator.error(f"{validator.label(index_path)}: invalid setting entry")
            continue
        path = root / "runnable_settings" / row["path"] / "setting.json"
        if row.get("id") in ids:
            validator.error(f"{validator.label(index_path)}: duplicate setting id {row.get('id')}")
        ids.add(row.get("id"))
        setting = validator.read_json(path)
        if isinstance(setting, dict) and row.get("id") != setting.get("setting_id"):
            validator.error(f"{validator.label(index_path)}: index id disagrees with {validator.label(path)}")
        validator.validate_setting(path)
    # Historical JSONs are syntax-checked, but their archived result paths are
    # not runtime dependencies. XML dependencies and links are checked globally.
    for path in sorted(root.rglob("*")):
        if any(part in {".git", "__pycache__"} for part in path.relative_to(root).parts):
            continue
        if path.is_symlink():
            validator.report.counts["symlinks"] += 1
            if not path.exists():
                validator.error(f"{validator.label(path)}: broken symlink")
            elif not path.resolve().is_relative_to(root.resolve()):
                validator.error(f"{validator.label(path)}: symlink leaves release")
            if os.path.isabs(os.readlink(path)):
                validator.error(f"{validator.label(path)}: absolute symlink is not portable")
        if path.is_file() and path.suffix == ".json":
            validator.report.counts["json_syntax"] += 1
            validator.read_json(path)
        elif path.is_file() and path.suffix == ".xml":
            validator.visit(path, "bundled XML")
    validator.report.external_errors = sorted(set(validator.report.external_errors))
    return validator.report
