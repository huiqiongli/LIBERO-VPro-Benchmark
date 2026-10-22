#!/usr/bin/env python3
"""Materialise model-independent setting contracts and reference launchers.

The paper-facing manifest defines the 96 settings. This builder creates
setting-scoped launchers and filtered manifests from the bundled definitions.
The contracts describe perturbations and cases independently of policy;
the launchers use the FastWAM reference implementation.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
RELEASE = HERE.parent
sys.path.insert(0, str(RELEASE))
from libero_vpro.asset_contract import contract_for

PAPER = RELEASE / "PAPER_FACING_SETTINGS.json"


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _rewrite_legacy_path(value: str) -> str:
    """Re-root copied manifests when they still contain checkout paths.

    The source snapshots were assembled from several runs of the benchmark,
    so their manifests may contain either ``/data/.../reviewed_script`` or
    ``/home/.../reviewed_script``.  Generated setting manifests must point to
    copy shipped in this release so copied checkouts use their local inputs.
    """
    text = value
    # Physical perturbation assets have one canonical public location.  The
    # old source trees remain compatibility aliases, but generated manifests
    # should advertise the canonical ``assets/`` paths.
    asset_suffixes = {
        "source/A1_physical_target_object_ratio_20260822/precompiled_variants_full_surface_v3":
            "assets/A1_target_object/precompiled_variants_full_surface_v3",
        "source/A2_physical_receptacle_ratio_20260822/precompiled_variants_full_surface_v3":
            "assets/A2_target_receptacle/precompiled_variants_full_surface_v3",
        "source/D1_4_suites_full/bddl_overrides":
            "assets/D1_task_relevant_distractors/bddl_overrides",
        "source/D2_D3_state_20260823/assets":
            "assets/D2_D3_state_variation/assets",
        "source/D2_D3_state_20260823/bddl_overrides":
            "assets/D2_D3_state_variation/bddl_overrides",
        "source/D4_contact4_physical_3severity_40task_20260815/assets":
            "assets/D4_target_geometry/assets",
    }
    for old, new in asset_suffixes.items():
        if text == str(RELEASE / old) or text.startswith(str(RELEASE / old) + "/"):
            return str(RELEASE / new) + text[len(str(RELEASE / old)):]
    marker = "/reviewed_script/"
    if marker in text:
        tail = text.split(marker, 1)[1]
        if tail.startswith("release/"):
            tail = tail[len("release/"):]
        first = tail.split("/", 1)[0]
        source_root = RELEASE / "source" / first
        for old, new in asset_suffixes.items():
            # Legacy absolute manifests use the source path without the
            # release prefix; map those references to the canonical assets.
            if tail == old or tail.startswith(old + "/"):
                return str(RELEASE / new) + tail[len(old):]
        if source_root.exists():
            return str(source_root / tail.split("/", 1)[1]) if "/" in tail else str(source_root)
        # Preserve unresolved external references for input validation.
        return text
    return text


def _normalize(value):
    if isinstance(value, str):
        return _rewrite_legacy_path(value)
    if isinstance(value, list):
        return [_normalize(item) for item in value]
    if isinstance(value, dict):
        return {key: _normalize(item) for key, item in value.items()}
    return value


def dump(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_normalize(value), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def safe_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")


def ratio_manifest(category: str, setting: str) -> tuple[str, dict]:
    if category == "A1":
        root = "source/A1_physical_target_object_ratio_6models_full_20260822"
        full = RELEASE / root / "manifest.json"
    else:
        root = "source/A2_physical_receptacle_ratio_6models_full_20260822"
        full = RELEASE / root / "manifest.json"
    ratio = int(setting.split("_")[1].replace("pct", ""))
    payload = load(full)
    rows = [row for row in payload["cases"] if int(row["ratio"]) == ratio]
    payload["cases"] = rows
    payload["setting_id"] = f"{category}.{setting}"
    for key in ("case_count", "task_count", "base_case_count", "episode_count"):
        payload.pop(key, None)
    return root, payload


def ratio_batch_manifest(category: str, setting: str) -> dict:
    if category == "A1":
        full = RELEASE / "source/A1_physical_target_object_ratio_6models_full_20260822/fastwam_manifest.json"
    else:
        full = RELEASE / "source/A2_physical_receptacle_ratio_6models_full_20260822/fastwam_manifest.json"
    ratio = int(setting.split("_")[1].replace("pct", ""))
    payload = load(full)
    payload["cases"] = [row for row in payload["cases"] if int(row["ratio"]) == ratio]
    payload["setting_id"] = f"{category}.{setting}"
    return payload


def d4_manifest(setting: str) -> tuple[str, dict]:
    root = "source/D4_contact4_physical_3severity_40task_20260815"
    payload = load(RELEASE / root / "manifest.json")
    family, severity = setting.split(".")
    rows = [row for row in payload["cases"]
            if row.get("mechanism") == family and row.get("severity") == severity]
    payload = {"schema_version": "libero_vpro.setting_manifest.v1", "setting_id": f"D4.{setting}",
               "trials_per_case": 10, "cases": rows}
    return root, payload


def d4_batch_manifest(setting: str, inventory: dict) -> dict:
    """Adapt case-level geometry definitions to one FastWAM row per trial."""
    rows = []
    for case in inventory["cases"]:
        case_id = f"{case['suite']}_t{int(case['task_id']):02d}__{case['case_id']}"
        for trial in range(10):
            rows.append({
                **case,
                "case_id": case_id,
                "semantic_case_id": case["case_id"],
                "spec": case["spec_path"],
                "condition": "perturbed",
                "trial_index": trial,
                "trial_start": trial,
                "num_trials": 1,
            })
    return {"schema_version": "libero_vpro.fastwam_batch_manifest.v1",
            "setting_id": f"D4.{setting}", "trials_per_case": 10, "cases": rows}


def c1_manifest(setting: str) -> tuple[str, dict]:
    root = "source/C1_cross_view_20260825"
    payload = load(RELEASE / root / "episode_manifests/full/all.json")
    subclass, direction = setting.split(".")
    rows = [row for row in payload["episodes"]
            if row.get("subtask") == subclass and row.get("direction") == direction]
    # The archived manifest contains the original absolute checkout.  Rewrite
    # only the case-file roots to the self-contained release copy; the case
    # contents and task selection remain byte-for-byte unchanged.
    release_cases = RELEASE / root / "cases"
    for row in rows:
        for key in ("case_json", "spec"):
            value = str(row.get(key, ""))
            marker = "/cases/"
            if marker in value:
                row[key] = str(release_cases / value.split(marker, 1)[1])
    payload = {"schema_version": "libero_vpro.setting_manifest.v1", "setting_id": f"C1.{setting}",
               "episodes": rows,
               "cases": [{**row, "spec": row.get("case_json", row.get("spec", ""))} for row in rows]}
    return root, payload


def a3_manifest(setting: str, setting_dir: Path) -> tuple[str, dict]:
    """Materialise the accepted A3 task inventory as runnable case rows.

    The archived A3 launch record preserves counts and model coverage but not
    the original case-file directory.  The task IDs below are recovered from
    that accepted result archive (the same inventory used by the statistics),
    while the interaction mask itself comes from the release candidate spec.
    """
    root = "source/A3_interaction_part_occlusion"
    task_ids = {
        "gripper": {
            "libero_10": list(range(10)), "libero_goal": list(range(10)),
            "libero_object": list(range(10)), "libero_spatial": list(range(10)),
        },
        "target_contact": {
            "libero_10": [2], "libero_goal": [2, 9],
            "libero_object": [0, 2, 3, 4, 5, 7, 9],
        },
        "fixture_contact": {
            "libero_10": [2, 3, 9], "libero_goal": [0, 3, 7],
        },
        "bounded_receiver": {
            "libero_10": [0, 1, 7], "libero_object": list(range(10)),
        },
    }
    spec_name = {
        "gripper": "A3_blur_gripper_candidate.json",
        "target_contact": "A3_blur_target_contact_candidate.json",
        "fixture_contact": "A3_blur_fixture_contact_candidate.json",
        "bounded_receiver": "A3_blur_bounded_opening_candidate.json",
    }[setting]
    base = load(RELEASE / root / "specs" / spec_name)
    for key in ("task_region_manifest", "geometry_reference_spec"):
        if base.get(key) and not Path(base[key]).is_absolute():
            base[key] = str((RELEASE / root / "specs" / base[key]).resolve())
    episodes: list[dict] = []
    cases_dir = setting_dir / "cases"
    for suite, ids in task_ids[setting].items():
        for task_id in ids:
            case_id = f"A3_{setting}__{suite}__task{int(task_id):02d}"
            case = {
                **base,
                "track": "A3",
                "subtask": setting,
                "suite": suite,
                "task_id": int(task_id),
                "case_id": case_id,
            }
            case_path = cases_dir / suite / f"task_{int(task_id):02d}.json"
            dump(case_path, case)
            for trial in range(10):
                episodes.append({
                    "case_json": str(case_path), "trial_index": trial,
                    "task_id": int(task_id), "case_id": case_id,
                    "suite": suite, "track": "A3", "subtask": setting,
                    "condition": "interaction_part",
                })
    payload = {
        "schema_version": "libero_vpro.setting_episode_manifest.v1",
        "setting_id": f"A3.{setting}", "trials_per_case": 10,
        "episodes": episodes,
        # FastWAM's current batch evaluator calls this field ``cases`` and
        # expects the same rows under the legacy ``spec`` alias.
        "cases": [{**row, "spec": row["case_json"]} for row in episodes],
    }
    return root, payload


def state_manifest(category: str, setting: str) -> tuple[str, dict, dict[str, dict]]:
    """Materialise the paper-facing D2/D3 cases as ten-trial episode rows.

    The archived state manifests are case-level inventories.  The actual
    model runners consume episode rows. This conversion selects the requested
    setting and expands each case into its ten initial-state trials.
    """
    root = "source/D2_D3_state_20260823"
    if category == "D2":
        cases = load(RELEASE / root / "d2_manifest.json")["cases"]
        if setting.startswith("open_"):
            pct = int(setting.split("_", 1)[1].replace("pct", ""))
            cases = [row for row in cases
                     if row.get("subtask") == "articulation_precondition"
                     and int(row.get("severity_pct", -1)) == pct]
        else:
            cases = [row for row in cases
                     if row.get("subtask") == "object_support_state"
                     and row.get("support_state") == setting]
    else:
        cases = load(RELEASE / root / "d3_manifest.json")["cases"]
        subclass = setting.split(".", 1)[0]
        kind = "articulation_state_ambiguity" if subclass.startswith("articulation") else "object_support_state_ambiguity"
        cases = [row for row in cases
                 if row.get("ambiguity_kind") == kind
                 and row.get("truth_state") == "truth_a"
                 and row.get("observation_mode") == "obfuscated"]
    episodes = []
    by_suite: dict[str, list[dict]] = {}
    for case in cases:
        for trial in range(10):
            row = {
                "case_json": case["case_path"],
                "spec": case["case_path"],
                "trial_index": trial,
                "task_id": int(case["task_id"]),
                "case_id": case["case_id"],
                "suite": case["suite"],
                "track": case.get("track", category),
                "subtask": case.get("subtask", case.get("ambiguity_kind", setting)),
                "condition": setting,
            }
            episodes.append(row)
            by_suite.setdefault(case["suite"], []).append(row)
    payload = {
        "schema_version": "libero_vpro.setting_episode_manifest.v1",
        "setting_id": f"{category}.{setting}",
        "trials_per_case": 10,
        "episodes": episodes,
        "cases": episodes,
    }
    return root, payload, by_suite


def source_and_configs(category: str, setting: str) -> tuple[list[str], list[str], str, dict]:
    """Return source roots, exact config references, backend, backend metadata."""
    src: list[str] = []
    cfg: list[str] = []
    meta: dict = {}
    backend = "shell"
    if category == "A1":
        if setting.startswith("ratio_"):
            root, _ = ratio_manifest(category, setting); src += [root]
            cfg += [f"{root}/manifest.json", "source/A1_physical_target_object_ratio_20260822/configs"]
            backend = "manifest_worker"; meta["worker"] = f"{root}/run_fastwam_worker.sh"
        else:
            src += ["source/A1_timing_variants", "source/formal_timing_manifests"]
            cfg += [f"source/A1_timing_variants/configs/trackC_A1t_{ {'initially_occluded_then_revealed':'masked_then_revealed','initially_visible_then_occluded':'visible_then_masked','pre_grasp':'pregrasp_only','post_grasp':'after_grasp'}[setting]}.json"]
            meta["observation_spec"] = "source/A1_timing_variants/configs/trackA_A1_precise_100pct.json"
            cfg.append(meta["observation_spec"])
            backend = "timing_client"
    elif category == "A2":
        if setting.startswith("ratio_"):
            root, _ = ratio_manifest(category, setting); src += [root]
            cfg += [f"{root}/manifest.json", "source/A2_physical_receptacle_ratio_20260822/configs"]
            backend = "manifest_worker"; meta["worker"] = f"{root}/run_fastwam_worker.sh"
        else:
            src += ["source/A2_placement_timing", "source/formal_timing_manifests"]
            aliases = {"initially_occluded_then_revealed":"masked_then_revealed_replan5", "initially_visible_then_occluded":"visible_then_masked_replan5", "pre_place":"pre_place_only", "post_place":"post_place_40steps"}
            cfg += [f"source/A2_placement_timing/configs/replan5/{aliases[setting]}.json"]
            meta["observation_spec"] = "source/A2_placement_timing/configs/trackA_A2_receptacle_black.json"
            cfg.append(meta["observation_spec"])
            backend = "timing_client"
    elif category == "A3":
        src += ["source/A3_interaction_part_occlusion"]
        aliases = {"gripper":"A3_blur_gripper_candidate.json", "target_contact":"A3_blur_target_contact_candidate.json", "fixture_contact":"A3_blur_fixture_contact_candidate.json", "bounded_receiver":"A3_blur_bounded_opening_candidate.json"}
        cfg += [f"source/A3_interaction_part_occlusion/specs/{aliases[setting]}"]
        backend = "a3_manifest"
    elif category == "D1":
        src += ["source/D1_4_suites_full"]
        cfg += [f"source/D1_4_suites_full/specs/{{suite}}/{{task}}/{setting}.json"]
        backend = "suite_condition"
    elif category in {"D2", "D3"}:
        src += ["source/D2_D3_state_20260823"]
        cfg += [f"source/D2_D3_state_20260823/{'d2_manifest.json' if category == 'D2' else 'd3_manifest.json'}", "source/D2_D3_state_20260823/state_runtime.py"]
        backend = "state_manifest"
    elif category == "D4":
        src += ["source/D4_contact4_physical_3severity_40task_20260815"]
        cfg += ["source/D4_contact4_physical_3severity_40task_20260815/manifest.json"]
        backend = "manifest_worker"; meta["worker"] = "source/D4_contact4_physical_3severity_40task_20260815/eval_6models/run_fastwam_worker.sh"
    elif category == "A4":
        src += ["source/A4_temporary_observation_corruption"]
        # A4 duration labels D1/D2/D3 select contiguous replan windows;
        # event-count labels N1/N2/N3 select discrete replan indices.
        if setting.startswith(("black_D", "noise_D")) and setting[-1:] in {"1", "2", "3"}:
            kind = "black" if setting.startswith("black") else "noise"
            duration = setting[-1]
            cfg += [
                f"source/A4_temporary_observation_corruption/configs/trackA_A4obs_fullscreen_{'black' if kind == 'black' else 'noise_v2'}.json",
                f"source/A4_temporary_observation_corruption/configs/schedule_D{duration}.json",
            ]
            backend = "a4_duration"
        elif setting == "target_region_noise":
            cfg += ["paper_facing_configs/A4/target_region_noise.json", "source/A4_temporary_observation_corruption/configs/trackA_A4rand_3replan.json"]
            backend = "a4_target_region_noise"
        elif setting.startswith(("black_", "noise_")) and setting.endswith("_3replan_once"):
            phases = ("reach", "pre_grasp", "grasp", "transport", "pre_place")
            phase = next((name for name in phases if f"_{name}_" in setting), None)
            if phase is None:
                raise ValueError(f"unrecognised A4 phase setting: {setting}")
            cfg += [
                "source/A4_temporary_observation_corruption/configs/trackA_A4obs_fullscreen_black.json",
                "source/A4_temporary_observation_corruption/configs/trackA_A4obs_fullscreen_noise_v2.json",
                f"source/A4_temporary_observation_corruption/configs/phase_full/{phase}.json",
            ]
            backend = "a4_phase"
        else:
            kind = "black" if setting.startswith("black") else "noise"
            n = setting.rsplit("N", 1)[-1] if "_N" in setting else setting.split("_")[1].replace("replan", "")
            cfg += [f"source/A4_temporary_observation_corruption/configs/trackA_A4obs_fullscreen_{'black' if kind == 'black' else 'noise_v2'}.json", f"source/A4_temporary_observation_corruption/configs/trackA_A4rand_{n}replan.json"]
            backend = "a4_fixed"
    elif category == "B1":
        src += ["source/B1_camera_staleness", "source/adapted"]
        c2_name = setting.replace("replay_", "replayv2_")
        cfg += [f"source/B1_camera_staleness/configs/trackB_B1_{c2_name}.json"]
        backend = "b1_condition"
    elif category == "C1":
        root, _ = c1_manifest(setting); src += [root]
        cfg += [f"{root}/episode_manifests/full/all.json", f"{root}/cases/{setting.split('.')[0]}"]
        backend = "c1_manifest"
    elif category == "C2":
        src += ["source/C2_prediction_reality_fastwam"]
        cfg += ["source/C2_prediction_reality_fastwam/recursive_feedback_schedule.json"]
        backend = "c2_feedback"
    elif category == "A5":
        src += ["source/A5_single_view", "source/A5_view_copy", "source/A5_blank_view_control", "source/A5_white_view_control"]
        cfg += ["source/A5_single_view/README.md", "source/A5_view_copy/README.md", "source/A5_blank_view_control/README.md", "source/A5_white_view_control/README.md"]
        backend = "a5_condition"
    return src, cfg, backend, meta


def main() -> None:
    manifest = load(PAPER)
    root = HERE / "settings"
    # Update only files owned by the builder. Preserve annotations and any
    # additional user files; stale directories are never recursively deleted.
    root.mkdir(parents=True, exist_ok=True)
    index = {"schema_version": "libero_vpro.runnable_settings.v1", "setting_count": 0, "settings": []}
    for category in manifest["categories"]:
        cid = category["id"]
        for setting in category["settings"]:
            sid = setting["id"]
            src, cfg, backend, meta = source_and_configs(cid, sid)
            assets = contract_for(cid)
            full_id = f"{cid}.{sid}"
            setting_dir = root / cid / safe_id(sid)
            setting_dir.mkdir(parents=True, exist_ok=True)
            result_source = f"external/full_results/by_experiment_subclass_model/{cid}/{sid}"
            if cid == "A4" and sid == "target_region_noise":
                # The setting uses the accepted segmented-target Gaussian-noise
                # episodes; the release name is normalized, but the archived
                # result directory is retained as its provenance.
                result_source = "external/full_results/by_experiment_subclass_model/A4/_legacy_segmented_target_noise"
            payload = {
                "schema_version": "libero_vpro.runnable_setting.v1",
                "setting_id": full_id,
                "category": cid,
                "trials_per_case": 10,
                "backend": backend,
                "source_roots": src,
                "config_files": cfg,
                "asset_contract_schema": assets["asset_contract_schema"],
                "asset_mode": assets["asset_mode"],
                "asset_roots": assets["asset_roots"],
                "external_dependencies": assets["external_dependencies"],
                "full_result_source": result_source,
                **meta,
            }
            dump(setting_dir / "setting.json", payload)
            (setting_dir / "run.sh").write_text(
                "#!/usr/bin/env bash\nset -Eeuo pipefail\nHERE=$(cd \"$(dirname \"${BASH_SOURCE[0]}\")\" && pwd)\nexec python3 \"$HERE/../../../run_setting.py\" --setting \"$HERE/setting.json\" \"$@\"\n",
                encoding="utf-8")
            (setting_dir / "run.sh").chmod(0o755)
            if cid in {"A1", "A2"} and sid.startswith("ratio_"):
                _, filtered = ratio_manifest(cid, sid)
                dump(setting_dir / "manifest.json", filtered)
                payload["manifest"] = "manifest.json"
                dump(setting_dir / "batch_manifest.json", ratio_batch_manifest(cid, sid))
                payload["batch_manifest"] = "batch_manifest.json"
            elif cid == "D4":
                _, filtered = d4_manifest(sid)
                dump(setting_dir / "manifest.json", filtered)
                payload["manifest"] = "manifest.json"
                dump(setting_dir / "batch_manifest.json", d4_batch_manifest(sid, filtered))
                payload["batch_manifest"] = "batch_manifest.json"
            elif cid == "C1":
                _, filtered = c1_manifest(sid)
                dump(setting_dir / "manifest.json", filtered)
                payload["manifest"] = "manifest.json"
            elif cid == "A3":
                _, filtered = a3_manifest(sid, setting_dir)
                dump(setting_dir / "manifest.json", filtered)
                payload["manifest"] = "manifest.json"
            elif cid in {"D2", "D3"}:
                _, filtered, by_suite = state_manifest(cid, sid)
                dump(setting_dir / "manifest.json", filtered)
                payload["manifest"] = "manifest.json"
                payload["manifests_by_suite"] = {}
                for suite, rows in sorted(by_suite.items()):
                    suite_payload = {
                        **filtered,
                        "suite": suite,
                        "episodes": rows,
                        "cases": rows,
                    }
                    name = f"manifest_{suite}.json"
                    dump(setting_dir / name, suite_payload)
                    payload["manifests_by_suite"][suite] = name
            dump(setting_dir / "setting.json", payload)
            index["settings"].append({"id": full_id, "path": str(setting_dir.relative_to(HERE)), "backend": backend})
    index["setting_count"] = len(index["settings"])
    dump(HERE / "index.json", index)
    print(f"generated {index['setting_count']} runnable setting directories under {root}")


if __name__ == "__main__":
    main()
