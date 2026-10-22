#!/usr/bin/env python3
"""Agent-view-referenced semantic validation for every C1 case.

This validator is intentionally fail-closed.  A generated C1 case is accepted
only when all ten official initial states expose the relevant fact in the
third-person ``agentview`` and a temporary simulator edit produces the intended
counterfactual there.  The wrist camera is still checked for direction isolation
but is not required to see the target in the initial state.  Reports are written
to a new directory; case definitions and result records are never edited.
"""

from __future__ import annotations

import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np


HERE = Path(__file__).resolve().parent
HOME_SIM = Path("/home/huiqiongli/Sim")
SOURCE = HERE.parent / "C1_case_revalidation_20260825/validated_cases.json"
OUTPUT = HERE / "agentview_reference_revalidation"
MIN_VISIBLE_PIXELS = 64
MIN_CHANGED_PIXELS = 64
MIN_TARGET_CHANGE_RATIO = 0.50
MIN_LOCAL_DIFF_RATIO = 0.55
MIN_POSITION_SHIFT_PIXELS = 12.0
VIEWS = ("agentview", "robot0_eye_in_hand")
VIEW_KEYS = {
    "agentview": ("agentview_image", "agentview_segmentation_instance"),
    "robot0_eye_in_hand": (
        "robot0_eye_in_hand_image",
        "robot0_eye_in_hand_segmentation_instance",
    ),
}
DUMMY = [0.0] * 7

sys.path[:0] = [
    str(HERE / "isolated_sources/lingbot_va/evaluation/libero"),
    str(HERE),
    str(HOME_SIM),
    str(HOME_SIM / "Repos/LIBERO"),
]

from libero.libero import benchmark  # noqa: E402
import client as libero_client  # noqa: E402
import robust_libero_perturbations as c1_runtime  # noqa: E402


def bddl_section(text: str, name: str, following: tuple[str, ...]) -> str:
    match = re.search(
        rf"\(:{re.escape(name)}\b(.*?)(?=\n\s*\(:(?:{'|'.join(following)})\b)",
        text,
        re.S | re.I,
    )
    return match.group(1) if match else ""


def parse_objects(text: str) -> dict[str, str]:
    result: dict[str, str] = {}
    body = bddl_section(text, "objects", ("obj_of_interest", "init", "goal"))
    for raw in body.splitlines():
        line = raw.strip().rstrip(")")
        if " - " not in line:
            continue
        names, category = line.split(" - ", 1)
        for name in names.split():
            result[name] = category.strip()
    return result


def named_tokens(section: str) -> set[str]:
    return set(re.findall(r"\b[a-zA-Z][a-zA-Z0-9_]*_\d+\b", section))


def clean_segmentation(obs: dict, view: str) -> np.ndarray:
    return np.asarray(obs[VIEW_KEYS[view][1]]).squeeze(-1)


def visible_pixels(obs: dict, env: Any, view: str, instance: str) -> int:
    instance_id = env.instance_to_id.get(instance)
    if instance_id is None:
        return 0
    return int(np.count_nonzero(clean_segmentation(obs, view) == int(instance_id)))


def instance_mask(obs: dict, env: Any, view: str, instance: str) -> np.ndarray:
    instance_id = env.instance_to_id.get(instance)
    if instance_id is None:
        return np.zeros(clean_segmentation(obs, view).shape, dtype=bool)
    return clean_segmentation(obs, view) == int(instance_id)


def mapped_render_segmentation(env: Any, view: str, height: int, width: int) -> np.ndarray:
    base = getattr(env, "env", env)
    raw = np.asarray(
        base.sim.render(
            camera_name=view,
            width=width,
            height=height,
            segmentation=True,
        )
    )
    geom_ids = raw[..., 1]
    mapped = np.zeros((height, width), dtype=np.int32)
    for geom_id, instance in base.model.geom_ids_to_instances.items():
        instance_id = env.instance_to_id.get(instance)
        if instance_id is not None:
            mapped[geom_ids == int(geom_id)] = int(instance_id)
    return mapped


def parent_instance(env: Any, target: str) -> str | None:
    if target in env.instance_to_id:
        return target
    candidates = [name for name in env.instance_to_id if target.startswith(f"{name}_")]
    return max(candidates, key=len) if candidates else None


def object_contacts(base_env: Any, target: str, movable_instances: set[str]) -> dict[str, float]:
    contacts: dict[str, float] = {}
    sim = base_env.sim
    mapping = base_env.model.geom_ids_to_instances
    for index in range(int(sim.data.ncon)):
        contact = sim.data.contact[index]
        first = mapping.get(int(contact.geom1))
        second = mapping.get(int(contact.geom2))
        if first == target and second in movable_instances and second != target:
            contacts[str(second)] = min(contacts.get(str(second), 0.0), float(contact.dist))
        elif second == target and first in movable_instances and first != target:
            contacts[str(first)] = min(contacts.get(str(first), 0.0), float(contact.dist))
    return contacts


def apply_edit_for_segmentation(
    env: Any,
    spec: dict,
    view: str,
    height: int,
    width: int,
    movable_instances: set[str],
) -> tuple[np.ndarray, dict]:
    """Render an edited instance map and restore the simulator byte-for-byte."""
    base = getattr(env, "env", env)
    sim = base.sim
    before_state = sim.get_state()
    before_qpos = np.array(sim.data.qpos, copy=True)
    before_qvel = np.array(sim.data.qvel, copy=True)
    edit = dict(spec["object_edits"][0])
    operation = str(edit["op"]).lower()
    metrics: dict[str, Any] = {"operation": operation}
    target_object = str(edit.get("object", ""))
    before_contacts = (
        object_contacts(base, target_object, movable_instances)
        if target_object else {}
    )
    try:
        if operation == "disappear":
            joint = c1_runtime._free_joint_name(base, str(edit["object"]))
            if joint is None:
                raise ValueError(f"no free joint for {edit['object']}")
            qpos = c1_runtime._get_joint_qpos(sim, joint)
            qpos[2] = float(edit.get("hidden_z", -10.0))
            c1_runtime._set_joint_qpos(sim, joint, qpos)
        elif operation == "relocate":
            joint = c1_runtime._free_joint_name(base, str(edit["object"]))
            if joint is None:
                raise ValueError(f"no free joint for {edit['object']}")
            qpos = c1_runtime._get_joint_qpos(sim, joint)
            old_xyz = np.asarray(qpos[:3], dtype=float).copy()
            offset = np.asarray(edit.get("offset", [0.12, 0.0, 0.0]), dtype=float)
            qpos[:3] += offset
            c1_runtime._set_joint_qpos(sim, joint, qpos)
            metrics.update({
                "old_xyz": old_xyz.tolist(),
                "new_xyz": np.asarray(qpos[:3], dtype=float).tolist(),
                "offset_m": offset.tolist(),
                "finite_pose": bool(np.all(np.isfinite(qpos))),
                "z_preserved": bool(abs(float(qpos[2] - old_xyz[2])) < 1e-9),
            })
        elif operation == "state":
            joint = str(edit["joint"])
            current = float(np.asarray(c1_runtime._get_joint_qpos(sim, joint)).reshape(-1)[0])
            joint_id = int(sim.model.joint_name2id(joint))
            low, high = (float(v) for v in sim.model.jnt_range[joint_id])
            if not high > low:
                raise ValueError(f"invalid joint range {joint}: {low}, {high}")
            value_mode = str(edit.get("value_mode", "joint_range_mirror"))
            if value_mode == "farthest_joint_limit":
                value = (
                    low
                    if abs(current - low) >= abs(high - current)
                    else high
                )
            else:
                value = low + high - current
            value = float(np.clip(value, low, high))
            c1_runtime._set_joint_qpos(sim, joint, np.array([value], dtype=float))
            metrics.update({
                "original_value": current,
                "counterfactual_value": value,
                "value_mode": value_mode,
                "joint_range": [low, high],
                "normalized_state_separation": abs(value - current) / (high - low),
            })
        else:
            raise ValueError(f"unsupported edit {operation}")
        c1_runtime._forward_sim(sim)
        if operation == "relocate":
            after_contacts = object_contacts(base, target_object, movable_instances)
            new_contacts = {
                name: distance
                for name, distance in after_contacts.items()
                if name not in before_contacts and distance < -1e-4
            }
            metrics.update({
                "movable_object_contacts_before": before_contacts,
                "movable_object_contacts_after": after_contacts,
                "new_penetrating_object_contacts": new_contacts,
            })
        segmentation = mapped_render_segmentation(env, view, height, width)
    finally:
        sim.set_state(before_state)
        c1_runtime._forward_sim(sim)
    restored = bool(
        np.allclose(np.asarray(sim.data.qpos), before_qpos, rtol=0.0, atol=1e-10)
        and np.allclose(np.asarray(sim.data.qvel), before_qvel, rtol=0.0, atol=1e-10)
    )
    metrics["sim_state_restored"] = restored
    return segmentation, metrics


def centroid(mask: np.ndarray) -> tuple[float, float] | None:
    rows, cols = np.nonzero(mask)
    if not len(rows):
        return None
    return float(rows.mean()), float(cols.mean())


def expanded(mask: np.ndarray, radius: int = 7) -> np.ndarray:
    kernel = np.ones((radius * 2 + 1, radius * 2 + 1), dtype=np.uint8)
    return cv2.dilate(mask.astype(np.uint8), kernel, iterations=1).astype(bool)


def locality(changed: np.ndarray, relevant: np.ndarray) -> float:
    total = int(changed.sum())
    return float(np.count_nonzero(changed & expanded(relevant)) / total) if total else 0.0


def static_validate(
    semantic: dict,
    pair: dict[str, dict],
    objects: dict[str, str],
    interest: set[str],
    goal: set[str],
) -> list[str]:
    errors: list[str] = []
    if set(pair) != {"agent", "wrist"}:
        errors.append("missing_symmetric_direction_pair")
        return errors
    expected = {
        "agent": ("agentview", "robot0_eye_in_hand"),
        "wrist": ("robot0_eye_in_hand", "agentview"),
    }
    for direction, (corrupted, truth) in expected.items():
        spec = pair[direction]
        if spec.get("corrupted_view") != corrupted or spec.get("truth_view") != truth:
            errors.append(f"{direction}:incorrect_view_direction")
        if spec.get("views") != [corrupted]:
            errors.append(f"{direction}:edit_not_limited_to_selected_view")
        if len(spec.get("object_edits") or []) != 1:
            errors.append(f"{direction}:not_exactly_one_counterfactual_edit")
    subclass = semantic["subclass"]
    target = semantic["target"]
    if subclass in {"existence", "position"}:
        if target not in objects:
            errors.append("target_not_a_movable_task_object")
        if target not in interest and target not in goal:
            errors.append("target_not_task_relevant")
    elif subclass == "count":
        family = semantic.get("semantic_family", "")
        target_category = objects.get(target, "")
        family_instances = [name for name, category in objects.items() if family in category]
        exact_category_relevant = any(
            name != target
            and objects.get(name) == target_category
            and (name in interest or name in goal)
            for name in family_instances
        )
        if len(family_instances) < 2:
            errors.append("fewer_than_two_same_family_instances")
        if target not in interest and target not in goal and not exact_category_relevant:
            errors.append("removed_instance_not_task_relevant_same_category")
    elif subclass == "state":
        if not target:
            errors.append("missing_articulated_target")
    return sorted(set(errors))


def validate_direction(
    semantic: dict,
    spec: dict,
    obs: dict,
    env: Any,
    objects: dict[str, str],
) -> dict:
    selected = spec["corrupted_view"]
    truth = spec["truth_view"]
    subclass = semantic["subclass"]
    target = semantic["target"]
    target_instance = parent_instance(env, target)
    selected_image_key = VIEW_KEYS[selected][0]
    truth_image_key = VIEW_KEYS[truth][0]

    # First exercise the case exactly as it will run: modify only the selected
    # camera and ensure the other camera remains byte-identical.  The selected
    # wrist view is allowed to contain no target pixels at the initial state.
    model_obs, audit = c1_runtime.apply_obs_noise_to_raw_obs(
        obs, env, spec, semantic["suite"], int(semantic["task_id"])
    )
    clean_selected = np.asarray(obs[selected_image_key])
    edited_selected = np.asarray(model_obs[selected_image_key])
    selected_changed = np.any(clean_selected != edited_selected, axis=-1)
    truth_unchanged = bool(
        np.array_equal(np.asarray(obs[truth_image_key]), np.asarray(model_obs[truth_image_key]))
    )

    # Independently render the same physical counterfactual through agentview.
    # This is the semantic reference for both contradiction directions, so a
    # wrist-target visibility failure cannot reject an otherwise valid case.
    reference_spec = dict(spec)
    reference_spec["views"] = ["agentview"]
    reference_spec["corrupted_view"] = "agentview"
    reference_spec["truth_view"] = "robot0_eye_in_hand"
    reference_obs, reference_audit = c1_runtime.apply_obs_noise_to_raw_obs(
        obs,
        env,
        reference_spec,
        semantic["suite"],
        int(semantic["task_id"]),
    )
    reference_image_key = VIEW_KEYS["agentview"][0]
    clean_reference = np.asarray(obs[reference_image_key])
    edited_reference = np.asarray(reference_obs[reference_image_key])
    reference_changed = np.any(clean_reference != edited_reference, axis=-1)
    edited_seg, edit_metrics = apply_edit_for_segmentation(
        env,
        reference_spec,
        "agentview",
        int(clean_reference.shape[0]),
        int(clean_reference.shape[1]),
        set(objects),
    )
    clean_target_reference = (
        instance_mask(obs, env, "agentview", target_instance)
        if target_instance else np.zeros(reference_changed.shape, dtype=bool)
    )
    edited_target_reference = (
        edited_seg == int(env.instance_to_id[target_instance])
        if target_instance in env.instance_to_id
        else np.zeros(reference_changed.shape, dtype=bool)
    )
    relevant = clean_target_reference | edited_target_reference
    selected_changed_pixels = int(selected_changed.sum())
    reference_changed_pixels = int(reference_changed.sum())
    target_change_ratio = float(
        np.count_nonzero(reference_changed & clean_target_reference)
        / max(1, int(clean_target_reference.sum()))
    )
    local_ratio = locality(reference_changed, relevant)
    checks = {
        "truth_view_unchanged": truth_unchanged,
        "selected_view_render_executed": bool(audit.get("enabled")),
        "sim_state_restored": bool(audit.get("sim_state_restored"))
        and bool(reference_audit.get("sim_state_restored"))
        and bool(edit_metrics.get("sim_state_restored")),
        "clean_fact_visible_agentview": (
            int(clean_target_reference.sum()) >= MIN_VISIBLE_PIXELS
        ),
        "semantic_reference_changed_agentview": (
            reference_changed_pixels >= MIN_CHANGED_PIXELS
        ),
    }
    if selected == "agentview":
        checks["selected_agentview_changed"] = (
            selected_changed_pixels >= MIN_CHANGED_PIXELS
        )
        checks["selected_agentview_matches_reference"] = bool(
            np.array_equal(edited_selected, edited_reference)
        )
    if subclass in {"existence", "count"}:
        checks.update({
            "target_removed_from_agentview_reference": (
                int(edited_target_reference.sum()) == 0
            ),
            "target_pixels_actually_changed_in_agentview": (
                target_change_ratio >= MIN_TARGET_CHANGE_RATIO
            ),
        })
    elif subclass == "position":
        old_center = centroid(clean_target_reference)
        new_center = centroid(edited_target_reference)
        shift = (
            float(np.hypot(old_center[0] - new_center[0], old_center[1] - new_center[1]))
            if old_center is not None and new_center is not None else 0.0
        )
        checks.update({
            "alternate_position_visible_in_agentview": (
                int(edited_target_reference.sum()) >= MIN_VISIBLE_PIXELS
            ),
            "positions_visually_distinct_in_agentview": (
                shift >= MIN_POSITION_SHIFT_PIXELS
            ),
            "alternate_pose_finite": bool(edit_metrics.get("finite_pose")),
            "alternate_pose_preserves_support_height": bool(edit_metrics.get("z_preserved")),
            "alternate_pose_has_no_new_object_penetration": not bool(
                edit_metrics.get("new_penetrating_object_contacts")
            ),
        })
        edit_metrics["centroid_shift_pixels"] = shift
    elif subclass == "state":
        checks.update({
            "opposite_state_is_meaningfully_separated": float(
                edit_metrics.get("normalized_state_separation", 0.0)
            ) >= 0.50,
            "state_change_visually_effective_in_agentview": (
                reference_changed_pixels >= MIN_CHANGED_PIXELS
            ),
        })

    if subclass == "count":
        family = semantic["semantic_family"]
        instances = [name for name, category in objects.items() if family in category]
        clean_counts = {
            name: visible_pixels(obs, env, "agentview", name)
            for name in instances
        }
        edited_visible_instances = sum(
            int(np.count_nonzero(edited_seg == int(env.instance_to_id[name]))) >= MIN_VISIBLE_PIXELS
            for name in instances if name in env.instance_to_id
        )
        expected_clean = len(instances)
        checks.update({
            "all_family_instances_visible_in_clean_agentview": all(
                pixels >= MIN_VISIBLE_PIXELS for pixels in clean_counts.values()
            ),
            "agentview_count_differs_by_exactly_one": (
                edited_visible_instances == expected_clean - 1
            ),
        })
        edit_metrics["clean_family_pixels_agentview"] = clean_counts
        edit_metrics["edited_visible_family_instances"] = edited_visible_instances
        edit_metrics["expected_clean_family_instances"] = expected_clean

    return {
        "passed": all(checks.values()),
        "selected_view": selected,
        "truth_view": truth,
        "checks": checks,
        "metrics": {
            "selected_view_changed_pixels": selected_changed_pixels,
            "agentview_reference_changed_pixels": reference_changed_pixels,
            "target_change_ratio_agentview": target_change_ratio,
            "local_diff_ratio_agentview": local_ratio,
            "local_diff_ratio_is_diagnostic_only": True,
            "clean_target_pixels_agentview": int(clean_target_reference.sum()),
            "clean_target_pixels_wrist_diagnostic": (
                visible_pixels(obs, env, "robot0_eye_in_hand", target_instance)
                if target_instance else 0
            ),
            "edited_target_pixels_agentview": int(edited_target_reference.sum()),
            **edit_metrics,
        },
    }


def main() -> int:
    started = time.time()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    source = json.loads(SOURCE.read_text())
    semantic_cases: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for subclass, rows in source["cases"].items():
        for row in rows:
            semantic_cases[(row["suite"], int(row["task_id"]))].append({
                **row,
                "subclass": subclass,
            })

    generated_pairs: dict[str, dict[str, dict]] = defaultdict(dict)
    for path in sorted((HERE / "cases").rglob("*.json")):
        spec = json.loads(path.read_text())
        generated_pairs[spec["semantic_case_id"]][spec["direction"]] = spec

    suites = benchmark.get_benchmark_dict()
    case_reports: dict[str, dict] = {}
    errors: list[dict] = []
    task_items = sorted(semantic_cases.items())
    for task_number, ((suite_name, task_id), rows) in enumerate(task_items, start=1):
        suite = suites[suite_name]()
        states = suite.get_task_init_states(task_id)
        bddl_path = Path(rows[0]["bddl_file"])
        text = bddl_path.read_text()
        objects = parse_objects(text)
        interest = named_tokens(bddl_section(text, "obj_of_interest", ("init", "goal")))
        goal = named_tokens(bddl_section(text, "goal", ("metric",)))
        env = libero_client.construct_single_env(
            {
                "bddl_file_name": suite.get_task_bddl_file_path(task_id),
                "camera_heights": 256,
                "camera_widths": 256,
            },
            use_segmentation=True,
        )
        if env is None:
            for semantic in rows:
                case_reports[semantic["case_id"]] = {
                    "passed": False,
                    "static_errors": ["environment_construction_failed"],
                    "trials": [],
                }
            continue
        try:
            env.reset()
            for semantic in rows:
                pair = generated_pairs.get(semantic["case_id"], {})
                case_reports[semantic["case_id"]] = {
                    "subclass": semantic["subclass"],
                    "suite": suite_name,
                    "task_id": task_id,
                    "instruction": semantic["instruction"],
                    "target": semantic["target"],
                    "static_errors": static_validate(
                        semantic, pair, objects, interest, goal
                    ),
                    "trials": [],
                }
            for init_index in range(10):
                obs = env.set_init_state(states[init_index])
                done = False
                for _ in range(5):
                    obs, _, done, _ = env.step(DUMMY)
                for semantic in rows:
                    report = case_reports[semantic["case_id"]]
                    pair = generated_pairs.get(semantic["case_id"], {})
                    trial = {"init_state_index": init_index, "directions": {}}
                    if done:
                        trial["error"] = "clean_environment_done_during_settle"
                    elif set(pair) == {"agent", "wrist"}:
                        for direction in ("agent", "wrist"):
                            try:
                                trial["directions"][direction] = validate_direction(
                                    semantic, pair[direction], obs, env, objects
                                )
                            except Exception as exc:  # fail closed, preserve diagnosis
                                trial["directions"][direction] = {
                                    "passed": False,
                                    "error": f"{type(exc).__name__}: {exc}",
                                }
                    trial["passed"] = (
                        not trial.get("error")
                        and len(trial["directions"]) == 2
                        and all(item.get("passed") for item in trial["directions"].values())
                    )
                    report["trials"].append(trial)
        finally:
            env.close()
        (OUTPUT / "progress.json").write_text(json.dumps({
            "status": "running",
            "completed_tasks": task_number,
            "expected_tasks": len(task_items),
            "current_suite": suite_name,
            "current_task_id": task_id,
            "elapsed_seconds": time.time() - started,
        }, indent=2) + "\n")
        print(
            f"validated task {task_number}/{len(task_items)}: {suite_name} t{task_id}",
            flush=True,
        )

    subclass_counts = Counter()
    failure_reasons = Counter()
    accepted: list[str] = []
    rejected: list[str] = []
    for case_id, report in sorted(case_reports.items()):
        report["passed_trials"] = sum(bool(item.get("passed")) for item in report["trials"])
        report["expected_trials"] = 10
        report["passed"] = (
            not report["static_errors"]
            and len(report["trials"]) == 10
            and report["passed_trials"] == 10
        )
        subclass_counts[(report.get("subclass", "unknown"), report["passed"])] += 1
        (accepted if report["passed"] else rejected).append(case_id)
        for reason in report["static_errors"]:
            failure_reasons[reason] += 1
        for trial in report["trials"]:
            for direction in trial.get("directions", {}).values():
                for name, passed in direction.get("checks", {}).items():
                    if not passed:
                        failure_reasons[name] += 1
                if direction.get("error"):
                    failure_reasons[direction["error"]] += 1

    payload = {
        "schema_version": "wam_syncbench.c1_agentview_reference_revalidation.v2",
        "passed": not rejected,
        "acceptance_policy": {
            "all_ten_init_states_required": True,
            "minimum_visible_pixels_256x256": MIN_VISIBLE_PIXELS,
            "minimum_changed_pixels": MIN_CHANGED_PIXELS,
            "minimum_target_change_ratio": MIN_TARGET_CHANGE_RATIO,
            "local_diff_ratio": "diagnostic_only_due_to_physical_shadows_and_occlusion",
            "minimum_position_centroid_shift_pixels": MIN_POSITION_SHIFT_PIXELS,
            "synchronized_dual_view": True,
            "single_selected_view_counterfactual": True,
            "semantic_reference_view": "agentview",
            "wrist_initial_target_visibility_required": False,
            "temporary_simulator_state_must_restore": True,
        },
        "semantic_case_count": len(case_reports),
        "direction_configuration_count": len(generated_pairs) * 2,
        "accepted_count": len(accepted),
        "rejected_count": len(rejected),
        "accepted_case_ids": accepted,
        "rejected_case_ids": rejected,
        "subclass_counts": {
            subclass: {
                "accepted": subclass_counts[(subclass, True)],
                "rejected": subclass_counts[(subclass, False)],
            }
            for subclass in ("existence", "position", "state", "count")
        },
        "failure_reason_occurrences": dict(failure_reasons.most_common()),
        "elapsed_seconds": time.time() - started,
        "cases": case_reports,
    }
    (OUTPUT / "strict_revalidation_report.json").write_text(
        json.dumps(payload, indent=2) + "\n"
    )
    lines = [
        "# C1 agentview-reference revalidation",
        "",
        f"- Overall pass: **{payload['passed']}**",
        f"- Accepted: **{len(accepted)} / {len(case_reports)}** semantic cases",
        "- Every accepted case passed both directions on all 10 official init states.",
        "- Semantic correctness is judged from agentview; initial wrist visibility is diagnostic only.",
        "",
        "| subclass | accepted | rejected |",
        "|---|---:|---:|",
    ]
    for subclass, counts in payload["subclass_counts"].items():
        lines.append(f"| {subclass} | {counts['accepted']} | {counts['rejected']} |")
    lines.extend(["", "## Rejected cases", ""])
    for case_id in rejected:
        report = case_reports[case_id]
        reasons = set(report["static_errors"])
        for trial in report["trials"]:
            for direction in trial.get("directions", {}).values():
                reasons.update(
                    name for name, passed in direction.get("checks", {}).items() if not passed
                )
                if direction.get("error"):
                    reasons.add(direction["error"])
        lines.append(f"- `{case_id}`: {', '.join(sorted(reasons))}")
    (OUTPUT / "STRICT_REVALIDATION_REPORT.md").write_text("\n".join(lines) + "\n")
    (OUTPUT / "progress.json").write_text(json.dumps({
        "status": "complete",
        "completed_tasks": len(task_items),
        "expected_tasks": len(task_items),
        "elapsed_seconds": time.time() - started,
        "accepted_cases": len(accepted),
        "rejected_cases": len(rejected),
    }, indent=2) + "\n")
    print(json.dumps({
        "passed": payload["passed"],
        "accepted": len(accepted),
        "rejected": len(rejected),
        "subclass_counts": payload["subclass_counts"],
        "report": str(OUTPUT / "strict_revalidation_report.json"),
    }, indent=2))
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
