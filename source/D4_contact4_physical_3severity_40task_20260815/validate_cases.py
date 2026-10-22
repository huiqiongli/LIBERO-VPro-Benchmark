#!/usr/bin/env python3
"""Compile, render, and physics-check every three-severity D4 case."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
SIM = Path("/home/huiqiongli/Sim")
sys.path[:0] = [str(SIM), str(SIM / "Repos/FastWAM"), str(SIM / "Repos/LIBERO")]
os.environ.setdefault("MUJOCO_GL", "egl")

from libero.libero import benchmark  # noqa: E402
from experiments.libero.libero_utils import get_libero_env, get_libero_dummy_action  # noqa: E402
import robust_libero_perturbations as perturb  # noqa: E402

PREVIEWS = HERE / "previews"
REPORTS = HERE / "validation"


def body_ids(base, target: str) -> list[int]:
    root = int(base.obj_body_id[target])
    found = [root]
    for bid in range(base.sim.model.nbody):
        cur = bid
        while cur > 0:
            cur = int(base.sim.model.body_parentid[cur])
            if cur == root:
                found.append(bid)
                break
    return sorted(set(found))


def audit_target(base, target: str) -> dict:
    ids = body_ids(base, target)
    root = ids[0]
    names = []
    for gid in range(base.sim.model.ngeom):
        if int(base.sim.model.geom_bodyid[gid]) in ids:
            names.append(base.sim.model.geom_id2name(gid) or "")
    return {
        "target": target,
        "root_body_id": root,
        "mass_kg": float(base.sim.model.body_mass[root]),
        "inertia": np.asarray(base.sim.model.body_inertia[root]).tolist(),
        "d4_collision_patch_count": sum("d4_contact_" in n for n in names),
        "position": np.asarray(base.sim.data.body_xpos[root]).tolist(),
        "quaternion": np.asarray(base.sim.data.body_xquat[root]).tolist(),
    }


def make_env(suite_name: str, task_id: int, spec: dict | None):
    suite = benchmark.get_benchmark_dict()[suite_name]()
    task = suite.get_task(task_id)
    mapping = perturb.object_xml_override_map(spec)
    env, _ = get_libero_env(task, 256, 42, object_xml_override_map=mapping or None)
    # A hard reset recompiles MJCF; keep the override active for that compile.
    with perturb.libero_env_override_context(object_xml_override_map_value=mapping or None):
        env.reset()
    state = suite.get_task_init_states(task_id)[0]
    obs = env.set_init_state(state)
    return env, obs


def image_from(obs: dict, key: str) -> np.ndarray:
    # Save raw simulator orientation; the validation is geometry/view based.
    return np.ascontiguousarray(np.asarray(obs[key])[::-1])


def paired_preview(clean: dict[str, np.ndarray], changed: dict[str, np.ndarray], labels: tuple[str, str]) -> Image.Image:
    rows = []
    for key in ("agentview_image", "robot0_eye_in_hand_image"):
        pair = np.concatenate([clean[key], changed[key]], axis=1)
        rows.append(pair)
    canvas = Image.fromarray(np.concatenate(rows, axis=0))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, canvas.width, 22), fill=(0, 0, 0))
    draw.text((8, 5), labels[0], fill=(255, 255, 255))
    draw.text((264, 5), labels[1], fill=(255, 255, 255))
    return canvas


def main() -> None:
    manifest = json.loads((HERE / "manifest.json").read_text())
    suite_filter = os.environ.get("D4_VALIDATE_SUITE", "").strip()
    case_filter = {x.strip() for x in os.environ.get("D4_VALIDATE_CASE_IDS", "").split(",") if x.strip()}
    grouped: dict[tuple[str, int], list[dict]] = {}
    for case in manifest["cases"]:
        if suite_filter and case["suite"] != suite_filter:
            continue
        if case_filter and case["case_id"] not in case_filter:
            continue
        grouped.setdefault((case["suite"], case["task_id"]), []).append(case)
    PREVIEWS.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    rows = []

    for (suite_name, task_id), cases in sorted(grouped.items()):
        clean_env, clean_obs = make_env(suite_name, task_id, None)
        try:
            clean_base = getattr(clean_env, "env", clean_env)
            clean_frames = {k: image_from(clean_obs, k) for k in ("agentview_image", "robot0_eye_in_hand_image")}
            clean_audits = {t: audit_target(clean_base, t) for t in cases[0]["targets"]}
            clean_before_pos = {t: np.asarray(clean_audits[t]["position"]) for t in cases[0]["targets"]}
            for _ in range(25):
                clean_obs, _, _, _ = clean_env.step(get_libero_dummy_action())
            clean_settle_drift = {
                t: float(np.linalg.norm(np.asarray(audit_target(clean_base, t)["position"]) - clean_before_pos[t]))
                for t in cases[0]["targets"]
            }
        finally:
            clean_env.close()

        for case in cases:
            spec = json.loads(Path(case["spec_path"]).read_text())
            row = {k: case[k] for k in ("suite", "task_id", "case_id", "mechanism", "targets")}
            try:
                env, obs = make_env(suite_name, task_id, spec)
                try:
                    base = getattr(env, "env", env)
                    frames = {k: image_from(obs, k) for k in ("agentview_image", "robot0_eye_in_hand_image")}
                    before = {t: audit_target(base, t) for t in case["targets"]}
                    for _ in range(25):
                        obs, _, _, _ = env.step(get_libero_dummy_action())
                    after = {t: audit_target(base, t) for t in case["targets"]}
                    per_target = []
                    for target in case["targets"]:
                        stock = clean_audits[target]
                        changed = before[target]
                        drift = float(np.linalg.norm(np.asarray(after[target]["position"]) - np.asarray(changed["position"])))
                        per_target.append({
                            "target": target,
                            "patch_count_ok": changed["d4_collision_patch_count"] == 24,
                            "mass_unchanged": abs(changed["mass_kg"] - stock["mass_kg"]) < 1e-8,
                            "inertia_unchanged": bool(np.allclose(changed["inertia"], stock["inertia"], atol=1e-10, rtol=1e-7)),
                            "settle_drift_m": drift,
                            "clean_settle_drift_m": clean_settle_drift[target],
                            # Some official states intentionally drop an object
                            # (e.g. a bowl into a drawer). Compare to clean rather
                            # than imposing an incorrect absolute drift bound.
                            "stable": np.isfinite(drift) and abs(drift - clean_settle_drift[target]) < 0.025,
                            "before": changed, "after": after[target], "clean": stock,
                        })
                    deltas = {k: float(np.mean(np.abs(frames[k].astype(float) - clean_frames[k].astype(float)))) for k in frames}
                    visible_change = max(deltas.values()) > 0.015
                    passed = all(x["patch_count_ok"] and x["mass_unchanged"] and x["inertia_unchanged"] and x["stable"] for x in per_target) and visible_change
                    row.update({"status": "passed" if passed else "failed", "view_pixel_deltas": deltas,
                                "visible_change": visible_change, "targets_audit": per_target})
                    out = PREVIEWS / suite_name / f"task_{task_id:02d}" / f"{case['case_id']}.png"
                    out.parent.mkdir(parents=True, exist_ok=True)
                    paired_preview(clean_frames, frames, ("clean (agent / wrist)", case["case_id"])).save(out)
                    row["preview_path"] = str(out)
                finally:
                    env.close()
            except Exception as exc:
                row.update({"status": "error", "error": f"{type(exc).__name__}: {exc}"})
            rows.append(row)
            print(f"{row['status']:>6} {suite_name} task={task_id:02d} {case['case_id']}", flush=True)

    summary = {
        "schema_version": "d4_contact4_3severity_validation.v1", "total": len(rows),
        "passed": sum(r["status"] == "passed" for r in rows),
        "failed": sum(r["status"] == "failed" for r in rows),
        "errors": sum(r["status"] == "error" for r in rows),
        "skipped_tasks": manifest["skipped_tasks"], "rows": rows,
    }
    if case_filter:
        suffix = suite_filter or "all"
        dump_path = REPORTS / f"validation_spotcheck_{suffix}.json"
    else:
        dump_path = REPORTS / (f"validation_report_{suite_filter}.json" if suite_filter else "validation_report.json")
    dump_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({k: summary[k] for k in ("total", "passed", "failed", "errors")}, ensure_ascii=False))
    if summary["failed"] or summary["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
