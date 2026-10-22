#!/usr/bin/env python3
"""Build four *physical* contact-band interventions at three severities.

The visible mesh is deformed locally in the grasp band and matching zero-mass
collision patches are added outside the stock collision envelope.  Therefore
the intervention changes what the policy sees and what the gripper contacts,
without changing friction, mass, inertia, the observation interface, or policy inference.
"""

from __future__ import annotations

import json
import math
import re
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path


HERE = Path(__file__).resolve().parent
SIM = Path("/home/huiqiongli/Sim")
ASSETS = SIM / "Repos/LIBERO/libero/libero/assets"
INVENTORY = Path("/data/huiqiongli/Sim/reviewed_script/D4_shape_grasp_40task_20260812/task_inventory.json")
ASSET_OUT = HERE / "assets"
SPEC_OUT = HERE / "specs"

# Four visually observable physical deformation families.  The parameter values
# used by the 2026-08-13 smoke are preserved exactly as ``low``; medium and high
# scale only the geometric displacement away from the stock mesh.
MECHANISMS = {
    "taper": "axial_slip_from_opposed_slopes",
    "twist": "height_dependent_contact_normal_misalignment",
    "multipeak": "multi_peak_contact_profile",
    "asymmetric": "left_right_asymmetric_contact_band",
}

SEVERITY_MULTIPLIERS = {
    "low": 1.0,
    "medium": 1.5,
    "high": 2.0,
}

# A scanned bowl has a wide rim and several benchmark states place it directly
# on another small object.  Applying the nominal multipliers to that rim changes
# the reset support contacts.  Use monotonic, category-safe effective strengths
# while preserving the original low setting exactly.
CATEGORY_EFFECTIVE_STRENGTHS = {
    "akita_black_bowl": {"low": 0.45, "medium": 0.55, "high": 0.65},
}

CATEGORY_MECHANISM_EFFECTIVE_STRENGTHS = {
    ("akita_black_bowl", "twist"): {"low": 0.45, "medium": 0.50, "high": 0.55},
}

# Some HOPE assets are authored on their side and rotated upright by LIBERO.
# The contact band must follow the authored longitudinal axis.
LOCAL_UP_AXIS = {
    "alphabet_soup": 1, "salad_dressing": 1, "ketchup": 1,
    "tomato_sauce": 1, "milk": 1, "orange_juice": 1,
}

# Goal task 6 grasps cream cheese and uses the bowl as the receptacle.
TARGET_FIXUPS = {("libero_goal", 6): ["cream_cheese_1"]}
FORCED_SKIPS = {
    ("libero_goal", 3): "target bowl starts inside a close-fitting top drawer; physical outward contact-band edits interpenetrate the drawer at reset",
}

# Official benchmark order from libero.libero.benchmark, not filesystem sorting.
OFFICIAL_ORDER = {
    "libero_object": [
        "pick_up_the_alphabet_soup_and_place_it_in_the_basket.bddl",
        "pick_up_the_cream_cheese_and_place_it_in_the_basket.bddl",
        "pick_up_the_salad_dressing_and_place_it_in_the_basket.bddl",
        "pick_up_the_bbq_sauce_and_place_it_in_the_basket.bddl",
        "pick_up_the_ketchup_and_place_it_in_the_basket.bddl",
        "pick_up_the_tomato_sauce_and_place_it_in_the_basket.bddl",
        "pick_up_the_butter_and_place_it_in_the_basket.bddl",
        "pick_up_the_milk_and_place_it_in_the_basket.bddl",
        "pick_up_the_chocolate_pudding_and_place_it_in_the_basket.bddl",
        "pick_up_the_orange_juice_and_place_it_in_the_basket.bddl",
    ],
    "libero_spatial": [
        "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate.bddl",
        "pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate.bddl",
        "pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate.bddl",
        "pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate.bddl",
        "pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate.bddl",
        "pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate.bddl",
        "pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate.bddl",
        "pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate.bddl",
        "pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate.bddl",
        "pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate.bddl",
    ],
    "libero_goal": [
        "open_the_middle_drawer_of_the_cabinet.bddl", "put_the_bowl_on_the_stove.bddl",
        "put_the_wine_bottle_on_top_of_the_cabinet.bddl", "open_the_top_drawer_and_put_the_bowl_inside.bddl",
        "put_the_bowl_on_top_of_the_cabinet.bddl", "push_the_plate_to_the_front_of_the_stove.bddl",
        "put_the_cream_cheese_in_the_bowl.bddl", "turn_on_the_stove.bddl",
        "put_the_bowl_on_the_plate.bddl", "put_the_wine_bottle_on_the_rack.bddl",
    ],
    "libero_10": [
        "LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket.bddl",
        "LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket.bddl",
        "KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it.bddl",
        "KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it.bddl",
        "LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate.bddl",
        "STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy.bddl",
        "LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate.bddl",
        "LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket.bddl",
        "KITCHEN_SCENE8_put_both_moka_pots_on_the_stove.bddl",
        "KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it.bddl",
    ],
}


def dump(value: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def category(instance: str) -> str:
    return re.sub(r"_\d+$", "", instance)


def source_dir(cat: str) -> Path:
    preferred = {
        "akita_black_bowl": "stable_scanned_objects",
        "wine_bottle": "turbosquid_objects",
        "moka_pot": "turbosquid_objects",
        "white_yellow_mug": "turbosquid_objects",
        "porcelain_mug": "turbosquid_objects",
        "black_book": "turbosquid_objects",
    }
    if cat in preferred:
        path = ASSETS / preferred[cat] / cat
        if (path / f"{cat}.xml").exists():
            return path
    matches = sorted(ASSETS.glob(f"*/{cat}/{cat}.xml"))
    if not matches:
        raise FileNotFoundError(f"No LIBERO asset XML for {cat}")
    return matches[0].parent


def numbers(text: str | None, default: tuple[float, ...]) -> tuple[float, ...]:
    if not text:
        return default
    return tuple(float(x) for x in text.split())


def primary_mesh(tree: ET.ElementTree, src: Path) -> tuple[ET.Element, Path, tuple[float, float, float]]:
    root = tree.getroot()
    body = root.find("./worldbody/body/body[@name='object']")
    if body is None:
        raise RuntimeError(f"No object body in {src}")
    mesh_names = [g.get("mesh") for g in body.findall("geom") if g.get("group") == "1" and g.get("mesh")]
    for name in mesh_names:
        mesh = root.find(f"./asset/mesh[@name='{name}']")
        if mesh is None:
            continue
        file_name = Path(mesh.get("file", "")).name
        candidates = [
            src / file_name.replace("_vis.msh", ".obj"),
            src / file_name.replace(".msh", ".obj"),
            src / file_name.replace("visual/", "").replace("_vis.msh", ".obj"),
        ]
        candidates += sorted(src.glob("*.obj"))
        for candidate in candidates:
            if candidate.exists() and "_col" not in candidate.stem and "coll" not in candidate.stem:
                scale = numbers(mesh.get("scale"), (1.0, 1.0, 1.0))
                if len(scale) == 1:
                    scale = scale * 3
                return mesh, candidate, scale[:3]
    raise RuntimeError(f"Could not resolve primary visual OBJ in {src}")


def obj_bounds(path: Path, scale: tuple[float, float, float]) -> tuple[list[float], list[float]]:
    points = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.startswith("v "):
            vals = [float(v) for v in line.split()[1:4]]
            points.append([vals[i] * scale[i] for i in range(3)])
    if not points:
        raise RuntimeError(f"No vertices in {path}")
    return [min(p[i] for p in points) for i in range(3)], [max(p[i] for p in points) for i in range(3)]


def profile(mechanism: str, t: float, strength: float = 1.0) -> tuple[float, float, float, float, float]:
    """Return x/y scale, rotation, and normalized x/y centre offset."""
    if mechanism == "taper":
        s = 1.01 + 0.17 * t
        raw = (s, s, 0.0, 0.0, 0.0)
    elif mechanism == "twist":
        raw = (1.20, 1.04, math.radians(-45.0 + 90.0 * t), 0.0, 0.0)
    elif mechanism == "multipeak":
        wave = 0.5 + 0.5 * math.cos(6.0 * math.pi * t)
        raw = (1.04 + 0.14 * wave, 1.03 + 0.09 * wave, 0.0, 0.0, 0.0)
    elif mechanism == "asymmetric":
        raw = (1.12, 1.06, 0.0, 0.06 * math.sin(math.pi * t), 0.0)
    else:
        raise KeyError(mechanism)
    sx, sy, angle, ox, oy = raw
    return 1 + (sx - 1) * strength, 1 + (sy - 1) * strength, angle * strength, ox * strength, oy * strength


def transform_obj(source: Path, destination: Path, scale: tuple[float, float, float], mechanism: str,
                  lo: list[float], hi: list[float], axis: int, strength: float) -> None:
    cross = [i for i in range(3) if i != axis]
    u, v = cross
    cu, cv = (lo[u] + hi[u]) / 2, (lo[v] + hi[v]) / 2
    hu, hv = max((hi[u] - lo[u]) / 2, 1e-6), max((hi[v] - lo[v]) / 2, 1e-6)
    a0, a1 = lo[axis] + 0.24 * (hi[axis] - lo[axis]), lo[axis] + 0.78 * (hi[axis] - lo[axis])
    blend_width = max(0.08 * (a1 - a0), 1e-5)
    output = []
    for line in source.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line.startswith("v "):
            output.append(line)
            continue
        parts = line.split()
        raw = [float(v) for v in parts[1:4]]
        xyz = [raw[i] * scale[i] for i in range(3)]
        axial = xyz[axis]
        if axial < a0 - blend_width or axial > a1 + blend_width:
            output.append(line)
            continue
        tc = min(max((axial - a0) / max(a1 - a0, 1e-9), 0.0), 1.0)
        sx, sy, angle, ox, oy = profile(mechanism, tc, strength)
        dx, dy = xyz[u] - cu, xyz[v] - cv
        tx, ty = sx * dx, sy * dy
        nu = math.cos(angle) * tx - math.sin(angle) * ty + cu + ox * hu
        nv = math.sin(angle) * tx + math.cos(angle) * ty + cv + oy * hv
        if axial < a0:
            blend = (axial - (a0 - blend_width)) / blend_width
        elif axial > a1:
            blend = ((a1 + blend_width) - axial) / blend_width
        else:
            blend = 1.0
        raw[u] = ((1 - blend) * xyz[u] + blend * nu) / scale[u]
        raw[v] = ((1 - blend) * xyz[v] + blend * nv) / scale[v]
        suffix = " " + " ".join(parts[4:]) if len(parts) > 4 else ""
        output.append(f"v {raw[0]:.9f} {raw[1]:.9f} {raw[2]:.9f}{suffix}")
    destination.write_text("\n".join(output) + "\n", encoding="utf-8")


def add_collision_patches(body: ET.Element, mechanism: str, lo: list[float], hi: list[float], axis: int, strength: float) -> int:
    """Add exterior patches; retain stock collisions to preserve hollow objects."""
    cross = [i for i in range(3) if i != axis]
    u, v = cross
    cu, cv = (lo[u] + hi[u]) / 2, (lo[v] + hi[v]) / 2
    hu, hv = max((hi[u] - lo[u]) / 2, 0.006), max((hi[v] - lo[v]) / 2, 0.006)
    a0, a1 = lo[axis] + 0.24 * (hi[axis] - lo[axis]), lo[axis] + 0.78 * (hi[axis] - lo[axis])
    patch = max(min(hu, hv) * 0.12, 0.0010)
    count = 0
    for iz in range(6):
        t = iz / 5
        sx, sy, angle, ox, oy = profile(mechanism, t, strength)
        axial = a0 + t * (a1 - a0)
        # Four cardinal contact patches cover arbitrary opposing approaches
        # without forming a second solid shell around hollow objects.
        for ip in range(4):
            theta = 2 * math.pi * ip / 4
            ex, ey = sx * hu * math.cos(theta), sy * hv * math.sin(theta)
            radius = max(math.hypot(ex, ey), patch)
            inset = max(radius - patch, 0.0) / radius
            ex, ey = ex * inset, ey * inset
            pu = math.cos(angle) * ex - math.sin(angle) * ey + cu + ox * hu
            pv = math.sin(angle) * ex + math.cos(angle) * ey + cv + oy * hv
            pos = [0.0, 0.0, 0.0]
            pos[u], pos[v], pos[axis] = pu, pv, axial
            size = [patch, patch, patch]
            size[axis] = max((a1-a0)/11, patch)
            ET.SubElement(body, "geom", {
                "name": f"d4_contact_{mechanism}_{iz:02d}_{ip:02d}",
                "type": "ellipsoid",
                "pos": " ".join(f"{x:.7f}" for x in pos),
                "size": " ".join(f"{x:.7f}" for x in size),
                "group": "0", "mass": "0", "friction": "0.95 0.3 0.1",
                "solimp": "0.998 0.998 0.001", "solref": "0.001 1",
                "rgba": "0.8 0.8 0.8 0.12",
            })
            count += 1
    return count


def build_asset(cat: str, mechanism: str, severity: str) -> tuple[Path, dict]:
    src = source_dir(cat)
    dst = ASSET_OUT / cat / mechanism / severity
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    xml_in = src / f"{cat}.xml"
    tree = ET.parse(xml_in)
    mesh, obj, scale = primary_mesh(tree, src)
    lo, hi = obj_bounds(obj, scale)
    transformed = dst / f"{obj.stem}_d4_{mechanism}.obj"
    axis = LOCAL_UP_AXIS.get(cat, 2)
    # Bowls often start inside close-fitting fixtures.  Retain their established
    # category safety factor, then apply the same low/medium/high multiplier as
    # every other category.
    category_strength = 0.45 if cat == "akita_black_bowl" else 1.0
    nominal_severity_multiplier = SEVERITY_MULTIPLIERS[severity]
    strength = CATEGORY_MECHANISM_EFFECTIVE_STRENGTHS.get((cat, mechanism), {}).get(
        severity,
        CATEGORY_EFFECTIVE_STRENGTHS.get(cat, {}).get(
            severity, category_strength * nominal_severity_multiplier
        ),
    )
    effective_severity_multiplier = strength / category_strength
    transform_obj(obj, transformed, scale, mechanism, lo, hi, axis, strength)
    mesh.set("file", transformed.name)
    body = tree.getroot().find("./worldbody/body/body[@name='object']")
    if body is None:
        raise RuntimeError(f"No object body in {xml_in}")
    patch_count = add_collision_patches(body, mechanism, lo, hi, axis, strength)
    xml_out = dst / f"{cat}.xml"
    ET.indent(tree, space="  ")
    tree.write(xml_out, encoding="unicode")
    return xml_out, {
        "category": cat, "mechanism": mechanism, "severity": severity,
        "source_xml": str(xml_in),
        "xml_path": str(xml_out), "visual_source_obj": str(obj),
        "visual_mesh_scale": list(scale), "local_bounds_m": {"min": lo, "max": hi},
        "contact_axis": axis,
        "category_strength": category_strength,
        "nominal_severity_multiplier": nominal_severity_multiplier,
        "effective_severity_multiplier": effective_severity_multiplier,
        "effective_strength": strength,
        "contact_band_axis_range_m": [lo[axis] + .24*(hi[axis]-lo[axis]), lo[axis] + .78*(hi[axis]-lo[axis])],
        "collision_patch_count": patch_count,
    }


def main() -> None:
    inv = json.loads(INVENTORY.read_text(encoding="utf-8"))
    by_bddl = {(r["suite"], Path(r["bddl"]).name): r for r in inv["tasks"]}
    official_rows = []
    for suite, names in OFFICIAL_ORDER.items():
        for task_id, name in enumerate(names):
            row = dict(by_bddl[(suite, name)])
            row["task_id"] = task_id
            # Remove scene prefixes from the old filename-derived text.
            row["instruction"] = name.rsplit("_SCENE", 1)[-1]
            row["instruction"] = re.sub(r"^\d+_", "", row["instruction"]).removesuffix(".bddl").replace("_", " ")
            official_rows.append(row)
    tasks = []
    categories = set()
    for row in official_rows:
        targets = TARGET_FIXUPS.get((row["suite"], row["task_id"]), row["shape_targets"])
        forced_reason = FORCED_SKIPS.get((row["suite"], row["task_id"]))
        if forced_reason:
            targets = []
        eligible = bool(targets)
        task = dict(row)
        task["shape_targets"] = targets
        task["eligible"] = eligible
        if forced_reason:
            task["reason"] = forced_reason
        elif not eligible:
            task["reason"] = "no grasped movable target; contact-band deformation is not applicable"
        tasks.append(task)
        categories.update(category(x) for x in targets)

    assets = {}
    for cat in sorted(categories):
        assets[cat] = {}
        for mechanism in MECHANISMS:
            assets[cat][mechanism] = {}
            for severity in SEVERITY_MULTIPLIERS:
                xml_path, audit = build_asset(cat, mechanism, severity)
                assets[cat][mechanism][severity] = {"xml_path": str(xml_path), "audit": audit}

    cases = []
    skipped = []
    for task in tasks:
        if not task["eligible"]:
            skipped.append({k: task[k] for k in ("suite", "task_id", "instruction", "reason")})
            continue
        for mechanism, mechanism_name in MECHANISMS.items():
            for severity, severity_multiplier in SEVERITY_MULTIPLIERS.items():
                case_id = f"{mechanism}_{severity}"
                overrides = []
                for target in task["shape_targets"]:
                    cat = category(target)
                    asset_audit = assets[cat][mechanism][severity]["audit"]
                    overrides.append({
                        "object": target, "category": cat, "scale": 1.0,
                        "xml_path": assets[cat][mechanism][severity]["xml_path"],
                        "effective_strength": asset_audit["effective_strength"],
                        "effective_severity_multiplier": asset_audit["effective_severity_multiplier"],
                        "apply_stage": "precompile_xml",
                    })
                spec = {
                    "schema_version": "d4_contact4_physical_3severity.v1", "status": "implemented",
                    "type": "pure_external_shape_grasp_difficulty", "suite": task["suite"],
                    "benchmark": task["suite"], "task_id": task["task_id"],
                    "instruction": task["instruction"], "base_bddl_file_name": task["bddl"],
                    "target_objects": task["shape_targets"], "object_scale_overrides": overrides,
                    "shape_intervention": {
                        "case_id": case_id, "mechanism": mechanism_name, "family": mechanism,
                        "severity": severity, "nominal_severity_multiplier": severity_multiplier,
                        "category_adaptive_safety_scaling": True,
                        "external_shape_modified": True,
                        "visual_mesh_modified": True, "collision_geometry_modified": True,
                        "friction_modified": False, "mass_or_inertia_modified": False,
                        "inference_modified": False, "apply_stage": "precompile_xml",
                    },
                }
                spec_path = SPEC_OUT / task["suite"] / f"task_{task['task_id']:02d}" / f"{case_id}.json"
                dump(spec, spec_path)
                cases.append({
                    "suite": task["suite"], "task_id": task["task_id"], "instruction": task["instruction"],
                    "case_id": case_id, "mechanism": mechanism, "severity": severity,
                    "severity_multiplier": severity_multiplier, "targets": task["shape_targets"],
                    "spec_path": str(spec_path), "trial_start": 0, "num_trials": 1,
                })

    manifest = {
        "schema_version": "d4_contact4_physical_3severity_manifest.v1",
        "scope": {"task_count_inspected": 40, "eligible_task_count": sum(t["eligible"] for t in tasks),
                  "skipped_task_count": len(skipped), "case_count": len(cases), "trials_per_case": 1},
        "retained_families": list(MECHANISMS),
        "severity_multipliers": SEVERITY_MULTIPLIERS,
        "category_effective_strengths": CATEGORY_EFFECTIVE_STRENGTHS,
        "category_mechanism_effective_strengths": {
            f"{cat}:{mechanism}": levels
            for (cat, mechanism), levels in CATEGORY_MECHANISM_EFFECTIVE_STRENGTHS.items()
        },
        "low_matches_20260813_parameters": True,
        "excluded_families": ["narrow_platform", "height_offset", "convex", "waist"],
        "cases": cases, "skipped_tasks": skipped, "assets": assets,
    }
    dump(manifest, HERE / "manifest.json")
    print(json.dumps(manifest["scope"], ensure_ascii=False))


if __name__ == "__main__":
    main()
