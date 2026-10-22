#!/usr/bin/env python3
"""Build the unified six D1 semantic distractors for LIBERO-Object."""

from __future__ import annotations

import json
import math
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps


HERE = Path(__file__).resolve().parent
SIM_ROOT = Path.home() / "Sim"
ASSETS = SIM_ROOT / "Repos/LIBERO/libero/libero/assets"
BDDL_ROOT = SIM_ROOT / "Repos/LIBERO/libero/libero/bddl_files/libero_object"
sys.path.insert(0, str(SIM_ROOT))

from scripts.generate_libero10_same_category_distractor_env_edit import (  # noqa: E402
    add_init_lines,
    add_objects_to_objects_section,
    add_regions_to_regions_section,
    extract_section_bounds,
    region_block,
    tint_texture_file,
)
from scripts.generate_libero10_target_size_env_edits import scale_mujoco_xml  # noqa: E402


CONDITIONS = [
    "same_category",
    "same_color",
    "same_shape",
    "same_affordance",
    "same_position",
    "trajectory",
]
WORKSPACE_BOUNDS = [[-0.32, 0.32], [-0.32, 0.34]]
PARKING_XY = (0.255, 0.305)

# Geometry is measured from official init state 0.  extents are the target
# collision AABB in world coordinates; top_offset is target top minus body z.
TASKS = [
    {
        "task_id": 0,
        "name": "pick_up_the_alphabet_soup_and_place_it_in_the_basket",
        "target": "alphabet_soup_1",
        "category": "alphabet_soup",
        "extents": [0.062429, 0.062398, 0.076190],
        "top_offset": 0.037600,
        "mass": 0.02585145,
        "shape_kind": "paper_roll",
        "same_category_variant": "can",
        "same_category_size_xyz": [0.072, 0.072, 0.060],
        "same_shape_semantic": "candle",
        "label_crop_norm": [0.00, 0.55, 1.00, 0.98],
        "tint": [0.05, 0.85, 1.0, 1.0],
        "texture_crop_top": 0.64,
        "trajectory_side": 0.052,
        "trajectory_xy": [-0.0067, -0.0007],
        "trajectory_spawn_offset": 0.067,
        "trajectory_settled_range": [0.058, 0.065],
    },
    {
        "task_id": 1,
        "name": "pick_up_the_cream_cheese_and_place_it_in_the_basket",
        "target": "cream_cheese_1",
        "category": "cream_cheese",
        "extents": [0.081216, 0.042672, 0.017872],
        "top_offset": 0.008936,
        "mass": 0.00619381,
        "shape_kind": "book",
        "same_category_variant": "tub",
        "same_category_size_xyz": [0.060, 0.060, 0.034],
        "same_shape_semantic": "smart_device",
        "label_crop_norm": [0.25, 0.48, 1.00, 1.00],
        "tint": [0.92, 0.15, 0.82, 1.0],
        "texture_crop_top": 0.58,
        "trajectory_side": 0.043,
    },
    {
        "task_id": 2,
        "name": "pick_up_the_salad_dressing_and_place_it_in_the_basket",
        "target": "salad_dressing_1",
        "category": "salad_dressing",
        "extents": [0.035512, 0.052740, 0.145570],
        "top_offset": 0.072850,
        "mass": 0.02178453,
        "shape_kind": "wine_bottle",
        "same_category_variant": "bottle",
        "same_category_size_xyz": [0.055, 0.055, 0.105],
        "same_shape_semantic": "bowling_pin",
        "label_crop_norm": [0.66, 0.43, 1.00, 0.98],
        "tint": [0.20, 0.45, 1.0, 1.0],
        "texture_crop_top": 0.58,
    },
    {
        "task_id": 3,
        "name": "pick_up_the_bbq_sauce_and_place_it_in_the_basket",
        "target": "bbq_sauce_1",
        "category": "bbq_sauce",
        "extents": [0.029110, 0.046980, 0.107100],
        "top_offset": 0.052460,
        "mass": 0.01218031,
        "shape_kind": "wine_bottle",
        "same_category_variant": "jar",
        "same_category_size_xyz": [0.062, 0.062, 0.070],
        "same_shape_semantic": "bowling_pin",
        "label_crop_norm": [0.40, 0.35, 0.88, 0.92],
        "tint": [0.05, 0.85, 1.0, 1.0],
        "texture_crop_top": 0.58,
    },
    {
        "task_id": 4,
        "name": "pick_up_the_ketchup_and_place_it_in_the_basket",
        "target": "ketchup_1",
        "category": "ketchup",
        "extents": [0.036760, 0.056220, 0.145610],
        "top_offset": 0.073190,
        "mass": 0.02321683,
        "shape_kind": "wine_bottle",
        "same_category_variant": "bottle",
        "same_category_size_xyz": [0.060, 0.060, 0.105],
        "same_shape_semantic": "bowling_pin",
        "label_crop_norm": [0.30, 0.40, 0.80, 0.92],
        "tint": [0.15, 0.90, 0.25, 1.0],
        "texture_crop_top": 0.58,
    },
    {
        "task_id": 5,
        "name": "pick_up_the_tomato_sauce_and_place_it_in_the_basket",
        "target": "tomato_sauce_1",
        "category": "tomato_sauce",
        "extents": [0.062192, 0.061638, 0.076190],
        "top_offset": 0.037600,
        "mass": 0.02663278,
        "shape_kind": "paper_roll",
        "same_category_variant": "jar",
        "same_category_size_xyz": [0.068, 0.068, 0.065],
        "same_shape_semantic": "candle",
        "label_crop_norm": [0.62, 0.52, 1.00, 0.95],
        "tint": [0.05, 0.85, 1.0, 1.0],
        "texture_crop_top": 0.64,
    },
    {
        "task_id": 6,
        "name": "pick_up_the_butter_and_place_it_in_the_basket",
        "target": "butter_1",
        "category": "butter",
        "extents": [0.076180, 0.039540, 0.017420],
        "top_offset": 0.008710,
        "mass": 0.00524718,
        "shape_kind": "book",
        "same_category_variant": "tub",
        "same_category_size_xyz": [0.060, 0.060, 0.032],
        "same_shape_semantic": "smart_device",
        "label_crop_norm": [0.35, 0.25, 0.95, 0.90],
        "tint": [0.25, 0.35, 1.0, 1.0],
        "texture_crop_top": 0.58,
    },
    {
        "task_id": 7,
        "name": "pick_up_the_milk_and_place_it_in_the_basket",
        "target": "milk_1",
        "category": "milk",
        "extents": [0.052500, 0.053070, 0.131190],
        "top_offset": 0.061440,
        "mass": 0.03166591,
        "shape_kind": "book",
        "same_category_variant": "bottle",
        "same_category_size_xyz": [0.060, 0.060, 0.110],
        "same_shape_semantic": "toy_house",
        "label_crop_norm": [0.36, 0.22, 0.82, 0.86],
        "tint": [0.95, 0.20, 0.75, 1.0],
        "texture_crop_top": 0.58,
    },
    {
        "task_id": 8,
        "name": "pick_up_the_chocolate_pudding_and_place_it_in_the_basket",
        "target": "chocolate_pudding_1",
        "category": "chocolate_pudding",
        "extents": [0.080200, 0.046320, 0.027420],
        "top_offset": 0.014340,
        "mass": 0.01018616,
        "shape_kind": "book",
        "same_category_variant": "cup",
        "same_category_size_xyz": [0.060, 0.060, 0.050],
        "same_shape_semantic": "smart_device",
        "label_crop_norm": [0.00, 0.43, 0.78, 0.95],
        "tint": [0.05, 0.85, 1.0, 1.0],
        "texture_crop_top": 0.58,
    },
    {
        "task_id": 9,
        "name": "pick_up_the_orange_juice_and_place_it_in_the_basket",
        "target": "orange_juice_1",
        "category": "orange_juice",
        "extents": [0.052500, 0.053070, 0.131190],
        "top_offset": 0.061440,
        "mass": 0.03166591,
        "shape_kind": "book",
        "same_category_variant": "bottle",
        "same_category_size_xyz": [0.060, 0.060, 0.110],
        "same_shape_semantic": "toy_house",
        "label_crop_norm": [0.20, 0.35, 0.83, 0.90],
        "tint": [0.25, 0.40, 1.0, 1.0],
        "texture_crop_top": 0.58,
    },
]


def write_json(data: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def source_xml(category: str) -> Path:
    candidates = [
        ASSETS / "stable_hope_objects" / category / f"{category}.xml",
        ASSETS / "stable_scanned_objects" / category / f"{category}.xml",
        ASSETS / "turbosquid_objects" / category / f"{category}.xml",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(category)


def bddl_objects(path: Path) -> list[str]:
    _, _, section = extract_section_bounds(path.read_text(encoding="utf-8"), "objects")
    objects = []
    for line in section.splitlines():
        stripped = line.strip()
        if " - " in stripped:
            objects.append(stripped.split(" - ", 1)[0])
    return objects


def write_override_bddl(task: dict, condition: str, name: str, category: str) -> Path:
    source = BDDL_ROOT / f"{task['name']}.bddl"
    text = source.read_text(encoding="utf-8")
    region_name = f"{name}_parking_region"
    text = add_regions_to_regions_section(
        text, [region_block(region_name, "floor", PARKING_XY)]
    )
    text = add_objects_to_objects_section(text, {category: [name]})
    text = add_init_lines(text, [f"(On {name} floor_{region_name})"])
    path = (
        HERE
        / "bddl_overrides"
        / "libero_object"
        / f"task_{task['task_id']:02d}"
        / f"{condition}.bddl"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path.resolve()


def base_spec(task: dict, condition: str, bddl: Path, definition: dict) -> dict:
    source = BDDL_ROOT / f"{task['name']}.bddl"
    return {
        "schema_version": "d1_4_suites_full.v1",
        "status": "implemented",
        "type": condition,
        "suite": "libero_object",
        "benchmark": "libero_object",
        "task_id": task["task_id"],
        "task_name": task["name"],
        "instruction": task["name"].replace("_", " "),
        "base_bddl_file_name": str(source.resolve()),
        "bddl_file_name_override": str(bddl),
        "definition": definition,
        "target_geometry": {
            "collision_aabb_extents_m": task["extents"],
            "top_offset_from_body_m": task["top_offset"],
            "mass_kg": task["mass"],
            "friction": [0.95, 0.3, 0.1],
            "calibration_init_state": 0,
        },
        "settle_steps": 18,
    }


def adjacent_pose(
    task: dict,
    name: str,
    target: str,
    distance: float,
    *,
    quat_from: str | None = None,
    quat: list[float] | None = None,
) -> dict:
    original_objects = bddl_objects(BDDL_ROOT / f"{task['name']}.bddl")
    if task["task_id"] == 8:
        distance = min(0.120, distance + 0.025)
    offsets = [
        [0.0, distance, 0.0],
        [distance, 0.0, 0.0],
        [-distance, 0.0, 0.0],
        [0.0, -distance, 0.0],
    ]
    if task["task_id"] == 8:
        # The default +y placement is visually hidden behind an existing bottle
        # in this scene; prefer the clear left side while retaining fallbacks.
        offsets = [
            [-distance, 0.0, 0.0],
            [distance, 0.0, 0.0],
            [0.0, -distance, 0.0],
            [0.0, distance, 0.0],
        ]
    pose = {
        "object": name,
        "relative_to": target,
        "candidate_offsets": offsets,
        "candidate_selection": "max_clearance",
        "avoid_objects": original_objects,
        "workspace_bounds_xy": WORKSPACE_BOUNDS,
        "min_distance_from_others": 0.045,
        "z_from": target,
    }
    if quat_from:
        pose["quat_from"] = quat_from
    if quat is not None:
        pose["quat"] = quat
    return pose


def _scaled_triplet(value: str | None, scale_xyz: list[float]) -> str | None:
    if not value:
        return value
    numbers = [float(item) for item in value.split()]
    if len(numbers) != 3:
        return value
    return " ".join(f"{number * scale:.8g}" for number, scale in zip(numbers, scale_xyz))


def anisotropic_scale_xml(
    source_path: Path,
    target_path: Path,
    scale_xyz: list[float],
) -> Path:
    """Scale an object in its local xyz axes while retaining its semantic mesh."""
    tree = ET.parse(source_path)
    root = tree.getroot()
    radial_scale = math.sqrt(scale_xyz[0] * scale_xyz[1])
    for element in root.iter():
        if element.tag == "mesh":
            existing = [
                float(value)
                for value in element.get("scale", "1 1 1").split()
            ]
            if len(existing) == 1:
                existing *= 3
            element.set(
                "scale",
                " ".join(
                    f"{value * scale:.8g}"
                    for value, scale in zip(existing, scale_xyz)
                ),
            )
        if element.tag in {"body", "geom", "site", "joint", "inertial"}:
            scaled_pos = _scaled_triplet(element.get("pos"), scale_xyz)
            if scaled_pos is not None:
                element.set("pos", scaled_pos)
        if element.tag == "geom" and element.get("fromto"):
            values = [float(value) for value in element.get("fromto").split()]
            if len(values) == 6:
                scaled = [
                    values[index] * scale_xyz[index % 3]
                    for index in range(6)
                ]
                element.set("fromto", " ".join(f"{value:.8g}" for value in scaled))
        if element.tag in {"geom", "site"} and element.get("size"):
            values = [float(value) for value in element.get("size").split()]
            geom_type = element.get("type", "")
            if len(values) == 3:
                values = [
                    value * scale
                    for value, scale in zip(values, scale_xyz)
                ]
            elif len(values) == 2 and geom_type in {"cylinder", "capsule"}:
                values = [values[0] * radial_scale, values[1] * scale_xyz[2]]
            elif len(values) == 1:
                values = [values[0] * min(scale_xyz)]
            element.set("size", " ".join(f"{value:.8g}" for value in values))
    target_path.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(tree, space="  ")
    tree.write(target_path, encoding="utf-8", xml_declaration=False)
    return target_path.resolve()


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    font_path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
    return ImageFont.truetype(str(font_path), size) if font_path.exists() else ImageFont.load_default()


def make_label_texture(path: Path, task: dict) -> None:
    """Transfer official category graphics to a differently colored package."""
    width, height = 640, 320
    source_texture = source_xml(task["category"]).parent / "texture_map.png"
    with Image.open(source_texture) as source:
        source = source.convert("RGB")
        left, top, right, bottom = task["label_crop_norm"]
        crop = source.crop(
            (
                int(round(left * source.width)),
                int(round(top * source.height)),
                int(round(right * source.width)),
                int(round(bottom * source.height)),
            )
        )
    image = ImageOps.fit(
        crop,
        (width, height),
        method=Image.Resampling.LANCZOS,
        centering=(0.5, 0.5),
    )
    pixels = np.asarray(image, dtype=np.float32) / 255.0
    luminance = np.mean(pixels, axis=2, keepdims=True)
    tint = np.asarray(task["tint"][:3], dtype=np.float32).reshape(1, 1, 3)
    recolored = np.clip(
        0.42 * pixels + 0.58 * (0.18 + 0.82 * luminance) * tint,
        0.0,
        1.0,
    )
    image = Image.fromarray((recolored * 255).astype(np.uint8), mode="RGB")
    draw = ImageDraw.Draw(image)
    foreground = (12, 12, 16)
    draw.rounded_rectangle(
        (18, 18, width - 18, height - 18),
        radius=36,
        outline=foreground,
        width=12,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def make_same_category_asset(task: dict) -> Path:
    """Build a natural alternate package instead of warping the target mesh."""
    target_dir = (
        HERE
        / "assets"
        / "libero_object"
        / f"task_{task['task_id']:02d}"
        / "same_category"
    )
    if target_dir.exists():
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True)
    label_path = target_dir / "alternate_product_label.png"
    make_label_texture(label_path, task)

    x, y, z = task["same_category_size_xyz"]
    hx, hy, hz = x / 2.0, y / 2.0, z / 2.0
    radius = min(hx, hy)
    tint = " ".join(f"{value:.4f}" for value in task["tint"])
    dark = " ".join(f"{max(0.02, value * 0.42):.4f}" for value in task["tint"][:3]) + " 1"
    light = " ".join(f"{min(1.0, 0.55 + value * 0.45):.4f}" for value in task["tint"][:3]) + " 1"
    variant = task["same_category_variant"]

    visuals: list[str] = []
    collisions: list[str] = []
    if variant == "can":
        visuals.extend([
            f'<geom name="package_body" type="cylinder" pos="0 0 {hz:.7f}" size="{radius:.7f} {hz:.7f}" group="1" contype="0" conaffinity="0" rgba="{tint}"/>',
            f'<geom name="package_top_rim" type="cylinder" pos="0 0 {z-0.0012:.7f}" size="{radius*1.02:.7f} 0.0012" group="1" contype="0" conaffinity="0" rgba="0.72 0.75 0.78 1"/>',
            f'<geom name="package_bottom_rim" type="cylinder" pos="0 0 0.0012" size="{radius*1.02:.7f} 0.0012" group="1" contype="0" conaffinity="0" rgba="0.58 0.61 0.64 1"/>',
        ])
        collisions.append(
            f'<geom name="package_collision" type="cylinder" pos="0 0 {hz:.7f}" size="{radius:.7f} {hz:.7f}" group="0" density="180" friction="0.95 0.3 0.1" rgba="0.8 0.8 0.8 0.2"/>'
        )
    elif variant in {"jar", "bottle"}:
        body_top = z * (0.72 if variant == "bottle" else 0.78)
        body_radius = radius * (0.82 if variant == "bottle" else 0.94)
        shoulder_z = z * (0.73 if variant == "bottle" else 0.79)
        shoulder_h = z * (0.13 if variant == "bottle" else 0.08)
        neck_radius = radius * (0.42 if variant == "bottle" else 0.72)
        neck_half = z * (0.09 if variant == "bottle" else 0.065)
        neck_z = z - 2.0 * neck_half
        visuals.extend([
            f'<geom name="package_body" type="cylinder" pos="0 0 {body_top/2:.7f}" size="{body_radius:.7f} {body_top/2:.7f}" group="1" contype="0" conaffinity="0" rgba="{tint}"/>',
            f'<geom name="package_shoulder" type="ellipsoid" pos="0 0 {shoulder_z:.7f}" size="{body_radius:.7f} {body_radius:.7f} {shoulder_h:.7f}" group="1" contype="0" conaffinity="0" rgba="{tint}"/>',
            f'<geom name="package_neck" type="cylinder" pos="0 0 {neck_z:.7f}" size="{neck_radius:.7f} {neck_half:.7f}" group="1" contype="0" conaffinity="0" rgba="{light}"/>',
            f'<geom name="package_cap" type="cylinder" pos="0 0 {z-z*0.035:.7f}" size="{neck_radius*1.10:.7f} {z*0.035:.7f}" group="1" contype="0" conaffinity="0" rgba="{dark}"/>',
        ])
        collisions.extend([
            f'<geom name="package_body_collision" type="cylinder" pos="0 0 {body_top/2:.7f}" size="{body_radius:.7f} {body_top/2:.7f}" group="0" density="180" friction="0.95 0.3 0.1" rgba="0.8 0.8 0.8 0.2"/>',
            f'<geom name="package_neck_collision" type="cylinder" pos="0 0 {neck_z:.7f}" size="{neck_radius:.7f} {neck_half:.7f}" group="0" density="180" friction="0.95 0.3 0.1" rgba="0.8 0.8 0.8 0.2"/>',
        ])
    elif variant in {"tub", "cup"}:
        body_radius = radius * (0.92 if variant == "tub" else 0.78)
        lid_radius = radius if variant == "tub" else radius * 1.02
        visuals.extend([
            f'<geom name="package_body" type="cylinder" pos="0 0 {hz:.7f}" size="{body_radius:.7f} {hz:.7f}" group="1" contype="0" conaffinity="0" rgba="{tint}"/>',
            f'<geom name="package_lid" type="cylinder" pos="0 0 {z-z*0.04:.7f}" size="{lid_radius:.7f} {z*0.04:.7f}" group="1" contype="0" conaffinity="0" rgba="{light}"/>',
            f'<geom name="package_lid_band" type="cylinder" pos="0 0 {z-z*0.018:.7f}" size="{lid_radius*0.72:.7f} {z*0.018:.7f}" group="1" contype="0" conaffinity="0" rgba="{dark}"/>',
        ])
        collisions.append(
            f'<geom name="package_collision" type="cylinder" pos="0 0 {hz:.7f}" size="{body_radius:.7f} {hz:.7f}" group="0" density="180" friction="0.95 0.3 0.1" rgba="0.8 0.8 0.8 0.2"/>'
        )
    else:
        raise ValueError(variant)

    label_z = min(z * 0.48, z - 0.012)
    label_h = min(z * 0.17, 0.021)
    label_w = radius * 0.62
    label_offset = radius + 0.00035
    labels = [
        f'<geom name="product_label_front" type="box" pos="0 {-label_offset:.7f} {label_z:.7f}" size="{label_w:.7f} 0.00035 {label_h:.7f}" group="1" contype="0" conaffinity="0" material="product_label"/>',
        f'<geom name="product_label_back" type="box" pos="0 {label_offset:.7f} {label_z:.7f}" size="{label_w:.7f} 0.00035 {label_h:.7f}" group="1" contype="0" conaffinity="0" material="product_label"/>',
        f'<geom name="product_label_left" type="box" pos="{-label_offset:.7f} 0 {label_z:.7f}" size="0.00035 {label_w:.7f} {label_h:.7f}" group="1" contype="0" conaffinity="0" material="product_label"/>',
        f'<geom name="product_label_right" type="box" pos="{label_offset:.7f} 0 {label_z:.7f}" size="0.00035 {label_w:.7f} {label_h:.7f}" group="1" contype="0" conaffinity="0" material="product_label"/>',
    ]
    path = target_dir / f"{task['category']}_alternate_package.xml"
    xml = f"""<mujoco model="{task['category']}_alternate_package">
  <asset>
    <texture name="product_label_texture" type="2d" file="{label_path.name}"/>
    <material name="product_label" texture="product_label_texture" texuniform="true" reflectance="0.15"/>
  </asset>
  <worldbody>
    <body>
      <body name="object">
        {' '.join(visuals)}
        {' '.join(labels)}
        {' '.join(collisions)}
      </body>
      <site rgba="0 0 0 0" size="0.003" pos="0 0 0" name="bottom_site"/>
      <site rgba="0 0 0 0" size="0.003" pos="0 0 {z:.7f}" name="top_site"/>
      <site rgba="0 0 0 0" size="0.003" pos="{radius:.7f} {radius:.7f} {hz:.7f}" name="horizontal_radius_site"/>
    </body>
  </worldbody>
</mujoco>
"""
    path.write_text(xml, encoding="utf-8")
    return path.resolve()


def make_target_texture_mug(task: dict, scale: float = 0.55) -> Path:
    native_dir = ASSETS / "turbosquid_objects/red_coffee_mug"
    target_dir = (
        HERE
        / "assets"
        / "libero_object"
        / f"task_{task['task_id']:02d}"
        / "same_color_mug"
    )
    if target_dir.exists():
        shutil.rmtree(target_dir)
    shutil.copytree(native_dir, target_dir)
    target_texture = source_xml(task["category"]).parent / "texture_map.png"
    with Image.open(target_texture) as source_image:
        texture = source_image.convert("RGBA")
        crop_top = int(texture.height * task["texture_crop_top"])
        label = texture.crop((0, crop_top, texture.width, texture.height))
    mug_tint = None
    if task["task_id"] != 0:
        pixels = np.asarray(label.convert("RGB"))
        foreground = np.max(pixels, axis=2) > 18
        column_density = foreground.mean(axis=0)
        runs = []
        start = None
        for index, active in enumerate(column_density >= 0.40):
            if active and start is None:
                start = index
            if start is not None and (not active or index == len(column_density) - 1):
                end = index if active and index == len(column_density) - 1 else index - 1
                if end - start + 1 >= max(32, pixels.shape[1] // 14):
                    runs.append((start, end + 1))
                start = None
        candidates = []
        for x0, x1 in runs:
            strip_mask = foreground[:, x0:x1]
            row_density = strip_mask.mean(axis=1)
            active_rows = np.where(row_density >= 0.40)[0]
            if not len(active_rows):
                continue
            y0, y1 = int(active_rows.min()), int(active_rows.max()) + 1
            patch = pixels[y0:y1, x0:x1]
            patch_mask = np.max(patch, axis=2) > 18
            if patch.size == 0 or patch_mask.mean() < 0.55:
                continue
            foreground_pixels = patch[patch_mask]
            color_variation = float(np.std(foreground_pixels))
            area_score = math.sqrt(float(patch.shape[0] * patch.shape[1]))
            candidates.append((area_score * (color_variation + 15.0), patch, patch_mask))
        if candidates:
            _, patch, patch_mask = max(candidates, key=lambda item: item[0])
            fill = np.median(patch[patch_mask], axis=0).astype(np.uint8)
            patch = patch.copy()
            patch[~patch_mask] = fill
            label = Image.fromarray(patch, mode="RGB").convert("RGBA")
            mug_tint = fill.astype(float)
    mug_texture_path = target_dir / "red_coffee_mug_texture.png"
    with Image.open(mug_texture_path) as native_image:
        mug_size = native_image.size
    label = label.resize(mug_size, Image.Resampling.LANCZOS)
    label.convert("RGB").save(mug_texture_path)
    scaled_xml = target_dir / "red_coffee_mug_scaled.xml"
    scale_mujoco_xml(target_dir / "red_coffee_mug.xml", scaled_xml, scale)
    if mug_tint is not None:
        normalized = mug_tint / max(float(mug_tint.max()), 1.0)
        normalized = np.clip(normalized, 0.18, 1.0)
        tree = ET.parse(scaled_xml)
        material = tree.getroot().find(".//material[@name='red_coffee_mug']")
        if material is not None:
            material.set(
                "rgba",
                " ".join(f"{value:.6f}" for value in [*normalized.tolist(), 1.0]),
            )
        tree.write(scaled_xml, encoding="utf-8", xml_declaration=False)
    return scaled_xml.resolve()


def make_shape_book(task: dict) -> Path:
    x, y, z = task["extents"]
    # Preserve the target's cuboid silhouette while exposing unambiguous book
    # cues: blue covers, cream pages, and a contrasting spine.
    cover = max(0.0012, min(0.0025, z * 0.08))
    hx, hy, hz = x / 2.0, y / 2.0, z / 2.0
    path = (
        HERE
        / "assets"
        / "libero_object"
        / f"task_{task['task_id']:02d}"
        / "same_shape_book.xml"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    xml = f"""<mujoco model="semantic_shape_book">
  <worldbody>
    <body>
      <body name="object">
        <geom name="pages_visual" type="box" pos="0 0 {hz:.6f}" size="{max(0.001, hx-cover):.6f} {max(0.001, hy-cover):.6f} {max(0.001, hz-cover):.6f}" group="1" contype="0" conaffinity="0" rgba="0.92 0.86 0.68 1"/>
        <geom name="top_cover" type="box" pos="0 0 {z-cover/2:.6f}" size="{hx:.6f} {hy:.6f} {cover/2:.6f}" group="1" contype="0" conaffinity="0" rgba="0.12 0.28 0.78 1"/>
        <geom name="bottom_cover" type="box" pos="0 0 {cover/2:.6f}" size="{hx:.6f} {hy:.6f} {cover/2:.6f}" group="1" contype="0" conaffinity="0" rgba="0.12 0.28 0.78 1"/>
        <geom name="book_spine" type="box" pos="{-hx + cover/2:.6f} 0 {hz:.6f}" size="{cover/2:.6f} {hy:.6f} {hz:.6f}" group="1" contype="0" conaffinity="0" rgba="0.04 0.12 0.48 1"/>
        <geom name="cover_label" type="box" pos="{hx*0.12:.6f} 0 {z+0.00025:.6f}" size="{hx*0.40:.6f} {hy*0.32:.6f} 0.00025" group="1" contype="0" conaffinity="0" rgba="0.94 0.82 0.28 1"/>
        <geom name="title_bar" type="box" pos="{hx*0.12:.6f} 0 {z+0.00055:.6f}" size="{hx*0.25:.6f} {max(0.0008, hy*0.055):.6f} 0.00020" group="1" contype="0" conaffinity="0" rgba="0.12 0.12 0.18 1"/>
        <geom name="book_collision" type="box" pos="0 0 {hz:.6f}" size="{hx:.6f} {hy:.6f} {hz:.6f}" group="0" density="100" friction="0.95 0.3 0.1" rgba="0.8 0.8 0.8 0.25"/>
      </body>
      <site rgba="0 0 0 0" size="0.003" pos="0 0 0" name="bottom_site"/>
      <site rgba="0 0 0 0" size="0.003" pos="0 0 {z:.6f}" name="top_site"/>
      <site rgba="0 0 0 0" size="0.003" pos="{hx:.6f} {hy:.6f} {hz:.6f}" name="horizontal_radius_site"/>
    </body>
  </worldbody>
</mujoco>
"""
    path.write_text(xml, encoding="utf-8")
    return path.resolve()


def make_shape_paper_roll(task: dict) -> Path:
    source = ASSETS / "d1_distractor_objects/paper_roll/paper_roll.xml"
    target = (
        HERE
        / "assets"
        / "libero_object"
        / f"task_{task['task_id']:02d}"
        / "same_shape_paper_roll.xml"
    )
    x, y, z = task["extents"]
    scale_xyz = [x / 0.062, y / 0.062, z / 0.082]
    return anisotropic_scale_xml(source, target, scale_xyz)


def make_shape_wine_proxy(task: dict) -> Path:
    x, y, z = task["extents"]
    hx, hy = x / 2.0, y / 2.0
    body_h = z * 0.68
    neck_radius = min(hx, hy) * 0.34
    path = (
        HERE
        / "assets"
        / "libero_object"
        / f"task_{task['task_id']:02d}"
        / "same_shape_wine_bottle.xml"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    xml = f"""<mujoco model="shape_matched_wine_bottle">
  <worldbody>
    <body>
      <body name="object">
        <geom name="wine_body_visual" type="ellipsoid" pos="0 0 {body_h/2:.7f}" size="{hx:.7f} {hy:.7f} {body_h/2:.7f}" group="1" contype="0" conaffinity="0" rgba="0.025 0.19 0.075 1"/>
        <geom name="wine_shoulder_visual" type="ellipsoid" pos="0 0 {z*0.69:.7f}" size="{hx*0.88:.7f} {hy*0.88:.7f} {z*0.16:.7f}" group="1" contype="0" conaffinity="0" rgba="0.025 0.19 0.075 1"/>
        <geom name="wine_neck_visual" type="cylinder" pos="0 0 {z*0.88:.7f}" size="{neck_radius:.7f} {z*0.12:.7f}" group="1" contype="0" conaffinity="0" rgba="0.018 0.13 0.05 1"/>
        <geom name="wine_label" type="box" pos="0 {-hy-0.00035:.7f} {z*0.35:.7f}" size="{hx*0.62:.7f} 0.00035 {z*0.16:.7f}" group="1" contype="0" conaffinity="0" rgba="0.82 0.73 0.50 1"/>
        <geom name="wine_label_band" type="box" pos="0 {-hy-0.00075:.7f} {z*0.35:.7f}" size="{hx*0.40:.7f} 0.00020 {z*0.025:.7f}" group="1" contype="0" conaffinity="0" rgba="0.42 0.04 0.07 1"/>
        <geom name="wine_cork" type="cylinder" pos="0 0 {z*0.985:.7f}" size="{neck_radius*0.88:.7f} {z*0.015:.7f}" group="1" contype="0" conaffinity="0" rgba="0.48 0.27 0.11 1"/>
        <geom name="wine_body_collision" type="ellipsoid" pos="0 0 {body_h/2:.7f}" size="{hx:.7f} {hy:.7f} {body_h/2:.7f}" group="0" density="100" friction="0.95 0.3 0.1" rgba="0.8 0.8 0.8 0.2"/>
        <geom name="wine_shoulder_collision" type="ellipsoid" pos="0 0 {z*0.69:.7f}" size="{hx*0.88:.7f} {hy*0.88:.7f} {z*0.16:.7f}" group="0" density="100" friction="0.95 0.3 0.1" rgba="0.8 0.8 0.8 0.2"/>
        <geom name="wine_neck_collision" type="cylinder" pos="0 0 {z*0.88:.7f}" size="{neck_radius:.7f} {z*0.12:.7f}" group="0" density="100" friction="0.95 0.3 0.1" rgba="0.8 0.8 0.8 0.2"/>
      </body>
      <site rgba="0 0 0 0" size="0.003" pos="0 0 0" name="bottom_site"/>
      <site rgba="0 0 0 0" size="0.003" pos="0 0 {z:.7f}" name="top_site"/>
      <site rgba="0 0 0 0" size="0.003" pos="{hx:.7f} {hy:.7f} {z/2:.7f}" name="horizontal_radius_site"/>
    </body>
  </worldbody>
</mujoco>
"""
    path.write_text(xml, encoding="utf-8")
    return path.resolve()


SHAPE_SEMANTICS = {
    # The first value is an existing LIBERO registry key used only to construct
    # the overridden object. The visible semantic category is the second value.
    "candle": ("paper_roll", "带蜡芯的彩色蜡烛"),
    "bowling_pin": ("wine_bottle", "白色红环保龄球瓶"),
    "smart_device": ("black_book", "带屏幕图标的电子设备"),
    "toy_house": ("black_book", "带门窗和砖纹的玩具小屋"),
}


def semantic_pattern_texture(path: Path, semantic: str) -> None:
    """Retexture the exact target mesh with unmistakable different-category cues."""
    with Image.open(path) as source:
        rgba = source.convert("RGBA")
    original = np.asarray(rgba).copy()
    mask = (original[:, :, 3] > 16) & (np.max(original[:, :, :3], axis=2) > 8)
    width, height = rgba.size
    pattern = Image.new("RGB", (width, height))
    draw = ImageDraw.Draw(pattern)
    unit = max(24, min(width, height) // 8)
    if semantic == "candle":
        draw.rectangle((0, 0, width, height), fill=(246, 220, 168))
        for y in range(0, height, unit * 2):
            draw.rectangle((0, y, width, min(height, y + unit // 3)), fill=(184, 95, 156))
        word, color = "CANDLE", (66, 33, 49)
    elif semantic == "bowling_pin":
        draw.rectangle((0, 0, width, height), fill=(245, 242, 224))
        for y in range(unit, height, unit * 3):
            draw.rectangle((0, y, width, min(height, y + unit // 3)), fill=(200, 28, 38))
            draw.rectangle(
                (0, y + unit // 2, width, min(height, y + 5 * unit // 6)),
                fill=(200, 28, 38),
            )
        word, color = "BOWLING", (155, 12, 24)
    elif semantic == "smart_device":
        draw.rectangle((0, 0, width, height), fill=(12, 23, 45))
        icon_colors = [(49, 190, 255), (255, 194, 46), (241, 75, 130), (78, 210, 126)]
        for row, y in enumerate(range(unit // 2, height, unit * 2)):
            for col, x in enumerate(range(unit // 2, width, unit * 2)):
                color = icon_colors[(row + col) % len(icon_colors)]
                draw.rounded_rectangle(
                    (x, y, min(width - 1, x + unit), min(height - 1, y + unit)),
                    radius=max(3, unit // 7),
                    fill=color,
                )
        word, color = "DEVICE", (230, 245, 255)
    elif semantic == "toy_house":
        draw.rectangle((0, 0, width, height), fill=(222, 167, 91))
        for y in range(0, height, unit):
            draw.line((0, y, width, y), fill=(126, 65, 39), width=max(2, unit // 12))
            offset = 0 if (y // unit) % 2 == 0 else unit
            for x in range(offset, width, unit * 2):
                draw.line((x, y, x, min(height, y + unit)), fill=(126, 65, 39), width=max(2, unit // 12))
        for y in range(unit // 2, height, unit * 3):
            for x in range(unit // 2, width, unit * 3):
                draw.rectangle(
                    (x, y, min(width - 1, x + unit), min(height - 1, y + unit)),
                    fill=(75, 167, 214),
                    outline=(245, 242, 210),
                    width=max(2, unit // 10),
                )
        word, color = "HOUSE", (93, 38, 29)
    else:
        raise ValueError(semantic)

    font = _font(max(18, unit))
    for y in range(unit, height, unit * 4):
        for x in range(0, width, unit * 5):
            draw.text((x, y), word, font=font, fill=color)
    pattern_array = np.asarray(pattern)
    original[:, :, :3][mask] = pattern_array[mask]
    Image.fromarray(original, mode="RGBA").convert("RGB").save(path)


def make_exact_shape_semantic_asset(task: dict) -> Path:
    """Keep the target mesh/collisions exactly, changing only semantic appearance."""
    semantic = task["same_shape_semantic"]
    source = source_xml(task["category"])
    target_dir = (
        HERE
        / "assets"
        / "libero_object"
        / f"task_{task['task_id']:02d}"
        / f"same_shape_{semantic}"
    )
    if target_dir.exists():
        shutil.rmtree(target_dir)
    shutil.copytree(source.parent, target_dir)
    for path in target_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
            semantic_pattern_texture(path, semantic)

    xml_path = target_dir / source.name
    tree = ET.parse(xml_path)
    root = tree.getroot()
    root.set("model", f"exact_shape_{semantic}")
    root.insert(0, ET.Comment(
        f"Exact geometry clone of {task['category']}; semantic category is {semantic}."
    ))
    for material in root.findall(".//material"):
        material.set("rgba", "1 1 1 1")
    if semantic == "candle":
        object_body = root.find(".//body[@name='object']")
        top_site = root.find(".//site[@name='top_site']")
        top_z = float((top_site.get("pos") if top_site is not None else "0 0 0.04").split()[2])
        if object_body is not None:
            ET.SubElement(
                object_body,
                "geom",
                {
                    "name": "semantic_candle_wick",
                    "type": "cylinder",
                    "pos": f"0 0 {top_z + 0.0025:.7f}",
                    "size": "0.0014 0.0030",
                    "group": "1",
                    "contype": "0",
                    "conaffinity": "0",
                    "rgba": "0.04 0.025 0.02 1",
                },
            )
    ET.indent(tree, space="  ")
    output = target_dir / f"same_shape_{semantic}.xml"
    tree.write(output, encoding="utf-8", xml_declaration=False)
    return output.resolve()


def make_exact_shape_color_asset(task: dict) -> Path:
    """Copy the target geometry exactly and change only its rendered color."""
    source = source_xml(task["category"])
    task_asset_dir = (
        HERE / "assets" / "libero_object" / f"task_{task['task_id']:02d}"
    )
    target_dir = task_asset_dir / "same_category_color_copy"
    if target_dir.exists():
        shutil.rmtree(target_dir)
    shutil.copytree(source.parent, target_dir)
    for path in target_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
            tint_texture_file(path, task["tint"])
    tree = ET.parse(target_dir / source.name)
    root = tree.getroot()
    root.set("model", f"{task['category']}_exact_shape_color_copy")
    root.insert(
        0,
        ET.Comment(
            f"Exact visual and collision geometry copy of {task['category']}; color only changed."
        ),
    )
    for material in root.findall(".//material"):
        material.set("rgba", "1 1 1 1")
    ET.indent(tree, space="  ")
    output = target_dir / f"{task['category']}_exact_shape_color_copy.xml"
    tree.write(output, encoding="utf-8", xml_declaration=False)
    return output.resolve()


def _shape_proxy_shell(task: dict, model: str, contents: str) -> Path:
    x, y, z = task["extents"]
    path = (
        HERE
        / "assets"
        / "libero_object"
        / f"task_{task['task_id']:02d}"
        / f"same_shape_{task['same_shape_semantic']}.xml"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    xml = f"""<mujoco model="{model}">
  <worldbody>
    <body>
      <body name="object">
        {contents}
      </body>
      <site rgba="0 0 0 0" size="0.003" pos="0 0 0" name="bottom_site"/>
      <site rgba="0 0 0 0" size="0.003" pos="0 0 {z:.7f}" name="top_site"/>
      <site rgba="0 0 0 0" size="0.003" pos="{x/2:.7f} {y/2:.7f} {z/2:.7f}" name="horizontal_radius_site"/>
    </body>
  </worldbody>
</mujoco>
"""
    path.write_text(xml, encoding="utf-8")
    return path.resolve()


def make_shape_candle(task: dict) -> Path:
    x, y, z = task["extents"]
    radius = min(x, y) / 2.0
    half_z = z / 2.0
    contents = f"""
        <geom name="candle_wax" type="cylinder" pos="0 0 {half_z:.7f}" size="{radius:.7f} {half_z:.7f}" group="1" contype="0" conaffinity="0" rgba="0.94 0.78 0.52 1"/>
        <geom name="candle_color_band" type="cylinder" pos="0 0 {z*0.48:.7f}" size="{radius*1.005:.7f} {z*0.055:.7f}" group="1" contype="0" conaffinity="0" rgba="0.64 0.18 0.48 1"/>
        <geom name="candle_wick" type="cylinder" pos="0 0 {z+0.0025:.7f}" size="0.0014 0.0030" group="1" contype="0" conaffinity="0" rgba="0.04 0.025 0.02 1"/>
        <geom name="candle_collision" type="cylinder" pos="0 0 {half_z:.7f}" size="{radius:.7f} {half_z:.7f}" group="0" density="180" friction="0.95 0.3 0.1" rgba="0.8 0.8 0.8 0.2"/>
    """
    return _shape_proxy_shell(task, "independent_shape_candle", contents)


def make_shape_device(task: dict) -> Path:
    x, y, z = task["extents"]
    hx, hy, hz = x / 2.0, y / 2.0, z / 2.0
    contents = f"""
        <geom name="device_body" type="box" pos="0 0 {hz:.7f}" size="{hx:.7f} {hy:.7f} {hz:.7f}" group="1" contype="0" conaffinity="0" rgba="0.035 0.06 0.11 1"/>
        <geom name="device_screen" type="box" pos="0 0 {z+0.00025:.7f}" size="{hx*0.76:.7f} {hy*0.72:.7f} 0.00025" group="1" contype="0" conaffinity="0" rgba="0.06 0.52 0.82 1"/>
        <geom name="device_icon_red" type="box" pos="{-hx*0.34:.7f} {-hy*0.30:.7f} {z+0.00055:.7f}" size="{max(0.0015,hx*0.10):.7f} {max(0.0015,hy*0.10):.7f} 0.00020" group="1" contype="0" conaffinity="0" rgba="0.94 0.18 0.35 1"/>
        <geom name="device_icon_yellow" type="box" pos="{hx*0.30:.7f} {-hy*0.30:.7f} {z+0.00055:.7f}" size="{max(0.0015,hx*0.10):.7f} {max(0.0015,hy*0.10):.7f} 0.00020" group="1" contype="0" conaffinity="0" rgba="0.98 0.78 0.12 1"/>
        <geom name="device_button" type="cylinder" pos="0 {hy*0.35:.7f} {z+0.0008:.7f}" size="{max(0.0018,min(hx,hy)*0.10):.7f} 0.00030" group="1" contype="0" conaffinity="0" rgba="0.82 0.88 0.94 1"/>
        <geom name="device_collision" type="box" pos="0 0 {hz:.7f}" size="{hx:.7f} {hy:.7f} {hz:.7f}" group="0" density="180" friction="0.95 0.3 0.1" rgba="0.8 0.8 0.8 0.2"/>
    """
    return _shape_proxy_shell(task, "independent_shape_device", contents)


def make_shape_bowling_pin(task: dict) -> Path:
    x, y, z = task["extents"]
    hx, hy = x / 2.0, y / 2.0
    neck = min(hx, hy) * 0.38
    contents = f"""
        <geom name="bowling_body" type="ellipsoid" pos="0 0 {z*0.32:.7f}" size="{hx:.7f} {hy:.7f} {z*0.32:.7f}" group="1" contype="0" conaffinity="0" rgba="0.96 0.95 0.88 1"/>
        <geom name="bowling_shoulder" type="ellipsoid" pos="0 0 {z*0.62:.7f}" size="{hx*0.84:.7f} {hy*0.84:.7f} {z*0.16:.7f}" group="1" contype="0" conaffinity="0" rgba="0.96 0.95 0.88 1"/>
        <geom name="bowling_neck" type="cylinder" pos="0 0 {z*0.86:.7f}" size="{neck:.7f} {z*0.14:.7f}" group="1" contype="0" conaffinity="0" rgba="0.96 0.95 0.88 1"/>
        <geom name="bowling_red_ring_lower" type="cylinder" pos="0 0 {z*0.76:.7f}" size="{neck*1.08:.7f} {z*0.018:.7f}" group="1" contype="0" conaffinity="0" rgba="0.82 0.035 0.06 1"/>
        <geom name="bowling_red_ring_upper" type="cylinder" pos="0 0 {z*0.82:.7f}" size="{neck*1.03:.7f} {z*0.018:.7f}" group="1" contype="0" conaffinity="0" rgba="0.82 0.035 0.06 1"/>
        <geom name="bowling_body_collision" type="ellipsoid" pos="0 0 {z*0.32:.7f}" size="{hx:.7f} {hy:.7f} {z*0.32:.7f}" group="0" density="180" friction="0.95 0.3 0.1" rgba="0.8 0.8 0.8 0.2"/>
        <geom name="bowling_shoulder_collision" type="ellipsoid" pos="0 0 {z*0.62:.7f}" size="{hx*0.84:.7f} {hy*0.84:.7f} {z*0.16:.7f}" group="0" density="180" friction="0.95 0.3 0.1" rgba="0.8 0.8 0.8 0.2"/>
        <geom name="bowling_neck_collision" type="cylinder" pos="0 0 {z*0.86:.7f}" size="{neck:.7f} {z*0.14:.7f}" group="0" density="180" friction="0.95 0.3 0.1" rgba="0.8 0.8 0.8 0.2"/>
    """
    return _shape_proxy_shell(task, "independent_shape_bowling_pin", contents)


def make_shape_toy_house(task: dict) -> Path:
    x, y, z = task["extents"]
    hx, hy = x / 2.0, y / 2.0
    body_h = z * 0.76
    roof_h = z - body_h
    path = (
        HERE
        / "assets"
        / "libero_object"
        / f"task_{task['task_id']:02d}"
        / "same_shape_toy_house.xml"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    roof_path = path.with_name("same_shape_toy_house_roof.obj")
    roof_obj = f"""v {-hx} {-hy} {body_h}
v {hx} {-hy} {body_h}
v 0 {-hy} {z}
v {-hx} {hy} {body_h}
v {hx} {hy} {body_h}
v 0 {hy} {z}
f 1 2 3
f 4 6 5
f 1 4 5 2
f 2 5 6 3
f 3 6 4 1
"""
    roof_path.write_text(roof_obj, encoding="utf-8")
    contents = f"""
        <geom name="house_body" type="box" pos="0 0 {body_h/2:.7f}" size="{hx:.7f} {hy:.7f} {body_h/2:.7f}" group="1" contype="0" conaffinity="0" rgba="0.82 0.58 0.30 1"/>
        <geom name="house_roof" type="mesh" mesh="house_roof_mesh" group="1" contype="0" conaffinity="0" rgba="0.46 0.12 0.08 1"/>
        <geom name="house_door" type="box" pos="0 {-hy-0.00030:.7f} {body_h*0.28:.7f}" size="{hx*0.24:.7f} 0.00030 {body_h*0.28:.7f}" group="1" contype="0" conaffinity="0" rgba="0.20 0.10 0.055 1"/>
        <geom name="house_window_left" type="box" pos="{-hx*0.52:.7f} {-hy-0.00045:.7f} {body_h*0.64:.7f}" size="{hx*0.18:.7f} 0.00025 {body_h*0.13:.7f}" group="1" contype="0" conaffinity="0" rgba="0.22 0.70 0.92 1"/>
        <geom name="house_window_right" type="box" pos="{hx*0.52:.7f} {-hy-0.00045:.7f} {body_h*0.64:.7f}" size="{hx*0.18:.7f} 0.00025 {body_h*0.13:.7f}" group="1" contype="0" conaffinity="0" rgba="0.22 0.70 0.92 1"/>
        <geom name="house_back_door" type="box" pos="0 {hy+0.00030:.7f} {body_h*0.28:.7f}" size="{hx*0.24:.7f} 0.00030 {body_h*0.28:.7f}" group="1" contype="0" conaffinity="0" rgba="0.20 0.10 0.055 1"/>
        <geom name="house_back_window_left" type="box" pos="{-hx*0.52:.7f} {hy+0.00045:.7f} {body_h*0.64:.7f}" size="{hx*0.18:.7f} 0.00025 {body_h*0.13:.7f}" group="1" contype="0" conaffinity="0" rgba="0.22 0.70 0.92 1"/>
        <geom name="house_back_window_right" type="box" pos="{hx*0.52:.7f} {hy+0.00045:.7f} {body_h*0.64:.7f}" size="{hx*0.18:.7f} 0.00025 {body_h*0.13:.7f}" group="1" contype="0" conaffinity="0" rgba="0.22 0.70 0.92 1"/>
        <geom name="house_side_window_left" type="box" pos="{-hx-0.00035:.7f} 0 {body_h*0.62:.7f}" size="0.00025 {hy*0.30:.7f} {body_h*0.14:.7f}" group="1" contype="0" conaffinity="0" rgba="0.22 0.70 0.92 1"/>
        <geom name="house_side_window_right" type="box" pos="{hx+0.00035:.7f} 0 {body_h*0.62:.7f}" size="0.00025 {hy*0.30:.7f} {body_h*0.14:.7f}" group="1" contype="0" conaffinity="0" rgba="0.22 0.70 0.92 1"/>
        <geom name="house_collision" type="box" pos="0 0 {z/2:.7f}" size="{hx:.7f} {hy:.7f} {z/2:.7f}" group="0" density="180" friction="0.95 0.3 0.1" rgba="0.8 0.8 0.8 0.2"/>
    """
    xml = f"""<mujoco model="independent_shape_toy_house">
  <asset><mesh name="house_roof_mesh" file="{roof_path.name}"/></asset>
  <worldbody>
    <body>
      <body name="object">{contents}</body>
      <site rgba="0 0 0 0" size="0.003" pos="0 0 0" name="bottom_site"/>
      <site rgba="0 0 0 0" size="0.003" pos="0 0 {z:.7f}" name="top_site"/>
      <site rgba="0 0 0 0" size="0.003" pos="{hx:.7f} {hy:.7f} {z/2:.7f}" name="horizontal_radius_site"/>
    </body>
  </worldbody>
</mujoco>
"""
    path.write_text(xml, encoding="utf-8")
    return path.resolve()


def make_shape_asset(task: dict) -> tuple[str, Path | None, str]:
    semantic = task["same_shape_semantic"]
    category, label = SHAPE_SEMANTICS[semantic]
    builders = {
        "candle": make_shape_candle,
        "bowling_pin": make_shape_bowling_pin,
        "smart_device": make_shape_device,
        "toy_house": make_shape_toy_house,
    }
    return category, builders[semantic](task), label


def make_scaled_tray() -> Path:
    source = ASSETS / "turbosquid_objects/wooden_tray/wooden_tray.xml"
    target = HERE / "assets" / "shared" / "wooden_tray_scaled.xml"
    scale_mujoco_xml(source, target, 0.68)
    return target.resolve()


def make_position_block() -> Path:
    side = 0.040
    half = side / 2.0
    output = HERE / "assets" / "shared" / "same_position_block.xml"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        f"""<mujoco model="d1_same_position_block">
  <worldbody>
    <body>
      <body name="object">
        <geom name="position_block_visual" type="box" pos="0 0 0" size="{half} {half} {half}" group="1" contype="0" conaffinity="0" rgba="0.95 0.55 0.10 1"/>
        <geom name="position_block_collision" type="box" pos="0 0 0" size="{half} {half} {half}" group="0" density="180" friction="0.95 0.3 0.1" rgba="0.8 0.8 0.8 0.25"/>
      </body>
      <site rgba="0 0 0 0" size="0.003" pos="0 0 {-half}" name="bottom_site"/>
      <site rgba="0 0 0 0" size="0.003" pos="0 0 {half}" name="top_site"/>
      <site rgba="0 0 0 0" size="0.003" pos="{half} {half} 0" name="horizontal_radius_site"/>
    </body>
  </worldbody>
</mujoco>
""",
        encoding="utf-8",
    )
    return output.resolve()


def trajectory_side(task: dict) -> float:
    if "trajectory_side" in task:
        return task["trajectory_side"]
    x, y, _ = task["extents"]
    rectangular = task["shape_kind"] == "book"
    area = x * y if rectangular else math.pi * (x / 2.0) * (y / 2.0)
    return max(0.034, min(0.062, math.sqrt(area)))


def make_trajectory_block(task: dict) -> tuple[Path, dict]:
    side = trajectory_side(task)
    half = side / 2.0
    density = task["mass"] / (side ** 3)
    path = (
        HERE
        / "assets"
        / "libero_object"
        / f"task_{task['task_id']:02d}"
        / "trajectory_block.xml"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    xml = f"""<mujoco model="trajectory_block">
  <worldbody>
    <body>
      <body name="object">
        <geom name="block_visual" type="box" pos="0 0 0" size="{half:.7f} {half:.7f} {half:.7f}" group="1" contype="0" conaffinity="0" rgba="0.20 0.45 0.95 1"/>
        <geom name="block_collision" type="box" pos="0 0 0" size="{half:.7f} {half:.7f} {half:.7f}" group="0" density="{density:.7f}" friction="0.95 0.3 0.1" solimp="0.998 0.998 0.001" solref="0.001 1" rgba="0.8 0.8 0.8 0.25"/>
      </body>
      <site rgba="0 0 0 0" size="0.003" pos="0 0 {-half:.7f}" name="bottom_site"/>
      <site rgba="0 0 0 0" size="0.003" pos="0 0 {half:.7f}" name="top_site"/>
      <site rgba="0 0 0 0" size="0.003" pos="{half:.7f} {half:.7f} 0" name="horizontal_radius_site"/>
    </body>
  </worldbody>
</mujoco>
"""
    path.write_text(xml, encoding="utf-8")
    footprint_area = task["extents"][0] * task["extents"][1]
    spawn_offset = task.get(
        "trajectory_spawn_offset",
        task["top_offset"] + half + 0.003,
    )
    settled_range = task.get(
        "trajectory_settled_range",
        [task["top_offset"] + half - 0.010, task["top_offset"] + half + 0.010],
    )
    calibration = {
        "trajectory_phase": f"pre-grasp approach to {task['target']}",
        "placement_rule": f"centered on the physical top surface of {task['target']}",
        "block_half_size_m": half,
        "block_side_length_m": side,
        "target_top_reference_area_m2": footprint_area,
        "block_to_target_bbox_top_area_ratio": side ** 2 / footprint_area,
        "target_mass_kg": task["mass"],
        "block_density_kg_m3": density,
        "matched_friction": [0.95, 0.3, 0.1],
        "initial_spawn_center_offset_m": spawn_offset,
        "initial_spawn_xy_compensation_m": task.get("trajectory_xy", [0.0, 0.0]),
        "settled_block_center_offset_range_m": settled_range,
        "physical_support": f"direct contact with the top of {task['target']}",
        "grasp_equivalence": "target-scaled footprint, matched mass and contact friction",
    }
    return path.resolve(), calibration


def build_task(task: dict, tray_xml: Path, position_xml: Path) -> dict:
    task_dir = (
        HERE / "specs" / "libero_object" / f"task_{task['task_id']:02d}"
    )
    task_dir.mkdir(parents=True, exist_ok=True)
    original_objects = bddl_objects(BDDL_ROOT / f"{task['name']}.bddl")
    target = task["target"]
    category = task["category"]
    target_span = max(task["extents"][:2])
    adjacent_distance = max(0.075, min(0.105, target_span + 0.014))
    cases = {}

    distractor = f"{category}_2"
    bddl = write_override_bddl(task, "same_category", distractor, category)
    same_category_xml = make_exact_shape_color_asset(task)
    definition = {
        "zh": "同类别干扰物",
        "definition": "保持与目标完全相同的类别和几何结构，但使用明显不同的颜色，并紧挨目标放置。",
        "target": target,
        "target_category": category,
        "distractor": distractor,
        "category": category,
        "construction": "same_category_exact_geometry_recolored",
        "source_xml": str(source_xml(category).resolve()),
        "appearance_change": "color_changed",
        "color_tint_rgba": task["tint"],
    }
    spec = base_spec(task, "same_category", bddl, definition)
    spec["object_scale_overrides"] = [{
        "object": distractor,
        "category": category,
        "scale": 1.0,
        "xml_path": str(same_category_xml),
        "apply_stage": "precompile_xml",
    }]
    spec["object_pose_overrides"] = [
        adjacent_pose(
            task,
            distractor,
            target,
            adjacent_distance,
            quat_from=target,
        )
    ]
    write_json(spec, task_dir / "same_category.json")
    cases["same_category"] = str((task_dir / "same_category.json").resolve())

    distractor = "red_coffee_mug_1"
    bddl = write_override_bddl(task, "same_color", distractor, "red_coffee_mug")
    definition = {
        "zh": "同颜色/纹理/图案干扰物",
        "definition": "真实带把手杯复用目标物的真实颜色、纹理和图案；类别明确不同，紧挨目标且不接触。",
        "target": target,
        "target_category": category,
        "distractor": distractor,
        "category": "red_coffee_mug",
        "construction": "different_category_target_texture_transfer",
    }
    spec = base_spec(task, "same_color", bddl, definition)
    mug_xml = make_target_texture_mug(task)
    spec["object_scale_overrides"] = [{
        "object": distractor,
        "category": "red_coffee_mug",
        "scale": 1.0,
        "xml_path": str(mug_xml),
        "apply_stage": "precompile_xml",
    }]
    mug_distance = max(0.082, min(0.105, target_span / 2.0 + 0.055))
    spec["object_pose_overrides"] = [
        adjacent_pose(
            task,
            distractor,
            target,
            mug_distance,
            quat=[0.0, 0.0, 0.0, 1.0],
        )
    ]
    if task["task_id"] == 9:
        pose = spec["object_pose_overrides"][0]
        pose["candidate_offsets"] = [
            [0.0, -mug_distance, 0.0],
            [0.0, mug_distance, 0.0],
            [mug_distance, 0.0, 0.0],
            [-mug_distance, 0.0, 0.0],
        ]
        pose["candidate_selection"] = "first_clear"
        pose["quat"] = [1.0, 0.0, 0.0, 0.0]
    write_json(spec, task_dir / "same_color.json")
    cases["same_color"] = str((task_dir / "same_color.json").resolve())

    shape_category, shape_xml, shape_label = make_shape_asset(task)
    distractor = f"{shape_category}_1"
    bddl = write_override_bddl(task, "same_shape", distractor, shape_category)
    definition = {
        "zh": "同形状干扰物",
        "definition": "匹配目标轮廓和尺度、具有独立功能结构的不同语义类别代理。",
        "target": target,
        "target_category": category,
        "distractor": distractor,
        "category": shape_category,
        "proxy_identity": task["same_shape_semantic"],
        "proxy_label": shape_label,
        "shape_kind": task["shape_kind"],
        "semantic_relation": "different_category_same_outline",
        "target_extent_match": "custom proxy calibrated to target collision AABB",
        "construction": "different_category_outline_matched_proxy",
        "appearance_change": "native_proxy_identity",
    }
    spec = base_spec(task, "same_shape", bddl, definition)
    if shape_xml is not None:
        spec["object_scale_overrides"] = [{
            "object": distractor,
            "category": shape_category,
            "scale": 1.0,
            "xml_path": str(shape_xml),
            "apply_stage": "precompile_xml",
        }]
    spec["object_pose_overrides"] = [
        adjacent_pose(
            task,
            distractor,
            target,
            adjacent_distance,
            quat=[1.0, 0.0, 0.0, 0.0],
        )
    ]
    write_json(spec, task_dir / "same_shape.json")
    cases["same_shape"] = str((task_dir / "same_shape.json").resolve())

    distractor = "wooden_tray_1"
    bddl = write_override_bddl(task, "same_affordance", distractor, "wooden_tray")
    definition = {
        "zh": "同可供性干扰物",
        "definition": "不同类别的真实木托盘，具有和目标篮子相同的承放/容纳功能。",
        "target": "basket_1",
        "target_category": "basket",
        "distractor": distractor,
        "category": "wooden_tray",
        "construction": "different_category_same_function",
    }
    spec = base_spec(task, "same_affordance", bddl, definition)
    spec["object_scale_overrides"] = [{
        "object": distractor,
        "category": "wooden_tray",
        "scale": 0.68,
        "xml_path": str(tray_xml),
        "apply_stage": "precompile_xml",
    }]
    spec["object_pose_overrides"] = [{
        "object": distractor,
        "relative_to": "basket_1",
        "candidate_offsets": [[0.0, 0.20, 0.0], [0.20, 0.0, 0.0], [-0.20, 0.0, 0.0], [0.0, -0.20, 0.0]],
        "candidate_selection": "max_clearance",
        "avoid_objects": original_objects,
        "workspace_bounds_xy": WORKSPACE_BOUNDS,
        "min_distance_from_others": 0.045,
        "z_from": "basket_1",
    }]
    write_json(spec, task_dir / "same_affordance.json")
    cases["same_affordance"] = str((task_dir / "same_affordance.json").resolve())

    distractor = "trajectory_block_d1_same_position"
    bddl = write_override_bddl(task, "same_position", distractor, "trajectory_block")
    definition = {
        "zh": "同位置干扰物",
        "definition": "把原目标物向旁边挪动，并在目标原始位置放置与任务无关的不同类别实体。",
        "target": target,
        "target_category": category,
        "distractor": distractor,
        "category": "trajectory_block",
        "construction": "move_target_and_replace_original_position",
    }
    spec = base_spec(task, "same_position", bddl, definition)
    spec["object_scale_overrides"] = [{
        "object": distractor,
        "category": "trajectory_block",
        "scale": 1.0,
        "xml_path": str(position_xml),
        "apply_stage": "precompile_xml",
    }]
    move_distance = max(0.145, min(0.180, target_span + 0.060))
    spec["same_position_replacements"] = [{
        "target": target,
        "replacement": distractor,
        "candidate_offsets": [[0.0, move_distance, 0.0], [move_distance, 0.0, 0.0], [-move_distance, 0.0, 0.0], [0.0, -move_distance, 0.0]],
        "candidate_selection": "max_clearance",
        "avoid_objects": [name for name in original_objects if name != target],
        "workspace_bounds_xy": WORKSPACE_BOUNDS,
        "min_distance_from_others": 0.055,
        "require_min_distance": True,
        "replacement_preserve_support_z": False,
    }]
    spec["object_pose_overrides"] = [{
        "object": distractor,
        "relative_to": distractor,
        "offset": [0.0, 0.0, 0.020],
        "quat": [1.0, 0.0, 0.0, 0.0],
    }]
    write_json(spec, task_dir / "same_position.json")
    cases["same_position"] = str((task_dir / "same_position.json").resolve())

    distractor = "trajectory_block_d1"
    bddl = write_override_bddl(task, "trajectory", distractor, "trajectory_block")
    definition = {
        "zh": "轨迹干扰物",
        "definition": "在目标物顶部放置真实碰撞方块，使其直接占据目标抓取路径；方块宽度近似目标所需夹爪开度。",
        "target": target,
        "target_category": category,
        "distractor": distractor,
        "category": "trajectory_block",
        "construction": "target_supported_grasp_width_matched_block",
    }
    spec = base_spec(task, "trajectory", bddl, definition)
    block_xml, calibration = make_trajectory_block(task)
    spec["trajectory_xml_path"] = str(block_xml)
    spec["object_scale_overrides"] = [{
        "object": distractor,
        "category": "trajectory_block",
        "scale": 1.0,
        "xml_path": str(block_xml),
        "apply_stage": "precompile_xml",
    }]
    spec["object_pose_overrides"] = [{
        "object": distractor,
        "on_top_of": target,
        "support_gap_m": 0.001,
        "quat": [1.0, 0.0, 0.0, 0.0],
    }]
    calibration.update({
        "placement_mode": "top",
        "placement_rule": "center on the target top and settle with direct target support",
        "gripper_opening_reference_m": calibration["block_side_length_m"],
        "gripper_opening_basis": "target graspable cross-section, capped to Panda usable opening",
        "physical_support": target,
        "floating": False,
    })
    spec["trajectory_calibration"] = calibration
    write_json(spec, task_dir / "trajectory.json")
    cases["trajectory"] = str((task_dir / "trajectory.json").resolve())

    return {
        "task_id": task["task_id"],
        "task_name": task["name"],
        "instruction": task["name"].replace("_", " "),
        "target": target,
        "target_category": category,
        "target_is_fixture": False,
        "receiver": "basket_1",
        "receiver_category": "basket",
        "primary_target_policy": "first required manipulation for multi-stage tasks",
        "cases": cases,
    }


def build_suite() -> dict:
    tray_xml = make_scaled_tray()
    position_xml = make_position_block()
    tasks = []
    for task in TASKS:
        print(f"[D1-object] building task {task['task_id']:02d}: {task['category']}", flush=True)
        tasks.append(build_task(task, tray_xml, position_xml))
    manifest = {
        "schema_version": "d1_4_suites_full_suite_manifest.v1",
        "suite": "libero_object",
        "conditions": CONDITIONS,
        "task_count": len(tasks),
        "case_count": len(tasks) * len(CONDITIONS),
        "tasks": tasks,
    }
    write_json(manifest, HERE / "manifest_libero_object.json")
    write_json(TASKS, HERE / "target_calibration_libero_object.json")
    return manifest


def main() -> None:
    manifest = build_suite()
    print(
        json.dumps(
            {
                "tasks": manifest["task_count"],
                "cases": manifest["case_count"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
