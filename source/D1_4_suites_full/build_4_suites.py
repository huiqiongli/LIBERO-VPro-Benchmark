#!/usr/bin/env python3
"""Build the approved six D1 semantic distractors for all four LIBERO suites."""

from __future__ import annotations

import json
import gc
import math
import re
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps


HERE = Path(__file__).resolve().parent
SIM_ROOT = Path.home() / "Sim"
ASSETS = SIM_ROOT / "Repos/LIBERO/libero/libero/assets"
BDDL_BASE = SIM_ROOT / "Repos/LIBERO/libero/libero/bddl_files"
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

SUITE_TASKS = {
    "libero_spatial": [
        ("pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate", "akita_black_bowl_1", "akita_black_bowl", "plate_1", "plate"),
        ("pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate", "akita_black_bowl_1", "akita_black_bowl", "plate_1", "plate"),
        ("pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate", "akita_black_bowl_1", "akita_black_bowl", "plate_1", "plate"),
        ("pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate", "akita_black_bowl_1", "akita_black_bowl", "plate_1", "plate"),
        ("pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate", "akita_black_bowl_1", "akita_black_bowl", "plate_1", "plate"),
        ("pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate", "akita_black_bowl_1", "akita_black_bowl", "plate_1", "plate"),
        ("pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate", "akita_black_bowl_1", "akita_black_bowl", "plate_1", "plate"),
        ("pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate", "akita_black_bowl_1", "akita_black_bowl", "plate_1", "plate"),
        ("pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate", "akita_black_bowl_1", "akita_black_bowl", "plate_1", "plate"),
        ("pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate", "akita_black_bowl_1", "akita_black_bowl", "plate_1", "plate"),
    ],
    "libero_goal": [
        ("open_the_middle_drawer_of_the_cabinet", "wooden_cabinet_1", "wooden_cabinet", "wooden_cabinet_1", "wooden_cabinet"),
        ("put_the_bowl_on_the_stove", "akita_black_bowl_1", "akita_black_bowl", "flat_stove_1", "flat_stove"),
        ("put_the_wine_bottle_on_top_of_the_cabinet", "wine_bottle_1", "wine_bottle", "wooden_cabinet_1", "wooden_cabinet"),
        ("open_the_top_drawer_and_put_the_bowl_inside", "akita_black_bowl_1", "akita_black_bowl", "wooden_cabinet_1", "wooden_cabinet"),
        ("put_the_bowl_on_top_of_the_cabinet", "akita_black_bowl_1", "akita_black_bowl", "wooden_cabinet_1", "wooden_cabinet"),
        ("push_the_plate_to_the_front_of_the_stove", "plate_1", "plate", "flat_stove_1", "flat_stove"),
        ("put_the_cream_cheese_in_the_bowl", "cream_cheese_1", "cream_cheese", "akita_black_bowl_1", "akita_black_bowl"),
        ("turn_on_the_stove", "flat_stove_1", "flat_stove", "flat_stove_1", "flat_stove"),
        ("put_the_bowl_on_the_plate", "akita_black_bowl_1", "akita_black_bowl", "plate_1", "plate"),
        ("put_the_wine_bottle_on_the_rack", "wine_bottle_1", "wine_bottle", "wine_rack_1", "wine_rack"),
    ],
    "libero_10": [
        ("LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket", "alphabet_soup_1", "alphabet_soup", "basket_1", "basket"),
        ("LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket", "cream_cheese_1", "cream_cheese", "basket_1", "basket"),
        ("KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it", "moka_pot_1", "moka_pot", "flat_stove_1", "flat_stove"),
        ("KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it", "akita_black_bowl_1", "akita_black_bowl", "white_cabinet_1", "white_cabinet"),
        ("LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate", "porcelain_mug_1", "porcelain_mug", "plate_1", "plate"),
        ("STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy", "black_book_1", "black_book", "desk_caddy_1", "desk_caddy"),
        ("LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate", "porcelain_mug_1", "porcelain_mug", "plate_1", "plate"),
        ("LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket", "alphabet_soup_1", "alphabet_soup", "basket_1", "basket"),
        ("KITCHEN_SCENE8_put_both_moka_pots_on_the_stove", "moka_pot_1", "moka_pot", "flat_stove_1", "flat_stove"),
        ("KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it", "white_yellow_mug_1", "white_yellow_mug", "microwave_1", "microwave"),
    ],
}

FIXTURE_TARGETS = {
    ("libero_goal", 0),
    ("libero_goal", 7),
}

TINTS = [
    [0.05, 0.85, 1.0, 1.0],
    [0.95, 0.12, 0.72, 1.0],
    [0.18, 0.88, 0.25, 1.0],
    [0.35, 0.22, 0.95, 1.0],
]

SAME_COLOR_PRIMARY_TEXTURE = {
    "alphabet_soup": "texture_map.png",
    "cream_cheese": "texture_map.png",
    "moka_pot": "metal_diff.png",
    "akita_black_bowl": "texture.png",
    "porcelain_mug": "porcelain_mug_texture.png",
    "black_book": "black_book_texture.png",
    "white_yellow_mug": "white_yellow_mug_texture.png",
}

# Normalized crops select the visible label / surface rather than the black
# unused area present in several UV atlases.
SAME_COLOR_TEXTURE_CROP = {
    "alphabet_soup": (0.00, 0.64, 0.52, 1.00),
    "cream_cheese": (0.23, 0.31, 0.96, 1.00),
}

# Different official object identities with a similar global silhouette.
# The proxy keeps its own mesh details and native texture; it is not a tinted
# clone of the target.
SHAPE_PROXY_CONFIG = {
    "alphabet_soup": ("tomato_sauce", 1.00),
    "cream_cheese": ("butter", 1.00),
    "moka_pot": ("red_coffee_mug", 0.80),
    "akita_black_bowl": ("glazed_rim_porcelain_ramekin", 1.00),
    "wine_bottle": ("paper_roll", 1.00),
    "plate": ("glazed_rim_porcelain_ramekin", 1.00),
    "wooden_cabinet": ("wooden_cabinet", 1.00),
    "flat_stove": ("flat_stove", 1.00),
    "porcelain_mug": ("white_yellow_mug", 1.00),
    "black_book": ("yellow_book", 1.00),
    "white_yellow_mug": ("porcelain_mug", 1.00),
}

SHAPE_PROXY_IDENTITY = {
    "alphabet_soup": "portable_speaker",
    "cream_cheese": "compact_electronic_device",
    "moka_pot": "handled_signal_lamp",
    "akita_black_bowl": "robotic_vacuum",
    "wine_bottle": "lighthouse_figurine",
    "plate": "analog_wall_clock",
    "wooden_cabinet": "vending_machine",
    "flat_stove": "computer_keyboard",
    "black_book": "tablet_device",
    "porcelain_mug": "handled_candle_lantern",
    "white_yellow_mug": "handled_candle_lantern",
}

PARKING = {
    "main_table": (0.28, 0.28),
    "living_room_table": (0.24, 0.20),
    "kitchen_table": (0.25, 0.20),
    "study_table": (0.24, 0.20),
}

WORKSPACE_BOUNDS = {
    "main_table": [[-0.48, 0.48], [-0.36, 0.38]],
    "living_room_table": [[-0.34, 0.34], [-0.34, 0.38]],
    "kitchen_table": [[-0.36, 0.36], [-0.35, 0.35]],
    "study_table": [[-0.32, 0.32], [-0.30, 0.30]],
}

# Approximate gripper openings used to grasp / actuate each target category.
# Values are capped by the Panda gripper's usable opening and intentionally
# describe the target's graspable cross-section rather than its full AABB.
TRAJECTORY_GRASP_WIDTH_M = {
    "akita_black_bowl": 0.070,
    "wine_bottle": 0.045,
    "plate": 0.075,
    "cream_cheese": 0.055,
    "alphabet_soup": 0.055,
    "moka_pot": 0.070,
    "porcelain_mug": 0.065,
    "white_yellow_mug": 0.065,
    "black_book": 0.025,
    "wooden_cabinet": 0.030,
    "flat_stove": 0.030,
}

TRAJECTORY_INSIDE_CATEGORIES = {
    "akita_black_bowl",
    "plate",
}


def write_json(value: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def source_xml(category: str) -> Path:
    for family in (
        "stable_hope_objects",
        "stable_scanned_objects",
        "turbosquid_objects",
        "d1_distractor_objects",
    ):
        candidate = ASSETS / family / category / f"{category}.xml"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"missing official object XML for {category}")


def section_items(text: str, section_name: str) -> dict[str, str]:
    _, _, section = extract_section_bounds(text, section_name)
    result: dict[str, str] = {}
    for line in section.splitlines():
        stripped = line.strip()
        if " - " not in stripped:
            continue
        left, category = stripped.split(" - ", 1)
        for name in left.split():
            result[name] = category.strip()
    return result


def add_name_to_section(
    text: str,
    section_name: str,
    category: str,
    name: str,
) -> str:
    if name in section_items(text, section_name):
        return text
    start, end, section = extract_section_bounds(text, section_name)
    lines = section.splitlines()
    inserted = False
    for index, line in enumerate(lines):
        match = re.match(rf"^(\s*)(.*?)\s+-\s+{re.escape(category)}\s*$", line)
        if not match:
            continue
        indent, existing = match.groups()
        lines[index] = f"{indent}{existing} {name} - {category}"
        inserted = True
        break
    if not inserted:
        lines.insert(max(1, len(lines) - 1), f"    {name} - {category}")
    return text[:start] + "\n".join(lines) + text[end:]


def infer_workspace(text: str) -> str:
    fixtures = section_items(text, "fixtures")
    for name, category in fixtures.items():
        if category == "table":
            return name
    for name in PARKING:
        if name in fixtures:
            return name
    raise ValueError("could not infer the supporting table fixture")


def source_bddl(suite: str, task_name: str) -> Path:
    return BDDL_BASE / suite / f"{task_name}.bddl"


def bddl_override(
    suite: str,
    task_id: int,
    task_name: str,
    condition: str,
    name: str,
    category: str,
    *,
    fixture: bool = False,
) -> tuple[Path, str, list[str]]:
    source = source_bddl(suite, task_name)
    text = source.read_text(encoding="utf-8")
    workspace = infer_workspace(text)
    existing = list(section_items(text, "objects")) + list(section_items(text, "fixtures"))
    region_name = f"d1_{condition}_parking"
    text = add_regions_to_regions_section(
        text,
        [region_block(region_name, workspace, PARKING[workspace])],
    )
    if fixture:
        text = add_name_to_section(text, "fixtures", category, name)
    else:
        text = add_objects_to_objects_section(text, {category: [name]})
    text = add_init_lines(text, [f"(On {name} {workspace}_{region_name})"])
    output = HERE / "bddl_overrides" / suite / f"task_{task_id:02d}" / f"{condition}.bddl"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")
    return output.resolve(), workspace, existing


def exact_shape_color_asset(
    suite: str,
    task_id: int,
    category: str,
    tint: list[float],
) -> Path:
    source = source_xml(category)
    tint_key = "_".join(str(int(round(value * 255))) for value in tint[:3])
    target_dir = HERE / "assets" / "shared" / "same_shape" / category / tint_key
    output = target_dir / f"{category}_d1_exact_shape_color_copy.xml"
    if output.exists():
        return output.resolve()
    if target_dir.exists():
        shutil.rmtree(target_dir)
    shutil.copytree(source.parent, target_dir)
    for path in target_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
            tint_texture_file(path, tint)
    tree = ET.parse(target_dir / source.name)
    root = tree.getroot()
    root.set("model", f"{category}_d1_exact_shape_color_copy")
    root.insert(
        0,
        ET.Comment(
            f"Exact visual/collision geometry clone of {category}; color only changed."
        ),
    )
    rgba = " ".join(f"{value:.6g}" for value in tint)
    for material in root.findall(".//material"):
        material.set("rgba", rgba)
    ET.indent(tree, space="  ")
    tree.write(output, encoding="utf-8", xml_declaration=False)
    return output.resolve()


def texture_files(directory: Path) -> list[Path]:
    return sorted(
        (
            path
            for path in directory.rglob("*")
            if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg"}
        ),
        key=lambda path: path.stat().st_size,
        reverse=True,
    )


def same_color_mug_asset(
    suite: str,
    task_id: int,
    category: str,
    texture_source_dir: Path | None = None,
) -> Path:
    native_dir = ASSETS / "turbosquid_objects/red_coffee_mug"
    target_dir = HERE / "assets" / "shared" / "same_color" / category
    output = target_dir / "red_coffee_mug_d1_same_color_v2.xml"
    if target_dir.exists():
        shutil.rmtree(target_dir)
    shutil.copytree(native_dir, target_dir)
    source_dir = (
        texture_source_dir
        if texture_source_dir is not None
        else source_xml(category).parent
    )
    primary_name = SAME_COLOR_PRIMARY_TEXTURE.get(category)
    primary_texture = (
        source_dir / primary_name
        if primary_name is not None
        else None
    )
    target_images = (
        [primary_texture]
        if primary_texture is not None and primary_texture.exists()
        else texture_files(source_dir)
    )
    mug_texture = target_dir / "red_coffee_mug_texture.png"
    if category == "black_book":
        with Image.open(mug_texture) as native:
            mapped = Image.new("RGB", native.size, (10, 10, 12))
        draw = ImageDraw.Draw(mapped)
        width, height = mapped.size
        draw.rectangle(
            (int(0.66 * width), 0, int(0.90 * width), height),
            fill=(215, 213, 205),
        )
        draw.rectangle(
            (int(0.90 * width), 0, width, height),
            fill=(48, 48, 52),
        )
        mapped.save(mug_texture)
    elif category == "wine_bottle":
        with Image.open(mug_texture) as native:
            mapped = Image.new("RGB", native.size, (12, 18, 10))
        draw = ImageDraw.Draw(mapped)
        width, height = mapped.size
        draw.rectangle(
            (
                int(0.20 * width),
                int(0.40 * height),
                int(0.80 * width),
                int(0.61 * height),
            ),
            fill=(166, 139, 91),
        )
        draw.rectangle(
            (
                int(0.20 * width),
                int(0.47 * height),
                int(0.80 * width),
                int(0.54 * height),
            ),
            fill=(76, 45, 28),
        )
        mapped.save(mug_texture)
    elif category == "plate":
        with Image.open(mug_texture) as native:
            mapped = Image.new("RGB", native.size, (250, 245, 238))
        draw = ImageDraw.Draw(mapped)
        width, height = mapped.size
        stripe = max(5, height // 18)
        for center in (int(0.28 * height), int(0.72 * height)):
            draw.rectangle(
                (0, center - stripe, width, center + stripe),
                fill=(220, 145, 140),
            )
            draw.rectangle(
                (
                    0,
                    center - max(1, stripe // 3),
                    width,
                    center + max(1, stripe // 3),
                ),
                fill=(255, 244, 232),
            )
        mapped.save(mug_texture)
    elif category == "akita_black_bowl":
        with Image.open(mug_texture) as native:
            mapped = Image.new("RGB", native.size, (218, 218, 214))
        draw = ImageDraw.Draw(mapped)
        width, height = mapped.size
        band = max(4, height // 22)
        draw.rectangle(
            (0, int(0.18 * height), width, int(0.18 * height) + band),
            fill=(72, 74, 75),
        )
        draw.rectangle(
            (0, int(0.82 * height), width, int(0.82 * height) + band),
            fill=(72, 74, 75),
        )
        mapped.save(mug_texture)
    elif category == "moka_pot":
        with Image.open(mug_texture) as native:
            mapped = Image.new("RGB", native.size, (235, 233, 228))
        draw = ImageDraw.Draw(mapped)
        width, height = mapped.size
        band = max(5, height // 18)
        draw.rectangle(
            (0, int(0.35 * height), width, int(0.35 * height) + band),
            fill=(54, 56, 58),
        )
        draw.rectangle(
            (0, int(0.68 * height), width, int(0.68 * height) + band),
            fill=(86, 88, 90),
        )
        mapped.save(mug_texture)
    elif target_images:
        with Image.open(target_images[0]) as source:
            source = source.convert("RGB")
            crop = SAME_COLOR_TEXTURE_CROP.get(category)
            if crop is not None:
                width, height = source.size
                source = source.crop(
                    (
                        int(round(crop[0] * width)),
                        int(round(crop[1] * height)),
                        int(round(crop[2] * width)),
                        int(round(crop[3] * height)),
                    )
                )
            elif category in {"porcelain_mug", "white_yellow_mug"}:
                # The official mug atlases reserve most pixels as pure black.
                # Mapping that entire atlas onto a different UV layout makes
                # the distractor visibly black although the target is
                # white/gray or white/yellow. Crop to the actual painted
                # object region before transferring the texture.
                non_black = ImageOps.grayscale(source).point(
                    lambda value: 255 if value > 24 else 0
                )
                content_bbox = non_black.getbbox()
                if content_bbox is not None:
                    source = source.crop(content_bbox)
                sample_pixels = list(
                    source.resize((192, 192), Image.Resampling.BILINEAR).getdata()
                )
                painted_pixels = [
                    pixel for pixel in sample_pixels if max(pixel) > 24
                ]
                if painted_pixels:
                    midpoint = len(painted_pixels) // 2
                    fill_color = tuple(
                        sorted(pixel[channel] for pixel in painted_pixels)[
                            midpoint
                        ]
                        for channel in range(3)
                    )
                    atlas_background = ImageOps.grayscale(source).point(
                        lambda value: 255 if value <= 24 else 0
                    )
                    source = Image.composite(
                        Image.new("RGB", source.size, fill_color),
                        source,
                        atlas_background,
                    )
            mapped = ImageOps.fit(
                source,
                Image.open(mug_texture).size,
                method=Image.Resampling.LANCZOS,
            )
        mapped.save(mug_texture)
    scale_mujoco_xml(target_dir / "red_coffee_mug.xml", output, 0.55)
    return output.resolve()


def different_identity_shape_asset(
    target_category: str,
    proxy_category: str,
    scale: float,
) -> Path:
    if target_category == "alphabet_soup":
        return can_outline_speaker_asset()
    if target_category == "cream_cheese":
        return box_outline_device_asset()
    if target_category == "moka_pot":
        return moka_outline_lantern_asset()
    if target_category == "akita_black_bowl":
        return bowl_outline_robot_vacuum_asset()
    if target_category == "wine_bottle":
        return wine_outline_lighthouse_asset()
    if target_category == "plate":
        return plate_outline_clock_asset()
    if target_category == "wooden_cabinet":
        return fixture_identity_asset("wooden_cabinet")
    if target_category == "flat_stove":
        return fixture_identity_asset("flat_stove")
    if target_category == "black_book":
        return book_outline_tablet_asset()
    if target_category in {"porcelain_mug", "white_yellow_mug"}:
        return mug_outline_lantern_asset()
    source = source_xml(proxy_category)
    scale_key = str(int(round(scale * 1000.0)))
    target_dir = (
        HERE
        / "assets"
        / "shared"
        / "same_shape_proxy"
        / target_category
        / f"{proxy_category}_{scale_key}"
    )
    output = target_dir / f"{proxy_category}_d1_outline_proxy.xml"
    if target_dir.exists():
        shutil.rmtree(target_dir)
    shutil.copytree(source.parent, target_dir)
    scale_mujoco_xml(target_dir / source.name, output, scale)
    return output.resolve()


def can_outline_speaker_asset() -> Path:
    """A speaker skin over the official can outline and collision geometry."""
    source = source_xml("tomato_sauce")
    target_dir = (
        HERE
        / "assets"
        / "shared"
        / "same_shape_proxy"
        / "alphabet_soup"
        / "portable_speaker"
    )
    output = target_dir / "portable_speaker.xml"
    if target_dir.exists():
        shutil.rmtree(target_dir)
    shutil.copytree(source.parent, target_dir)

    texture = target_dir / "texture_map.png"
    with Image.open(texture) as native:
        size = native.size
    speaker = Image.new("RGB", size, (12, 24, 30))
    draw = ImageDraw.Draw(speaker)
    width, height = size
    spacing = max(5, width // 42)
    dot = max(1, spacing // 4)
    for y in range(spacing // 2, height, spacing):
        for x in range(spacing // 2, width, spacing):
            draw.ellipse((x - dot, y - dot, x + dot, y + dot), fill=(60, 82, 88))
    band = max(8, height // 16)
    draw.rectangle((0, 0, width, band), fill=(8, 155, 190))
    draw.rectangle((0, height - band, width, height), fill=(8, 155, 190))
    panel = (int(0.36 * width), int(0.38 * height), int(0.64 * width), int(0.62 * height))
    draw.rounded_rectangle(panel, radius=max(3, width // 80), fill=(20, 28, 32))
    cx = width // 2
    cy = height // 2
    draw.line((cx - width // 18, cy, cx + width // 18, cy), fill=(235, 235, 225), width=max(2, width // 100))
    draw.line((cx, cy - width // 18, cx, cy + width // 18), fill=(235, 235, 225), width=max(2, width // 100))
    speaker.save(texture)

    tree = ET.parse(target_dir / source.name)
    root = tree.getroot()
    root.set("model", "d1_portable_speaker")
    root.insert(
        0,
        ET.Comment(
            "Official can outline/collision retained; speaker grille, controls, "
            "and accent bands establish a different functional identity."
        ),
    )
    ET.indent(tree, space="  ")
    tree.write(output, encoding="utf-8", xml_declaration=False)
    return output.resolve()


def box_outline_device_asset() -> Path:
    """A compact electronic device in the cream-cheese box envelope."""
    output = (
        HERE
        / "assets"
        / "shared"
        / "same_shape_proxy"
        / "cream_cheese"
        / "compact_electronic_device.xml"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        """<mujoco model="d1_compact_electronic_device">
  <worldbody>
    <body>
      <body name="object">
        <geom name="device_case_visual" type="box" pos="0 0 0" size="0.0406 0.0213 0.0089" group="1" contype="0" conaffinity="0" rgba="0.08 0.11 0.14 1"/>
        <geom name="device_screen_visual" type="box" pos="-0.007 0 0.0094" size="0.025 0.015 0.0007" group="1" contype="0" conaffinity="0" rgba="0.05 0.55 0.68 1"/>
        <geom name="device_key_1_visual" type="cylinder" pos="0.029 -0.010 0.0101" size="0.0035 0.0008" group="1" contype="0" conaffinity="0" rgba="0.92 0.45 0.10 1"/>
        <geom name="device_key_2_visual" type="cylinder" pos="0.029 0 0.0101" size="0.0035 0.0008" group="1" contype="0" conaffinity="0" rgba="0.90 0.90 0.86 1"/>
        <geom name="device_key_3_visual" type="cylinder" pos="0.029 0.010 0.0101" size="0.0035 0.0008" group="1" contype="0" conaffinity="0" rgba="0.90 0.90 0.86 1"/>
        <geom name="device_collision" type="box" pos="0 0 0" size="0.0406 0.0213 0.0089" group="0" density="110" friction="0.95 0.3 0.1" rgba="0.8 0.8 0.8 0.2"/>
      </body>
      <site rgba="0 0 0 0" size="0.003" pos="0 0 -0.0089" name="bottom_site"/>
      <site rgba="0 0 0 0" size="0.003" pos="0 0 0.0109" name="top_site"/>
      <site rgba="0 0 0 0" size="0.003" pos="0.0406 0.0213 0" name="horizontal_radius_site"/>
    </body>
  </worldbody>
</mujoco>
""",
        encoding="utf-8",
    )
    return output.resolve()


def bowl_outline_robot_vacuum_asset() -> Path:
    """A flat robotic-vacuum proxy matching the bowl's circular footprint."""
    source = source_xml("glazed_rim_porcelain_ramekin")
    target_dir = (
        HERE
        / "assets"
        / "shared"
        / "same_shape_proxy"
        / "akita_black_bowl"
        / "robotic_vacuum"
    )
    output = target_dir / "robotic_vacuum.xml"
    if target_dir.exists():
        shutil.rmtree(target_dir)
    shutil.copytree(source.parent, target_dir)
    tree = ET.parse(target_dir / source.name)
    root = tree.getroot()
    root.set("model", "d1_robotic_vacuum")
    object_body = root.find(".//body[@name='object']")
    if object_body is None:
        raise ValueError("ramekin proxy XML has no object body")
    for geom in list(object_body.findall("geom")):
        object_body.remove(geom)
    visual_geoms = [
        ("vacuum_body_visual", "cylinder", "0 0 0.013", "0.043 0.015", "0.12 0.15 0.18 1"),
        ("vacuum_top_visual", "cylinder", "0 0 0.029", "0.039 0.002", "0.20 0.24 0.27 1"),
        ("vacuum_lidar_visual", "cylinder", "0.014 0.008 0.033", "0.009 0.004", "0.04 0.05 0.06 1"),
        ("vacuum_button_visual", "cylinder", "-0.015 -0.007 0.032", "0.005 0.002", "0.08 0.70 0.78 1"),
    ]
    for name, geom_type, pos, size, rgba in visual_geoms:
        ET.SubElement(
            object_body,
            "geom",
            {
                "name": name,
                "type": geom_type,
                "pos": pos,
                "size": size,
                "group": "1",
                "contype": "0",
                "conaffinity": "0",
                "rgba": rgba,
            },
        )
    collision_geoms = [
        (
            "vacuum_base_collision",
            "0 0 0.002",
            "0.043 0.004",
            "520",
        ),
        (
            "vacuum_upper_collision",
            "0 0 0.023",
            "0.039 0.017",
            "24",
        ),
    ]
    for name, pos, size, density in collision_geoms:
        ET.SubElement(
            object_body,
            "geom",
            {
                "name": name,
                "type": "cylinder",
                "pos": pos,
                "size": size,
                "group": "0",
                "density": density,
                "friction": "1.20 0.4 0.1",
                "rgba": "0.8 0.8 0.8 0.2",
            },
        )
    sites = {site.get("name"): site for site in root.findall(".//site")}
    sites["bottom_site"].set("pos", "0 0 -0.002")
    sites["top_site"].set("pos", "0 0 0.037")
    sites["horizontal_radius_site"].set("pos", "0.043 0.043 0")
    root.insert(
        0,
        ET.Comment(
            "Flat circular visual and collision geometry, lidar turret, and "
            "status button establish a robotic-vacuum identity."
        ),
    )
    ET.indent(tree, space="  ")
    tree.write(output, encoding="utf-8", xml_declaration=False)
    return output.resolve()


def wine_outline_lighthouse_asset() -> Path:
    """A lighthouse figurine calibrated to the official wine-bottle AABB."""
    x, y, z = 0.0425932, 0.0442396, 0.1569306
    hx, hy, hz = x / 2.0, y / 2.0, z / 2.0
    radius = min(hx, hy)
    output = (
        HERE
        / "assets"
        / "shared"
        / "same_shape_proxy"
        / "wine_bottle"
        / "lighthouse_figurine.xml"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        f"""<mujoco model="d1_lighthouse_figurine">
  <worldbody>
    <body>
      <body name="object">
        <geom name="tower_white_visual" type="cylinder" pos="0 0 {-hz + z*0.39:.7f}" size="{radius:.7f} {z*0.39:.7f}" group="1" contype="0" conaffinity="0" rgba="0.94 0.92 0.84 1"/>
        <geom name="tower_red_lower_visual" type="cylinder" pos="0 0 {-hz + z*0.20:.7f}" size="{radius*1.002:.7f} {z*0.055:.7f}" group="1" contype="0" conaffinity="0" rgba="0.78 0.06 0.08 1"/>
        <geom name="tower_red_upper_visual" type="cylinder" pos="0 0 {-hz + z*0.52:.7f}" size="{radius*1.002:.7f} {z*0.055:.7f}" group="1" contype="0" conaffinity="0" rgba="0.78 0.06 0.08 1"/>
        <geom name="lantern_room_visual" type="cylinder" pos="0 0 {hz-z*0.18:.7f}" size="{radius*0.68:.7f} {z*0.10:.7f}" group="1" contype="0" conaffinity="0" rgba="0.20 0.68 0.86 1"/>
        <geom name="lantern_window_x_visual" type="box" pos="0 0 {hz-z*0.18:.7f}" size="{radius*0.12:.7f} {radius*0.70:.7f} {z*0.10:.7f}" group="1" contype="0" conaffinity="0" rgba="0.05 0.08 0.10 1"/>
        <geom name="lantern_window_y_visual" type="box" pos="0 0 {hz-z*0.18:.7f}" size="{radius*0.70:.7f} {radius*0.12:.7f} {z*0.10:.7f}" group="1" contype="0" conaffinity="0" rgba="0.05 0.08 0.10 1"/>
        <geom name="lighthouse_roof_visual" type="cylinder" pos="0 0 {hz-z*0.055:.7f}" size="{radius:.7f} {z*0.045:.7f}" group="1" contype="0" conaffinity="0" rgba="0.10 0.12 0.14 1"/>
        <geom name="tower_collision" type="cylinder" pos="0 0 {-hz + z*0.39:.7f}" size="{radius:.7f} {z*0.39:.7f}" group="0" density="120" friction="0.95 0.3 0.1" rgba="0.8 0.8 0.8 0.2"/>
        <geom name="lantern_collision" type="cylinder" pos="0 0 {hz-z*0.18:.7f}" size="{radius*0.68:.7f} {z*0.10:.7f}" group="0" density="80" friction="0.95 0.3 0.1" rgba="0.8 0.8 0.8 0.2"/>
        <geom name="roof_collision" type="cylinder" pos="0 0 {hz-z*0.055:.7f}" size="{radius:.7f} {z*0.045:.7f}" group="0" density="80" friction="0.95 0.3 0.1" rgba="0.8 0.8 0.8 0.2"/>
      </body>
      <site rgba="0 0 0 0" size="0.003" pos="0 0 {-hz:.7f}" name="bottom_site"/>
      <site rgba="0 0 0 0" size="0.003" pos="0 0 {hz:.7f}" name="top_site"/>
      <site rgba="0 0 0 0" size="0.003" pos="{hx:.7f} {hy:.7f} 0" name="horizontal_radius_site"/>
    </body>
  </worldbody>
</mujoco>
""",
        encoding="utf-8",
    )
    return output.resolve()


def plate_outline_clock_asset() -> Path:
    """A shallow analog clock matching the official plate's circular AABB."""
    x, y, z = 0.1374699, 0.1376212, 0.0189525
    radius = min(x, y) / 2.0
    half_z = z / 2.0
    output = (
        HERE
        / "assets"
        / "shared"
        / "same_shape_proxy"
        / "plate"
        / "analog_wall_clock.xml"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    tick_geoms = []
    for index in range(12):
        angle = math.radians(index * 30.0)
        tick_geoms.append(
            (
                f'<geom name="clock_tick_{index:02d}" type="box" '
                f'pos="{0.052*math.sin(angle):.7f} '
                f'{0.052*math.cos(angle):.7f} {half_z+0.0012:.7f}" '
                f'quat="{math.cos(angle/2):.7f} 0 0 {math.sin(angle/2):.7f}" '
                f'size="0.0014 0.0060 0.0005" group="1" contype="0" '
                f'conaffinity="0" rgba="0.06 0.07 0.08 1"/>'
            )
        )
    output.write_text(
        f"""<mujoco model="d1_analog_wall_clock">
  <worldbody>
    <body>
      <body name="object">
        <geom name="clock_case_visual" type="cylinder" pos="0 0 0" size="{radius:.7f} {half_z:.7f}" group="1" contype="0" conaffinity="0" rgba="0.10 0.12 0.14 1"/>
        <geom name="clock_face_visual" type="cylinder" pos="0 0 {half_z:.7f}" size="{radius*0.94:.7f} 0.0008" group="1" contype="0" conaffinity="0" rgba="0.92 0.90 0.82 1"/>
        {' '.join(tick_geoms)}
        <geom name="clock_hour_hand" type="box" pos="-0.012 0 {half_z+0.0018:.7f}" size="0.012 0.0015 0.0005" group="1" contype="0" conaffinity="0" rgba="0.08 0.09 0.10 1"/>
        <geom name="clock_minute_hand" type="box" pos="0 0.019 {half_z+0.0020:.7f}" size="0.0012 0.019 0.0005" group="1" contype="0" conaffinity="0" rgba="0.08 0.09 0.10 1"/>
        <geom name="clock_center_visual" type="cylinder" pos="0 0 {half_z+0.0028:.7f}" size="0.0035 0.0010" group="1" contype="0" conaffinity="0" rgba="0.72 0.08 0.08 1"/>
        <geom name="clock_collision" type="cylinder" pos="0 0 0" size="{radius:.7f} {half_z:.7f}" group="0" density="120" friction="0.95 0.3 0.1" rgba="0.8 0.8 0.8 0.2"/>
      </body>
      <site rgba="0 0 0 0" size="0.003" pos="0 0 {-half_z:.7f}" name="bottom_site"/>
      <site rgba="0 0 0 0" size="0.003" pos="0 0 {half_z+0.0038:.7f}" name="top_site"/>
      <site rgba="0 0 0 0" size="0.003" pos="{radius:.7f} {radius:.7f} 0" name="horizontal_radius_site"/>
    </body>
  </worldbody>
</mujoco>
""",
        encoding="utf-8",
    )
    return output.resolve()


def fixture_identity_asset(category: str) -> Path:
    """Retain fixture outline/physics while adding a different clear identity."""
    if category not in {"wooden_cabinet", "flat_stove"}:
        raise ValueError(category)
    source = ASSETS / "articulated_objects" / f"{category}.xml"
    identity = (
        "vending_machine"
        if category == "wooden_cabinet"
        else "computer_keyboard"
    )
    target_dir = (
        HERE
        / "assets"
        / "shared"
        / "same_shape_proxy"
        / category
        / identity
    )
    if target_dir.exists():
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True)
    shutil.copy2(source, target_dir / source.name)
    asset_subdir = ASSETS / "articulated_objects" / category
    if asset_subdir.exists():
        shutil.copytree(asset_subdir, target_dir / category)
    tree = ET.parse(target_dir / source.name)
    root = tree.getroot()
    root.set("model", f"d1_{identity}")
    object_body = root.find(".//body[@name='object']")
    if object_body is None:
        raise ValueError(f"{category} fixture XML has no object body")
    for material in root.findall(".//material"):
        material.set(
            "rgba",
            "0.08 0.22 0.36 1"
            if category == "wooden_cabinet"
            else "0.12 0.14 0.18 1",
        )
    if category == "wooden_cabinet":
        additions = [
            ("vending_screen_visual", "box", "0 -0.118 0.158", "0.048 0.003 0.034", "0.04 0.62 0.78 1"),
            ("vending_slot_visual", "box", "0 -0.122 0.080", "0.050 0.003 0.009", "0.04 0.05 0.06 1"),
            ("vending_button_red_visual", "sphere", "0.073 -0.123 0.152", "0.008", "0.85 0.08 0.08 1"),
            ("vending_button_green_visual", "sphere", "0.073 -0.123 0.126", "0.008", "0.08 0.75 0.30 1"),
        ]
    else:
        additions = []
        for row in range(3):
            for column in range(8):
                additions.append(
                    (
                        f"keyboard_key_{row}_{column}",
                        "box",
                        f"{0.075 + column*0.020:.4f} {-0.055 + row*0.036:.4f} 0.048",
                        "0.0075 0.0120 0.0030",
                        "0.78 0.82 0.86 1",
                    )
                )
    for name, geom_type, pos, size, rgba in additions:
        ET.SubElement(
            object_body,
            "geom",
            {
                "name": name,
                "type": geom_type,
                "pos": pos,
                "size": size,
                "group": "1",
                "contype": "0",
                "conaffinity": "0",
                "rgba": rgba,
            },
        )
    root.insert(
        0,
        ET.Comment(
            f"Original {category} outline/physics retained; visual identity "
            f"changed to {identity}."
        ),
    )
    output = target_dir / f"{identity}.xml"
    ET.indent(tree, space="  ")
    tree.write(output, encoding="utf-8", xml_declaration=False)
    return output.resolve()


def moka_outline_lantern_asset() -> Path:
    output = (
        HERE
        / "assets"
        / "shared"
        / "same_shape_proxy"
        / "moka_pot"
        / "handled_signal_lamp.xml"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    def write_octagonal_frustum(
        path: Path,
        bottom_radius: float,
        top_radius: float,
        bottom_z: float,
        top_z: float,
    ) -> None:
        vertices = []
        for z, radius in (
            (bottom_z, bottom_radius),
            (top_z, top_radius),
        ):
            for index in range(8):
                angle = math.radians(22.5 + 45.0 * index)
                vertices.append(
                    (
                        radius * math.cos(angle),
                        radius * math.sin(angle),
                        z,
                    )
                )
        lines = [
            f"v {x:.8f} {y:.8f} {z:.8f}"
            for x, y, z in vertices
        ]
        lines.append("f 8 7 6 5 4 3 2 1")
        lines.append("f 9 10 11 12 13 14 15 16")
        for index in range(8):
            next_index = (index + 1) % 8
            lines.append(
                f"f {index + 1} {next_index + 1} "
                f"{next_index + 9} {index + 9}"
            )
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    lower_mesh = output.parent / "signal_lamp_lower_octagon.obj"
    upper_mesh = output.parent / "signal_lamp_upper_octagon.obj"
    write_octagonal_frustum(lower_mesh, 0.040, 0.030, -0.065, 0.010)
    write_octagonal_frustum(upper_mesh, 0.030, 0.039, 0.012, 0.064)
    output.write_text(
        """<mujoco model="d1_handled_signal_lamp">
  <asset>
    <mesh file="signal_lamp_lower_octagon.obj" name="signal_lamp_lower_octagon"/>
    <mesh file="signal_lamp_upper_octagon.obj" name="signal_lamp_upper_octagon"/>
  </asset>
  <worldbody>
    <body>
      <body name="object">
        <geom name="signal_lower_visual" type="mesh" mesh="signal_lamp_lower_octagon" group="1" contype="0" conaffinity="0" rgba="0.07 0.13 0.20 1"/>
        <geom name="signal_globe_visual" type="mesh" mesh="signal_lamp_upper_octagon" group="1" contype="0" conaffinity="0" rgba="0.96 0.65 0.12 1"/>
        <geom name="signal_waist_visual" type="cylinder" pos="0 0 0.011" size="0.031 0.006" group="1" contype="0" conaffinity="0" rgba="0.08 0.11 0.14 1"/>
        <geom name="signal_cap_visual" type="cylinder" pos="0 0 0.069" size="0.041 0.005" group="1" contype="0" conaffinity="0" rgba="0.08 0.11 0.14 1"/>
        <geom name="signal_button_visual" type="sphere" pos="0 0 0.080" size="0.008" group="1" contype="0" conaffinity="0" rgba="0.78 0.10 0.08 1"/>
        <geom name="signal_cage_x_visual" type="box" pos="0 0 0.039" size="0.004 0.039 0.027" group="1" contype="0" conaffinity="0" rgba="0.12 0.14 0.16 1"/>
        <geom name="signal_cage_y_visual" type="box" pos="0 0 0.039" size="0.039 0.004 0.027" group="1" contype="0" conaffinity="0" rgba="0.12 0.14 0.16 1"/>
        <geom name="handle_outer_visual" type="box" pos="0 0.069 0.030" size="0.005 0.005 0.044" group="1" contype="0" conaffinity="0" rgba="0.10 0.10 0.12 1"/>
        <geom name="handle_low_visual" type="box" pos="0 0.053 -0.010" size="0.005 0.020 0.005" group="1" contype="0" conaffinity="0" rgba="0.10 0.10 0.12 1"/>
        <geom name="handle_high_visual" type="box" pos="0 0.053 0.070" size="0.005 0.020 0.005" group="1" contype="0" conaffinity="0" rgba="0.10 0.10 0.12 1"/>
        <geom name="reflector_housing_visual" type="box" pos="0 -0.055 0.052" size="0.019 0.017 0.016" group="1" contype="0" conaffinity="0" rgba="0.18 0.20 0.22 1"/>
        <geom name="reflector_lens_visual" type="cylinder" pos="0 -0.073 0.052" quat="0.70710678 0.70710678 0 0" size="0.018 0.002" group="1" contype="0" conaffinity="0" rgba="0.88 0.94 0.98 1"/>
        <geom name="signal_lower_collision" type="cylinder" pos="0 0 -0.027" size="0.040 0.038" group="0" density="95" friction="1.0 0.3 0.1" rgba="0.8 0.8 0.8 0.2"/>
        <geom name="signal_upper_collision" type="cylinder" pos="0 0 0.039" size="0.039 0.027" group="0" density="70" friction="0.95 0.3 0.1" rgba="0.8 0.8 0.8 0.2"/>
        <geom name="handle_outer_collision" type="box" pos="0 0.069 0.030" size="0.005 0.005 0.044" group="0" density="25" friction="0.95 0.3 0.1" rgba="0.8 0.8 0.8 0.2"/>
        <geom name="handle_low_collision" type="box" pos="0 0.053 -0.010" size="0.005 0.020 0.005" group="0" density="25" friction="0.95 0.3 0.1" rgba="0.8 0.8 0.8 0.2"/>
        <geom name="handle_high_collision" type="box" pos="0 0.053 0.070" size="0.005 0.020 0.005" group="0" density="25" friction="0.95 0.3 0.1" rgba="0.8 0.8 0.8 0.2"/>
        <geom name="reflector_collision" type="box" pos="0 -0.055 0.052" size="0.019 0.017 0.016" group="0" density="25" friction="0.95 0.3 0.1" rgba="0.8 0.8 0.8 0.2"/>
      </body>
      <site rgba="0 0 0 0" size="0.003" pos="0 0 -0.065" name="bottom_site"/>
      <site rgba="0 0 0 0" size="0.003" pos="0 0 0.088" name="top_site"/>
      <site rgba="0 0 0 0" size="0.003" pos="0.041 0.074 0" name="horizontal_radius_site"/>
    </body>
  </worldbody>
</mujoco>
""",
        encoding="utf-8",
    )
    return output.resolve()


def book_outline_tablet_asset() -> Path:
    output = (
        HERE
        / "assets"
        / "shared"
        / "same_shape_proxy"
        / "black_book"
        / "tablet_device.xml"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        """<mujoco model="d1_tablet_device">
  <worldbody>
    <body>
      <body name="object">
        <geom name="tablet_frame_visual" type="box" pos="0 0 0" size="0.0115 0.051 0.067" group="1" contype="0" conaffinity="0" rgba="0.16 0.20 0.22 1"/>
        <geom name="tablet_screen_visual" type="box" pos="0.012 0 0" size="0.0015 0.040 0.056" group="1" contype="0" conaffinity="0" rgba="0.05 0.42 0.52 1"/>
        <geom name="tablet_screen_back_visual" type="box" pos="-0.012 0 0" size="0.0015 0.040 0.056" group="1" contype="0" conaffinity="0" rgba="0.05 0.42 0.52 1"/>
        <geom name="tablet_button_visual" type="cylinder" pos="0.014 -0.044 0" quat="0.7071 0 0.7071 0" size="0.004 0.0015" group="1" contype="0" conaffinity="0" rgba="0.85 0.85 0.82 1"/>
        <geom name="tablet_collision" type="box" pos="0 0 0" size="0.0115 0.051 0.067" group="0" density="90" friction="0.90 0.3 0.1" rgba="0.8 0.8 0.8 0.2"/>
      </body>
      <site rgba="0 0 0 0" size="0.003" pos="0 0 -0.067" name="bottom_site"/>
      <site rgba="0 0 0 0" size="0.003" pos="0 0 0.067" name="top_site"/>
      <site rgba="0 0 0 0" size="0.003" pos="0.015 0.051 0" name="horizontal_radius_site"/>
    </body>
  </worldbody>
</mujoco>
""",
        encoding="utf-8",
    )
    return output.resolve()


def mug_outline_lantern_asset() -> Path:
    output = (
        HERE
        / "assets"
        / "shared"
        / "same_shape_proxy"
        / "mug"
        / "handled_candle_lantern.xml"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        """<mujoco model="d1_handled_candle_lantern">
  <worldbody>
    <body>
      <body name="object">
        <geom name="lantern_glow_visual" type="cylinder" pos="0 0 0" size="0.040 0.046" group="1" contype="0" conaffinity="0" rgba="0.92 0.46 0.10 1"/>
        <geom name="lantern_base_visual" type="box" pos="0 0 -0.049" size="0.046 0.046 0.004" group="1" contype="0" conaffinity="0" rgba="0.08 0.10 0.12 1"/>
        <geom name="lantern_bottom_band_visual" type="cylinder" pos="0 0 -0.044" size="0.042 0.006" group="1" contype="0" conaffinity="0" rgba="0.10 0.14 0.18 1"/>
        <geom name="lantern_top_band_visual" type="cylinder" pos="0 0 0.044" size="0.042 0.006" group="1" contype="0" conaffinity="0" rgba="0.10 0.14 0.18 1"/>
        <geom name="lantern_cage_x_visual" type="box" pos="0 0 0" size="0.004 0.041 0.050" group="1" contype="0" conaffinity="0" rgba="0.12 0.16 0.20 1"/>
        <geom name="lantern_cage_y_visual" type="box" pos="0 0 0" size="0.041 0.004 0.050" group="1" contype="0" conaffinity="0" rgba="0.12 0.16 0.20 1"/>
        <geom name="handle_outer_visual" type="box" pos="0 0.068 0" size="0.005 0.005 0.038" group="1" contype="0" conaffinity="0" rgba="0.10 0.12 0.14 1"/>
        <geom name="handle_low_visual" type="box" pos="0 0.054 -0.035" size="0.005 0.018 0.005" group="1" contype="0" conaffinity="0" rgba="0.10 0.12 0.14 1"/>
        <geom name="handle_high_visual" type="box" pos="0 0.054 0.035" size="0.005 0.018 0.005" group="1" contype="0" conaffinity="0" rgba="0.10 0.12 0.14 1"/>
        <geom name="lantern_body_collision" type="cylinder" pos="0 0 0" size="0.040 0.050" group="0" density="70" friction="0.95 0.3 0.1" rgba="0.8 0.8 0.8 0.2"/>
        <geom name="lantern_base_collision" type="box" pos="0 0 -0.049" size="0.046 0.046 0.004" group="0" density="180" friction="1.20 0.4 0.1" rgba="0.8 0.8 0.8 0.2"/>
        <geom name="handle_outer_collision" type="box" pos="0 0.068 0" size="0.005 0.005 0.038" group="0" density="35" friction="0.95 0.3 0.1" rgba="0.8 0.8 0.8 0.2"/>
        <geom name="handle_low_collision" type="box" pos="0 0.054 -0.035" size="0.005 0.018 0.005" group="0" density="35" friction="0.95 0.3 0.1" rgba="0.8 0.8 0.8 0.2"/>
        <geom name="handle_high_collision" type="box" pos="0 0.054 0.035" size="0.005 0.018 0.005" group="0" density="35" friction="0.95 0.3 0.1" rgba="0.8 0.8 0.8 0.2"/>
      </body>
      <site rgba="0 0 0 0" size="0.003" pos="0 0 -0.050" name="bottom_site"/>
      <site rgba="0 0 0 0" size="0.003" pos="0 0 0.050" name="top_site"/>
      <site rgba="0 0 0 0" size="0.003" pos="0.045 0.073 0" name="horizontal_radius_site"/>
    </body>
  </worldbody>
</mujoco>
""",
        encoding="utf-8",
    )
    return output.resolve()


def trajectory_block_asset(
    side: float = 0.055,
    *,
    cap_fit: bool = False,
) -> Path:
    half = side / 2.0
    millimeters = int(round(side * 1000.0))
    suffix = "_capfit" if cap_fit else ""
    output = (
        HERE
        / "assets"
        / "shared"
        / f"trajectory_block_{millimeters:03d}mm{suffix}.xml"
    )
    if cap_fit:
        cavity_half = 0.008
        wall_half = (half - cavity_half) / 2.0
        wall_center = cavity_half + wall_half
        lower_half_z = half * 0.28
        lower_center_z = -half + lower_half_z
        slab_half_z = half * 0.22
        slab_center_z = half - slab_half_z
        collision_geoms = f"""
        <geom name="block_collision_top" type="box" pos="0 0 {slab_center_z}" size="{half} {half} {slab_half_z}" group="0" density="30" friction="1.2 0.4 0.1" rgba="0.8 0.8 0.8 0.25"/>
        <geom name="block_collision_xp" type="box" pos="{wall_center} 0 {lower_center_z}" size="{wall_half} {half} {lower_half_z}" group="0" density="30" friction="1.2 0.4 0.1" rgba="0.8 0.8 0.8 0.25"/>
        <geom name="block_collision_xn" type="box" pos="{-wall_center} 0 {lower_center_z}" size="{wall_half} {half} {lower_half_z}" group="0" density="30" friction="1.2 0.4 0.1" rgba="0.8 0.8 0.8 0.25"/>
        <geom name="block_collision_yp" type="box" pos="0 {wall_center} {lower_center_z}" size="{cavity_half} {wall_half} {lower_half_z}" group="0" density="30" friction="1.2 0.4 0.1" rgba="0.8 0.8 0.8 0.25"/>
        <geom name="block_collision_yn" type="box" pos="0 {-wall_center} {lower_center_z}" size="{cavity_half} {wall_half} {lower_half_z}" group="0" density="30" friction="1.2 0.4 0.1" rgba="0.8 0.8 0.8 0.25"/>"""
    else:
        collision_geoms = f"""
        <geom name="block_collision" type="box" pos="0 0 0" size="{half} {half} {half}" group="0" density="30" friction="1.2 0.4 0.1" rgba="0.8 0.8 0.8 0.25"/>"""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        f"""<mujoco model="d1_supported_trajectory_block">
  <worldbody>
    <body>
      <body name="object">
        <geom name="block_visual" type="box" pos="0 0 0" size="{half} {half} {half}" group="1" contype="0" conaffinity="0" rgba="0.20 0.45 0.95 1"/>
        {collision_geoms}
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


def position_block_asset() -> Path:
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


def scaled_tray_asset() -> Path:
    source = source_xml("wooden_tray")
    output = HERE / "assets" / "shared" / "wooden_tray_scaled.xml"
    output.parent.mkdir(parents=True, exist_ok=True)
    scale_mujoco_xml(source, output, 0.68)
    return output.resolve()


def adjacent_pose(
    name: str,
    target: str,
    existing: list[str],
    workspace: str,
    distance: float,
    *,
    fixture: bool = False,
    quat_from: str | None = None,
    match_target_z: bool = True,
) -> dict:
    pose = {
        "object": name,
        "relative_to": target,
        "candidate_offsets": [
            [0.0, distance, 0.0],
            [distance, 0.0, 0.0],
            [-distance, 0.0, 0.0],
            [0.0, -distance, 0.0],
        ],
        "candidate_selection": "max_clearance",
        "avoid_objects": existing,
        "workspace_bounds_xy": WORKSPACE_BOUNDS[workspace],
        "min_distance_from_others": 0.045 if not fixture else 0.18,
    }
    if fixture or not match_target_z:
        pose["preserve_z"] = True
    else:
        pose["z_from"] = target
    if fixture:
        pose["fixed_fixture"] = True
    if quat_from and not fixture:
        pose["quat_from"] = quat_from
    return pose


def base_spec(
    suite: str,
    task_id: int,
    task_name: str,
    target: str,
    target_category: str,
    condition: str,
    bddl: Path,
    definition: dict,
) -> dict:
    source = source_bddl(suite, task_name).resolve()
    return {
        "schema_version": "d1_4_suites_full.v1",
        "status": "implemented",
        "type": condition,
        "suite": suite,
        "benchmark": suite,
        "task_id": task_id,
        "task_name": task_name,
        "instruction": task_name.replace("_", " "),
        "base_bddl_file_name": str(source),
        "bddl_file_name_override": str(bddl),
        "target": target,
        "target_category": target_category,
        "definition": definition,
        "settle_steps": 18,
    }


def affordance_choice(
    suite: str,
    task_id: int,
    receiver_category: str,
) -> tuple[str, str, bool, str]:
    if (suite, task_id) == ("libero_goal", 0):
        return "white_cabinet_d1_affordance", "white_cabinet", True, "可开合并可容纳物体的不同材质柜体"
    if (suite, task_id) == ("libero_goal", 7):
        return "microwave_d1_affordance", "microwave", True, "可开关并可加热的不同类别电器"
    if receiver_category in {"wooden_cabinet", "white_cabinet", "microwave"}:
        return "basket_d1_affordance", "basket", False, "不同类别但同样提供容纳/放置功能的篮子"
    if receiver_category == "basket":
        return "wooden_tray_d1_affordance", "wooden_tray", False, "不同类别但同样提供承放/容纳功能的木托盘"
    if receiver_category in {"akita_black_bowl", "desk_caddy", "wine_rack"}:
        return "basket_d1_affordance", "basket", False, "不同类别但同样提供容纳功能的篮子"
    return "wooden_tray_d1_affordance", "wooden_tray", False, "不同类别但同样提供承放功能的木托盘"


def build_task(
    suite: str,
    task_id: int,
    record: tuple[str, str, str, str, str],
    tray_xml: Path,
    trajectory_xml: Path,
    position_xml: Path,
) -> dict:
    task_name, target, category, receiver, receiver_category = record
    fixture_target = (suite, task_id) in FIXTURE_TARGETS
    task_dir = HERE / "specs" / suite / f"task_{task_id:02d}"
    task_dir.mkdir(parents=True, exist_ok=True)
    cases: dict[str, str] = {}
    tint = TINTS[task_id % len(TINTS)]
    adjacent_distance = {
        "plate": 0.160,
        "moka_pot": 0.145,
        "porcelain_mug": 0.130,
        "white_yellow_mug": 0.130,
        "black_book": 0.205,
        "akita_black_bowl": 0.120,
    }.get(category, 0.105)
    if (suite, task_id) == ("libero_spatial", 4):
        adjacent_distance = 0.140
    if (suite, task_id) == ("libero_goal", 8):
        adjacent_distance = 0.145
    use_table_z = (suite, task_id) in {
        ("libero_spatial", 3),
        ("libero_spatial", 4),
        ("libero_spatial", 5),
        ("libero_spatial", 7),
    }

    name = f"{category}_d1_same_category"
    bddl, workspace, existing = bddl_override(
        suite,
        task_id,
        task_name,
        "same_category",
        name,
        category,
        fixture=fixture_target,
    )
    definition = {
        "zh": "同类别干扰物",
        "definition": "保持与目标完全相同的类别和几何结构，但使用明显不同的颜色，并紧挨目标放置。",
        "target": target,
        "target_category": category,
        "distractor": name,
        "category": category,
        "construction": "same_category_exact_geometry_recolored",
        "appearance_change": "color_changed",
        "color_tint_rgba": tint,
    }
    spec = base_spec(suite, task_id, task_name, target, category, "same_category", bddl, definition)
    if fixture_target:
        spec["geom_color_overrides"] = [{
            "body_name_contains": [name],
            "rgba": tint,
            "mesh_only": False,
            "include_descendants": True,
        }]
    else:
        category_color_xml = exact_shape_color_asset(
            suite,
            task_id,
            category,
            tint,
        )
        spec["object_scale_overrides"] = [{
            "object": name,
            "category": category,
            "scale": 1.0,
            "xml_path": str(category_color_xml),
            "apply_stage": "precompile_xml",
        }]
    spec["object_pose_overrides"] = [
        adjacent_pose(
            name,
            target,
            existing,
            workspace,
            (
                0.34
                if fixture_target
                else 0.160
                if (suite, task_id) == ("libero_goal", 1)
                else adjacent_distance
            ),
            fixture=fixture_target,
            quat_from=target,
            match_target_z=not use_table_z,
        )
    ]
    if (suite, task_id) == ("libero_spatial", 4):
        spec["object_pose_overrides"][0]["candidate_offsets"] = [
            [adjacent_distance, 0.0, 0.0],
            [0.0, adjacent_distance, 0.0],
            [0.0, -adjacent_distance, 0.0],
            [-adjacent_distance, 0.0, 0.0],
        ]
        spec["object_pose_overrides"][0]["candidate_selection"] = "first_clear"
    if suite == "libero_goal" and category == "akita_black_bowl":
        spec["object_pose_overrides"][0]["candidate_offsets"] = [
            [-0.100, 0.120, 0.0],
            [-0.130, 0.100, 0.0],
        ]
        spec["object_pose_overrides"][0]["candidate_selection"] = "first_clear"
    if (suite, task_id) == ("libero_spatial", 2):
        spec["object_pose_overrides"][0]["candidate_offsets"] = [
            [-0.160, 0.0, 0.0],
        ]
        spec["object_pose_overrides"][0]["candidate_selection"] = "first_clear"
    if (suite, task_id) == ("libero_spatial", 9):
        spec["object_pose_overrides"][0].pop("z_from", None)
        spec["object_pose_overrides"][0]["preserve_z"] = True
        spec["object_pose_overrides"][0]["candidate_offsets"] = [
            [0.0, 0.200, 0.0],
        ]
        spec["object_pose_overrides"][0]["candidate_selection"] = "first_clear"
    if (suite, task_id) == ("libero_10", 5):
        # The book is elevated beside the caddy.  The rear sector is the only
        # common supported surface across all official init states.
        spec["object_pose_overrides"][0]["candidate_offsets"] = [
            [0.0, -adjacent_distance, 0.0],
        ]
        spec["object_pose_overrides"][0]["candidate_selection"] = "first_clear"
    if (suite, task_id) == ("libero_10", 7):
        # The target starts on a raised support.  The left sector stays on that
        # support in all official init states; max-clearance can select an
        # unsupported side and launch the inserted can during settling.
        spec["object_pose_overrides"][0]["candidate_offsets"] = [
            [-adjacent_distance, 0.0, 0.0],
        ]
        spec["object_pose_overrides"][0]["candidate_selection"] = "first_clear"
    write_json(spec, task_dir / "same_category.json")
    cases["same_category"] = str((task_dir / "same_category.json").resolve())

    name = "red_coffee_mug_d1_same_color"
    bddl, workspace, existing = bddl_override(
        suite, task_id, task_name, "same_color", name, "red_coffee_mug"
    )
    definition = {
        "zh": "同颜色/纹理/图案干扰物",
        "definition": "不同类别的真实带把手杯映射目标物的颜色、纹理和图案，并紧挨目标放置。",
        "target": target,
        "target_category": category,
        "distractor": name,
        "category": "red_coffee_mug",
        "construction": "different_category_target_texture_transfer",
    }
    spec = base_spec(suite, task_id, task_name, target, category, "same_color", bddl, definition)
    if fixture_target:
        mug_xml = same_color_mug_asset(
            suite,
            task_id,
            category,
            ASSETS / "articulated_objects" / category,
        )
        spec["object_scale_overrides"] = [{
            "object": name,
            "category": "red_coffee_mug",
            "scale": 1.0,
            "xml_path": str(mug_xml),
            "apply_stage": "precompile_xml",
        }]
    else:
        mug_xml = same_color_mug_asset(suite, task_id, category)
        spec["object_scale_overrides"] = [{
            "object": name,
            "category": "red_coffee_mug",
            "scale": 1.0,
            "xml_path": str(mug_xml),
            "apply_stage": "precompile_xml",
        }]
    spec["object_pose_overrides"] = [
        adjacent_pose(
            name,
            target,
            existing,
            workspace,
            0.32 if fixture_target else adjacent_distance,
            match_target_z=not use_table_z,
        )
    ]
    if (suite, task_id) == ("libero_spatial", 4):
        spec["object_pose_overrides"][0]["candidate_offsets"] = [
            [adjacent_distance, 0.0, 0.0],
            [0.0, adjacent_distance, 0.0],
            [0.0, -adjacent_distance, 0.0],
            [-adjacent_distance, 0.0, 0.0],
        ]
        spec["object_pose_overrides"][0]["candidate_selection"] = "first_clear"
    if suite == "libero_goal" and category == "akita_black_bowl":
        spec["object_pose_overrides"][0]["candidate_offsets"] = [
            [-0.100, 0.120, 0.0],
            [-0.130, 0.100, 0.0],
        ]
        spec["object_pose_overrides"][0]["candidate_selection"] = "first_clear"
    if (suite, task_id) == ("libero_spatial", 2):
        spec["object_pose_overrides"][0]["candidate_offsets"] = [
            [-0.160, 0.0, 0.0],
        ]
        spec["object_pose_overrides"][0]["candidate_selection"] = "first_clear"
    if (suite, task_id) == ("libero_spatial", 9):
        spec["object_pose_overrides"][0].pop("z_from", None)
        spec["object_pose_overrides"][0]["preserve_z"] = True
        spec["object_pose_overrides"][0]["candidate_offsets"] = [
            [0.0, 0.200, 0.0],
        ]
        spec["object_pose_overrides"][0]["candidate_selection"] = "first_clear"
    if (suite, task_id) == ("libero_goal", 0):
        fixture_color_distance = 0.240
        spec["object_pose_overrides"][0]["candidate_offsets"] = [
            [fixture_color_distance, 0.0, 0.0],
            [0.0, fixture_color_distance, 0.0],
            [-fixture_color_distance, 0.0, 0.0],
            [0.0, -fixture_color_distance, 0.0],
        ]
        spec["object_pose_overrides"][0]["candidate_selection"] = "first_clear"
    write_json(spec, task_dir / "same_color.json")
    cases["same_color"] = str((task_dir / "same_color.json").resolve())

    shape_registry_category, proxy_scale = SHAPE_PROXY_CONFIG[category]
    shape_identity = SHAPE_PROXY_IDENTITY[category]
    name = f"{shape_registry_category}_d1_same_shape"
    bddl, workspace, existing = bddl_override(
        suite,
        task_id,
        task_name,
        "same_shape",
        name,
        shape_registry_category,
        fixture=fixture_target,
    )
    shape_override = different_identity_shape_asset(
        category,
        shape_registry_category,
        proxy_scale,
    )
    definition = {
        "zh": "同形状干扰物",
        "definition": "匹配目标轮廓和尺度、具有独立功能结构的不同语义类别代理。",
        "target": target,
        "target_category": category,
        "distractor": name,
        "category": shape_identity,
        "registry_category": shape_registry_category,
        "proxy_identity": shape_identity,
        "construction": "different_category_outline_matched_proxy",
        "appearance_change": "native_proxy_identity",
    }
    spec = base_spec(suite, task_id, task_name, target, category, "same_shape", bddl, definition)
    if shape_override:
        spec["object_scale_overrides"] = [{
            "object": name,
            "category": shape_registry_category,
            "scale": 1.0,
            "xml_path": str(shape_override),
            "apply_stage": "precompile_xml",
        }]
    spec["object_pose_overrides"] = [
        adjacent_pose(
            name,
            target,
            existing,
            workspace,
            0.34 if fixture_target else adjacent_distance,
            fixture=fixture_target,
            quat_from=target,
            match_target_z=not use_table_z,
        )
    ]
    if category in {"wine_bottle", "akita_black_bowl"}:
        # These target meshes have category-specific intrinsic rotations.
        # The lighthouse and robotic-vacuum proxies are authored upright in
        # world Z and must not inherit those mesh-frame quaternions.
        spec["object_pose_overrides"][0].pop("quat_from", None)
        spec["object_pose_overrides"][0].pop("z_from", None)
        spec["object_pose_overrides"][0]["quat"] = [1.0, 0.0, 0.0, 0.0]
        spec["object_pose_overrides"][0]["preserve_z"] = True
    if (suite, task_id, category) == (
        "libero_spatial",
        7,
        "akita_black_bowl",
    ):
        # The target bowl is on the raised stove, while the adjacent proxy is
        # placed on the tabletop.  Do not preserve the inserted object's
        # sampled stove/cabinet height; use the calibrated tabletop height.
        pose = spec["object_pose_overrides"][0]
        pose.pop("preserve_z", None)
        for offset in pose["candidate_offsets"]:
            offset[2] = -0.0261
    if (suite, task_id, category) == (
        "libero_spatial",
        1,
        "akita_black_bowl",
    ):
        pose = spec["object_pose_overrides"][0]
        pose.pop("preserve_z", None)
        pose["z_value"] = 0.902
    if suite == "libero_10" and category == "akita_black_bowl":
        # Leave a narrow physical gap between the two circular silhouettes.
        bowl_proxy_distance = 0.145
        spec["object_pose_overrides"][0]["candidate_offsets"] = [
            [0.0, bowl_proxy_distance, 0.0],
            [bowl_proxy_distance, 0.0, 0.0],
            [-bowl_proxy_distance, 0.0, 0.0],
            [0.0, -bowl_proxy_distance, 0.0],
        ]
    if (
        suite == "libero_10"
        and category in {"porcelain_mug", "white_yellow_mug"}
    ):
        # Official mug meshes use category-specific intrinsic axes. The
        # identity-distinct lantern proxy is authored upright in world Z.
        spec["object_pose_overrides"][0].pop("quat_from", None)
        spec["object_pose_overrides"][0].pop("z_from", None)
        spec["object_pose_overrides"][0]["quat"] = [1.0, 0.0, 0.0, 0.0]
        spec["object_pose_overrides"][0]["preserve_z"] = True
        lantern_distance = 0.160
        spec["object_pose_overrides"][0]["candidate_offsets"] = [
            [0.0, lantern_distance, 0.0],
            [lantern_distance, 0.0, 0.0],
            [-lantern_distance, 0.0, 0.0],
            [0.0, -lantern_distance, 0.0],
        ]
        spec["object_pose_overrides"][0]["candidate_selection"] = "first_clear"
        spec["object_pose_overrides"][0]["min_distance_from_others"] = 0.130
        spec["object_pose_overrides"][0]["require_min_distance"] = True
        if task_id in {4, 6}:
            spec["object_pose_overrides"][0]["candidate_offsets"] = [
                [-0.180, 0.0, 0.0],
                [0.0, -0.180, 0.0],
                [0.0, 0.180, 0.0],
                [0.180, 0.0, 0.0],
            ]
    if (suite, task_id) == ("libero_spatial", 4):
        spec["object_pose_overrides"][0]["candidate_offsets"] = [
            [adjacent_distance, 0.0, 0.0],
            [0.0, adjacent_distance, 0.0],
            [0.0, -adjacent_distance, 0.0],
            [-adjacent_distance, 0.0, 0.0],
        ]
        spec["object_pose_overrides"][0]["candidate_selection"] = "first_clear"
    if suite == "libero_goal" and category == "akita_black_bowl":
        spec["object_pose_overrides"][0]["candidate_offsets"] = [
            [-0.100, 0.120, 0.0],
            [-0.130, 0.100, 0.0],
        ]
        spec["object_pose_overrides"][0]["candidate_selection"] = "first_clear"
    if (suite, task_id) == ("libero_spatial", 2):
        spec["object_pose_overrides"][0]["candidate_offsets"] = [
            [-0.160, 0.0, 0.0],
        ]
        spec["object_pose_overrides"][0]["candidate_selection"] = "first_clear"
    if (suite, task_id) == ("libero_spatial", 9):
        spec["object_pose_overrides"][0].pop("z_from", None)
        spec["object_pose_overrides"][0]["preserve_z"] = True
        spec["object_pose_overrides"][0]["candidate_offsets"] = [
            [0.0, 0.200, 0.0],
        ]
        spec["object_pose_overrides"][0]["candidate_selection"] = "first_clear"
    if suite == "libero_goal" and category == "akita_black_bowl":
        pose = spec["object_pose_overrides"][0]
        pose.pop("preserve_z", None)
        pose.pop("z_from", None)
        # The diagonal used by the narrower cup/bowl variants overlaps the
        # stove base for this wider circular proxy. The target's rear side is
        # the only common collision-clear adjacent sector in all four tasks.
        pose["candidate_offsets"] = [
            [0.0, -0.160, 0.0],
        ]
        pose["candidate_selection"] = "first_clear"
        # The tabletop surface is z=0.900 m in LIBERO_Tabletop_Manipulation;
        # the proxy collision bottom is -0.002 m in its local frame.
        # Place it directly at resting height to avoid a stiff-contact impact.
        pose["z_value"] = 0.902
        if task_id == 1:
            # The rear sector is occupied by the stove footprint in this
            # layout; the proxy slides off it and contacts the target bowl.
            # The left tabletop sector is level and collision-clear.
            pose["candidate_offsets"] = [
                [-0.180, 0.0, 0.0],
            ]
        if task_id == 3:
            # Keep clearance from both the target rim and the rear cabinet.
            pose["candidate_offsets"] = [
                [-0.035, -0.170, 0.0],
            ]
        if task_id == 8:
            # In these initial states the rear sector is occupied by the
            # cabinet. Use the stove-supported diagonal, calibrated from the
            # stable same-category bowl placement.
            pose["candidate_offsets"] = [
                [-0.125, 0.125, 0.0],
            ]
            pose["z_value"] = 0.929
        if task_id == 4:
            # Add rim clearance while remaining inside the adjacent-distance
            # acceptance band after MuJoCo settling.
            pose["candidate_offsets"] = [[-0.175, 0.0, 0.0]]
    if (suite, task_id) == ("libero_goal", 5):
        # The +X sector tips the thin clock proxy onto its rim in two init
        # states.  The +Y tabletop sector keeps it flat and adjacent.
        spec["object_scale_overrides"][0]["scale"] = 1.0
        spec["object_pose_overrides"][0]["candidate_offsets"] = [
            [0.0, 0.175, 0.0],
        ]
        spec["object_pose_overrides"][0]["candidate_selection"] = "first_clear"
    if (suite, task_id) == ("libero_spatial", 5):
        spec["object_pose_overrides"][0]["candidate_offsets"] = [
            [0.0, adjacent_distance, 0.0],
        ]
        spec["object_pose_overrides"][0]["candidate_selection"] = "first_clear"
    if (suite, task_id) == ("libero_spatial", 7):
        spec["object_pose_overrides"][0]["candidate_offsets"] = [
            [adjacent_distance, 0.0, -0.0261],
        ]
        spec["object_pose_overrides"][0]["candidate_selection"] = "first_clear"
    if (suite, task_id) == ("libero_10", 7):
        spec["object_pose_overrides"][0]["candidate_offsets"] = [
            [-adjacent_distance, 0.0, 0.0],
        ]
        spec["object_pose_overrides"][0]["candidate_selection"] = "first_clear"
    write_json(spec, task_dir / "same_shape.json")
    cases["same_shape"] = str((task_dir / "same_shape.json").resolve())

    name, affordance_category, affordance_fixture, affordance_note = affordance_choice(
        suite, task_id, receiver_category
    )
    bddl, workspace, existing = bddl_override(
        suite,
        task_id,
        task_name,
        "same_affordance",
        name,
        affordance_category,
        fixture=affordance_fixture,
    )
    definition = {
        "zh": "同可供性干扰物",
        "definition": affordance_note,
        "target": receiver,
        "target_category": receiver_category,
        "distractor": name,
        "category": affordance_category,
        "construction": "different_category_same_function",
    }
    spec = base_spec(suite, task_id, task_name, target, category, "same_affordance", bddl, definition)
    if affordance_category == "wooden_tray":
        spec["object_scale_overrides"] = [{
            "object": name,
            "category": affordance_category,
            "scale": 1.0,
            "xml_path": str(tray_xml),
            "apply_stage": "precompile_xml",
        }]
    if affordance_fixture:
        distance = 0.34
    elif receiver_category in {"wooden_cabinet", "white_cabinet", "microwave"}:
        distance = 0.34
    elif receiver_category == "wine_rack":
        distance = 0.30
    elif receiver_category in {"akita_black_bowl", "desk_caddy"}:
        distance = 0.24
    else:
        distance = 0.20
    spec["object_pose_overrides"] = [
        adjacent_pose(
            name,
            receiver,
            existing,
            workspace,
            distance,
            fixture=affordance_fixture,
        )
    ]
    if (suite, task_id) == ("libero_goal", 9):
        spec["object_pose_overrides"][0]["candidate_offsets"] = [
            [-0.14, 0.36, 0.0],
        ]
        spec["object_pose_overrides"][0]["candidate_selection"] = "first_clear"
    if (suite, task_id) == ("libero_10", 5):
        spec["object_pose_overrides"][0]["candidate_offsets"] = [
            [0.34, 0.0, 0.0],
        ]
        spec["object_pose_overrides"][0]["candidate_selection"] = "first_clear"
    if (suite, task_id) == ("libero_10", 9):
        spec["object_pose_overrides"][0]["candidate_offsets"] = [
            [distance, 0.0, 0.0],
        ]
        spec["object_pose_overrides"][0]["candidate_selection"] = "first_clear"
    write_json(spec, task_dir / "same_affordance.json")
    cases["same_affordance"] = str((task_dir / "same_affordance.json").resolve())

    position_category = "trajectory_block"
    name = "trajectory_block_d1_same_position"
    bddl, workspace, existing = bddl_override(
        suite, task_id, task_name, "same_position", name, position_category
    )
    definition = {
        "zh": "同位置干扰物",
        "definition": "把原目标物向旁边挪动，并在目标原始位置放置与任务无关的不同类别实体。",
        "target": target,
        "target_category": category,
        "distractor": name,
        "category": position_category,
        "construction": "move_target_and_replace_original_position",
    }
    spec = base_spec(suite, task_id, task_name, target, category, "same_position", bddl, definition)
    spec["object_scale_overrides"] = [{
        "object": name,
        "category": position_category,
        "scale": 1.0,
        "xml_path": str(position_xml),
        "apply_stage": "precompile_xml",
    }]
    move_distance = {
        "plate": 0.190,
        "moka_pot": 0.180,
        "porcelain_mug": 0.160,
        "white_yellow_mug": 0.160,
        "black_book": 0.220,
        "akita_black_bowl": 0.165,
    }.get(category, 0.145)
    if (suite, task_id) == ("libero_10", 1):
        move_distance = 0.125
    candidate_offsets = (
        [[0.30, 0.0, 0.0], [-0.30, 0.0, 0.0], [0.0, 0.30, 0.0], [0.0, -0.30, 0.0]]
        if fixture_target
        else [[0.0, move_distance, 0.0], [move_distance, 0.0, 0.0], [-move_distance, 0.0, 0.0], [0.0, -move_distance, 0.0]]
    )
    # These targets start near a fixture / camera boundary. Prefer a direction
    # that moves them onto clear, visible tabletop space instead of maximizing
    # center-point clearance and accidentally selecting an off-screen edge.
    if (suite, task_id) == ("libero_spatial", 7):
        candidate_offsets = [
            [move_distance, 0.0, 0.0],
            [0.0, move_distance, 0.0],
            [-move_distance, 0.0, 0.0],
            [0.0, -move_distance, 0.0],
        ]
    elif (suite, task_id) == ("libero_spatial", 9):
        candidate_offsets = [
            [0.0, move_distance, 0.0],
            [-move_distance, 0.0, 0.0],
            [move_distance, 0.0, 0.0],
            [0.0, -move_distance, 0.0],
        ]
    elif (suite, task_id) == ("libero_10", 5):
        candidate_offsets = [[0.0, -move_distance, 0.0]]
    spec["same_position_replacements"] = [{
        "target": target,
        "replacement": name,
        "candidate_offsets": candidate_offsets,
        "candidate_selection": (
            "first_clear"
            if (suite, task_id)
            in {
                ("libero_spatial", 7),
                ("libero_spatial", 9),
                ("libero_10", 0),
                ("libero_10", 5),
                ("libero_10", 7),
            }
            else "max_clearance"
        ),
        "avoid_objects": [item for item in existing if item != target],
        "workspace_bounds_xy": WORKSPACE_BOUNDS[workspace],
        "min_distance_from_others": 0.055,
        "require_min_distance": True,
        "replacement_preserve_support_z": False,
    }]
    if (suite, task_id) == ("libero_spatial", 9):
        # The bowl begins on top of the cabinet. Place it aside on the main
        # table directly; retaining the cabinet-height Z makes it fall and roll
        # out of view during settling.
        spec["same_position_replacements"][0]["target_aside_z"] = 0.8984
    spec["object_pose_overrides"] = [{
        "object": name,
        "relative_to": name,
        "offset": [0.0, 0.0, 0.020],
        "quat": [1.0, 0.0, 0.0, 0.0],
    }]
    write_json(spec, task_dir / "same_position.json")
    cases["same_position"] = str((task_dir / "same_position.json").resolve())

    name = "trajectory_block_d1"
    bddl, workspace, existing = bddl_override(
        suite, task_id, task_name, "trajectory", name, "trajectory_block"
    )
    definition = {
        "zh": "轨迹干扰物",
        "definition": "在容器型目标内部或非容器目标顶部放置真实碰撞方块，使其直接占据目标抓取/交互路径；方块宽度近似目标所需夹爪开度。",
        "target": target,
        "target_category": category,
        "distractor": name,
        "category": "trajectory_block",
        "construction": "target_supported_grasp_width_matched_block",
    }
    spec = base_spec(suite, task_id, task_name, target, category, "trajectory", bddl, definition)
    trajectory_side = TRAJECTORY_GRASP_WIDTH_M[category]
    trajectory_xml = trajectory_block_asset(
        trajectory_side,
        cap_fit=(category == "wine_bottle"),
    )
    placement_mode = (
        "inside"
        if category in TRAJECTORY_INSIDE_CATEGORIES
        else "top"
    )
    spec["object_scale_overrides"] = [{
        "object": name,
        "category": "trajectory_block",
        "scale": 1.0,
        "xml_path": str(trajectory_xml),
        "apply_stage": "precompile_xml",
    }]
    spec["object_pose_overrides"] = [{
        "object": name,
        "on_top_of": target,
        "support_gap_m": -0.010 if category == "wine_bottle" else 0.001,
        "quat": [1.0, 0.0, 0.0, 0.0],
    }]
    spec["trajectory_calibration"] = {
        "trajectory_phase": f"pre-grasp/pre-interaction approach to {target}",
        "placement_mode": placement_mode,
        "placement_rule": (
            "drop into the target interior and settle on its inner support"
            if placement_mode == "inside"
            else "center on the target top and settle with direct target support"
        ),
        "block_side_length_m": trajectory_side,
        "gripper_opening_reference_m": trajectory_side,
        "gripper_opening_basis": (
            "target graspable cross-section, capped to Panda usable opening"
        ),
        "physical_support": target,
        "floating": False,
    }
    write_json(spec, task_dir / "trajectory.json")
    cases["trajectory"] = str((task_dir / "trajectory.json").resolve())

    return {
        "task_id": task_id,
        "task_name": task_name,
        "instruction": task_name.replace("_", " "),
        "target": target,
        "target_category": category,
        "target_is_fixture": fixture_target,
        "receiver": receiver,
        "receiver_category": receiver_category,
        "primary_target_policy": "first required manipulation for multi-stage tasks",
        "cases": cases,
    }


def main() -> None:
    tray_xml = scaled_tray_asset()
    trajectory_xml = trajectory_block_asset()
    position_xml = position_block_asset()
    root_manifest = {
        "schema_version": "d1_4_suites_full_manifest.v1",
        "conditions": CONDITIONS,
        "suite_count": 0,
        "task_count": 0,
        "case_count": 0,
        "model_evaluation": "not_run_by_user_request",
        "suites": {},
    }
    for suite, records in SUITE_TASKS.items():
        tasks = []
        for task_id, record in enumerate(records):
            print(f"[D1-build] {suite} task {task_id:02d}", flush=True)
            tasks.append(
                build_task(
                    suite,
                    task_id,
                    record,
                    tray_xml,
                    trajectory_xml,
                    position_xml,
                )
            )
            gc.collect()
        suite_manifest = {
            "suite": suite,
            "conditions": CONDITIONS,
            "task_count": len(tasks),
            "case_count": len(tasks) * len(CONDITIONS),
            "tasks": tasks,
        }
        write_json(suite_manifest, HERE / f"manifest_{suite}.json")
        root_manifest["suites"][suite] = suite_manifest

    # LIBERO-Object needs target-specific shape proxies and calibrated grasp
    # widths, but emits the same schema and the same six perturbation types.
    from build_libero_object_suite import build_suite as build_object_suite

    object_manifest = build_object_suite()
    root_manifest["suites"]["libero_object"] = object_manifest
    root_manifest["suite_count"] = len(root_manifest["suites"])
    root_manifest["task_count"] = sum(
        suite_manifest["task_count"]
        for suite_manifest in root_manifest["suites"].values()
    )
    root_manifest["case_count"] = sum(
        suite_manifest["case_count"]
        for suite_manifest in root_manifest["suites"].values()
    )
    write_json(root_manifest, HERE / "manifest.json")
    print(
        json.dumps(
            {
                "suites": root_manifest["suite_count"],
                "tasks": root_manifest["task_count"],
                "cases": root_manifest["case_count"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
