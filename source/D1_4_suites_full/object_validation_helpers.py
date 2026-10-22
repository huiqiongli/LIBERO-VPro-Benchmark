#!/usr/bin/env python3
"""Render and validate all 60 LIBERO-Object D1 semantic distractor cases."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import traceback
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps


HERE = Path(__file__).resolve().parent
SIM_ROOT = Path.home() / "Sim"
ASSETS = SIM_ROOT / "Repos/LIBERO/libero/libero/assets"
sys.path[:0] = [
    str(SIM_ROOT / "Repos/LIBERO"),
    str(SIM_ROOT / "Repos/lingbot-va"),
    str(SIM_ROOT / "reviewed_script/_shared_core"),
]
os.environ.setdefault("MUJOCO_GL", "egl")

import client  # noqa: E402
from libero.libero import benchmark  # noqa: E402


CONDITIONS = [
    "same_category",
    "same_color",
    "same_shape",
    "same_affordance",
    "same_position",
    "trajectory",
]
PREVIEWS = HERE / "previews"
VALIDATION = HERE / "validation"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(data: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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


def original_objects(bddl_path: Path) -> list[str]:
    text = bddl_path.read_text(encoding="utf-8")
    start = text.index("(:objects")
    end = text.index("(:obj_of_interest", start)
    result = []
    for line in text[start:end].splitlines():
        stripped = line.strip()
        if " - " in stripped:
            result.append(stripped.split(" - ", 1)[0])
    return result


def body_position(base_env, name: str) -> np.ndarray:
    return client._object_body_position(base_env, name)


def collision_aabb_extents(base_env, name: str) -> np.ndarray:
    sim = base_env.sim
    model = sim.model
    root_body = int(base_env.obj_body_id[name])
    descendants = {root_body}
    changed = True
    while changed:
        changed = False
        for body_id in range(model.nbody):
            if (
                body_id not in descendants
                and int(model.body_parentid[body_id]) in descendants
            ):
                descendants.add(body_id)
                changed = True
    lows = []
    highs = []
    for geom_id in range(model.ngeom):
        if (
            int(model.geom_bodyid[geom_id]) not in descendants
            or int(model.geom_group[geom_id]) != 0
        ):
            continue
        geom_type = int(model.geom_type[geom_id])
        size = np.asarray(model.geom_size[geom_id], dtype=float)
        if geom_type == 2:  # sphere
            local_half = np.repeat(size[0], 3)
        elif geom_type == 3:  # capsule
            local_half = np.array([size[0], size[0], size[1] + size[0]])
        elif geom_type == 4:  # ellipsoid
            local_half = size[:3]
        elif geom_type == 5:  # cylinder
            local_half = np.array([size[0], size[0], size[1]])
        elif geom_type == 6:  # box
            local_half = size[:3]
        else:
            local_half = np.repeat(float(model.geom_rbound[geom_id]), 3)
        rotation = np.asarray(sim.data.geom_xmat[geom_id]).reshape(3, 3)
        world_half = np.abs(rotation) @ local_half
        center = np.asarray(sim.data.geom_xpos[geom_id], dtype=float)
        lows.append(center - world_half)
        highs.append(center + world_half)
    if not lows:
        raise ValueError(f"No collision geoms found for {name}")
    return np.max(highs, axis=0) - np.min(lows, axis=0)


def observation(env) -> dict:
    return getattr(env, "env", env)._get_observations()


def rgb(obs: dict, key: str) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(obs[key])[::-1])


def segmentation(obs: dict, key: str) -> np.ndarray:
    value = np.asarray(obs[key])[::-1]
    return value[:, :, 0] if value.ndim == 3 else value


def visible_pixels(env, obs: dict, object_name: str) -> dict:
    object_id = env.instance_to_id.get(object_name)
    result = {}
    for view, key in {
        "agentview": "agentview_segmentation_instance",
        "wrist": "robot0_eye_in_hand_segmentation_instance",
    }.items():
        result[view] = (
            int(np.count_nonzero(segmentation(obs, key) == object_id))
            if object_id is not None
            else 0
        )
    return result


def bbox(seg_map: np.ndarray, instance_id: int | None) -> list[int] | None:
    if instance_id is None:
        return None
    ys, xs = np.where(seg_map == instance_id)
    if len(xs) == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def bbox_overlap_fraction(seg_map: np.ndarray, first_id: int, second_id: int) -> dict:
    first = bbox(seg_map, first_id)
    second = bbox(seg_map, second_id)
    if first is None or second is None:
        return {"first_bbox": first, "second_bbox": second, "overlap_over_first": None}
    ix0, iy0 = max(first[0], second[0]), max(first[1], second[1])
    ix1, iy1 = min(first[2], second[2]), min(first[3], second[3])
    intersection = max(0, ix1 - ix0 + 1) * max(0, iy1 - iy0 + 1)
    first_area = (first[2] - first[0] + 1) * (first[3] - first[1] + 1)
    return {
        "first_bbox": first,
        "second_bbox": second,
        "intersection_px": int(intersection),
        "overlap_over_first": float(intersection) / float(first_area),
    }


def normalized_silhouette_iou(
    seg_map: np.ndarray,
    first_id: int,
    second_id: int,
    canvas_size: int = 80,
    object_size: int = 68,
) -> float | None:
    normalized = []
    for instance_id in (first_id, second_id):
        ys, xs = np.where(seg_map == instance_id)
        if len(xs) == 0:
            return None
        crop = (seg_map[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1] == instance_id)
        height, width = crop.shape
        scale = min(object_size / max(width, 1), object_size / max(height, 1))
        resized_width = max(1, int(round(width * scale)))
        resized_height = max(1, int(round(height * scale)))
        resized = Image.fromarray((crop * 255).astype(np.uint8)).resize(
            (resized_width, resized_height),
            Image.Resampling.NEAREST,
        )
        canvas = np.zeros((canvas_size, canvas_size), dtype=bool)
        x0 = (canvas_size - resized_width) // 2
        y0 = (canvas_size - resized_height) // 2
        canvas[
            y0 : y0 + resized_height,
            x0 : x0 + resized_width,
        ] = np.asarray(resized) > 0
        normalized.append(canvas)
    intersection = np.count_nonzero(normalized[0] & normalized[1])
    union = np.count_nonzero(normalized[0] | normalized[1])
    return float(intersection) / float(union) if union else None


def owner_name(model, geom_id: int) -> str:
    body_id = int(model.geom_bodyid[geom_id])
    names = []
    while body_id > 0:
        name = model.body_id2name(body_id)
        if name:
            names.append(name)
        parent = int(model.body_parentid[body_id])
        if parent == body_id:
            break
        body_id = parent
    return "/".join(names)


def contact_pairs(base_env, focus: str) -> list[dict]:
    sim = base_env.sim
    model = sim.model
    pairs = []
    for idx in range(int(sim.data.ncon)):
        contact = sim.data.contact[idx]
        first = owner_name(model, int(contact.geom1))
        second = owner_name(model, int(contact.geom2))
        if focus not in first and focus not in second:
            continue
        other = second if focus in first else first
        if any(support in other for support in ("floor", "table", "world")):
            continue
        pairs.append(
            {
                "first": first,
                "second": second,
                "distance_m": float(contact.dist),
                "geom1": model.geom_id2name(int(contact.geom1)),
                "geom2": model.geom_id2name(int(contact.geom2)),
            }
        )
    return pairs


def font(size: int):
    try:
        return ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size
        )
    except OSError:
        return ImageFont.load_default()


def annotated_preview(
    env,
    obs: dict,
    image: np.ndarray,
    view: str,
    target: str,
    container: str,
    distractor: str,
) -> np.ndarray:
    seg_key = (
        "agentview_segmentation_instance"
        if view == "agentview"
        else "robot0_eye_in_hand_segmentation_instance"
    )
    seg_map = segmentation(obs, seg_key)
    labels = [
        (target, (255, 50, 50), "TARGET"),
        (container, (40, 220, 80), "GOAL"),
        (distractor, (30, 170, 255), "DISTRACTOR"),
    ]
    canvas = Image.fromarray(image.copy())
    draw = ImageDraw.Draw(canvas)
    preview_font = font(11)
    for name, color, label in labels:
        bounds = bbox(seg_map, env.instance_to_id.get(name))
        if bounds is None:
            continue
        x0, y0, x1, y1 = bounds
        draw.rectangle((x0, y0, x1, y1), outline=color, width=2)
        text_y = max(0, y0 - 13)
        text_box = draw.textbbox((x0, text_y), label, font=preview_font)
        draw.rectangle(text_box, fill=(0, 0, 0))
        draw.text((x0, text_y), label, fill=color, font=preview_font)
    return np.asarray(canvas)


def save_sheet(images: dict[str, np.ndarray], path: Path) -> None:
    labels = list(images)
    tile_h, tile_w = next(iter(images.values())).shape[:2]
    header = 32
    canvas = Image.new("RGB", (tile_w * 3, (tile_h + header) * 2), "white")
    draw = ImageDraw.Draw(canvas)
    sheet_font = font(17)
    for idx, label in enumerate(labels):
        x = (idx % 3) * tile_w
        y = (idx // 3) * (tile_h + header)
        draw.text((x + 8, y + 5), label, fill="black", font=sheet_font)
        canvas.paste(Image.fromarray(images[label]), (x, y + header))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def save_comparison_row(images: dict[str, np.ndarray], path: Path) -> None:
    labels = list(images)
    tile_h, tile_w = next(iter(images.values())).shape[:2]
    header = 32
    canvas = Image.new("RGB", (tile_w * len(labels), tile_h + header), "white")
    draw = ImageDraw.Draw(canvas)
    row_font = font(17)
    for index, label in enumerate(labels):
        x = index * tile_w
        draw.text((x + 8, 5), label, fill="black", font=row_font)
        canvas.paste(Image.fromarray(images[label]), (x, header))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def object_pair_closeup(
    image: np.ndarray,
    seg_map: np.ndarray,
    target_id: int,
    distractor_id: int,
) -> np.ndarray:
    """Magnify the target and distractor independently for semantic review."""
    tile = 190
    header = 30
    canvas = Image.new("RGB", (tile * 2, tile + header), "white")
    draw = ImageDraw.Draw(canvas)
    pair_font = font(16)
    for index, (label, instance_id) in enumerate(
        (("TARGET", target_id), ("DISTRACTOR", distractor_id))
    ):
        bounds = bbox(seg_map, instance_id)
        draw.text((index * tile + 8, 5), label, fill="black", font=pair_font)
        if bounds is None:
            continue
        x0, y0, x1, y1 = bounds
        pad = max(3, int(0.15 * max(x1 - x0 + 1, y1 - y0 + 1)))
        x0, y0 = max(0, x0 - pad), max(0, y0 - pad)
        x1, y1 = min(image.shape[1] - 1, x1 + pad), min(image.shape[0] - 1, y1 + pad)
        crop = Image.fromarray(image[y0 : y1 + 1, x0 : x1 + 1])
        scale = min((tile - 12) / crop.width, (tile - 12) / crop.height)
        crop = crop.resize(
            (
                max(1, int(round(crop.width * scale))),
                max(1, int(round(crop.height * scale))),
            ),
            Image.Resampling.NEAREST,
        )
        px = index * tile + (tile - crop.width) // 2
        py = header + (tile - crop.height) // 2
        canvas.paste(crop, (px, py))
    return np.asarray(canvas)


def save_task_overview(
    task_images: dict[int, np.ndarray], labels: dict[int, str], path: Path
) -> None:
    ids = sorted(task_images)
    tile_h, tile_w = task_images[ids[0]].shape[:2]
    header = 29
    canvas = Image.new("RGB", (tile_w * 5, (tile_h + header) * 2), "white")
    draw = ImageDraw.Draw(canvas)
    overview_font = font(14)
    for idx, task_id in enumerate(ids):
        x = (idx % 5) * tile_w
        y = (idx // 5) * (tile_h + header)
        draw.text(
            (x + 5, y + 5),
            f"task {task_id}: {labels[task_id]}",
            fill="black",
            font=overview_font,
        )
        canvas.paste(Image.fromarray(task_images[task_id]), (x, y + header))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def prepare_clean(task_suite, task_id: int, init_state, names: list[str]) -> dict:
    bddl = task_suite.get_task_bddl_file_path(task_id)
    env = client.construct_single_env(
        {"bddl_file_name": bddl, "camera_heights": 256, "camera_widths": 256},
        use_segmentation=True,
    )
    client.init_single_env(
        env,
        init_state,
        {
            "_base_bddl_file_name_for_init_state": bddl,
            "settle_steps": 12,
        },
        None,
        "libero_object",
        task_id,
    )
    obs = observation(env)
    base = env.env
    data = {
        "positions": {name: body_position(base, name).tolist() for name in names},
        "visible": {name: visible_pixels(env, obs, name) for name in names},
        "images": {
            "agentview": rgb(obs, "agentview_image"),
            "wrist": rgb(obs, "robot0_eye_in_hand_image"),
        },
    }
    env.close()
    return data


def static_checks(condition: str, spec: dict) -> list[dict]:
    checks = []

    def add(name: str, passed: bool, evidence) -> None:
        checks.append({"name": name, "passed": bool(passed), "evidence": evidence})

    definition = spec["definition"]
    target_category = definition["target_category"]
    category = definition["category"]
    override = spec.get("object_scale_overrides", [{}])[0]
    override_xml = Path(override["xml_path"]) if override.get("xml_path") else None

    if condition == "same_category":
        add("类别相同", category == target_category, [category, target_category])
        add(
            "注册类别沿用目标类别",
            Path(definition.get("source_xml", "")).resolve()
            == source_xml(target_category).resolve(),
            definition.get("source_xml"),
        )
        add(
            "使用同类别异色实体而非原样复制",
            override_xml is not None
            and override_xml.exists()
            and definition.get("construction")
            == "same_category_different_color_variant",
            {
                "override_xml": None if override_xml is None else str(override_xml),
                "construction": definition.get("construction"),
            },
        )
        add(
            "颜色已明确改变",
            definition.get("appearance_change") == "color_changed"
            and definition.get("color_tint_rgba") is not None,
            {
                "appearance_change": definition.get("appearance_change"),
                "color_tint_rgba": definition.get("color_tint_rgba"),
            },
        )
    elif condition == "same_color":
        native_mug = ASSETS / "turbosquid_objects/red_coffee_mug/red_coffee_mug_texture.png"
        edited_mug = override_xml.parent / "red_coffee_mug_texture.png"
        xml = ET.parse(override_xml)
        meshes = {item.get("name") for item in xml.findall(".//mesh")}
        add("类别明显不同", category != target_category, [category, target_category])
        add(
            "目标真实纹理已映射且替换原杯纹理",
            edited_mug.exists() and sha256(edited_mug) != sha256(native_mug),
            {"edited_mug_texture": str(edited_mug)},
        )
        add(
            "使用真实带把手杯网格",
            "red_coffee_mug_vis" in meshes,
            sorted(name for name in meshes if name),
        )
    elif condition == "same_shape":
        edited_tree = ET.parse(override_xml)
        add(
            "类别明显不同",
            category != target_category,
            [category, target_category],
        )
        add(
            "使用独立类别身份的轮廓代理",
            definition.get("construction")
            == "different_category_outline_matched_proxy"
            and definition.get("appearance_change") == "native_proxy_identity"
            and bool(definition.get("proxy_identity")),
            {
                "construction": definition.get("construction"),
                "appearance_change": definition.get("appearance_change"),
                "proxy_identity": definition.get("proxy_identity"),
            },
        )
        visual_geoms = [
            geom
            for geom in edited_tree.findall(".//geom")
            if geom.get("group") == "1"
        ]
        collision_geoms = [
            geom
            for geom in edited_tree.findall(".//geom")
            if geom.get("group") == "0"
        ]
        add(
            "代理具有独立视觉结构和真实碰撞",
            bool(visual_geoms) and bool(collision_geoms),
            {
                "visual_geom_count": len(visual_geoms),
                "collision_geom_count": len(collision_geoms),
            },
        )
    elif condition == "same_affordance":
        xml = ET.parse(override_xml)
        site = xml.find(".//site[@name='contain_region']")
        size = [] if site is None else [float(value) for value in site.get("size").split()]
        add("容器类别明显不同", category != target_category, [category, target_category])
        add("具有真实容纳区域", site is not None and max(size, default=0.0) > 0.055, size)
    elif condition == "same_position":
        add("占位物与任务无关", category == "black_book", category)
    elif condition == "trajectory":
        calibration = spec["trajectory_calibration"]
        block_xml = (
            override_xml
            if override_xml is not None
            else Path(spec["trajectory_xml_path"])
        )
        xml = ET.parse(block_xml)
        collision = xml.find(".//geom[@name='block_collision']")
        half_sizes = (
            []
            if collision is None
            else [float(value) for value in collision.get("size").split()]
        )
        add(
            "实体方块具有真实碰撞",
            collision is not None
            and collision.get("type") == "box"
            and np.allclose(half_sizes, [calibration["block_half_size_m"]] * 3),
            half_sizes,
        )
        add(
            "方块按目标顶部尺度标定",
            0.034 <= calibration["block_side_length_m"] <= 0.062,
            {
                "side_m": calibration["block_side_length_m"],
                "target_extents_m": spec["target_geometry"]["collision_aabb_extents_m"],
            },
        )
        add(
            "方块质量与摩擦按目标标定",
            calibration["matched_friction"] == [0.95, 0.3, 0.1]
            and calibration["target_mass_kg"] > 0,
            {
                "target_mass_kg": calibration["target_mass_kg"],
                "block_density_kg_m3": calibration["block_density_kg_m3"],
                "friction": calibration["matched_friction"],
            },
        )
    return checks


def render_case(
    task_suite,
    task: dict,
    init_state,
    condition: str,
    clean: dict,
    names: list[str],
) -> tuple[dict, dict[str, np.ndarray], dict[str, np.ndarray]]:
    task_id = task["task_id"]
    spec = load_json(Path(task["cases"][condition]))
    spec["_base_bddl_file_name_for_init_state"] = spec["base_bddl_file_name"]
    xml_map = client._libero_object_xml_override_map(spec)
    env = client.construct_single_env(
        {
            "bddl_file_name": spec["bddl_file_name_override"],
            "camera_heights": 256,
            "camera_widths": 256,
        },
        use_segmentation=True,
        object_xml_override_map=xml_map or None,
    )
    if env is None:
        raise RuntimeError(f"could not construct task {task_id} {condition}")
    try:
        client.init_single_env(env, init_state, spec, None, "libero_object", task_id)
        obs = observation(env)
        base = env.env
        target = task["target"]
        container = "basket_1"
        distractor = spec["definition"]["distractor"]
        before = body_position(base, distractor).copy()
        target_pos = body_position(base, target).copy()
        container_pos = body_position(base, container).copy()
        target_extents = collision_aabb_extents(base, target)
        distractor_extents = collision_aabb_extents(base, distractor)
        extent_ratio = distractor_extents / np.maximum(target_extents, 1e-9)
        shape_log_error = float(np.max(np.abs(np.log(extent_ratio))))
        contacts = contact_pairs(base, distractor)
        images = {
            "agentview": rgb(obs, "agentview_image"),
            "wrist": rgb(obs, "robot0_eye_in_hand_image"),
        }
        annotated = {
            view: annotated_preview(
                env, obs, image, view, target, container, distractor
            )
            for view, image in images.items()
        }
        silhouette_iou = normalized_silhouette_iou(
            segmentation(obs, "agentview_segmentation_instance"),
            env.instance_to_id[target],
            env.instance_to_id[distractor],
        )
        wrist_silhouette_iou = normalized_silhouette_iou(
            segmentation(obs, "robot0_eye_in_hand_segmentation_instance"),
            env.instance_to_id[target],
            env.instance_to_id[distractor],
        )
        agent_segmentation = segmentation(
            obs, "agentview_segmentation_instance"
        )

        def instance_mean_rgb(name: str) -> list[float] | None:
            mask = agent_segmentation == env.instance_to_id[name]
            if not np.any(mask):
                return None
            return np.mean(images["agentview"][mask], axis=0).tolist()

        target_mean_rgb = instance_mean_rgb(target)
        distractor_mean_rgb = instance_mean_rgb(distractor)
        color_distance = (
            None
            if target_mean_rgb is None or distractor_mean_rgb is None
            else float(
                np.linalg.norm(
                    np.asarray(target_mean_rgb)
                    - np.asarray(distractor_mean_rgb)
                )
            )
        )
        task_output = PREVIEWS / f"task_{task_id:02d}"
        for view in images:
            view_dir = task_output / view
            view_dir.mkdir(parents=True, exist_ok=True)
            Image.fromarray(images[view]).save(view_dir / f"{condition}.png")
            Image.fromarray(annotated[view]).save(
                view_dir / f"{condition}_annotated.png"
            )
            seg_key = (
                "agentview_segmentation_instance"
                if view == "agentview"
                else "robot0_eye_in_hand_segmentation_instance"
            )
            closeup = object_pair_closeup(
                images[view],
                segmentation(obs, seg_key),
                env.instance_to_id[target],
                env.instance_to_id[distractor],
            )
            Image.fromarray(closeup).save(
                view_dir / f"{condition}_object_closeup.png"
            )

        for _ in range(30):
            env.step([0.0] * 7)
        after = body_position(base, distractor).copy()
        after_contacts = contact_pairs(base, distractor)
        visible = {name: visible_pixels(env, obs, name) for name in names}
        relevant = (
            target
            if condition
            in {"same_category", "same_color", "same_shape", "same_position", "trajectory"}
            else container
        )
        checks = static_checks(condition, spec)

        def add(name: str, passed: bool, evidence) -> None:
            checks.append({"name": name, "passed": bool(passed), "evidence": evidence})

        if condition in {"same_category", "same_color", "same_shape"}:
            distance = float(np.linalg.norm(before[:2] - target_pos[:2]))
            add("紧挨目标物", 0.055 <= distance <= 0.125, {"center_xy_distance_m": distance})
            target_contacts = [
                pair
                for pair in contacts
                if target in pair["first"] or target in pair["second"]
            ]
            add("与目标物无碰撞/重叠", not target_contacts, target_contacts)
            if condition == "same_category":
                add(
                    "同类别异色实例尺度合理",
                    bool(np.all((extent_ratio >= 0.35) & (extent_ratio <= 2.50))),
                    {
                        "target_extents_m": target_extents.tolist(),
                        "distractor_extents_m": distractor_extents.tolist(),
                        "extent_ratio": extent_ratio.tolist(),
                        "max_log_extent_error": shape_log_error,
                    },
                )
                add(
                    "同类别实例可见颜色不同",
                    color_distance is not None and color_distance >= 22.0,
                    {
                        "target_mean_rgb": target_mean_rgb,
                        "distractor_mean_rgb": distractor_mean_rgb,
                        "rgb_distance": color_distance,
                        "minimum_rgb_distance": 22.0,
                    },
                )
            elif condition == "same_color":
                add(
                    "异类别实例可见主色接近",
                    color_distance is not None and color_distance <= 78.0,
                    {
                        "target_mean_rgb": target_mean_rgb,
                        "distractor_mean_rgb": distractor_mean_rgb,
                        "rgb_distance": color_distance,
                        "maximum_rgb_distance": 78.0,
                    },
                )
            elif condition == "same_shape":
                add(
                    "三维碰撞轮廓逐轴严格匹配目标",
                    bool(np.all((extent_ratio >= 0.95) & (extent_ratio <= 1.05))),
                    {
                        "target_extents_m": target_extents.tolist(),
                        "distractor_extents_m": distractor_extents.tolist(),
                        "extent_ratio": extent_ratio.tolist(),
                        "max_log_extent_error": shape_log_error,
                    },
                )
        elif condition == "same_affordance":
            distance = float(np.linalg.norm(before[:2] - container_pos[:2]))
            add("干扰容器位于目标容器附近", distance <= 0.23, {"center_xy_distance_m": distance})
            goal_contacts = [
                pair
                for pair in contacts
                if container in pair["first"] or container in pair["second"]
            ]
            add("与目标容器无碰撞/重叠", not goal_contacts, goal_contacts)
        elif condition == "same_position":
            original = np.asarray(clean["positions"][target])
            replacement_error = float(np.linalg.norm(before[:2] - original[:2]))
            target_move = float(np.linalg.norm(target_pos[:2] - original[:2]))
            add("干扰物占据目标原位置", replacement_error <= 0.008, {
                "xy_error_m": replacement_error,
                "original_target_xy": original[:2].tolist(),
                "replacement_xy": before[:2].tolist(),
            })
            add("原目标已移到旁边", 0.080 <= target_move <= 0.155, {
                "target_move_xy_m": target_move,
                "moved_target_xy": target_pos[:2].tolist(),
            })
        else:
            calibration = spec["trajectory_calibration"]
            xy_error = float(np.linalg.norm(before[:2] - target_pos[:2]))
            z_offset = float(before[2] - target_pos[2])
            target_contacts = [
                pair
                for pair in contacts
                if target in pair["first"] or target in pair["second"]
            ]
            unexpected = [
                pair
                for pair in contacts
                if target not in pair["first"] and target not in pair["second"]
            ]
            add("方块位于目标顶部中心", xy_error <= 0.015, {
                "xy_error_m": xy_error,
                "block_xyz": before.tolist(),
                "target_xyz": target_pos.tolist(),
            })
            expected = calibration["settled_block_center_offset_range_m"]
            add("方块位于目标物上方", expected[0] <= z_offset <= expected[1], {
                "center_z_offset_m": z_offset,
                "expected_range_m": expected,
            })
            target_mass = float(base.sim.model.body_mass[base.obj_body_id[target]])
            block_mass = float(base.sim.model.body_mass[base.obj_body_id[distractor]])
            ratio = block_mass / target_mass
            add("方块质量与目标物匹配", 0.85 <= ratio <= 1.15, {
                "block_mass_kg": block_mass,
                "target_mass_kg": target_mass,
                "mass_ratio": ratio,
            })
            add("方块由目标物真实支撑", bool(target_contacts), target_contacts)
            add("方块未接触其他物体或机器人", not unexpected, unexpected)

        drift = float(np.linalg.norm(after - before))
        add("静置30步位置稳定", drift <= 0.018, {
            "drift_m": drift,
            "before_xyz": before.tolist(),
            "after_xyz": after.tolist(),
        })
        unexpected_after = after_contacts
        if condition == "trajectory":
            unexpected_after = [
                pair
                for pair in after_contacts
                if target not in pair["first"] and target not in pair["second"]
            ]
        add("无非预期接触", not unexpected_after, {
            "unexpected": unexpected_after,
            "all_contacts": after_contacts,
        })

        overlap = bbox_overlap_fraction(
            segmentation(obs, "agentview_segmentation_instance"),
            env.instance_to_id[relevant],
            env.instance_to_id[distractor],
        )
        if condition == "trajectory":
            add(
                "方块位于抓取接近路径",
                bool(target_contacts) and xy_error <= 0.015,
                {
                    "physical_target_contact": bool(target_contacts),
                    "xy_center_error_m": xy_error,
                    "agentview_projection": overlap,
                },
            )
        else:
            add(
                "agentview中不遮挡关键原物体",
                overlap["overlap_over_first"] is not None
                and overlap["overlap_over_first"] <= 0.08,
                overlap,
            )

        visibility_ratios = {}
        for name in names:
            if condition in {"same_position", "trajectory"} and name == target:
                continue
            denominator = clean["visible"][name]["agentview"]
            visibility_ratios[name] = (
                float(visible[name]["agentview"]) / float(denominator)
                if denominator
                else None
            )
        min_visibility = min(
            (value for value in visibility_ratios.values() if value is not None),
            default=1.0,
        )
        add(
            "其他原场景物体保留至少80%可见像素",
            min_visibility >= 0.80,
            {
                "minimum_visibility_ratio": min_visibility,
                "per_object_ratio": visibility_ratios,
            },
        )

        result = {
            "task_id": task_id,
            "task_name": task["task_name"],
            "instruction": task["instruction"],
            "condition": condition,
            "definition": spec["definition"],
            "passed": all(check["passed"] for check in checks),
            "checks": checks,
            "runtime": {
                "distractor_initial_xyz": before.tolist(),
                "distractor_after_settle_xyz": after.tolist(),
                "target_xyz": target_pos.tolist(),
                "container_xyz": container_pos.tolist(),
                "initial_non_support_contacts": contacts,
                "post_settle_non_support_contacts": after_contacts,
                "visibility": visible,
                "shape_metrics": {
                    "target_collision_aabb_extents_m": target_extents.tolist(),
                    "distractor_collision_aabb_extents_m": distractor_extents.tolist(),
                    "extent_ratio": extent_ratio.tolist(),
                    "max_log_extent_error": shape_log_error,
                    "normalized_agentview_silhouette_iou": silhouette_iou,
                    "normalized_wrist_silhouette_iou": wrist_silhouette_iou,
                },
            },
            "preview_files": {
                view: str((task_output / view / f"{condition}.png").resolve())
                for view in ("agentview", "wrist")
            },
            "annotated_preview_files": {
                view: str(
                    (task_output / view / f"{condition}_annotated.png").resolve()
                )
                for view in ("agentview", "wrist")
            },
            "object_closeup_files": {
                view: str(
                    (
                        task_output
                        / view
                        / f"{condition}_object_closeup.png"
                    ).resolve()
                )
                for view in ("agentview", "wrist")
            },
        }
        return result, images, annotated
    finally:
        env.close()


def write_markdown(report: dict) -> None:
    lines = [
        "# D1 LIBERO-Object full-suite validation",
        "",
        "- Suite: `libero_object`",
        "- Tasks: 10",
        "- Conditions per task: 6",
        "- Preview/validation state: official init state 0",
        "",
        f"- Automatic checks: **{'PASS' if report['all_passed'] else 'FAIL'}**",
        "",
    ]
    for task in report["tasks"]:
        lines.extend(
            [
                f"## Task {task['task_id']}: {task['instruction']}",
                "",
            ]
        )
        for result in task["results"]:
            mark = "PASS" if result["passed"] else "FAIL"
            lines.append(
                f"- {result['definition']['zh']} (`{result['condition']}`): **{mark}**"
            )
            for check in result["checks"]:
                if not check["passed"]:
                    lines.append(
                        f"  - ✗ {check['name']}: `{json.dumps(check['evidence'], ensure_ascii=False)}`"
                    )
        contrast = task.get("contrast_check")
        if contrast:
            mark = "PASS" if contrast["passed"] else "FAIL"
            category_value = contrast["same_category_max_log_extent_error"]
            shape_value = contrast["same_shape_max_log_extent_error"]
            category_iou = contrast["same_category_normalized_silhouette_iou"]
            shape_iou = contrast["same_shape_normalized_silhouette_iou"]
            category_wrist_iou = contrast["same_category_normalized_wrist_silhouette_iou"]
            shape_wrist_iou = contrast["same_shape_normalized_wrist_silhouette_iou"]
            category_text = "n/a" if category_value is None else f"{category_value:.4f}"
            shape_text = "n/a" if shape_value is None else f"{shape_value:.4f}"
            category_iou_text = "n/a" if category_iou is None else f"{category_iou:.4f}"
            shape_iou_text = "n/a" if shape_iou is None else f"{shape_iou:.4f}"
            category_wrist_text = (
                "n/a" if category_wrist_iou is None else f"{category_wrist_iou:.4f}"
            )
            shape_wrist_text = (
                "n/a" if shape_wrist_iou is None else f"{shape_wrist_iou:.4f}"
            )
            lines.append(
                f"- 两类条件的精确几何一致性: **{mark}** "
                f"(category error={category_text}, shape error={shape_text}; "
                f"agent category/shape IoU={category_iou_text}/{shape_iou_text}; "
                f"wrist category/shape IoU={category_wrist_text}/{shape_wrist_text})"
            )
        lines.append("")
    (VALIDATION / "validation_report.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tasks",
        nargs="*",
        type=int,
        help="Optional task ids; default renders all ten.",
    )
    parser.add_argument(
        "--conditions",
        nargs="*",
        choices=CONDITIONS,
        help="Optional conditions; use both same_category and same_shape for contrast validation.",
    )
    args = parser.parse_args()
    selected = set(args.tasks) if args.tasks else None
    selected_conditions = args.conditions or CONDITIONS
    manifest = load_json(HERE / "manifest.json")
    task_suite = benchmark.get_benchmark_dict()["libero_object"]()
    task_reports = []
    failures = []
    overview_images = {
        condition: {"agentview": {}, "wrist": {}}
        for condition in CONDITIONS
    }
    labels = {}

    for task in manifest["tasks"]:
        task_id = task["task_id"]
        if selected is not None and task_id not in selected:
            continue
        labels[task_id] = task["target_category"]
        init_state = task_suite.get_task_init_states(task_id)[0]
        names = original_objects(Path(task_suite.get_task_bddl_file_path(task_id)))
        print(f"[D1-object] clean task {task_id:02d}", flush=True)
        clean = prepare_clean(task_suite, task_id, init_state, names)
        task_output = PREVIEWS / f"task_{task_id:02d}"
        task_output.mkdir(parents=True, exist_ok=True)
        Image.fromarray(clean["images"]["agentview"]).save(task_output / "clean_agentview.png")
        Image.fromarray(clean["images"]["wrist"]).save(task_output / "clean_wrist.png")
        task_results = []
        sheets = {
            "agentview": {},
            "wrist": {},
            "agentview_annotated": {},
            "wrist_annotated": {},
        }
        for condition in selected_conditions:
            print(f"[D1-object] task {task_id:02d} {condition}", flush=True)
            try:
                result, images, annotated = render_case(
                    task_suite, task, init_state, condition, clean, names
                )
            except Exception as error:
                traceback.print_exc()
                result = {
                    "task_id": task_id,
                    "task_name": task["task_name"],
                    "instruction": task["instruction"],
                    "condition": condition,
                    "definition": {
                        "zh": condition,
                        "definition": "case construction failed before validation",
                    },
                    "passed": False,
                    "error": f"{type(error).__name__}: {error}",
                    "checks": [],
                }
                failures.append({"task_id": task_id, "condition": condition, "error": str(error)})
                task_results.append(result)
                continue
            task_results.append(result)
            if not result["passed"]:
                failures.append({"task_id": task_id, "condition": condition, "error": "checks_failed"})
            for view in ("agentview", "wrist"):
                sheets[view][condition] = images[view]
                sheets[f"{view}_annotated"][condition] = annotated[view]
                overview_images[condition][view][task_id] = annotated[view]
            print(
                f"[D1-object] task {task_id:02d} {condition}: "
                f"{'PASS' if result['passed'] else 'FAIL'}",
                flush=True,
            )
        for key, values in sheets.items():
            if len(values) == len(selected_conditions):
                save_sheet(values, task_output / f"six_conditions_{key}.png")
        if {
            "same_category",
            "same_shape",
        }.issubset(sheets["agentview_annotated"]):
            save_comparison_row(
                {
                    "clean target scene": clean["images"]["agentview"],
                    "same_category": sheets["agentview_annotated"]["same_category"],
                    "same_shape": sheets["agentview_annotated"]["same_shape"],
                },
                task_output / "category_vs_shape_agentview.png",
            )
            save_comparison_row(
                {
                    "clean target scene": clean["images"]["wrist"],
                    "same_category": sheets["wrist_annotated"]["same_category"],
                    "same_shape": sheets["wrist_annotated"]["same_shape"],
                },
                task_output / "category_vs_shape_wrist.png",
            )
            for view in ("agentview", "wrist"):
                closeups = {}
                for condition in ("same_category", "same_shape"):
                    closeup_path = (
                        task_output
                        / view
                        / f"{condition}_object_closeup.png"
                    )
                    if closeup_path.exists():
                        closeups[condition] = np.asarray(
                            Image.open(closeup_path).convert("RGB")
                        )
                if len(closeups) == 2:
                    save_comparison_row(
                        closeups,
                        task_output / f"category_vs_shape_{view}_closeups.png",
                    )
        by_condition = {
            result["condition"]: result
            for result in task_results
            if "runtime" in result
        }
        category_error = (
            by_condition.get("same_category", {})
            .get("runtime", {})
            .get("shape_metrics", {})
            .get("max_log_extent_error")
        )
        shape_error = (
            by_condition.get("same_shape", {})
            .get("runtime", {})
            .get("shape_metrics", {})
            .get("max_log_extent_error")
        )
        category_iou = (
            by_condition.get("same_category", {})
            .get("runtime", {})
            .get("shape_metrics", {})
            .get("normalized_agentview_silhouette_iou")
        )
        shape_iou = (
            by_condition.get("same_shape", {})
            .get("runtime", {})
            .get("shape_metrics", {})
            .get("normalized_agentview_silhouette_iou")
        )
        category_wrist_iou = (
            by_condition.get("same_category", {})
            .get("runtime", {})
            .get("shape_metrics", {})
            .get("normalized_wrist_silhouette_iou")
        )
        shape_wrist_iou = (
            by_condition.get("same_shape", {})
            .get("runtime", {})
            .get("shape_metrics", {})
            .get("normalized_wrist_silhouette_iou")
        )
        standards_passed = (
            by_condition.get("same_category", {}).get("passed", False)
            and by_condition.get("same_shape", {}).get("passed", False)
            and shape_error is not None
            and shape_error <= float(np.log(1.05))
        )
        contrast_check = {
            "name": "同类别异色与异类别同轮廓标准均满足",
            "passed": bool(standards_passed),
            "standards_passed": bool(standards_passed),
            "same_category_max_log_extent_error": category_error,
            "same_shape_max_log_extent_error": shape_error,
            "same_shape_maximum_allowed_log_extent_error": float(np.log(1.05)),
            "same_category_normalized_silhouette_iou": category_iou,
            "same_shape_normalized_silhouette_iou": shape_iou,
            "same_category_normalized_wrist_silhouette_iou": category_wrist_iou,
            "same_shape_normalized_wrist_silhouette_iou": shape_wrist_iou,
            "same_category_shape_role": "same category may use a natural alternate package shape",
            "silhouette_iou_role": "supporting evidence only; positions differ",
        }
        task_report = {
            "task_id": task_id,
            "task_name": task["task_name"],
            "instruction": task["instruction"],
            "target": task["target"],
            "target_category": task["target_category"],
            "init_state_index": 0,
            "clean": {key: value for key, value in clean.items() if key != "images"},
            "contrast_check": contrast_check,
            "all_passed": all(result["passed"] for result in task_results)
            and contrast_check["passed"],
            "results": task_results,
        }
        write_json(task_report, VALIDATION / f"task_{task_id:02d}.json")
        task_reports.append(task_report)

    if selected is None and selected_conditions == CONDITIONS:
        for condition in CONDITIONS:
            for view in ("agentview", "wrist"):
                values = overview_images[condition][view]
                if len(values) == 10:
                    save_task_overview(
                        values,
                        labels,
                        PREVIEWS / "overview" / f"{condition}_{view}_annotated.png",
                    )
    report = {
        "suite": "libero_object",
        "task_count": len(task_reports),
        "case_count": sum(len(task["results"]) for task in task_reports),
        "all_passed": all(task["all_passed"] for task in task_reports),
        "failure_count": len(failures),
        "failures": failures,
        "tasks": task_reports,
    }
    write_json(report, VALIDATION / "validation_report.json")
    write_markdown(report)
    print(
        json.dumps(
            {
                "tasks": report["task_count"],
                "cases": report["case_count"],
                "all_passed": report["all_passed"],
                "failures": failures,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
