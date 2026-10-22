#!/usr/bin/env python3
"""Build complete-surface, visual-only A2 XML variants.

The original visual geoms remain completely unchanged.  A massless, non-colliding
black triangle overlay is cut as a contiguous, camera-aligned slice of the
target's full three-dimensional render surface. The cutoff is solved against the
complete surface area, including initially hidden, inner, rear, and later
revealed regions. Surface enumeration comes from MuJoCo geometry, never RGB
brightness, so dark target receptacles are included normally.
This preserves the untouched material exactly while collision geoms, bodies,
joints, sites and task predicates remain unchanged.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np


SIM = Path("/home/huiqiongli/Sim")
HERE = Path(__file__).resolve().parent
LIBERO = SIM / "Repos/LIBERO/libero/libero"
sys.path[:0] = [
    str(SIM),
    str(SIM / "Repos/LIBERO"),
    str(SIM / "Repos/lingbot-va/evaluation/libero"),
]

from libero.libero import benchmark  # noqa: E402
from libero.libero.envs import SegmentationRenderEnv  # noqa: E402
import robust_libero_perturbations as perturb  # noqa: E402


SUITES = ("libero_object", "libero_spatial", "libero_goal", "libero_10")
COVERAGES = (0.25, 0.50, 0.75)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def ring_mesh(rings: list[tuple[float, float]], segments: int = 48) -> np.ndarray:
    vertices = []
    for radius, z_value in rings:
        vertices.append(
            np.asarray(
                [
                    [radius * math.cos(2.0 * math.pi * i / segments),
                     radius * math.sin(2.0 * math.pi * i / segments), z_value]
                    for i in range(segments)
                ],
                dtype=float,
            )
        )
    triangles = []
    for ring_index in range(len(vertices) - 1):
        lower = vertices[ring_index]
        upper = vertices[ring_index + 1]
        for i in range(segments):
            j = (i + 1) % segments
            if np.linalg.norm(lower[i] - lower[j]) > 1e-12:
                triangles.append([lower[i], lower[j], upper[j]])
            if np.linalg.norm(upper[i] - upper[j]) > 1e-12:
                triangles.append([lower[i], upper[j], upper[i]])
    return np.asarray(triangles, dtype=float)


def local_primitive_triangles(geom_type: int, size: np.ndarray) -> np.ndarray:
    a, b, c = [float(value) for value in size]
    if geom_type == 6:  # box
        vertices = np.asarray(
            [
                [x * a, y * b, z * c]
                for x, y, z in (
                    (-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1),
                    (-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1),
                )
            ],
            dtype=float,
        )
        faces = np.asarray(
            [
                (0, 2, 1), (0, 3, 2), (4, 5, 6), (4, 6, 7),
                (0, 1, 5), (0, 5, 4), (1, 2, 6), (1, 6, 5),
                (2, 3, 7), (2, 7, 6), (3, 0, 4), (3, 4, 7),
            ],
            dtype=np.int64,
        )
        return vertices[faces]
    if geom_type == 5:  # cylinder
        return ring_mesh([(0.0, -b), (a, -b), (a, b), (0.0, b)])
    if geom_type == 3:  # capsule
        hemi = 12
        rings = [(0.0, -b - a)]
        rings.extend(
            (a * math.cos(theta), -b + a * math.sin(theta))
            for theta in np.linspace(-math.pi / 2.0, 0.0, hemi + 1)[1:]
        )
        rings.append((a, b))
        rings.extend(
            (a * math.cos(theta), b + a * math.sin(theta))
            for theta in np.linspace(0.0, math.pi / 2.0, hemi + 1)[1:]
        )
        return ring_mesh(rings)
    if geom_type in {2, 4}:  # sphere / ellipsoid
        radii = np.asarray([a, a, a] if geom_type == 2 else [a, b, c])
        unit = ring_mesh(
            [(0.0, -1.0)]
            + [
                (math.cos(theta), math.sin(theta))
                for theta in np.linspace(-math.pi / 2.0, math.pi / 2.0, 25)[1:-1]
            ]
            + [(0.0, 1.0)]
        )
        return unit * radii[None, None, :]
    raise ValueError(f"Unsupported visual primitive geom type {geom_type}")


def geom_triangles_world(base: object, geom_id: int) -> np.ndarray:
    sim = base.sim
    model = sim.model
    geom_type = int(model.geom_type[geom_id])
    if geom_type == 7:
        mesh_id = int(model.geom_dataid[geom_id])
        va = int(model.mesh_vertadr[mesh_id])
        vn = int(model.mesh_vertnum[mesh_id])
        fa = int(model.mesh_faceadr[mesh_id])
        fn = int(model.mesh_facenum[mesh_id])
        vertices = np.asarray(model.mesh_vert[va : va + vn], dtype=float)
        faces = np.asarray(model.mesh_face[fa : fa + fn], dtype=np.int64)
        local = vertices[faces]
    else:
        local = local_primitive_triangles(
            geom_type, np.asarray(model.geom_size[geom_id], dtype=float)
        )
    rotation = np.asarray(sim.data.geom_xmat[geom_id], dtype=float).reshape(3, 3)
    translation = np.asarray(sim.data.geom_xpos[geom_id], dtype=float)
    return local @ rotation.T + translation


def world_to_body(base: object, body_id: int, triangles: np.ndarray) -> np.ndarray:
    sim = base.sim
    rotation = np.asarray(sim.data.body_xmat[body_id], dtype=float).reshape(3, 3)
    translation = np.asarray(sim.data.body_xpos[body_id], dtype=float)
    return (triangles - translation[None, None, :]) @ rotation


def triangle_area(triangles: np.ndarray) -> float:
    if not len(triangles):
        return 0.0
    return float(
        (
            0.5
            * np.linalg.norm(
                np.cross(
                    triangles[:, 1] - triangles[:, 0],
                    triangles[:, 2] - triangles[:, 0],
                ),
                axis=1,
            )
        ).sum()
    )


def clip_triangle_to_halfspace(triangle: np.ndarray, signed: np.ndarray) -> np.ndarray:
    """Clip one triangle to signed <= 0 and return zero or more triangles."""
    polygon: list[np.ndarray] = []
    previous = np.asarray(triangle[-1], dtype=float)
    previous_signed = float(signed[-1])
    previous_inside = previous_signed <= 0.0
    for current, current_signed_raw in zip(triangle, signed):
        current = np.asarray(current, dtype=float)
        current_signed = float(current_signed_raw)
        current_inside = current_signed <= 0.0
        if current_inside != previous_inside:
            denominator = previous_signed - current_signed
            fraction = previous_signed / denominator if abs(denominator) > 1e-15 else 0.5
            polygon.append(previous + fraction * (current - previous))
        if current_inside:
            polygon.append(current)
        previous = current
        previous_signed = current_signed
        previous_inside = current_inside
    if len(polygon) < 3:
        return np.empty((0, 3, 3), dtype=float)
    anchor = polygon[0]
    return np.asarray(
        [[anchor, polygon[index], polygon[index + 1]] for index in range(1, len(polygon) - 1)],
        dtype=float,
    )


def best_contiguous_cut(mask: np.ndarray, coverage: float) -> tuple[str, float, int, float]:
    """Choose the longer image axis and the closest contiguous pixel prefix."""
    rows, columns = np.nonzero(mask)
    if not len(rows):
        raise ValueError("Target has no visible agentview segmentation pixels")
    row_span = int(rows.max() - rows.min() + 1)
    column_span = int(columns.max() - columns.min() + 1)
    axis = "screen_x" if column_span >= row_span else "screen_y"
    coordinates = columns if axis == "screen_x" else rows
    unique = np.unique(coordinates)
    desired = float(coverage) * float(len(coordinates))
    candidates: list[tuple[float, int]] = []
    for coordinate in unique:
        count = int(np.count_nonzero(coordinates <= coordinate))
        candidates.append((float(coordinate) + 0.5, count))
    cutoff, count = min(candidates, key=lambda item: abs(float(item[1]) - desired))
    return axis, cutoff, count, float(count) / float(len(coordinates))


def split_agentview_visible_triangles(
    base: object,
    world_groups: list[np.ndarray],
    visual_geom_ids: list[int],
    coverage: float,
) -> tuple[list[np.ndarray], dict[str, object]]:
    """Clip physical faces by a ray plane chosen from visible agentview pixels."""
    height = width = 256
    segmentation = base.sim.render(
        camera_name="agentview", height=height, width=width, segmentation=True
    )[::-1]
    target_mask = np.isin(segmentation[:, :, 1], np.asarray(visual_geom_ids, dtype=int))
    axis, cutoff, selected_pixels, expected_coverage = best_contiguous_cut(
        target_mask, coverage
    )

    camera_id = int(base.sim.model.camera_name2id("agentview"))
    fovy = float(base.sim.model.cam_fovy[camera_id])
    focal_length = 0.5 * float(height) / math.tan(fovy * math.pi / 360.0)
    intrinsic = np.asarray(
        [
            [focal_length, 0.0, width / 2.0],
            [0.0, focal_length, height / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    camera_to_world = np.eye(4, dtype=float)
    camera_to_world[:3, :3] = (
        np.asarray(base.sim.data.cam_xmat[camera_id], dtype=float).reshape(3, 3)
        @ np.diag([1.0, -1.0, -1.0])
    )
    camera_to_world[:3, 3] = np.asarray(
        base.sim.data.cam_xpos[camera_id], dtype=float
    )
    world_to_camera = np.linalg.inv(camera_to_world)
    focal = float(intrinsic[0, 0] if axis == "screen_x" else intrinsic[1, 1])
    center = float(intrinsic[0, 2] if axis == "screen_x" else intrinsic[1, 2])
    camera_coordinate = 0 if axis == "screen_x" else 1

    black_groups: list[np.ndarray] = []
    total_surface_area = 0.0
    black_surface_area = 0.0
    source_face_count = 0
    clipped_face_count = 0
    for group in world_groups:
        total_surface_area += triangle_area(group)
        source_face_count += int(len(group))
        homogeneous = np.concatenate(
            [group, np.ones(group.shape[:-1] + (1,), dtype=float)], axis=-1
        )
        camera = homogeneous @ world_to_camera.T
        signed = focal * camera[:, :, camera_coordinate] + (center - cutoff) * camera[:, :, 2]
        clipped: list[np.ndarray] = []
        for triangle, triangle_signed in zip(group, signed):
            pieces = clip_triangle_to_halfspace(triangle, triangle_signed)
            if len(pieces):
                clipped.extend(pieces)
        clipped_array = np.asarray(clipped, dtype=float).reshape(-1, 3, 3)
        black_groups.append(clipped_array)
        black_surface_area += triangle_area(clipped_array)
        clipped_face_count += int(len(clipped_array))

    return black_groups, {
        "coverage_definition": "agentview_visible_target_pixels",
        "requested_visible_coverage": float(coverage),
        "baseline_agentview_target_pixels": int(target_mask.sum()),
        "predicted_agentview_black_pixels": int(selected_pixels),
        "predicted_agentview_visible_coverage": float(expected_coverage),
        "selection_axis": axis,
        "screen_cutoff_px": float(cutoff),
        "selection_side": "low_coordinate_prefix",
        "visibility_source": "mujoco_geom_segmentation",
        "brightness_independent_visibility": True,
        "surface_area_total": float(total_surface_area),
        "surface_area_black": float(black_surface_area),
        "actual_surface_coverage": float(black_surface_area / total_surface_area),
        "selected_face_count": int(clipped_face_count),
        "total_face_count": int(source_face_count),
    }


def split_complete_surface_camera_slice(
    base: object,
    world_groups: list[np.ndarray],
    visual_geom_ids: list[int],
    coverage: float,
) -> tuple[list[np.ndarray], dict[str, object]]:
    """Cut a camera-aligned slice whose denominator is the complete 3-D area."""
    height = width = 256
    segmentation = base.sim.render(
        camera_name="agentview", height=height, width=width, segmentation=True
    )[::-1]
    target_mask = np.isin(segmentation[:, :, 1], np.asarray(visual_geom_ids, dtype=int))
    rows, columns = np.nonzero(target_mask)
    if not len(rows):
        raise ValueError("Target has no visible agentview segmentation pixels")
    row_span = int(rows.max() - rows.min() + 1)
    column_span = int(columns.max() - columns.min() + 1)
    axis = "screen_x" if column_span >= row_span else "screen_y"

    camera_id = int(base.sim.model.camera_name2id("agentview"))
    fovy = float(base.sim.model.cam_fovy[camera_id])
    focal = 0.5 * float(height) / math.tan(fovy * math.pi / 360.0)
    center = float(width / 2.0 if axis == "screen_x" else height / 2.0)
    camera_coordinate = 0 if axis == "screen_x" else 1
    camera_to_world = np.eye(4, dtype=float)
    camera_to_world[:3, :3] = (
        np.asarray(base.sim.data.cam_xmat[camera_id], dtype=float).reshape(3, 3)
        @ np.diag([1.0, -1.0, -1.0])
    )
    camera_to_world[:3, 3] = np.asarray(
        base.sim.data.cam_xpos[camera_id], dtype=float
    )
    world_to_camera = np.linalg.inv(camera_to_world)
    camera_groups = []
    projected_groups = []
    areas_by_group = []
    for group in world_groups:
        homogeneous = np.concatenate(
            [group, np.ones(group.shape[:-1] + (1,), dtype=float)], axis=-1
        )
        camera = homogeneous @ world_to_camera.T
        if np.any(camera[:, :, 2] <= 0.0):
            raise ValueError("Target surface contains vertices behind agentview")
        camera_groups.append(camera)
        projected_groups.append(
            focal * camera[:, :, camera_coordinate] / camera[:, :, 2] + center
        )
        areas_by_group.append(
            0.5
            * np.linalg.norm(
                np.cross(
                    group[:, 1] - group[:, 0], group[:, 2] - group[:, 0]
                ),
                axis=1,
            )
        )
    total_surface_area = float(sum(float(areas.sum()) for areas in areas_by_group))
    target_surface_area = total_surface_area * float(coverage)

    def clipped_area(cutoff: float) -> float:
        total = 0.0
        for group, camera, full_areas in zip(
            world_groups, camera_groups, areas_by_group
        ):
            signed = (
                focal * camera[:, :, camera_coordinate]
                + (center - cutoff) * camera[:, :, 2]
            )
            inside = signed <= 0.0
            counts = inside.sum(axis=1)
            total += float(full_areas[counts == 3].sum())
            for lone_inside in range(3):
                mask = np.logical_and(counts == 1, inside[:, lone_inside])
                if np.any(mask):
                    i = lone_inside
                    j = (i + 1) % 3
                    k = (i + 2) % 3
                    si = signed[mask, i]
                    pi = group[mask, i]
                    pj = group[mask, j]
                    pk = group[mask, k]
                    qj = pi + (si / (si - signed[mask, j]))[:, None] * (pj - pi)
                    qk = pi + (si / (si - signed[mask, k]))[:, None] * (pk - pi)
                    total += triangle_area(np.stack([pi, qj, qk], axis=1))
            for lone_outside in range(3):
                mask = np.logical_and(counts == 2, ~inside[:, lone_outside])
                if np.any(mask):
                    o = lone_outside
                    j = (o + 1) % 3
                    k = (o + 2) % 3
                    so = signed[mask, o]
                    po = group[mask, o]
                    pj = group[mask, j]
                    pk = group[mask, k]
                    qj = po + (so / (so - signed[mask, j]))[:, None] * (pj - po)
                    qk = po + (so / (so - signed[mask, k]))[:, None] * (pk - po)
                    removed = triangle_area(np.stack([po, qj, qk], axis=1))
                    total += float(full_areas[mask].sum()) - removed
        return total

    projected = np.concatenate([values.reshape(-1) for values in projected_groups])
    lower = float(projected.min()) - 1.0
    upper = float(projected.max()) + 1.0
    for _ in range(40):
        middle = 0.5 * (lower + upper)
        if clipped_area(middle) < target_surface_area:
            lower = middle
        else:
            upper = middle
    cutoff = 0.5 * (lower + upper)

    black_groups: list[np.ndarray] = []
    black_surface_area = 0.0
    clipped_face_count = 0
    for group, camera in zip(world_groups, camera_groups):
        signed = (
            focal * camera[:, :, camera_coordinate]
            + (center - cutoff) * camera[:, :, 2]
        )
        clipped: list[np.ndarray] = []
        for triangle, triangle_signed in zip(group, signed):
            pieces = clip_triangle_to_halfspace(triangle, triangle_signed)
            if len(pieces):
                clipped.extend(pieces)
        clipped_array = np.asarray(clipped, dtype=float).reshape(-1, 3, 3)
        black_groups.append(clipped_array)
        black_surface_area += triangle_area(clipped_array)
        clipped_face_count += int(len(clipped_array))

    visible_coordinates = columns if axis == "screen_x" else rows
    predicted_visible_pixels = int(np.count_nonzero(visible_coordinates <= cutoff))
    return black_groups, {
        "coverage_definition": "complete_3d_render_surface_area",
        "requested_surface_coverage": float(coverage),
        "actual_surface_coverage": float(black_surface_area / total_surface_area),
        "surface_area_total": float(total_surface_area),
        "surface_area_black": float(black_surface_area),
        "selected_face_count": int(clipped_face_count),
        "total_face_count": int(sum(len(group) for group in world_groups)),
        "selection_axis": axis,
        "screen_cutoff_px": float(cutoff),
        "selection_side": "low_coordinate_prefix",
        "baseline_agentview_target_pixels": int(target_mask.sum()),
        "predicted_agentview_black_pixels": predicted_visible_pixels,
        "predicted_agentview_visible_coverage": float(
            predicted_visible_pixels / float(target_mask.sum())
        ),
        "surface_enumeration": "mujoco_render_geometries",
        "brightness_independent_surface_enumeration": True,
        "includes_initially_hidden_surface": True,
        "includes_inner_and_rear_surface": True,
    }


def write_obj(path: Path, triangles: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if len(triangles) == 1:
        triangle = np.asarray(triangles[0], dtype=float)
        midpoint = 0.5 * (triangle[0] + triangle[1])
        triangles = np.asarray(
            [
                [triangle[0], midpoint, triangle[2]],
                [midpoint, triangle[1], triangle[2]],
            ],
            dtype=float,
        )
    lines = ["# A2 visual-only surface partition"]
    index = 1
    for triangle in triangles:
        for vertex in triangle:
            lines.append("v {:.10g} {:.10g} {:.10g}".format(*vertex))
        lines.append(f"f {index} {index + 1} {index + 2}")
        index += 3
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def xml_visual_geoms(root: ET.Element, category: str) -> list[tuple[ET.Element, ET.Element]]:
    rows = []
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError("Object XML has no worldbody")
    for parent in worldbody.iter():
        for child in list(parent):
            if child.tag != "geom":
                continue
            group = child.get("group")
            class_name = str(child.get("class", "")).lower()
            if group == "0" or "collision" in class_name:
                continue
            # LIBERO visual geoms either set group=1 explicitly or inherit a
            # non-collision default class (for example Microwave).
            if group == "1" or group is None:
                rows.append((parent, child))
    return rows


def absolute_asset_paths(root: ET.Element, source_xml: Path) -> None:
    compiler = root.find("compiler")
    meshdir = (compiler.get("meshdir", "") if compiler is not None else "").strip()
    texturedir = (compiler.get("texturedir", "") if compiler is not None else "").strip()
    asset = root.find("asset")
    if asset is None:
        return
    for child in asset:
        file_value = child.get("file")
        if not file_value or os.path.isabs(file_value):
            continue
        subdir = meshdir if child.tag == "mesh" else texturedir
        child.set("file", str((source_xml.parent / subdir / file_value).resolve()))
    if compiler is not None:
        compiler.attrib.pop("meshdir", None)
        compiler.attrib.pop("texturedir", None)


def make_variant(
    base: object,
    visual_geom_ids: list[int],
    source_xml: Path,
    category: str,
    output_dir: Path,
    coverage: float,
) -> tuple[Path, dict[str, object]]:
    tree = ET.parse(source_xml)
    root = tree.getroot()
    absolute_asset_paths(root, source_xml)
    xml_geoms = xml_visual_geoms(root, category)
    if len(xml_geoms) != len(visual_geom_ids):
        raise ValueError(
            f"{category}: XML has {len(xml_geoms)} visual geoms but model has "
            f"{len(visual_geom_ids)}"
        )
    asset = root.find("asset")
    if asset is None:
        asset = ET.SubElement(root, "asset")
    black_material = ET.SubElement(asset, "material")
    black_material.set("name", "a2_surface_black")
    black_material.set("rgba", "0 0 0 1")
    black_material.set("reflectance", "0")
    black_material.set("shininess", "0")
    black_material.set("specular", "0")

    world_groups = [geom_triangles_world(base, geom_id) for geom_id in visual_geom_ids]
    black_world, audit = split_complete_surface_camera_slice(
        base, world_groups, visual_geom_ids, coverage
    )
    camera_id = int(base.sim.model.camera_name2id("agentview"))
    camera_position = np.asarray(base.sim.data.cam_xpos[camera_id], dtype=float)
    target_center = np.concatenate(world_groups, axis=0).reshape(-1, 3).mean(axis=0)
    agentview_direction = camera_position - target_center
    agentview_direction /= max(float(np.linalg.norm(agentview_direction)), 1e-12)
    agentview_offset = agentview_direction * 1e-3
    for index, ((parent, original), geom_id) in enumerate(zip(xml_geoms, visual_geom_ids)):
        body_id = int(base.sim.model.geom_bodyid[geom_id])
        overlay_world = np.asarray(black_world[index], dtype=float).copy()
        if len(overlay_world):
            overlay_world += agentview_offset[None, None, :]
        black_local = world_to_body(base, body_id, overlay_world)
        insert_at = list(parent).index(original) + 1
        if not len(black_local):
            continue
        mesh_name = f"a2_{index}_black"
        obj_path = output_dir / f"{mesh_name}.obj"
        write_obj(obj_path, black_local)
        mesh_asset = ET.SubElement(asset, "mesh")
        mesh_asset.set("name", mesh_name)
        mesh_asset.set("file", str(obj_path.resolve()))
        new_geom = ET.Element("geom")
        new_geom.set("name", mesh_name)
        new_geom.set("type", "mesh")
        new_geom.set("mesh", mesh_name)
        new_geom.set("group", "1")
        new_geom.set("contype", "0")
        new_geom.set("conaffinity", "0")
        new_geom.set("mass", "0")
        new_geom.set("material", "a2_surface_black")
        new_geom.set("rgba", "0 0 0 1")
        parent.insert(insert_at, new_geom)

    xml_path = output_dir / f"{category}.xml"
    ET.indent(tree, space="  ")
    tree.write(xml_path, encoding="utf-8", xml_declaration=True)
    return xml_path, audit


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=Path, default=HERE / "precompiled_variants_full_surface_v3"
    )
    parser.add_argument("--suite", choices=SUITES)
    parser.add_argument("--task-id", type=int, choices=range(10))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / "manifest.json"
    entries = []
    if (args.suite is not None or args.task_id is not None) and manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        entries = [
            entry
            for entry in previous.get("entries", [])
            if not (
                (args.suite is None or str(entry["suite"]) == args.suite)
                and (args.task_id is None or int(entry["task_id"]) == args.task_id)
            )
        ]

    for suite_name in SUITES:
        if args.suite is not None and suite_name != args.suite:
            continue
        suite = benchmark.get_benchmark_dict()[suite_name]()
        target_spec = json.loads(
            (HERE.parent / "adapted" / suite_name / "A2_receptacle_100pct.json").read_text(
                encoding="utf-8"
            )
        )["target_objects_by_task"]
        for task_id in range(10):
            if args.task_id is not None and task_id != args.task_id:
                continue
            targets = target_spec[f"{suite_name}:{task_id}"]
            state = suite.get_task_init_states(task_id)[0]
            env = SegmentationRenderEnv(
                bddl_file_name=suite.get_task_bddl_file_path(task_id),
                camera_heights=256,
                camera_widths=256,
            )
            try:
                env.seed(0)
                env.reset()
                env.set_init_state(state)
                base = env.env
                perturb._forward_sim(base.sim)
                for target in targets:
                    target_object = base.get_object(target)
                    category = str(target_object.category_name)
                    source_xml = Path(str(target_object.file)).resolve()
                    root_body_id = int(base.obj_body_id[target])
                    body_ids = set(
                        perturb._descendant_body_ids(base.sim, root_body_id)
                    )
                    visual_geom_ids = [
                        geom_id
                        for geom_id in range(int(base.sim.model.ngeom))
                        if int(base.sim.model.geom_bodyid[geom_id]) in body_ids
                        and int(base.sim.model.geom_group[geom_id]) != 0
                        and float(base.sim.model.geom_rgba[geom_id][3]) > 0.0
                    ]
                    for coverage in COVERAGES:
                        label = int(round(coverage * 100))
                        output_dir = (
                            args.output
                            / suite_name
                            / f"task_{task_id:02d}"
                            / target
                            / f"{label}pct"
                        )
                        xml_path, audit = make_variant(
                            base,
                            visual_geom_ids,
                            source_xml,
                            category,
                            output_dir,
                            coverage,
                        )
                        entries.append(
                            {
                                "suite": suite_name,
                                "task_id": task_id,
                                "object": target,
                                "category": category,
                                "coverage": coverage,
                                "label": f"{label}pct",
                                "xml_path": str(xml_path.resolve()),
                                "audit": {
                                    "object": target,
                                    "mode": "precompiled_visual_surface_partition",
                                    **audit,
                                    "selection_axis": audit["selection_axis"],
                                    "collision_unchanged": True,
                                    "pose_unchanged": True,
                                },
                            }
                        )
            finally:
                env.close()
    write_json(
        manifest_path,
        {
            "schema_version": "wam_syncbench.A2.precompiled_surface_variants.v1",
            "entries": entries,
        },
    )
    print(json.dumps({"generated": len(entries), "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
