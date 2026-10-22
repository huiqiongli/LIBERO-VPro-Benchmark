"""MuJoCo scene-level A3 interaction-part patches for LIBERO.

Each occluder is a 3-D MuJoCo geom compiled into the scene.  In the selected
protocol, each contact-surface patch is one thin body-attached box and the two
grasping fingers are covered together by one moderately sized box whose pose is
locked to the robosuite ``grip_site``.  The boxes therefore move and rotate with their
physical subjects.  Every patch is rendered by the environment's normal camera
observables.  Source-object RGBA values are never changed, and no returned image
is painted, re-rendered, or replaced by this wrapper.  Three localizers are
supported:

``gripper``
    The inter-finger ``grip_site`` provided by the active robosuite gripper.
``object_contact``
    The centroid of active gripper/target contacts.  Before contact, the exact
    closest point on the target collision surface is obtained with
    ``mujoco.mj_geomDistance``.
``receptacle_contact``
    The centroid of active target/receptacle contacts.  Before contact, the
    closest point on the receptacle collision surface is used.  Goal regions
    without their own collision geometry use their MuJoCo site as the intended
    placement anchor.

Target/receptacle roles are derived from the parsed BDDL goal predicates, not
from the order of ``obj_of_interest``.  This matters for multi-object tasks and
for fixtures that also appear in ``obj_of_interest``.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
import xml.etree.ElementTree as ET

import mujoco
import numpy as np


PARTS = ("gripper", "object_contact", "receptacle_contact")
PLACEMENT_PREDICATES = frozenset(("in", "on"))
CAMERA_BY_IMAGE_KEY = {
    "agentview_image": "agentview",
    "robot0_eye_in_hand_image": "robot0_eye_in_hand",
}
BODY_NAME_BY_PART = {part: f"wam_a3_patch_{part}_body" for part in PARTS}
GEOM_NAME_BY_PART = {part: f"wam_a3_patch_{part}_geom" for part in PARTS}
HIDDEN_POSITION = np.asarray((0.0, 0.0, -10.0), dtype=np.float64)
SUPPORTED_GEOM_TYPES = frozenset(("ellipsoid", "box", "source_mesh"))
MUJOCO_GEOM_TYPE_BY_NAME = {
    "ellipsoid": int(mujoco.mjtGeom.mjGEOM_ELLIPSOID),
    "box": int(mujoco.mjtGeom.mjGEOM_BOX),
    "source_mesh": int(mujoco.mjtGeom.mjGEOM_MESH),
}


def _gripper_surface_geom_name(index: int) -> str:
    return f"wam_a3_patch_gripper_finger_{index}_geom"


def _fixed_region_geom_name(part: str, index: int) -> str:
    return f"wam_a3_patch_{part}_region_{index}_geom"


def _normal_tangent_quaternion(
    normal: Sequence[float], tangent_y: Sequence[float]
) -> np.ndarray:
    """Build a wxyz quaternion with local z normal and local y on the surface."""

    z_axis = np.asarray(normal, dtype=np.float64)
    z_norm = float(np.linalg.norm(z_axis))
    if z_axis.shape != (3,) or not np.isfinite(z_norm) or z_norm < 1.0e-9:
        raise ValueError(f"invalid fixed-region normal: {normal!r}")
    z_axis /= z_norm
    y_axis = np.asarray(tangent_y, dtype=np.float64)
    if y_axis.shape != (3,) or not np.all(np.isfinite(y_axis)):
        raise ValueError(f"invalid fixed-region tangent_y: {tangent_y!r}")
    y_axis = y_axis - z_axis * float(np.dot(y_axis, z_axis))
    y_norm = float(np.linalg.norm(y_axis))
    if y_norm < 1.0e-9:
        raise ValueError(
            f"fixed-region tangent_y is parallel to its normal: "
            f"normal={normal!r}, tangent_y={tangent_y!r}"
        )
    y_axis /= y_norm
    x_axis = np.cross(y_axis, z_axis)
    x_axis /= float(np.linalg.norm(x_axis))
    rotation = np.column_stack((x_axis, y_axis, z_axis))
    quaternion = np.empty(4, dtype=np.float64)
    mujoco.mju_mat2Quat(quaternion, np.ascontiguousarray(rotation).ravel())
    return quaternion


@dataclass(frozen=True)
class EntityRef:
    """One BDDL entity resolved to MuJoCo geometry and/or an anchor site."""

    raw_name: str
    resolved_name: str | None
    geom_ids: tuple[int, ...]
    anchor_site_id: int | None

    def as_dict(self, model: Any) -> dict[str, Any]:
        return {
            "raw_name": self.raw_name,
            "resolved_name": self.resolved_name,
            "geom_names": [model.geom_id2name(gid) for gid in self.geom_ids],
            "anchor_site": (
                model.site_id2name(self.anchor_site_id)
                if self.anchor_site_id is not None
                else None
            ),
        }


@dataclass(frozen=True)
class NearestResult:
    distance_m: float
    point_a: np.ndarray
    point_b: np.ndarray
    geom_a: int
    geom_b: int
    endpoint_order_repaired: bool = False
    endpoint_assignment_cost: float = 0.0


@dataclass(frozen=True)
class PartLocation:
    part: str
    active: bool
    world_point: np.ndarray | None
    source: str
    world_normal: np.ndarray | None = None
    target: str | None = None
    receptacle: str | None = None
    distance_m: float | None = None
    contact_count: int = 0
    detail: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "part": self.part,
            "active": self.active,
            "world_point": (
                [float(value) for value in self.world_point]
                if self.world_point is not None
                else None
            ),
            "world_normal": (
                [float(value) for value in self.world_normal]
                if self.world_normal is not None
                else None
            ),
            "source": self.source,
            "target": self.target,
            "receptacle": self.receptacle,
            "distance_m": (
                None if self.distance_m is None else float(self.distance_m)
            ),
            "contact_count": int(self.contact_count),
            "detail": self.detail,
        }


def _read_spec(path: str | os.PathLike[str]) -> tuple[Path, dict[str, Any]]:
    spec_path = Path(path).expanduser().resolve()
    payload = json.loads(spec_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"A3 spec must be a JSON object: {spec_path}")
    if not payload.get("enabled", False):
        raise ValueError(f"A3 spec is disabled: {spec_path}")
    parts = payload.get("interaction_parts")
    if not isinstance(parts, list) or not parts:
        raise ValueError("A3 spec needs a non-empty interaction_parts list")
    unknown = sorted(set(parts) - set(PARTS))
    if unknown:
        raise ValueError(f"unknown A3 interaction parts: {unknown}")
    if payload.get("occlusion_space", "mujoco_world") != "mujoco_world":
        raise ValueError("optimized A3 only accepts occlusion_space=mujoco_world")
    dynamics_mode = payload.get("dynamics_mode", "non_colliding_scene_geometry")
    if dynamics_mode != "non_colliding_scene_geometry":
        raise ValueError(
            "optimized A3 requires dynamics_mode=non_colliding_scene_geometry"
        )
    return spec_path, payload


def _as_rgba(value: Any) -> np.ndarray:
    rgba = np.asarray(value, dtype=np.float64)
    if rgba.shape != (4,) or not np.all(np.isfinite(rgba)):
        raise ValueError(f"occluder_rgba must have four finite values, got {value!r}")
    if np.any((rgba < 0.0) | (rgba > 1.0)):
        raise ValueError(f"occluder_rgba values must be in [0, 1], got {value!r}")
    return rgba


class A3PhysicalOcclusionEnv:
    """Transparent wrapper that adds non-colliding 3-D patches to a LIBERO env."""

    def __init__(
        self,
        env: Any,
        *,
        spec_path: str | os.PathLike[str],
        suite: str,
        task_idx: int,
    ) -> None:
        self._env = env
        self.spec_path, self.spec = _read_spec(spec_path)
        self.suite = str(suite)
        self.task_idx = int(task_idx)
        self.parts = tuple(dict.fromkeys(self.spec["interaction_parts"]))
        self.views = tuple(
            self.spec.get("views", ("agentview", "robot0_eye_in_hand"))
        )
        invalid_views = sorted(set(self.views) - set(CAMERA_BY_IMAGE_KEY.values()))
        if invalid_views:
            raise ValueError(f"unsupported A3 views: {invalid_views}")

        self.localization_mode = str(
            self.spec.get("localization_mode", "online_dynamic_contact")
        )
        if self.localization_mode not in (
            "online_dynamic_contact",
            "frozen_task_local_regions",
        ):
            raise ValueError(
                f"unsupported A3 localization_mode: {self.localization_mode}"
            )
        activation = str(self.spec.get("activation", "entire_episode"))
        if self.localization_mode == "frozen_task_local_regions" and activation != "entire_episode":
            raise ValueError(
                "frozen A3 regions require activation=entire_episode, got "
                f"{activation!r}"
            )
        self.activation = activation
        self._fixed_manifest_path: Path | None = None
        self._fixed_manifest: dict[str, Any] | None = None
        self._fixed_task_config: dict[str, Any] | None = None
        self._fixed_template_by_part: dict[str, dict[str, Any]] = {}
        self._fixed_patch_definitions_by_part: dict[
            str, tuple[dict[str, Any], ...]
        ] = {}
        self._fixed_patch_names_by_part: dict[str, tuple[str, ...]] = {}
        self._fixed_parts: tuple[str, ...] = ()
        if self.localization_mode == "frozen_task_local_regions":
            self._load_fixed_region_manifest()

        geom_type_value = self.spec.get("patch_geom_type", "ellipsoid")
        if isinstance(geom_type_value, Mapping):
            self.geom_type_by_part = {
                part: str(geom_type_value.get(part, "ellipsoid"))
                for part in self.parts
            }
        else:
            self.geom_type_by_part = {
                part: str(geom_type_value) for part in self.parts
            }
        invalid_geom_types = {
            part: geom_type
            for part, geom_type in self.geom_type_by_part.items()
            if geom_type not in SUPPORTED_GEOM_TYPES
        }
        if invalid_geom_types:
            raise ValueError(f"unsupported A3 patch geom types: {invalid_geom_types}")
        invalid_source_mesh_parts = {
            part
            for part, geom_type in self.geom_type_by_part.items()
            if geom_type == "source_mesh" and part != "gripper"
        }
        if invalid_source_mesh_parts:
            raise ValueError(
                "source_mesh is only valid for the articulated gripper fingers: "
                f"{sorted(invalid_source_mesh_parts)}"
            )
        for part, template in self._fixed_template_by_part.items():
            manifest_type = str(template["geom_type"])
            if self.geom_type_by_part[part] != manifest_type:
                raise ValueError(
                    f"A3 fixed template {template['template_key']} uses "
                    f"{manifest_type}, but the spec configures "
                    f"{self.geom_type_by_part[part]} for {part}"
                )

        radius_value = self.spec.get(
            "patch_radius_m",
            self.spec.get(
                "proxy_radius_m",
                self.spec.get(
                    "proxy_size_m", self.spec.get("proxy_size_m_default", 0.03)
                ),
            ),
        )
        if isinstance(radius_value, Mapping):
            self.radius_m = {
                part: float(radius_value.get(part, 0.015)) for part in self.parts
            }
        else:
            self.radius_m = {part: float(radius_value) for part in self.parts}
        if any(not 0.002 <= radius <= 0.15 for radius in self.radius_m.values()):
            raise ValueError(f"A3 patch radii must be in [0.002, 0.15] m: {self.radius_m}")

        thickness_value = self.spec.get("patch_thickness_m", 0.036)
        if isinstance(thickness_value, Mapping):
            self.thickness_m = {
                part: float(thickness_value.get(part, 0.036)) for part in self.parts
            }
        else:
            self.thickness_m = {part: float(thickness_value) for part in self.parts}
        if any(
            not 0.0005 <= thickness <= 0.20
            for thickness in self.thickness_m.values()
        ):
            raise ValueError(
                f"A3 patch thickness must be in [0.0005, 0.20] m: "
                f"{self.thickness_m}"
            )

        self.gripper_axis_offset_m = float(
            self.spec.get("gripper_patch_axis_offset_m", 0.0)
        )
        if not -0.12 <= self.gripper_axis_offset_m <= 0.12:
            raise ValueError(
                "A3 gripper patch axis offset must be in [-0.12, 0.12] m: "
                f"{self.gripper_axis_offset_m}"
            )
        self.gripper_local_offset_m = np.asarray(
            self.spec.get("gripper_patch_local_offset_m", (0.0, 0.0, 0.0)),
            dtype=np.float64,
        )
        if (
            self.gripper_local_offset_m.shape != (3,)
            or not np.all(np.isfinite(self.gripper_local_offset_m))
            or np.any(np.abs(self.gripper_local_offset_m) > 0.03)
        ):
            raise ValueError(
                "A3 gripper local offset must be three finite values in "
                "[-0.03, 0.03] m: "
                f"{self.gripper_local_offset_m}"
            )
        gripper_extents = self.spec.get("gripper_patch_half_extents_m")
        if gripper_extents is None:
            self.gripper_half_extents_m = np.asarray(
                (
                    self.radius_m.get("gripper", 0.035),
                    self.radius_m.get("gripper", 0.035),
                    self.thickness_m.get("gripper", 0.035) / 2.0,
                ),
                dtype=np.float64,
            )
        else:
            self.gripper_half_extents_m = np.asarray(
                gripper_extents, dtype=np.float64
            )
        if (
            self.gripper_half_extents_m.shape != (3,)
            or not np.all(np.isfinite(self.gripper_half_extents_m))
            or np.any(self.gripper_half_extents_m < 0.00025)
            or np.any(self.gripper_half_extents_m > 0.15)
        ):
            raise ValueError(
                "A3 gripper patch half extents must be three values in "
                "[0.00025, 0.15] m: "
                f"{self.gripper_half_extents_m}"
            )

        self.gripper_attachment_mode = str(
            self.spec.get("gripper_attachment_mode", "mocap_grip_site")
        )
        if self.gripper_attachment_mode not in (
            "mocap_grip_site",
            "two_finger_bodies",
        ):
            raise ValueError(
                "unsupported A3 gripper_attachment_mode: "
                f"{self.gripper_attachment_mode!r}"
            )
        self._gripper_body_attached = bool(
            "gripper" in self.parts
            and (
                self.geom_type_by_part["gripper"] == "source_mesh"
                or self.gripper_attachment_mode == "two_finger_bodies"
            )
        )
        if (
            self._gripper_body_attached
            and self.geom_type_by_part["gripper"] not in ("source_mesh", "box")
        ):
            raise ValueError(
                "two-finger body attachment accepts only source_mesh or box, got "
                f"{self.geom_type_by_part['gripper']!r}"
            )

        self.rgba = _as_rgba(self.spec.get("occluder_rgba", (0.0, 0.0, 0.0, 1.0)))
        self.hysteresis_m = float(self.spec.get("selection_hysteresis_m", 0.015))
        self.region_contact_radius_m = float(
            self.spec.get("region_contact_radius_m", 0.12)
        )
        self._records: list[dict[str, Any]] = []
        self._active_target: EntityRef | None = None
        self._body_ids: dict[str, int] = {}
        self._geom_ids: dict[str, int] = {}
        self._geom_ids_by_part: dict[str, tuple[int, ...]] = {}
        self._mocap_ids: dict[str, int] = {}
        self._gripper_source_names: tuple[str, ...] = ()
        self._gripper_patch_names: tuple[str, ...] = ()
        self._gripper_source_geom_ids: tuple[int, ...] = ()
        self._gripper_source_rgba: dict[int, np.ndarray] = {}
        self._observation_update_count = 0
        self._last_recorded_timestep: int | None = None
        self._original_update_observables = None
        self._fixed_contact_stats: dict[str, dict[str, Any]] = {}
        automatic_metadata_dir = os.environ.get(
            "WAM_A3_METADATA_DIR", ""
        ).strip()
        self._automatic_metadata_dir = (
            None
            if not automatic_metadata_dir
            else Path(automatic_metadata_dir).expanduser().resolve()
        )
        self._automatic_trial_index = int(
            os.environ.get("WAM_A3_TRIAL_START", "0")
        )
        if self._automatic_trial_index < 0:
            raise ValueError("WAM_A3_TRIAL_START must be non-negative")
        self._episode_step_count = 0

        if self.localization_mode == "frozen_task_local_regions":
            # Resolve the original scene once so the frozen task entry can be
            # checked before its geoms are compiled into the target bodies.
            self._resolve_task_semantics()
            self._validate_fixed_task_semantics()
        self._install_scene_patches()
        self._resolve_task_semantics()
        if self.localization_mode == "frozen_task_local_regions":
            self._validate_fixed_task_semantics()
        self._hide_patches(forward=True)
        self._install_observable_hook()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._env, name)

    @property
    def _inner(self) -> Any:
        return self._env.env

    @property
    def sim(self) -> Any:
        return self._env.sim

    def _load_fixed_region_manifest(self) -> None:
        selected = str(self.spec.get("task_region_manifest", "")).strip()
        if not selected:
            raise ValueError(
                "frozen_task_local_regions requires task_region_manifest"
            )
        manifest_path = Path(selected).expanduser()
        if not manifest_path.is_absolute():
            manifest_path = self.spec_path.parent / manifest_path
        manifest_path = manifest_path.resolve()
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise TypeError(f"A3 fixed-region manifest is not an object: {manifest_path}")
        task_key = f"{self.suite}/{self.task_idx}"
        tasks = payload.get("tasks", {})
        templates = payload.get("templates", {})
        if task_key not in tasks:
            raise ValueError(
                f"A3 task {task_key} is not in the selected fixed-region manifest"
            )
        task_config = tasks[task_key]
        if not isinstance(task_config, dict):
            raise TypeError(f"invalid A3 task entry for {task_key}")

        definitions_by_part: dict[str, tuple[dict[str, Any], ...]] = {}
        template_by_part: dict[str, dict[str, Any]] = {}
        names_by_part: dict[str, tuple[str, ...]] = {}
        for part in ("object_contact", "receptacle_contact"):
            if part not in self.parts:
                continue
            template_key = str(task_config.get(f"{part}_template", ""))
            if template_key not in templates:
                raise ValueError(
                    f"A3 task {task_key} has no valid {part} template: "
                    f"{template_key!r}"
                )
            template = templates[template_key]
            if not isinstance(template, dict):
                raise TypeError(f"invalid A3 fixed template: {template_key}")
            expected_role = "target" if part == "object_contact" else "receptacle"
            if template.get("role") != expected_role:
                raise ValueError(
                    f"A3 template {template_key} has role={template.get('role')!r}; "
                    f"expected {expected_role!r}"
                )
            body_name = str(template.get("body_name", "")).strip()
            geom_type = str(template.get("geom_type", ""))
            if not body_name or geom_type not in ("ellipsoid", "box"):
                raise ValueError(f"invalid body/type in A3 template {template_key}")
            raw_patches = template.get("patches", [])
            if not isinstance(raw_patches, list):
                raise TypeError(f"A3 template {template_key} patches is not a list")
            patches = list(raw_patches)
            radial_rings = template.get("radial_rings", [])
            if not isinstance(radial_rings, list):
                raise TypeError(
                    f"A3 template {template_key} radial_rings is not a list"
                )
            for ring_index, ring in enumerate(radial_rings):
                if not isinstance(ring, dict):
                    raise TypeError(
                        f"A3 template {template_key} radial ring {ring_index} "
                        "is not an object"
                    )
                count = int(ring.get("count", 8))
                radius = float(ring.get("radius_m"))
                height = float(ring.get("height_m"))
                size = ring.get("size")
                if (
                    count < 3
                    or count > 64
                    or not np.isfinite(radius)
                    or radius <= 0.0
                    or not np.isfinite(height)
                ):
                    raise ValueError(
                        f"invalid A3 radial ring in {template_key}[{ring_index}]"
                    )
                for sample_index in range(count):
                    angle = 2.0 * np.pi * sample_index / count
                    cosine = float(np.cos(angle))
                    sine = float(np.sin(angle))
                    patches.append(
                        {
                            "pos": [radius * cosine, radius * sine, height],
                            "normal": [cosine, sine, 0.0],
                            "tangent_y": [0.0, 0.0, 1.0],
                            "size": size,
                        }
                    )
            if not patches:
                raise ValueError(f"A3 template {template_key} has no patches")
            normalized: list[dict[str, Any]] = []
            for index, raw_patch in enumerate(patches):
                if not isinstance(raw_patch, dict):
                    raise TypeError(
                        f"A3 template {template_key} patch {index} is not an object"
                    )
                position = np.asarray(raw_patch.get("pos"), dtype=np.float64)
                size = np.asarray(raw_patch.get("size"), dtype=np.float64)
                if (
                    position.shape != (3,)
                    or size.shape != (3,)
                    or not np.all(np.isfinite(position))
                    or not np.all(np.isfinite(size))
                    or np.any(size < 0.00025)
                    or np.any(size > 0.20)
                ):
                    raise ValueError(
                        f"invalid A3 patch position/size in {template_key}[{index}]"
                    )
                if "quat" in raw_patch:
                    quaternion = np.asarray(raw_patch["quat"], dtype=np.float64)
                    if quaternion.shape != (4,) or not np.all(np.isfinite(quaternion)):
                        raise ValueError(
                            f"invalid A3 patch quaternion in {template_key}[{index}]"
                        )
                    quaternion /= max(float(np.linalg.norm(quaternion)), 1.0e-12)
                else:
                    quaternion = _normal_tangent_quaternion(
                        raw_patch.get("normal"), raw_patch.get("tangent_y")
                    )
                normalized.append(
                    {
                        "position": position,
                        "size": size,
                        "quaternion": quaternion,
                        "normal": np.asarray(raw_patch.get("normal"), dtype=np.float64),
                    }
                )
            normalized_template = dict(template)
            normalized_template["template_key"] = template_key
            template_by_part[part] = normalized_template
            definitions_by_part[part] = tuple(normalized)
            names_by_part[part] = tuple(
                _fixed_region_geom_name(part, index)
                for index in range(len(normalized))
            )

        self._fixed_manifest_path = manifest_path
        self._fixed_manifest = payload
        self._fixed_task_config = task_config
        self._fixed_template_by_part = template_by_part
        self._fixed_patch_definitions_by_part = definitions_by_part
        self._fixed_patch_names_by_part = names_by_part
        self._fixed_parts = tuple(definitions_by_part)

    def _validate_fixed_task_semantics(self) -> None:
        if self._fixed_task_config is None:
            return
        if len(self.target_receptacle_pairs) != 1:
            raise ValueError(
                "selected A3 fixed-region tasks require exactly one BDDL "
                f"target/receptacle pair, got {len(self.target_receptacle_pairs)}"
            )
        target, receptacle = self.target_receptacle_pairs[0]
        expected_target = str(self._fixed_task_config["target"])
        expected_receptacle = str(self._fixed_task_config["receptacle"])
        if target.raw_name != expected_target or receptacle.raw_name != expected_receptacle:
            raise ValueError(
                "A3 fixed task semantics do not match the manifest: "
                f"resolved=({target.raw_name}, {receptacle.raw_name}), "
                f"expected=({expected_target}, {expected_receptacle})"
            )

    def _model_invariants(self) -> dict[str, int]:
        model = self.sim.model
        return {
            "nq": int(model.nq),
            "nv": int(model.nv),
            "nu": int(model.nu),
            "nbody": int(model.nbody),
            "ngeom": int(model.ngeom),
            "nsite": int(model.nsite),
            "nmocap": int(model.nmocap),
        }

    def _install_scene_patches(self) -> None:
        before = self._model_invariants()
        flat_state = np.asarray(self._env.get_sim_state()).copy()
        # Rebuild from robosuite's source model XML, not MuJoCo's serialized
        # compiled XML.  The latter is not mass-idempotent for these mesh
        # assets: recompiling it can count visual-mesh inertia a second time
        # even before a patch is added.
        xml_root = ET.fromstring(self._inner.model.get_xml())
        worldbody = xml_root.find("worldbody")
        if worldbody is None:
            raise RuntimeError("MuJoCo XML has no worldbody")

        parent_by_element = {
            child: parent
            for parent in xml_root.iter()
            for child in parent
        }
        original_gripper_rgba_by_name: dict[str, np.ndarray] = {}
        if self._gripper_body_attached:
            visual_names = tuple(
                str(name)
                for name in getattr(
                    self._inner.robots[0].gripper, "visual_geoms", ()
                )
            )
            self._gripper_source_names = tuple(
                name for name in visual_names if "finger" in name.lower()
            )
            if len(self._gripper_source_names) != 2:
                raise RuntimeError(
                    "A3 source-mesh gripper patch requires exactly two finger "
                    f"visual geoms, got {self._gripper_source_names}"
                )
            self._gripper_patch_names = tuple(
                _gripper_surface_geom_name(index)
                for index in range(len(self._gripper_source_names))
            )

        existing_names = {
            element.get("name")
            for tag in ("body", "geom")
            for element in xml_root.iter(tag)
            if element.get("name")
        }
        generated_names: list[str] = []
        for part in self.parts:
            if part == "gripper" and self._gripper_body_attached:
                generated_names.extend(self._gripper_patch_names)
            elif part in self._fixed_parts:
                generated_names.extend(self._fixed_patch_names_by_part[part])
            else:
                generated_names.extend(
                    (BODY_NAME_BY_PART[part], GEOM_NAME_BY_PART[part])
                )
        duplicates = [name for name in generated_names if name in existing_names]
        if duplicates:
            raise RuntimeError(f"A3 scene patches already exist: {duplicates}")

        for part in self.parts:
            if part == "gripper" and self._gripper_body_attached:
                geom_elements = {
                    element.get("name"): element
                    for element in xml_root.iter("geom")
                    if element.get("name")
                }
                for source_name, patch_name in zip(
                    self._gripper_source_names, self._gripper_patch_names
                ):
                    source = geom_elements.get(source_name)
                    if source is None:
                        raise RuntimeError(
                            f"gripper finger visual geom missing from XML: {source_name}"
                        )
                    if source.get("type", "sphere") != "mesh":
                        raise RuntimeError(
                            "gripper body-attached patch requires visual mesh geoms: "
                            f"{source_name} type={source.get('type')}"
                        )
                    parent = parent_by_element.get(source)
                    if parent is None or parent.tag != "body":
                        raise RuntimeError(
                            f"gripper finger visual geom has no body: {source_name}"
                        )
                    original_rgba = _as_rgba(
                        [float(value) for value in source.get("rgba", "1 1 1 1").split()]
                    )
                    original_gripper_rgba_by_name[source_name] = original_rgba
                    if self.geom_type_by_part[part] == "source_mesh":
                        patch_attributes = dict(source.attrib)
                        patch_attributes.pop("material", None)
                    else:
                        # A thin, moderately enlarged box uses the finger
                        # visual's local pose but remains parented to the same
                        # articulated body.  This makes the patch clearly
                        # visible without covering the palm or nearby scene.
                        source_id = int(self.sim.model.geom_name2id(source_name))
                        source_local_position = np.asarray(
                            self.sim.model.geom_pos[source_id], dtype=np.float64
                        )
                        source_local_quaternion = np.asarray(
                            self.sim.model.geom_quat[source_id], dtype=np.float64
                        )
                        patch_attributes = {
                            "type": "box",
                            # Compiled mesh assets can carry refpos/refquat not
                            # shown on the XML geom.  Use MuJoCo's resolved
                            # local pose so the box is centred on the visible
                            # finger instead of the finger-body origin.
                            "pos": " ".join(
                                f"{value:.9g}" for value in source_local_position
                            ),
                            "quat": " ".join(
                                f"{value:.9g}" for value in source_local_quaternion
                            ),
                            "size": " ".join(
                                f"{value:.9g}"
                                for value in self.gripper_half_extents_m
                            ),
                        }
                    patch_attributes.update(
                        {
                            "name": patch_name,
                            "rgba": " ".join(
                                f"{value:.9g}" for value in self.rgba
                            ),
                            "contype": "0",
                            "conaffinity": "0",
                            "group": "1",
                            # Visual-only geoms must not change the owning
                            # body's compiled mass or inertia.  Collision
                            # flags alone do not make a MuJoCo geom inertial.
                            "mass": "1e-12",
                            "density": "0",
                        }
                    )
                    ET.SubElement(parent, "geom", patch_attributes)
                    hidden_rgba = original_rgba.copy()
                    hidden_rgba[3] = 0.0
                    source.set(
                        "rgba", " ".join(f"{value:.9g}" for value in hidden_rgba)
                    )
                continue

            if part in self._fixed_parts:
                template = self._fixed_template_by_part[part]
                body_name = str(template["body_name"])
                parent = next(
                    (
                        element
                        for element in xml_root.iter("body")
                        if element.get("name") == body_name
                    ),
                    None,
                )
                if parent is None:
                    raise RuntimeError(
                        f"A3 fixed-region body is missing from XML: {body_name}"
                    )
                for patch_name, definition in zip(
                    self._fixed_patch_names_by_part[part],
                    self._fixed_patch_definitions_by_part[part],
                ):
                    ET.SubElement(
                        parent,
                        "geom",
                        {
                            "name": patch_name,
                            "type": str(template["geom_type"]),
                            "pos": " ".join(
                                f"{value:.9g}" for value in definition["position"]
                            ),
                            "quat": " ".join(
                                f"{value:.9g}" for value in definition["quaternion"]
                            ),
                            "size": " ".join(
                                f"{value:.9g}" for value in definition["size"]
                            ),
                            "rgba": " ".join(
                                f"{value:.9g}" for value in self.rgba
                            ),
                            "contype": "0",
                            "conaffinity": "0",
                            "group": "1",
                            "mass": "1e-12",
                            "density": "0",
                        },
                    )
                continue

            half_extents = (
                self.gripper_half_extents_m
                if part == "gripper"
                else np.asarray(
                    (
                        self.radius_m[part],
                        self.radius_m[part],
                        self.thickness_m[part] / 2.0,
                    ),
                    dtype=np.float64,
                )
            )
            body = ET.SubElement(
                worldbody,
                "body",
                {
                    "name": BODY_NAME_BY_PART[part],
                    "mocap": "true",
                    "pos": "0 0 -10",
                },
            )
            ET.SubElement(
                body,
                "geom",
                {
                    "name": GEOM_NAME_BY_PART[part],
                    "type": self.geom_type_by_part[part],
                    "size": " ".join(f"{value:.9g}" for value in half_extents),
                    "rgba": " ".join(f"{value:.9g}" for value in self.rgba),
                    "contype": "0",
                    "conaffinity": "0",
                    "group": "1",
                    "mass": "1e-12",
                    "density": "0",
                },
            )

        self._env.reset_from_xml_string(ET.tostring(xml_root, encoding="unicode"))
        # A later hard reset would rebuild the original XML and silently drop the
        # proxies.  Soft reset still performs all LIBERO placement randomization.
        self._inner.hard_reset = False
        self._env.set_state(flat_state)
        self.sim.forward()

        after = self._model_invariants()
        unchanged_fields = ("nq", "nv", "nu")
        changed = {
            key: (before[key], after[key])
            for key in unchanged_fields
            if before[key] != after[key]
        }
        mocap_parts = tuple(
            part
            for part in self.parts
            if not (part == "gripper" and self._gripper_body_attached)
            and part not in self._fixed_parts
        )
        fixed_geom_count = sum(
            len(self._fixed_patch_names_by_part[part])
            for part in self._fixed_parts
        )
        added_geom_count = (
            len(mocap_parts) + len(self._gripper_patch_names) + fixed_geom_count
        )
        expected = {
            "nbody": before["nbody"] + len(mocap_parts),
            "ngeom": before["ngeom"] + added_geom_count,
            "nsite": before["nsite"],
            "nmocap": before["nmocap"] + len(mocap_parts),
        }
        count_errors = {
            key: (after[key], value)
            for key, value in expected.items()
            if after[key] != value
        }
        if changed or count_errors:
            raise RuntimeError(
                "A3 scene-patch installation violated model invariants: "
                f"dynamic_dimensions={changed}, count_errors={count_errors}"
            )

        for part in self.parts:
            if part == "gripper" and self._gripper_body_attached:
                source_ids = tuple(
                    int(self.sim.model.geom_name2id(name))
                    for name in self._gripper_source_names
                )
                patch_ids = tuple(
                    int(self.sim.model.geom_name2id(name))
                    for name in self._gripper_patch_names
                )
                for source_id, patch_id in zip(source_ids, patch_ids):
                    if int(self.sim.model.geom_bodyid[source_id]) != int(
                        self.sim.model.geom_bodyid[patch_id]
                    ):
                        raise RuntimeError(
                            "gripper surface patch is not attached to its source finger"
                        )
                    if (
                        int(self.sim.model.geom_contype[patch_id]) != 0
                        or int(self.sim.model.geom_conaffinity[patch_id]) != 0
                    ):
                        raise RuntimeError(
                            "gripper surface patch unexpectedly participates in contact"
                        )
                    if int(self.sim.model.geom_group[patch_id]) != 1:
                        raise RuntimeError(
                            "gripper surface patch is outside the visual geom group"
                        )
                    expected_gripper_geom_type = MUJOCO_GEOM_TYPE_BY_NAME[
                        self.geom_type_by_part[part]
                    ]
                    if int(self.sim.model.geom_type[patch_id]) != expected_gripper_geom_type:
                        raise RuntimeError(
                            "gripper patch has the wrong configured geom type"
                        )
                self._gripper_source_geom_ids = source_ids
                self._gripper_source_rgba = {
                    source_id: original_gripper_rgba_by_name[source_name].copy()
                    for source_name, source_id in zip(
                        self._gripper_source_names, source_ids
                    )
                }
                self._body_ids[part] = int(
                    self.sim.model.geom_bodyid[patch_ids[0]]
                )
                self._geom_ids[part] = patch_ids[0]
                self._geom_ids_by_part[part] = patch_ids
                continue

            if part in self._fixed_parts:
                template = self._fixed_template_by_part[part]
                patch_ids = tuple(
                    int(self.sim.model.geom_name2id(name))
                    for name in self._fixed_patch_names_by_part[part]
                )
                expected_body_name = str(template["body_name"])
                expected_body_id = int(
                    self.sim.model.body_name2id(expected_body_name)
                )
                expected_geom_type = MUJOCO_GEOM_TYPE_BY_NAME[
                    self.geom_type_by_part[part]
                ]
                for patch_id in patch_ids:
                    if int(self.sim.model.geom_bodyid[patch_id]) != expected_body_id:
                        raise RuntimeError(
                            f"A3 fixed patch {part} is not attached to "
                            f"{expected_body_name}"
                        )
                    if (
                        int(self.sim.model.geom_contype[patch_id]) != 0
                        or int(self.sim.model.geom_conaffinity[patch_id]) != 0
                    ):
                        raise RuntimeError(
                            f"A3 fixed patch {part} unexpectedly participates in contact"
                        )
                    if int(self.sim.model.geom_group[patch_id]) != 1:
                        raise RuntimeError(
                            f"A3 fixed patch {part} is outside visual group 1"
                        )
                    if int(self.sim.model.geom_type[patch_id]) != expected_geom_type:
                        raise RuntimeError(
                            f"A3 fixed patch {part} has the wrong geom type"
                        )
                self._body_ids[part] = expected_body_id
                self._geom_ids[part] = patch_ids[0]
                self._geom_ids_by_part[part] = patch_ids
                continue

            bid = int(self.sim.model.body_name2id(BODY_NAME_BY_PART[part]))
            gid = int(self.sim.model.geom_name2id(GEOM_NAME_BY_PART[part]))
            mocap_id = int(self.sim.model.body_mocapid[bid])
            if mocap_id < 0:
                raise RuntimeError(f"A3 patch {part} is not a mocap body")
            if int(self.sim.model.geom_bodyid[gid]) != bid:
                raise RuntimeError(f"A3 patch geom {part} has the wrong parent body")
            if (
                int(self.sim.model.geom_contype[gid]) != 0
                or int(self.sim.model.geom_conaffinity[gid]) != 0
            ):
                raise RuntimeError(f"A3 patch {part} unexpectedly participates in contact")
            if int(self.sim.model.geom_group[gid]) != 1:
                raise RuntimeError(f"A3 patch {part} is outside the visual geom group")
            expected_geom_type = MUJOCO_GEOM_TYPE_BY_NAME[
                self.geom_type_by_part[part]
            ]
            if int(self.sim.model.geom_type[gid]) != expected_geom_type:
                raise RuntimeError(
                    f"A3 patch {part} has unexpected geom type: "
                    f"{int(self.sim.model.geom_type[gid])} != {expected_geom_type}"
                )
            self._body_ids[part] = bid
            self._geom_ids[part] = gid
            self._geom_ids_by_part[part] = (gid,)
            self._mocap_ids[part] = mocap_id

        self.model_invariants = {
            "before": before,
            "after": after,
            "dynamics_unchanged": True,
            "added_scene_bodies": len(mocap_parts),
            "added_visual_geoms": added_geom_count,
            "added_mocap_bodies": len(mocap_parts),
            "added_collision_geoms": 0,
            "all_patch_geoms_non_colliding": True,
            "patch_geom_group": 1,
            "patch_geom_types": dict(self.geom_type_by_part),
            "fixed_body_attached_parts": list(
                dict.fromkeys(
                    (
                        ("gripper",) if self._gripper_body_attached else ()
                    )
                    + self._fixed_parts
                )
            ),
            "fixed_region_manifest": (
                None
                if self._fixed_manifest_path is None
                else str(self._fixed_manifest_path)
            ),
        }

    def _resolve_entity(self, raw_name: str) -> EntityRef:
        raw_name = str(raw_name)
        objects = self._inner.objects_dict
        fixtures = self._inner.fixtures_dict
        site_objects = self._inner.object_sites_dict

        anchor_site_id: int | None = None
        try:
            anchor_site_id = int(self.sim.model.site_name2id(raw_name))
        except ValueError:
            pass

        resolved_name: str | None = None
        if raw_name in objects or raw_name in fixtures:
            resolved_name = raw_name
        elif raw_name in site_objects:
            parent = getattr(site_objects[raw_name], "parent_name", None)
            if parent in objects or parent in fixtures:
                resolved_name = str(parent)
        if resolved_name is None:
            possible = [
                name
                for name in tuple(objects) + tuple(fixtures)
                if raw_name == name or raw_name.startswith(f"{name}_")
            ]
            if possible:
                resolved_name = max(possible, key=len)

        geom_names: Sequence[str] = ()
        if resolved_name in objects:
            geom_names = tuple(getattr(objects[resolved_name], "contact_geoms", ()))
        elif resolved_name in fixtures:
            geom_names = tuple(getattr(fixtures[resolved_name], "contact_geoms", ()))

        geom_ids: list[int] = []
        for name in geom_names:
            try:
                geom_ids.append(int(self.sim.model.geom_name2id(name)))
            except ValueError:
                continue
        if resolved_name is not None and not geom_ids:
            prefix = f"{resolved_name}_"
            for gid in range(self.sim.model.ngeom):
                name = self.sim.model.geom_id2name(gid) or ""
                if name.startswith(prefix) and (
                    self.sim.model.geom_contype[gid] != 0
                    or self.sim.model.geom_conaffinity[gid] != 0
                ):
                    geom_ids.append(int(gid))

        return EntityRef(
            raw_name=raw_name,
            resolved_name=resolved_name,
            geom_ids=tuple(dict.fromkeys(geom_ids)),
            anchor_site_id=anchor_site_id,
        )

    def _resolve_task_semantics(self) -> None:
        parsed = self._inner.parsed_problem
        goals = parsed.get("goal_state", ())
        target_names: list[str] = []
        raw_pairs: list[tuple[str, str, str]] = []
        unary_targets: list[str] = []
        for atom in goals:
            if not atom:
                continue
            predicate = str(atom[0]).lower()
            if predicate in PLACEMENT_PREDICATES and len(atom) >= 3:
                target = str(atom[1])
                receptacle = str(atom[2])
                target_names.append(target)
                raw_pairs.append((predicate, target, receptacle))
            elif len(atom) >= 2:
                target = str(atom[1])
                target_names.append(target)
                unary_targets.append(target)

        semantic_source = "bddl_goal_state"
        if not target_names:
            # This should be rare, but is intentionally explicit in metadata.
            target_names = [str(name) for name in parsed.get("obj_of_interest", ())[:1]]
            semantic_source = "obj_of_interest_fallback"

        refs_by_raw = {
            name: self._resolve_entity(name)
            for name in dict.fromkeys(
                target_names + [receptacle for _, _, receptacle in raw_pairs]
            )
        }
        self.targets = tuple(refs_by_raw[name] for name in dict.fromkeys(target_names))
        self.target_receptacle_pairs = tuple(
            (refs_by_raw[target], refs_by_raw[receptacle])
            for _, target, receptacle in raw_pairs
        )
        self.placement_predicates = {
            (target, receptacle): predicate
            for predicate, target, receptacle in raw_pairs
        }

        gripper = self._inner.robots[0].gripper
        gripper_geom_names = tuple(getattr(gripper, "contact_geoms", ()))
        self.gripper_geom_ids = tuple(
            int(self.sim.model.geom_name2id(name))
            for name in gripper_geom_names
            if name in self.sim.model._geom_name2id
        )
        self.gripper_pad_geom_ids = tuple(
            int(self.sim.model.geom_name2id(name))
            for name in gripper_geom_names
            if "pad" in str(name).lower()
            and name in self.sim.model._geom_name2id
        )
        if len(self.gripper_pad_geom_ids) != 2:
            raise RuntimeError(
                "A3 requires exactly two gripper pad collision geoms, got "
                f"{[self.sim.model.geom_id2name(gid) for gid in self.gripper_pad_geom_ids]}"
            )
        important_sites = getattr(gripper, "important_sites", {})
        grip_site_name = important_sites.get("grip_site", "gripper0_grip_site")
        self.grip_site_id = int(self.sim.model.site_name2id(grip_site_name))

        model = self.sim.model
        self.task_semantics = {
            "source": semantic_source,
            "language": " ".join(
                str(value) for value in parsed.get("language_instruction", ())
            ),
            "goal_state": goals,
            "obj_of_interest": list(parsed.get("obj_of_interest", ())),
            "targets": [target.as_dict(model) for target in self.targets],
            "target_receptacle_pairs": [
                {
                    "predicate": self.placement_predicates[
                        (target.raw_name, receptacle.raw_name)
                    ],
                    "target": target.as_dict(model),
                    "receptacle": receptacle.as_dict(model),
                }
                for target, receptacle in self.target_receptacle_pairs
            ],
            "unary_targets_without_receptacle": unary_targets,
            "gripper_site": grip_site_name,
            "gripper_contact_geoms": [
                model.geom_id2name(gid) for gid in self.gripper_geom_ids
            ],
            "gripper_pad_contact_geoms": [
                model.geom_id2name(gid) for gid in self.gripper_pad_geom_ids
            ],
        }

    def _hide_patches(self, *, forward: bool) -> None:
        for mocap_id in self._mocap_ids.values():
            self.sim.data.mocap_pos[mocap_id] = HIDDEN_POSITION
            self.sim.data.mocap_quat[mocap_id] = (1.0, 0.0, 0.0, 0.0)
        if self._gripper_source_geom_ids:
            for source_id in self._gripper_source_geom_ids:
                self.sim.model.geom_rgba[source_id] = self._gripper_source_rgba[
                    source_id
                ]
            for patch_id in self._geom_ids_by_part["gripper"]:
                self.sim.model.geom_rgba[patch_id] = (0.0, 0.0, 0.0, 0.0)
        if forward:
            self.sim.forward()

    def _show_gripper_surface_patch(self) -> None:
        """Swap the two finger visuals for their body-attached black patches."""

        for source_id in self._gripper_source_geom_ids:
            hidden = self._gripper_source_rgba[source_id].copy()
            hidden[3] = 0.0
            self.sim.model.geom_rgba[source_id] = hidden
        for patch_id in self._geom_ids_by_part["gripper"]:
            self.sim.model.geom_rgba[patch_id] = self.rgba

    def _install_observable_hook(self) -> None:
        """Move scene patches immediately before robosuite samples its cameras."""

        if self._original_update_observables is not None:
            raise RuntimeError("A3 observable hook is already installed")
        self._original_update_observables = self._inner._update_observables

        def update_observables_with_scene_patches(force: bool = False) -> None:
            self._update_scene_patches()
            self._original_update_observables(force=force)

        # This is an instance-level closure: robosuite calls it with only
        # ``force``.  The original bound method remains responsible for every
        # camera render and for the returned observation cache.
        self._inner._update_observables = update_observables_with_scene_patches

    def _contact_points(
        self, group_a: Iterable[int], group_b: Iterable[int]
    ) -> list[np.ndarray]:
        a = frozenset(int(value) for value in group_a)
        b = frozenset(int(value) for value in group_b)
        if not a or not b:
            return []
        points: list[np.ndarray] = []
        for index in range(int(self.sim.data.ncon)):
            contact = self.sim.data.contact[index]
            geom1 = int(contact.geom1)
            geom2 = int(contact.geom2)
            if (geom1 in a and geom2 in b) or (geom1 in b and geom2 in a):
                points.append(np.asarray(contact.pos, dtype=np.float64).copy())
        return points

    def _assign_distance_endpoints(
        self,
        geom_a: int,
        geom_b: int,
        endpoint_0: np.ndarray,
        endpoint_1: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, bool, float]:
        """Associate ``mj_geomDistance`` endpoints with their named geoms.

        MuJoCo's mesh / primitive collision dispatch can internally reorder a
        pair without restoring the documented endpoint order.  A true surface
        point must lie inside its geom's bounding sphere, so we validate both
        assignments and choose the uniquely valid one.  Normalized centre
        distance resolves the rare case where both bounding spheres overlap.
        """

        model = self.sim.model
        data = self.sim.data

        def normalized_distance(point: np.ndarray, geom_id: int) -> float:
            radius = max(float(model.geom_rbound[geom_id]), 1.0e-9)
            return float(np.linalg.norm(point - data.geom_xpos[geom_id])) / radius

        direct_cost = normalized_distance(endpoint_0, geom_a) + normalized_distance(
            endpoint_1, geom_b
        )
        swapped_cost = normalized_distance(endpoint_1, geom_a) + normalized_distance(
            endpoint_0, geom_b
        )
        tolerance = 1.0 + 1.0e-6
        direct_valid = (
            normalized_distance(endpoint_0, geom_a) <= tolerance
            and normalized_distance(endpoint_1, geom_b) <= tolerance
        )
        swapped_valid = (
            normalized_distance(endpoint_1, geom_a) <= tolerance
            and normalized_distance(endpoint_0, geom_b) <= tolerance
        )

        if swapped_valid and not direct_valid:
            return endpoint_1.copy(), endpoint_0.copy(), True, swapped_cost
        if direct_valid and not swapped_valid:
            return endpoint_0.copy(), endpoint_1.copy(), False, direct_cost
        if swapped_cost + 1.0e-9 < direct_cost:
            return endpoint_1.copy(), endpoint_0.copy(), True, swapped_cost
        return endpoint_0.copy(), endpoint_1.copy(), False, direct_cost

    def _nearest_between(
        self, group_a: Iterable[int], group_b: Iterable[int]
    ) -> NearestResult | None:
        ids_a = tuple(dict.fromkeys(int(value) for value in group_a))
        ids_b = tuple(dict.fromkeys(int(value) for value in group_b))
        if not ids_a or not ids_b:
            return None

        model = self.sim.model
        data = self.sim.data
        candidates: list[tuple[float, int, int]] = []
        for geom_a in ids_a:
            for geom_b in ids_b:
                center_distance = float(
                    np.linalg.norm(data.geom_xpos[geom_a] - data.geom_xpos[geom_b])
                )
                lower_bound = center_distance - float(
                    model.geom_rbound[geom_a] + model.geom_rbound[geom_b]
                )
                candidates.append((lower_bound, geom_a, geom_b))
        candidates.sort(key=lambda value: value[0])

        best: NearestResult | None = None
        raw_model = model._model
        raw_data = data._data
        for lower_bound, geom_a, geom_b in candidates:
            if best is not None and lower_bound > best.distance_m + 1.0e-9:
                break
            fromto = np.empty(6, dtype=np.float64)
            try:
                distance = float(
                    mujoco.mj_geomDistance(
                        raw_model, raw_data, geom_a, geom_b, 5.0, fromto
                    )
                )
            except (mujoco.FatalError, TypeError, ValueError):
                continue
            if not np.all(np.isfinite(fromto)):
                continue
            if best is None or distance < best.distance_m:
                point_a, point_b, repaired, assignment_cost = (
                    self._assign_distance_endpoints(
                        geom_a, geom_b, fromto[:3], fromto[3:]
                    )
                )
                best = NearestResult(
                    distance_m=distance,
                    point_a=point_a,
                    point_b=point_b,
                    geom_a=geom_a,
                    geom_b=geom_b,
                    endpoint_order_repaired=repaired,
                    endpoint_assignment_cost=assignment_cost,
                )
        return best

    def _entity_anchor(self, entity: EntityRef) -> np.ndarray | None:
        if entity.anchor_site_id is not None:
            return np.asarray(
                self.sim.data.site_xpos[entity.anchor_site_id], dtype=np.float64
            ).copy()
        if entity.geom_ids:
            return np.mean(
                np.asarray(self.sim.data.geom_xpos[list(entity.geom_ids)]), axis=0
            )
        return None

    def _target_scores(
        self,
    ) -> list[tuple[EntityRef, list[np.ndarray], NearestResult | None, float]]:
        results = []
        grip_point = np.asarray(self.sim.data.site_xpos[self.grip_site_id])
        for target in self.targets:
            contacts = self._contact_points(self.gripper_geom_ids, target.geom_ids)
            nearest = self._nearest_between(self.gripper_geom_ids, target.geom_ids)
            if contacts:
                score = -1.0
            elif nearest is not None:
                score = nearest.distance_m
            else:
                anchor = self._entity_anchor(target)
                score = (
                    float(np.linalg.norm(anchor - grip_point))
                    if anchor is not None
                    else float("inf")
                )
            results.append((target, contacts, nearest, score))
        return results

    def _select_target(
        self,
    ) -> tuple[EntityRef | None, list[np.ndarray], NearestResult | None]:
        scored = self._target_scores()
        if not scored:
            return None, [], None

        contacting = [item for item in scored if item[1]]
        if contacting:
            chosen = max(contacting, key=lambda item: len(item[1]))
        else:
            chosen = min(scored, key=lambda item: item[3])
            if self._active_target is not None:
                previous = next(
                    (item for item in scored if item[0] == self._active_target), None
                )
                if previous is not None and previous[3] <= chosen[3] + self.hysteresis_m:
                    chosen = previous
        self._active_target = chosen[0]
        return chosen[0], chosen[1], chosen[2]

    def _pair_for_target(
        self, target: EntityRef
    ) -> tuple[EntityRef, EntityRef] | None:
        candidates = [
            pair
            for pair in self.target_receptacle_pairs
            if pair[0].raw_name == target.raw_name
        ]
        if not candidates:
            return None
        if len(candidates) == 1:
            return candidates[0]

        def score(pair: tuple[EntityRef, EntityRef]) -> float:
            nearest = self._nearest_between(pair[0].geom_ids, pair[1].geom_ids)
            if nearest is not None:
                return nearest.distance_m
            target_anchor = self._entity_anchor(pair[0])
            receptacle_anchor = self._entity_anchor(pair[1])
            if target_anchor is None or receptacle_anchor is None:
                return float("inf")
            return float(np.linalg.norm(target_anchor - receptacle_anchor))

        return min(candidates, key=score)

    def _placement_predicate(
        self, pair: tuple[EntityRef, EntityRef]
    ) -> str | None:
        return self.placement_predicates.get(
            (pair[0].raw_name, pair[1].raw_name)
        )

    def _goal_region_support(
        self, receptacle: EntityRef, target: EntityRef | None = None
    ) -> tuple[np.ndarray, np.ndarray, str] | None:
        """Return the actual support surface below a BDDL goal-region site.

        Once the target enters the goal region it can sit between the ray
        origin and the receptacle floor.  Exclude that articulated target body
        so localization remains locked to the intended receiving surface.
        """

        sid = receptacle.anchor_site_id
        if sid is None:
            return None
        model = self.sim.model
        data = self.sim.data
        site_type = int(model.site_type[sid])
        size = np.asarray(model.site_size[sid], dtype=np.float64)
        if site_type in (
            int(mujoco.mjtGeom.mjGEOM_BOX),
            int(mujoco.mjtGeom.mjGEOM_ELLIPSOID),
        ):
            half_height = float(size[2])
        elif site_type in (
            int(mujoco.mjtGeom.mjGEOM_CYLINDER),
            int(mujoco.mjtGeom.mjGEOM_CAPSULE),
        ):
            half_height = float(size[1])
        elif site_type == int(mujoco.mjtGeom.mjGEOM_SPHERE):
            half_height = float(size[0])
        else:
            half_height = 0.0
        # LIBERO goal-region site frames are not consistent: some ``in``
        # regions have local z pointing down.  Placement support is a gravity-
        # aligned surface, so always cast vertically downward and return an
        # upward-facing normal instead of trusting the site's local z axis.
        normal = np.asarray((0.0, 0.0, 1.0), dtype=np.float64)
        anchor = np.asarray(data.site_xpos[sid], dtype=np.float64)
        # Cast through collision group 0 only, excluding the visual A3 patches
        # in group 1.  Starting inside an ``in`` region makes the first hit the
        # true interior support.  Site-only regions (for example, the table
        # area in front of a stove) have no resolved parent entity, so their
        # actual first supporting collision geom is accepted explicitly.
        geom_group = np.zeros(6, dtype=np.uint8)
        geom_group[0] = 1
        hit_geom = np.empty(1, dtype=np.int32)
        target_body_ids = (
            {
                int(model.geom_bodyid[gid])
                for gid in target.geom_ids
            }
            if target is not None
            else set()
        )
        target_body_exclude = (
            next(iter(target_body_ids)) if len(target_body_ids) == 1 else -1
        )
        ray_clearance = half_height + 0.01
        ray_origin = anchor + normal * ray_clearance
        distance = float(
            mujoco.mj_ray(
                model._model,
                data._data,
                np.ascontiguousarray(ray_origin),
                np.ascontiguousarray(-normal),
                geom_group,
                1,
                target_body_exclude,
                hit_geom,
            )
        )
        hit_id = int(hit_geom[0])
        intended_hit = hit_id in receptacle.geom_ids
        site_only_support = not receptacle.geom_ids and hit_id >= 0
        if distance >= 0.0 and (intended_hit or site_only_support):
            return (
                ray_origin - normal * distance,
                normal,
                (
                    "top_down_ray_to_receptacle_collision_surface"
                    if intended_hit
                    else "top_down_ray_to_site_region_support_surface"
                ),
            )
        return (
            anchor - normal * half_height,
            normal,
            "goal_region_lower_face_fallback",
        )

    def _region_contact_points(
        self, target: EntityRef, receptacle: EntityRef
    ) -> list[np.ndarray]:
        anchor = self._entity_anchor(receptacle)
        if anchor is None or not target.geom_ids:
            return []
        excluded = set(target.geom_ids) | set(self.gripper_geom_ids)
        points: list[np.ndarray] = []
        for index in range(int(self.sim.data.ncon)):
            contact = self.sim.data.contact[index]
            geom1 = int(contact.geom1)
            geom2 = int(contact.geom2)
            if geom1 in target.geom_ids and geom2 not in excluded:
                point = np.asarray(contact.pos, dtype=np.float64)
            elif geom2 in target.geom_ids and geom1 not in excluded:
                point = np.asarray(contact.pos, dtype=np.float64)
            else:
                continue
            if float(np.linalg.norm(point - anchor)) <= self.region_contact_radius_m:
                points.append(point.copy())
        return points

    def _locate_parts(self) -> dict[str, PartLocation]:
        target, target_contacts, target_nearest = self._select_target()
        grip_point = np.asarray(
            self.sim.data.site_xpos[self.grip_site_id], dtype=np.float64
        ).copy()
        locations: dict[str, PartLocation] = {
            "gripper": PartLocation(
                part="gripper",
                active=True,
                world_point=grip_point,
                source="robosuite_gripper_grip_site",
                target=None if target is None else target.raw_name,
            )
        }

        if target is None:
            locations["object_contact"] = PartLocation(
                part="object_contact",
                active=False,
                world_point=None,
                source="unavailable",
                detail="no target entity resolved from BDDL goals",
            )
            locations["receptacle_contact"] = PartLocation(
                part="receptacle_contact",
                active=False,
                world_point=None,
                source="not_applicable",
                detail="no target entity resolved from BDDL goals",
            )
            return locations

        if target_contacts:
            object_point = np.mean(np.stack(target_contacts), axis=0)
            locations["object_contact"] = PartLocation(
                part="object_contact",
                active=True,
                world_point=object_point,
                source="active_mujoco_gripper_target_contact_centroid",
                target=target.raw_name,
                distance_m=0.0,
                contact_count=len(target_contacts),
            )
        elif target_nearest is not None:
            locations["object_contact"] = PartLocation(
                part="object_contact",
                active=True,
                world_point=target_nearest.point_b,
                source="mujoco_exact_nearest_target_surface_point",
                target=target.raw_name,
                distance_m=target_nearest.distance_m,
                detail=(
                    f"gripper_geom={self.sim.model.geom_id2name(target_nearest.geom_a)}, "
                    f"target_geom={self.sim.model.geom_id2name(target_nearest.geom_b)}, "
                    f"endpoint_order_repaired={target_nearest.endpoint_order_repaired}"
                ),
            )
        else:
            anchor = self._entity_anchor(target)
            locations["object_contact"] = PartLocation(
                part="object_contact",
                active=anchor is not None,
                world_point=anchor,
                source="target_anchor_fallback" if anchor is not None else "unavailable",
                target=target.raw_name,
                detail="target has no usable collision geometry",
            )

        pair = self._pair_for_target(target)
        if pair is None:
            locations["receptacle_contact"] = PartLocation(
                part="receptacle_contact",
                active=False,
                world_point=None,
                source="not_applicable",
                target=target.raw_name,
                detail="BDD goal has no target/receptacle placement predicate",
            )
            return locations

        _, receptacle = pair
        predicate = self._placement_predicate(pair)
        contacts = (
            self._contact_points(target.geom_ids, receptacle.geom_ids)
            if receptacle.geom_ids
            else self._region_contact_points(target, receptacle)
        )
        if contacts:
            locations["receptacle_contact"] = PartLocation(
                part="receptacle_contact",
                active=True,
                world_point=np.mean(np.stack(contacts), axis=0),
                source=(
                    "active_mujoco_target_receptacle_contact_centroid"
                    if receptacle.geom_ids
                    else "active_mujoco_target_contact_near_goal_region"
                ),
                target=target.raw_name,
                receptacle=receptacle.raw_name,
                distance_m=0.0,
                contact_count=len(contacts),
                detail=f"placement_predicate={predicate}",
            )
        else:
            goal_support = self._goal_region_support(receptacle, target)
            if goal_support is not None and predicate in PLACEMENT_PREDICATES:
                support_point, support_normal, support_method = goal_support
                locations["receptacle_contact"] = PartLocation(
                    part="receptacle_contact",
                    active=True,
                    world_point=support_point,
                    world_normal=support_normal,
                    source=f"bddl_{predicate}_goal_region_support_surface",
                    target=target.raw_name,
                    receptacle=receptacle.raw_name,
                    detail=(
                        f"goal_region_site="
                        f"{self.sim.model.site_id2name(receptacle.anchor_site_id)}, "
                        f"support_method={support_method}"
                    ),
                )
            elif receptacle.geom_ids:
                nearest = self._nearest_between(
                    target.geom_ids, receptacle.geom_ids
                )
                if nearest is not None:
                    locations["receptacle_contact"] = PartLocation(
                        part="receptacle_contact",
                        active=True,
                        world_point=nearest.point_b,
                        source="mujoco_exact_nearest_receptacle_surface_point",
                        target=target.raw_name,
                        receptacle=receptacle.raw_name,
                        distance_m=nearest.distance_m,
                        detail=(
                            f"target_geom={self.sim.model.geom_id2name(nearest.geom_a)}, "
                            f"receptacle_geom={self.sim.model.geom_id2name(nearest.geom_b)}, "
                            f"endpoint_order_repaired={nearest.endpoint_order_repaired}"
                        ),
                    )
                else:
                    locations["receptacle_contact"] = PartLocation(
                        part="receptacle_contact",
                        active=False,
                        world_point=None,
                        source="unavailable",
                        target=target.raw_name,
                        receptacle=receptacle.raw_name,
                    )
            else:
                anchor = self._entity_anchor(receptacle)
                locations["receptacle_contact"] = PartLocation(
                    part="receptacle_contact",
                    active=anchor is not None,
                    world_point=anchor,
                    source=(
                        "receptacle_anchor_fallback"
                        if anchor is not None
                        else "unavailable"
                    ),
                    target=target.raw_name,
                    receptacle=receptacle.raw_name,
                )
        return locations

    def _fixed_region_contact_points(self, part: str) -> list[np.ndarray]:
        if not self.target_receptacle_pairs:
            return []
        target, receptacle = self.target_receptacle_pairs[0]
        if part == "object_contact":
            # Only the two finger-pad contacts define the intended grasp
            # interface.  Palm / finger-link collisions are incidental and
            # must not expand the A3 target region around the object contour.
            return self._contact_points(self.gripper_pad_geom_ids, target.geom_ids)
        if part == "receptacle_contact":
            if receptacle.geom_ids:
                return self._contact_points(target.geom_ids, receptacle.geom_ids)
            return self._region_contact_points(target, receptacle)
        return []

    def _fixed_region_covers_point(self, part: str, point: np.ndarray) -> bool:
        """Return whether a physical contact lies under one frozen surface pad."""

        world_point = np.asarray(point, dtype=np.float64)
        for geom_id in self._geom_ids_by_part[part]:
            rotation = np.asarray(
                self.sim.data.geom_xmat[geom_id], dtype=np.float64
            ).reshape(3, 3)
            local = rotation.T @ (
                world_point
                - np.asarray(self.sim.data.geom_xpos[geom_id], dtype=np.float64)
            )
            size = np.asarray(self.sim.model.geom_size[geom_id], dtype=np.float64)
            if self.geom_type_by_part[part] == "box":
                tangent_inside = bool(
                    abs(float(local[0])) <= 1.05 * float(size[0])
                    and abs(float(local[1])) <= 1.05 * float(size[1])
                )
            else:
                tangent_metric = float(
                    (local[0] / max(size[0], 1.0e-12)) ** 2
                    + (local[1] / max(size[1], 1.0e-12)) ** 2
                )
                tangent_inside = tangent_metric <= 1.05**2
            # The physical source surface touches the pad's inner pole at
            # local z=-size[2].  A few millimetres tolerance accommodates
            # collision-mesh discretization without expanding tangential area.
            normal_error = abs(float(local[2] + size[2]))
            if tangent_inside and normal_error <= 0.006:
                return True
        return False

    def _fixed_region_payload(
        self,
        part: str,
        dynamic_location: PartLocation,
        *,
        accumulate_contacts: bool,
    ) -> dict[str, Any]:
        geom_payloads = []
        centers = []
        attachment_points = []
        normals = []
        for geom_id in self._geom_ids_by_part[part]:
            center = np.asarray(
                self.sim.data.geom_xpos[geom_id], dtype=np.float64
            ).copy()
            rotation = np.asarray(
                self.sim.data.geom_xmat[geom_id], dtype=np.float64
            ).reshape(3, 3)
            normal = rotation[:, 2].copy()
            size = np.asarray(
                self.sim.model.geom_size[geom_id], dtype=np.float64
            ).copy()
            attachment = center - normal * size[2]
            centers.append(center)
            normals.append(normal)
            attachment_points.append(attachment)
            geom_payloads.append(
                {
                    "geom": self.sim.model.geom_id2name(geom_id),
                    "body": self.sim.model.body_id2name(
                        int(self.sim.model.geom_bodyid[geom_id])
                    ),
                    "center": center.tolist(),
                    "attachment_point": attachment.tolist(),
                    "normal": normal.tolist(),
                    "half_extents_m": size.tolist(),
                    "local_position": np.asarray(
                        self.sim.model.geom_pos[geom_id], dtype=np.float64
                    ).tolist(),
                    "local_quaternion_wxyz": np.asarray(
                        self.sim.model.geom_quat[geom_id], dtype=np.float64
                    ).tolist(),
                }
            )

        contacts = self._fixed_region_contact_points(part)
        covered = sum(
            self._fixed_region_covers_point(part, point) for point in contacts
        )
        if accumulate_contacts and contacts:
            source_body_name = str(self._fixed_template_by_part[part]["body_name"])
            source_body_id = int(self.sim.model.body_name2id(source_body_name))
            source_rotation = np.asarray(
                self.sim.data.body_xmat[source_body_id], dtype=np.float64
            ).reshape(3, 3)
            source_position = np.asarray(
                self.sim.data.body_xpos[source_body_id], dtype=np.float64
            )
            source_local_points = [
                source_rotation.T @ (np.asarray(point) - source_position)
                for point in contacts
            ]
            stats = self._fixed_contact_stats.setdefault(
                part,
                {
                    "frames_with_contact": 0,
                    "contact_points": 0,
                    "covered_contact_points": 0,
                    "source_body": source_body_name,
                    "source_local_min_m": [float("inf")] * 3,
                    "source_local_max_m": [float("-inf")] * 3,
                    "source_local_sum_m": [0.0] * 3,
                    "source_local_samples_m": [],
                },
            )
            stats["frames_with_contact"] += 1
            stats["contact_points"] += len(contacts)
            stats["covered_contact_points"] += int(covered)
            local_stack = np.stack(source_local_points)
            stats["source_local_min_m"] = np.minimum(
                np.asarray(stats["source_local_min_m"], dtype=np.float64),
                np.min(local_stack, axis=0),
            ).tolist()
            stats["source_local_max_m"] = np.maximum(
                np.asarray(stats["source_local_max_m"], dtype=np.float64),
                np.max(local_stack, axis=0),
            ).tolist()
            stats["source_local_sum_m"] = (
                np.asarray(stats["source_local_sum_m"], dtype=np.float64)
                + np.sum(local_stack, axis=0)
            ).tolist()
            samples = stats["source_local_samples_m"]
            remaining = max(0, 512 - len(samples))
            samples.extend(local_stack[:remaining].tolist())

        center = np.mean(np.stack(centers), axis=0)
        attachment = np.mean(np.stack(attachment_points), axis=0)
        representative_normal = normals[0]
        template = self._fixed_template_by_part[part]
        target_name = (
            None
            if self._fixed_task_config is None
            else str(self._fixed_task_config["target"])
        )
        receptacle_name = (
            None
            if part == "object_contact" or self._fixed_task_config is None
            else str(self._fixed_task_config["receptacle"])
        )
        payload = dynamic_location.as_dict()
        payload.update(
            {
                "active": True,
                "world_point": center.tolist(),
                "world_normal": representative_normal.tolist(),
                "source": "frozen_task_local_region_manifest",
                "target": target_name,
                "receptacle": receptacle_name,
                "contact_count": len(contacts),
                "detail": (
                    f"template={template['template_key']}, "
                    "activation=entire_episode, runtime_retargeting=false"
                ),
                "patch_pose": {
                    "attachment_point": attachment.tolist(),
                    "center": center.tolist(),
                    "quaternion_wxyz": None,
                    "radius_m": None,
                    "thickness_m": None,
                    "half_extents_m": None,
                    "geom_type": "multi_surface_region",
                    "axis_offset_m": None,
                    "mujoco_body": str(template["body_name"]),
                    "mujoco_geoms": [item["geom"] for item in geom_payloads],
                },
                "patch_geoms": geom_payloads,
                "contacts_in_region_this_observation": int(covered),
            }
        )
        return payload

    def _fixed_contact_coverage_report(self) -> dict[str, dict[str, Any]]:
        threshold = float(self.spec.get("minimum_contact_coverage", 0.95))
        result: dict[str, dict[str, Any]] = {}
        for part in self._fixed_parts:
            stats = self._fixed_contact_stats.get(
                part,
                {
                    "frames_with_contact": 0,
                    "contact_points": 0,
                    "covered_contact_points": 0,
                },
            )
            total = int(stats["contact_points"])
            covered = int(stats["covered_contact_points"])
            fraction = None if total == 0 else float(covered / total)
            local_sum = stats.get("source_local_sum_m")
            local_mean = (
                None
                if total == 0 or local_sum is None
                else (np.asarray(local_sum, dtype=np.float64) / total).tolist()
            )
            result[part] = {
                **stats,
                "source_local_mean_m": local_mean,
                "coverage_fraction": fraction,
                "minimum_fraction": threshold,
                "passed": None if fraction is None else fraction >= threshold,
            }
        return result

    @staticmethod
    def _unit_vector(value: np.ndarray, fallback: np.ndarray) -> np.ndarray:
        vector = np.asarray(value, dtype=np.float64)
        norm = float(np.linalg.norm(vector))
        if np.isfinite(norm) and norm > 1.0e-8:
            return vector / norm
        return np.asarray(fallback, dtype=np.float64).copy()

    def _patch_normal(self, part: str, location: PartLocation) -> np.ndarray:
        if location.world_normal is not None:
            return self._unit_vector(
                location.world_normal, np.asarray((0.0, 0.0, 1.0))
            )
        if part == "gripper":
            grip_rotation = np.asarray(
                self.sim.data.site_xmat[self.grip_site_id], dtype=np.float64
            ).reshape(3, 3)
            return self._unit_vector(
                grip_rotation[:, 2], np.asarray((0.0, 0.0, 1.0))
            )
        if part == "object_contact" and location.world_point is not None:
            grip_point = np.asarray(self.sim.data.site_xpos[self.grip_site_id])
            return self._unit_vector(
                grip_point - location.world_point, np.asarray((0.0, 0.0, 1.0))
            )
        if part == "receptacle_contact" and location.world_point is not None:
            target_anchor = (
                None
                if self._active_target is None
                else self._entity_anchor(self._active_target)
            )
            if target_anchor is not None:
                return self._unit_vector(
                    target_anchor - location.world_point,
                    np.asarray((0.0, 0.0, 1.0)),
                )
        return np.asarray((0.0, 0.0, 1.0), dtype=np.float64)

    @staticmethod
    def _normal_to_quaternion(normal: np.ndarray) -> np.ndarray:
        """Return a MuJoCo wxyz quaternion whose local z axis is ``normal``."""

        z_axis = A3PhysicalOcclusionEnv._unit_vector(
            normal, np.asarray((0.0, 0.0, 1.0))
        )
        reference = (
            np.asarray((1.0, 0.0, 0.0))
            if abs(float(z_axis[2])) > 0.9
            else np.asarray((0.0, 0.0, 1.0))
        )
        x_axis = A3PhysicalOcclusionEnv._unit_vector(
            np.cross(reference, z_axis), np.asarray((1.0, 0.0, 0.0))
        )
        y_axis = np.cross(z_axis, x_axis)
        rotation = np.column_stack((x_axis, y_axis, z_axis))
        quaternion = np.empty(4, dtype=np.float64)
        mujoco.mju_mat2Quat(quaternion, np.ascontiguousarray(rotation).ravel())
        return quaternion

    def _update_scene_patches(self) -> None:
        """Update real MuJoCo patch poses before normal camera observables run."""

        # Frozen task-local regions already encode the exact source body,
        # local pose, and size.  Re-running the legacy online nearest-surface
        # localizer is both scientifically unnecessary and, with the MuJoCo
        # build in the OpenVLA environment, pathologically expensive.  Keep
        # online localization only when at least one requested part actually
        # uses that mode; fixed payloads still collect real contact coverage
        # below on every new simulator timestep.
        fixed_runtime_parts = set(self._fixed_parts)
        if (
            self.localization_mode == "frozen_task_local_regions"
            and "gripper" in self.parts
        ):
            # The accepted gripper rectangle is locked directly to robosuite's
            # grip_site and needs no target / nearest-surface query.
            fixed_runtime_parts.add("gripper")
        if set(self.parts) == fixed_runtime_parts:
            if self._active_target is None and self.target_receptacle_pairs:
                self._active_target = self.target_receptacle_pairs[0][0]
            locations = {}
            for part in self.parts:
                if part == "gripper":
                    locations[part] = PartLocation(
                        part=part,
                        active=True,
                        world_point=np.asarray(
                            self.sim.data.site_xpos[self.grip_site_id],
                            dtype=np.float64,
                        ).copy(),
                        source="robosuite_gripper_grip_site",
                        target=(
                            None
                            if self._active_target is None
                            else self._active_target.raw_name
                        ),
                    )
                else:
                    locations[part] = PartLocation(
                        part=part,
                        active=True,
                        world_point=None,
                        source="frozen_task_local_region_manifest",
                    )
        else:
            locations = self._locate_parts()
        self._hide_patches(forward=False)
        step = int(getattr(self._inner, "timestep", -1))
        accumulate_contacts = self._last_recorded_timestep != step
        part_payloads: dict[str, dict[str, Any]] = {}
        for part in self.parts:
            location = locations[part]
            if part in self._fixed_parts:
                part_payloads[part] = self._fixed_region_payload(
                    part,
                    location,
                    accumulate_contacts=accumulate_contacts,
                )
                continue
            payload = location.as_dict()
            if not location.active or location.world_point is None:
                part_payloads[part] = payload
                continue

            normal = self._patch_normal(part, location)
            attachment_point = np.asarray(location.world_point, dtype=np.float64)
            if part == "gripper" and self._gripper_source_geom_ids:
                self._show_gripper_surface_patch()
                patch_ids = self._geom_ids_by_part[part]
                source_ids = self._gripper_source_geom_ids
                centres = np.stack(
                    [np.asarray(self.sim.data.geom_xpos[gid]) for gid in patch_ids]
                )
                centre = np.mean(centres, axis=0)
                grip_rotation = np.asarray(
                    self.sim.data.site_xmat[self.grip_site_id], dtype=np.float64
                ).reshape(3, 3)
                quaternion = np.empty(4, dtype=np.float64)
                mujoco.mju_mat2Quat(
                    quaternion, np.ascontiguousarray(grip_rotation).ravel()
                )
                payload["world_normal"] = [float(value) for value in normal]
                payload["patch_pose"] = {
                    "attachment_point": [
                        float(value) for value in attachment_point
                    ],
                    "center": [float(value) for value in centre],
                    "quaternion_wxyz": [
                        float(value) for value in quaternion
                    ],
                    "radius_m": None,
                    "thickness_m": None,
                    "half_extents_m": (
                        None
                        if self.geom_type_by_part[part] == "source_mesh"
                        else self.gripper_half_extents_m.tolist()
                    ),
                    "geom_type": self.geom_type_by_part[part],
                    "axis_offset_m": 0.0,
                    "attachment_mode": self.gripper_attachment_mode,
                    "surface_scope": (
                        "two_finger_visual_mesh_geoms_only"
                        if self.geom_type_by_part[part] == "source_mesh"
                        else "moderately_enlarged_two_finger_boxes_only"
                    ),
                    "mujoco_bodies": [
                        self.sim.model.body_id2name(
                            int(self.sim.model.geom_bodyid[gid])
                        )
                        for gid in patch_ids
                    ],
                    "mujoco_geoms": [
                        self.sim.model.geom_id2name(gid) for gid in patch_ids
                    ],
                    "source_visual_geoms": [
                        self.sim.model.geom_id2name(gid) for gid in source_ids
                    ],
                }
                payload["patch_geoms"] = [
                    {
                        "geom": self.sim.model.geom_id2name(patch_id),
                        "source_visual_geom": self.sim.model.geom_id2name(source_id),
                        "body": self.sim.model.body_id2name(
                            int(self.sim.model.geom_bodyid[patch_id])
                        ),
                        "center": np.asarray(
                            self.sim.data.geom_xpos[patch_id], dtype=np.float64
                        ).tolist(),
                        "local_position": np.asarray(
                            self.sim.model.geom_pos[patch_id], dtype=np.float64
                        ).tolist(),
                        "local_quaternion_wxyz": np.asarray(
                            self.sim.model.geom_quat[patch_id], dtype=np.float64
                        ).tolist(),
                        "half_extents_m": (
                            None
                            if self.geom_type_by_part[part] == "source_mesh"
                            else np.asarray(
                                self.sim.model.geom_size[patch_id], dtype=np.float64
                            ).tolist()
                        ),
                    }
                    for source_id, patch_id in zip(source_ids, patch_ids)
                ]
                part_payloads[part] = payload
                continue

            # Contact patches sit fully outside the target surface: their inner
            # pole touches the localized interaction point.  This prevents half
            # of a rounded patch from being hidden inside the object itself.
            # Legacy box-mode gripper patches retain the optional local-z
            # offset.  The accepted source-mesh mode above needs no proxy pose:
            # it is parented directly to the two moving finger bodies.
            if part == "gripper":
                grip_rotation = np.asarray(
                    self.sim.data.site_xmat[self.grip_site_id], dtype=np.float64
                ).reshape(3, 3)
                centre = (
                    attachment_point
                    + normal * self.gripper_axis_offset_m
                    + grip_rotation @ self.gripper_local_offset_m
                )
                quaternion = np.empty(4, dtype=np.float64)
                mujoco.mju_mat2Quat(
                    quaternion, np.ascontiguousarray(grip_rotation).ravel()
                )
            else:
                centre = attachment_point + normal * (
                    self.thickness_m[part] / 2.0
                )
                quaternion = self._normal_to_quaternion(normal)
            mocap_id = self._mocap_ids[part]
            self.sim.data.mocap_pos[mocap_id] = centre
            self.sim.data.mocap_quat[mocap_id] = quaternion
            payload["world_normal"] = [float(value) for value in normal]
            payload["patch_pose"] = {
                "attachment_point": [float(value) for value in attachment_point],
                "center": [float(value) for value in centre],
                "quaternion_wxyz": [float(value) for value in quaternion],
                "radius_m": float(self.radius_m[part]),
                "thickness_m": float(self.thickness_m[part]),
                "half_extents_m": (
                    self.gripper_half_extents_m.tolist()
                    if part == "gripper"
                    else [
                        float(self.radius_m[part]),
                        float(self.radius_m[part]),
                        float(self.thickness_m[part] / 2.0),
                    ]
                ),
                "geom_type": self.geom_type_by_part[part],
                "axis_offset_m": (
                    float(self.gripper_axis_offset_m)
                    if part == "gripper"
                    else float(self.thickness_m[part] / 2.0)
                ),
                "local_offset_m": (
                    self.gripper_local_offset_m.tolist()
                    if part == "gripper"
                    else None
                ),
                "mujoco_body": BODY_NAME_BY_PART[part],
                "mujoco_geom": GEOM_NAME_BY_PART[part],
            }
            part_payloads[part] = payload

        # Required so geom_xpos / geom_xmat and the immediately following
        # robosuite camera sensors see the new mocap poses.
        self.sim.forward()

        record = {
            "observation_index": len(self._records),
            "sim_timestep": step,
            "active_target": (
                None if self._active_target is None else self._active_target.raw_name
            ),
            "parts": part_payloads,
            "rendering_path": "robosuite_standard_camera_observables",
        }
        if self._records and self._last_recorded_timestep == step:
            record["observation_index"] = self._records[-1]["observation_index"]
            self._records[-1] = record
        else:
            self._records.append(record)
            self._last_recorded_timestep = step
        self._observation_update_count += 1

    def _localizer_source_counts(self) -> dict[str, int]:
        counts: Counter[str] = Counter()
        for record in self._records:
            for part, location in record["parts"].items():
                counts[f"{part}:{location['source']}"] += 1
        return dict(counts)

    def _reset_tracking(self) -> None:
        self._hide_patches(forward=True)
        self._active_target = None
        self._records.clear()
        self._last_recorded_timestep = None
        self._observation_update_count = 0
        self._fixed_contact_stats.clear()
        self._episode_step_count = 0

    def _flush_automatic_metadata(self) -> Path | None:
        if (
            self._automatic_metadata_dir is None
            or self._episode_step_count < 1
            or not self._records
        ):
            return None
        output = self._automatic_metadata_dir / (
            f"trial_{self._automatic_trial_index:02d}_a3_parts.json"
        )
        self.write_a3_metadata(output)
        self._automatic_trial_index += 1
        self._episode_step_count = 0
        return output

    def reset(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        self._flush_automatic_metadata()
        self._reset_tracking()
        return self._env.reset(*args, **kwargs)

    def set_init_state(self, init_state: Any) -> dict[str, Any]:
        self._flush_automatic_metadata()
        self._reset_tracking()
        return self._env.set_init_state(init_state)

    def step(self, action: Any) -> tuple[dict[str, Any], Any, Any, Any]:
        # The standard environment observation is returned untouched.  The
        # internal hook has already positioned the scene geoms before cameras
        # were sampled.
        result = self._env.step(action)
        self._episode_step_count += 1
        return result

    def close(self) -> Any:
        self._flush_automatic_metadata()
        close_fn = getattr(self._env, "close", None)
        return close_fn() if callable(close_fn) else None

    def validation_report(self) -> dict[str, Any]:
        return {
            "schema_version": "wam_syncbench.a3.validation.v2",
            "suite": self.suite,
            "task_idx": self.task_idx,
            "spec_path": str(self.spec_path),
            "parts": list(self.parts),
            "views": list(self.views),
            "model_invariants": self.model_invariants,
            "task_semantics": self.task_semantics,
            "scene_observation_count": len(self._records),
            "internal_pose_update_count": self._observation_update_count,
            "localizer_source_counts": self._localizer_source_counts(),
            "localization_mode": self.localization_mode,
            "activation": self.activation,
            "fixed_region_manifest": (
                None
                if self._fixed_manifest_path is None
                else str(self._fixed_manifest_path)
            ),
            "fixed_task_config": self._fixed_task_config,
            "fixed_contact_coverage": self._fixed_contact_coverage_report(),
            "last_record": self._records[-1] if self._records else None,
            "standard_observation_passthrough": True,
            "manual_camera_render_calls": 0,
            "returned_image_replacements": 0,
            "full_model_inference_started": False,
        }

    def write_a3_metadata(self, path: str | os.PathLike[str]) -> Path:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": "wam_syncbench.a3.parts.v4",
            "suite": self.suite,
            "task_idx": self.task_idx,
            "spec_path": str(self.spec_path),
            "parts": list(self.parts),
            "spec": self.spec,
            "model_invariants": self.model_invariants,
            "task_semantics": self.task_semantics,
            "summary": {
                "scene_observation_count": len(self._records),
                "internal_pose_update_count": self._observation_update_count,
                "localizer_source_counts": self._localizer_source_counts(),
                "localization_mode": self.localization_mode,
                "activation": self.activation,
                "fixed_region_manifest": (
                    None
                    if self._fixed_manifest_path is None
                    else str(self._fixed_manifest_path)
                ),
                "fixed_task_config": self._fixed_task_config,
                "fixed_contact_coverage": self._fixed_contact_coverage_report(),
                "standard_observation_passthrough": True,
                "manual_camera_render_calls": 0,
                "returned_image_replacements": 0,
            },
            "frames": self._records,
        }
        temporary = output_path.with_suffix(output_path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        temporary.replace(output_path)
        return output_path


def maybe_wrap_env(
    env: Any,
    *,
    suite: str,
    task_idx: int,
    spec_path: str | os.PathLike[str] | None = None,
) -> Any:
    """Return an A3 wrapper only when an explicit spec was provided."""

    selected = str(spec_path or os.environ.get("WAM_A3_SPEC", "")).strip()
    if not selected:
        return env
    if os.environ.get("WAM_A5_VIEW_FILL", "").strip():
        raise RuntimeError("A3 physical occlusion and A5 view fill cannot be enabled together")
    if isinstance(env, A3PhysicalOcclusionEnv):
        raise RuntimeError("environment is already wrapped by A3PhysicalOcclusionEnv")
    return A3PhysicalOcclusionEnv(
        env, spec_path=selected, suite=suite, task_idx=task_idx
    )
