"""World-anchored, cross-view-consistent image blur for A3.

The perturbation is defined by a rectangle in a MuJoCo body's local frame.
At every observation, the same rectangle is transformed into world space and
projected independently into each configured camera.  Only allowed pixels
inside the projected polygon are blurred.  No MuJoCo body, geom, contact property, or
source visual is created or changed.

This deliberately differs from a free-hand 2-D mask: motion and cross-view
consistency come from the simulated body's 3-D pose, while the image-space
operation supplies blur instead of an opaque scene geom.
"""

from __future__ import annotations

from collections import Counter
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


PARTS = ("gripper", "object_contact", "receptacle_contact")
CAMERA_BY_IMAGE_KEY = {
    "agentview_image": "agentview",
    "robot0_eye_in_hand_image": "robot0_eye_in_hand",
}


def _read_spec(path: str | os.PathLike[str]) -> tuple[Path, dict[str, Any]]:
    spec_path = Path(path).expanduser().resolve()
    payload = json.loads(spec_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not payload.get("enabled", False):
        raise ValueError(f"A3 blur spec is invalid or disabled: {spec_path}")
    if payload.get("occlusion_space") != "world_anchored_camera_blur":
        raise ValueError(
            "A3 blur requires occlusion_space=world_anchored_camera_blur"
        )
    if payload.get("dynamics_mode") != "observation_only_no_sim_mutation":
        raise ValueError(
            "A3 blur requires dynamics_mode=observation_only_no_sim_mutation"
        )
    parts = payload.get("interaction_parts")
    if not isinstance(parts, list) or len(parts) != 1 or parts[0] not in PARTS:
        raise ValueError("A3 blur specs must select exactly one interaction part")
    return spec_path, payload


def _unit(value: Sequence[float], label: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    norm = float(np.linalg.norm(result))
    if result.shape != (3,) or not np.isfinite(norm) or norm < 1.0e-9:
        raise ValueError(f"invalid {label}: {value!r}")
    return result / norm


def _convex_hull(points: np.ndarray) -> list[tuple[float, float]]:
    """Monotone-chain hull returned as (column, row) coordinates."""

    unique = sorted({(float(point[0]), float(point[1])) for point in points})
    if len(unique) <= 2:
        return unique

    def cross(
        origin: tuple[float, float],
        first: tuple[float, float],
        second: tuple[float, float],
    ) -> float:
        return (first[0] - origin[0]) * (second[1] - origin[1]) - (
            first[1] - origin[1]
        ) * (second[0] - origin[0])

    lower: list[tuple[float, float]] = []
    for point in unique:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0.0:
            lower.pop()
        lower.append(point)
    upper: list[tuple[float, float]] = []
    for point in reversed(unique):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0.0:
            upper.pop()
        upper.append(point)
    return lower[:-1] + upper[:-1]


class A3WorldBlurEnv:
    """Transparent environment wrapper for body-anchored local blur regions."""

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
        self.parts = tuple(self.spec["interaction_parts"])
        self.part = self.parts[0]
        self.views = tuple(
            self.spec.get("views", ("agentview", "robot0_eye_in_hand"))
        )
        invalid_views = sorted(set(self.views) - set(CAMERA_BY_IMAGE_KEY.values()))
        if invalid_views:
            raise ValueError(f"unsupported A3 blur views: {invalid_views}")
        self.blur_radius_fraction = float(
            self.spec.get("gaussian_radius_fraction", 0.045)
        )
        self.blur_operator = str(self.spec.get("blur_operator", "gaussian"))
        self.privacy_cell_radius_ratio = float(
            self.spec.get("privacy_cell_radius_ratio", 0.34)
        )
        self.privacy_soften_radius_ratio = float(
            self.spec.get("privacy_soften_radius_ratio", 0.12)
        )
        self.privacy_blur_mode = str(
            self.spec.get("privacy_blur_mode", "downsample")
        )
        self.edge_feather_fraction = float(
            self.spec.get("edge_feather_fraction", 0.006)
        )
        self.observation_image_convention = str(
            self.spec.get("observation_image_convention", "opengl")
        )
        self.mask_to_source_body_visible_pixels = bool(
            self.spec.get("mask_to_source_body_visible_pixels", False)
        )
        self.protect_robot_visible_pixels = bool(
            self.spec.get("protect_robot_visible_pixels", False)
        )
        if not 0.01 <= self.blur_radius_fraction <= 0.20:
            raise ValueError("gaussian_radius_fraction must be in [0.01, 0.20]")
        if self.blur_operator not in ("gaussian", "privacy_blur"):
            raise ValueError("blur_operator must be gaussian or privacy_blur")
        if not 0.2 <= self.privacy_cell_radius_ratio <= 3.0:
            raise ValueError("privacy_cell_radius_ratio must be in [0.2, 3.0]")
        if not 0.05 <= self.privacy_soften_radius_ratio <= 1.0:
            raise ValueError("privacy_soften_radius_ratio must be in [0.05, 1.0]")
        if self.privacy_blur_mode not in ("downsample", "local_mean"):
            raise ValueError("privacy_blur_mode must be downsample or local_mean")
        if not 0.0 <= self.edge_feather_fraction <= 0.03:
            raise ValueError("edge_feather_fraction must be in [0, 0.03]")
        if self.observation_image_convention not in ("opengl", "opencv"):
            raise ValueError(
                "observation_image_convention must be opengl or opencv"
            )

        self._manifest_path: Path | None = None
        self._manifest: dict[str, Any] | None = None
        self._task_config: dict[str, Any] | None = None
        self._template: dict[str, Any] | None = None
        self._body_id: int | None = None
        self._source_geom_ids: tuple[int, ...] = ()
        self._local_rectangles: tuple[np.ndarray, ...] = ()
        self._region_kind = "body_attached_black_bar_geometry"
        self.gripper_half_extents_m: np.ndarray | None = None
        self.gripper_axis_offset_m = 0.0
        self.gripper_local_offset_m = np.zeros(3, dtype=np.float64)
        if self.part != "gripper":
            self._load_manifest_region()
        else:
            # These are the exact pose parameters of the accepted one-piece
            # black gripper bar.  The blur path projects the same 3-D box; only
            # its image-space appearance changes from opaque black to blur.
            self.gripper_half_extents_m = np.asarray(
                self.spec.get("gripper_patch_half_extents_m"), dtype=np.float64
            )
            self.gripper_axis_offset_m = float(
                self.spec.get("gripper_patch_axis_offset_m", 0.0)
            )
            self.gripper_local_offset_m = np.asarray(
                self.spec.get("gripper_patch_local_offset_m", (0.0, 0.0, 0.0)),
                dtype=np.float64,
            )
            if (
                self.gripper_half_extents_m.shape != (3,)
                or not np.all(np.isfinite(self.gripper_half_extents_m))
                or np.any(self.gripper_half_extents_m <= 0.0)
                or self.gripper_local_offset_m.shape != (3,)
                or not np.all(np.isfinite(self.gripper_local_offset_m))
                or not np.isfinite(self.gripper_axis_offset_m)
            ):
                raise ValueError("invalid exact-black-bar gripper geometry")
            self._region_kind = "single_grip_site_black_bar_geometry"

        self._records: list[dict[str, Any]] = []
        self._last_recorded_timestep: int | None = None
        self._observation_update_count = 0
        self._episode_step_count = 0
        self._automatic_metadata_dir = None
        selected_metadata = os.environ.get("WAM_A3_METADATA_DIR", "").strip()
        if selected_metadata:
            self._automatic_metadata_dir = Path(selected_metadata).expanduser().resolve()
        self._automatic_trial_index = int(os.environ.get("WAM_A3_TRIAL_START", "0"))
        self.model_invariants = self._model_invariants()
        self._resolve_runtime_region()
        model = self.sim.model
        protected_robot_geom_ids = []
        for geom_id in range(int(model.ngeom)):
            body_name = str(model.body_id2name(int(model.geom_bodyid[geom_id])) or "")
            geom_name = str(model.geom_id2name(geom_id) or "")
            if (
                body_name.startswith("robot0_")
                or body_name.startswith("gripper0_")
                or geom_name.startswith("robot0_")
                or geom_name.startswith("gripper0_")
            ):
                protected_robot_geom_ids.append(geom_id)
        self._protected_robot_geom_ids = tuple(protected_robot_geom_ids)
        self._original_update_observables = None
        self._install_observable_hook()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._env, name)

    @property
    def _inner(self) -> Any:
        return self._env.env

    @property
    def sim(self) -> Any:
        return self._env.sim

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

    def _load_manifest_region(self) -> None:
        selected = str(self.spec.get("task_region_manifest", "")).strip()
        if not selected:
            raise ValueError("A3 body-anchored blur requires task_region_manifest")
        path = Path(selected).expanduser()
        if not path.is_absolute():
            path = self.spec_path.parent / path
        path = path.resolve()
        manifest = json.loads(path.read_text(encoding="utf-8"))
        task_key = f"{self.suite}/{self.task_idx}"
        tasks = manifest.get("tasks", {})
        templates = manifest.get("templates", {})
        if task_key not in tasks:
            raise ValueError(f"A3 blur task {task_key} is absent from {path}")
        task = tasks[task_key]
        field = f"{self.part}_template"
        template_key = str(task.get(field, ""))
        if template_key not in templates:
            raise ValueError(f"A3 blur task {task_key} has no valid {field}")
        template = templates[template_key]
        expected_role = "target" if self.part == "object_contact" else "receptacle"
        if template.get("role") != expected_role:
            raise ValueError(
                f"A3 blur template {template_key} role is not {expected_role}"
            )
        raw_patches = template.get("patches")
        if not isinstance(raw_patches, list) or not raw_patches:
            raise ValueError(f"A3 blur template {template_key} has no rectangles")

        rectangles = []
        for index, patch in enumerate(raw_patches):
            position = np.asarray(patch.get("pos"), dtype=np.float64)
            size = np.asarray(patch.get("size"), dtype=np.float64)
            normal = _unit(patch.get("normal"), f"rectangle {index} normal")
            tangent_y = _unit(
                patch.get("tangent_y"), f"rectangle {index} tangent_y"
            )
            tangent_y -= normal * float(np.dot(tangent_y, normal))
            tangent_y = _unit(tangent_y, f"rectangle {index} tangent_y")
            tangent_x = _unit(
                np.cross(tangent_y, normal), f"rectangle {index} tangent_x"
            )
            if position.shape != (3,) or size.shape != (3,) or np.any(size <= 0.0):
                raise ValueError(f"invalid A3 blur rectangle {template_key}[{index}]")
            # Keep the centre, orientation, and all three half extents exactly
            # as defined for the accepted black MuJoCo box.  In particular, do
            # not snap the centre to a source surface and do not collapse the
            # thin box to a mathematical plane: either would change its
            # projected silhouette and violate black-bar / blur-bar parity.
            corners = np.asarray(
                [
                    position
                    + sx * tangent_x * size[0]
                    + sy * tangent_y * size[1]
                    + sz * normal * size[2]
                    for sx in (-1.0, 1.0)
                    for sy in (-1.0, 1.0)
                    for sz in (-1.0, 1.0)
                ],
                dtype=np.float64,
            )
            rectangles.append(corners)

        self._manifest_path = path
        self._manifest = manifest
        self._task_config = task
        self._template = dict(template)
        self._template["template_key"] = template_key
        self._local_rectangles = tuple(rectangles)
        self._region_kind = str(
            template.get("blur_region_kind", "body_attached_surface_rectangle")
        )

    def _resolve_runtime_region(self) -> None:
        if self.part == "gripper":
            # LIBERO's single-arm FixedBaseRobot exposes grippers as a
            # one-entry ``{"right": PandaGripper}`` mapping, while some
            # robosuite runners expose the gripper object directly.  Resolve
            # both forms without changing the active robot model.
            gripper_container = self._inner.robots[0].gripper
            if isinstance(gripper_container, Mapping):
                if not gripper_container:
                    raise RuntimeError("A3 gripper blur found an empty gripper mapping")
                gripper = gripper_container.get("right") or next(
                    iter(gripper_container.values())
                )
            else:
                gripper = gripper_container
            visual_names = tuple(
                str(name)
                for name in getattr(gripper, "visual_geoms", ())
                if "finger" in str(name).lower()
            )
            if len(visual_names) != 2:
                raise RuntimeError(
                    "A3 gripper blur requires exactly two finger visual geoms, got "
                    f"{visual_names}"
                )
            self._gripper_geom_ids = tuple(
                int(self.sim.model.geom_name2id(name)) for name in visual_names
            )
            self._source_geom_ids = self._gripper_geom_ids
            important_sites = getattr(gripper, "important_sites", {})
            grip_site_name = important_sites.get("grip_site", "gripper0_grip_site")
            self._grip_site_id = int(self.sim.model.site_name2id(grip_site_name))
            return
        assert self._template is not None
        body_name = str(self._template.get("body_name", ""))
        try:
            self._body_id = int(self.sim.model.body_name2id(body_name))
        except ValueError as error:
            raise RuntimeError(f"A3 blur body is missing: {body_name}") from error
        model = self.sim.model
        descendants = set()
        for candidate in range(int(model.nbody)):
            current = candidate
            while current > 0:
                if current == self._body_id:
                    descendants.add(candidate)
                    break
                current = int(model.body_parentid[current])
        descendants.add(self._body_id)
        self._source_geom_ids = tuple(
            geom_id
            for geom_id in range(int(model.ngeom))
            if int(model.geom_bodyid[geom_id]) in descendants
            and int(model.geom_group[geom_id]) == 1
        )
        if self.mask_to_source_body_visible_pixels and not self._source_geom_ids:
            raise RuntimeError(f"A3 blur body has no visible source geoms: {body_name}")

    @staticmethod
    def _camera_projection(
        sim: Any, camera_name: str, height: int, width: int
    ) -> tuple[np.ndarray, np.ndarray, float]:
        camera_id = int(sim.model.camera_name2id(camera_name))
        position = np.asarray(sim.data.cam_xpos[camera_id], dtype=np.float64)
        rotation = np.asarray(
            sim.data.cam_xmat[camera_id], dtype=np.float64
        ).reshape(3, 3)
        corrected_rotation = rotation @ np.diag((1.0, -1.0, -1.0))
        world_to_camera_rotation = corrected_rotation.T
        fovy = float(sim.model.cam_fovy[camera_id])
        focal = 0.5 * height / np.tan(fovy * np.pi / 360.0)
        return position, world_to_camera_rotation, float(focal)

    def _project(
        self,
        world_points: np.ndarray,
        camera_name: str,
        height: int,
        width: int,
    ) -> tuple[list[tuple[float, float]], list[float]]:
        camera_position, world_to_camera_rotation, focal = self._camera_projection(
            self.sim, camera_name, height, width
        )
        camera_points = (world_points - camera_position) @ world_to_camera_rotation.T
        depth = camera_points[:, 2]
        valid = depth > 1.0e-5
        if np.count_nonzero(valid) < 3:
            return [], depth.tolist()
        points = camera_points[valid]
        columns = focal * points[:, 0] / points[:, 2] + width / 2.0
        rows = focal * points[:, 1] / points[:, 2] + height / 2.0
        # robosuite's default ``opengl`` observable returns MuJoCo's native
        # bottom-up array (IMAGE_CONVENTION_MAPPING["opengl"] == 1).  The
        # pinhole equations above produce conventional top-down image rows,
        # so mirror exactly once for the actual array presented to the model.
        if self.observation_image_convention == "opengl":
            rows = (height - 1.0) - rows
        polygon = _convex_hull(np.column_stack((columns, rows)))
        return polygon, depth.tolist()

    def _body_world_rectangles(self) -> tuple[np.ndarray, ...]:
        assert self._body_id is not None
        position = np.asarray(self.sim.data.body_xpos[self._body_id], dtype=np.float64)
        rotation = np.asarray(
            self.sim.data.body_xmat[self._body_id], dtype=np.float64
        ).reshape(3, 3)
        return tuple(points @ rotation.T + position for points in self._local_rectangles)

    def _gripper_world_rectangles(self) -> tuple[np.ndarray, ...]:
        assert self.gripper_half_extents_m is not None
        rotation = np.asarray(
            self.sim.data.site_xmat[self._grip_site_id], dtype=np.float64
        ).reshape(3, 3)
        grip_point = np.asarray(
            self.sim.data.site_xpos[self._grip_site_id], dtype=np.float64
        )
        center = (
            grip_point
            + rotation[:, 2] * self.gripper_axis_offset_m
            + rotation @ self.gripper_local_offset_m
        )
        half = self.gripper_half_extents_m
        corners = np.asarray(
            [
                center + rotation @ np.asarray((sx * half[0], sy * half[1], sz * half[2]))
                for sx in (-1.0, 1.0)
                for sy in (-1.0, 1.0)
                for sz in (-1.0, 1.0)
            ],
            dtype=np.float64,
        )
        return (corners,)

    @staticmethod
    def _blur_image(
        image: np.ndarray,
        polygons: Sequence[Sequence[tuple[float, float]]],
        radius: float,
        edge_feather: float,
        operator: str = "gaussian",
        privacy_cell_radius_ratio: float = 0.34,
        privacy_soften_radius_ratio: float = 0.12,
        privacy_blur_mode: str = "downsample",
        allowed_mask: np.ndarray | None = None,
    ) -> tuple[np.ndarray, int]:
        source = Image.fromarray(np.asarray(image, dtype=np.uint8))
        mask = Image.new("L", source.size, 0)
        draw = ImageDraw.Draw(mask)
        for polygon in polygons:
            if len(polygon) >= 3:
                draw.polygon(polygon, fill=255)
        if edge_feather > 0.0:
            mask = mask.filter(ImageFilter.GaussianBlur(radius=edge_feather))
        mask_array = np.asarray(mask)
        if allowed_mask is not None:
            allowed = np.asarray(allowed_mask, dtype=bool)
            if allowed.shape != mask_array.shape:
                raise ValueError(
                    f"A3 blur source mask shape mismatch: {allowed.shape} != "
                    f"{mask_array.shape}"
                )
            mask_array = np.where(allowed, mask_array, 0).astype(np.uint8)
            mask = Image.fromarray(mask_array, mode="L")
        mask_pixels = int(np.count_nonzero(mask_array))
        if mask_pixels == 0:
            return np.asarray(image).copy(), 0
        if operator == "gaussian":
            blurred = source.filter(ImageFilter.GaussianBlur(radius=radius))
        elif operator == "privacy_blur" and privacy_blur_mode == "downsample":
            # A tiny fixture can remain recognizable after ordinary Gaussian
            # blur because its silhouette is only a few pixels wide.  Privacy
            # blur first removes high-frequency and silhouette detail through
            # area downsampling, then softens the enlarged cells.  It is still
            # a local image blur (never an opaque/black overlay).
            cell = max(5, int(round(radius * privacy_cell_radius_ratio)))
            small_size = (
                max(1, int(np.ceil(source.width / cell))),
                max(1, int(np.ceil(source.height / cell))),
            )
            blurred = source.resize(small_size, resample=Image.Resampling.BOX)
            blurred = blurred.resize(source.size, resample=Image.Resampling.BILINEAR)
            blurred = blurred.filter(
                ImageFilter.GaussianBlur(
                    radius=max(1.0, radius * privacy_soften_radius_ratio)
                )
            )
        elif operator == "privacy_blur" and privacy_blur_mode == "local_mean":
            # Infinite-kernel-style local blur: each projected privacy region
            # receives the mean colour of a much larger surrounding window.
            # This removes both high-frequency detail and the low-frequency
            # silhouette that can survive ordinary Gaussian/downsample blur.
            canvas = np.asarray(source, dtype=np.uint8).copy()
            margin = max(8, int(round(radius)))
            for polygon in polygons:
                if len(polygon) < 3:
                    continue
                xs = [float(point[0]) for point in polygon]
                ys = [float(point[1]) for point in polygon]
                x0 = max(0, int(np.floor(min(xs))))
                x1 = min(source.width, int(np.ceil(max(xs))) + 1)
                y0 = max(0, int(np.floor(min(ys))))
                y1 = min(source.height, int(np.ceil(max(ys))) + 1)
                if x1 <= x0 or y1 <= y0:
                    continue
                sx0, sx1 = max(0, x0 - margin), min(source.width, x1 + margin)
                sy0, sy1 = max(0, y0 - margin), min(source.height, y1 + margin)
                context = np.asarray(source, dtype=np.float32)[sy0:sy1, sx0:sx1]
                if context.size == 0:
                    continue
                mean_colour = np.rint(context.mean(axis=(0, 1))).astype(np.uint8)
                canvas[y0:y1, x0:x1] = mean_colour
            # Do not blur this candidate canvas again: for a narrow handle,
            # convolution would pull the original handle back in from both
            # sides.  Edge softness is already supplied by the feathered
            # composite mask, while the interior stays truly featureless.
            blurred = Image.fromarray(canvas)
        else:  # guarded during construction
            raise RuntimeError(f"unsupported A3 blur operator: {operator}")
        result = Image.composite(blurred, source, mask)
        return np.asarray(result, dtype=np.asarray(image).dtype), mask_pixels

    def _install_observable_hook(self) -> None:
        if self._original_update_observables is not None:
            raise RuntimeError("A3 blur observable hook is already installed")
        self._original_update_observables = self._inner._update_observables

        def update_observables_with_world_blur(force: bool = False) -> None:
            self._original_update_observables(force=force)
            self._apply_world_blur()

        self._inner._update_observables = update_observables_with_world_blur

    def _apply_world_blur(self) -> None:
        world_regions = (
            self._gripper_world_rectangles()
            if self.part == "gripper"
            else self._body_world_rectangles()
        )
        view_payloads: dict[str, Any] = {}
        for image_key, camera_name in CAMERA_BY_IMAGE_KEY.items():
            if camera_name not in self.views:
                continue
            observable = self._inner._observables.get(image_key)
            if observable is None or observable.obs is None:
                raise RuntimeError(f"A3 blur camera observable is missing: {image_key}")
            clean = np.asarray(observable.obs)
            if clean.ndim != 3 or clean.shape[-1] != 3:
                raise RuntimeError(f"A3 blur expects an RGB image: {clean.shape}")
            height, width = clean.shape[:2]
            polygons = []
            depths = []
            for region in world_regions:
                polygon, region_depths = self._project(
                    region, camera_name, height, width
                )
                if len(polygon) >= 3:
                    polygons.append(polygon)
                depths.append(region_depths)
            radius = max(2.0, min(height, width) * self.blur_radius_fraction)
            feather = min(height, width) * self.edge_feather_fraction
            allowed_mask = None
            source_visible_mask = None
            protected_robot_mask = None
            if (
                self.mask_to_source_body_visible_pixels
                or self.protect_robot_visible_pixels
            ):
                segmentation = np.asarray(
                    self.sim.render(
                        camera_name=camera_name,
                        width=width,
                        height=height,
                        segmentation=True,
                    )
                )
                if self.observation_image_convention == "opencv":
                    segmentation = segmentation[::-1]
                if self.mask_to_source_body_visible_pixels:
                    source_visible_mask = np.isin(
                        segmentation[..., 1], self._source_geom_ids
                    )
                    allowed_mask = source_visible_mask
                if self.protect_robot_visible_pixels:
                    protected_robot_mask = np.isin(
                        segmentation[..., 1], self._protected_robot_geom_ids
                    )
                    robot_allowed = ~protected_robot_mask
                    allowed_mask = (
                        robot_allowed
                        if allowed_mask is None
                        else allowed_mask & robot_allowed
                    )
            blurred, mask_pixels = self._blur_image(
                clean,
                polygons,
                radius,
                feather,
                operator=self.blur_operator,
                privacy_cell_radius_ratio=self.privacy_cell_radius_ratio,
                privacy_soften_radius_ratio=self.privacy_soften_radius_ratio,
                privacy_blur_mode=self.privacy_blur_mode,
                allowed_mask=allowed_mask,
            )
            observable._current_observed_value = blurred
            self._inner._obs_cache[image_key] = blurred.copy()
            view_payloads[camera_name] = {
                "image_key": image_key,
                "image_shape": list(clean.shape),
                "polygons_xy": [
                    [[round(x, 3), round(y, 3)] for x, y in polygon]
                    for polygon in polygons
                ],
                "corner_depths_m": depths,
                "mask_pixels": mask_pixels,
                "mask_fraction": float(mask_pixels / (height * width)),
                "gaussian_radius_pixels": float(radius),
                "blur_operator": self.blur_operator,
                "privacy_cell_radius_ratio": self.privacy_cell_radius_ratio,
                "privacy_soften_radius_ratio": self.privacy_soften_radius_ratio,
                "privacy_blur_mode": self.privacy_blur_mode,
                "edge_feather_pixels": float(feather),
                "changed_pixels": int(
                    np.count_nonzero(np.any(blurred != clean, axis=-1))
                ),
                "source_visible_pixels": (
                    None
                    if source_visible_mask is None
                    else int(np.count_nonzero(source_visible_mask))
                ),
                "mask_restricted_to_source_visible_pixels": bool(
                    self.mask_to_source_body_visible_pixels
                ),
                "protected_robot_pixels": (
                    None
                    if protected_robot_mask is None
                    else int(np.count_nonzero(protected_robot_mask))
                ),
                "mask_excludes_robot_visible_pixels": bool(
                    self.protect_robot_visible_pixels
                ),
            }

        timestep = int(getattr(self._inner, "timestep", -1))
        record = {
            "observation_index": len(self._records),
            "sim_timestep": timestep,
            "part": self.part,
            "region_kind": self._region_kind,
            "world_regions": [region.round(9).tolist() for region in world_regions],
            "views": view_payloads,
            "rendering_path": (
                "robosuite_rgb_then_world_anchored_local_" + self.blur_operator
            ),
        }
        if self._records and self._last_recorded_timestep == timestep:
            record["observation_index"] = self._records[-1]["observation_index"]
            self._records[-1] = record
        else:
            self._records.append(record)
            self._last_recorded_timestep = timestep
        self._observation_update_count += 1

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

    def _reset_tracking(self) -> None:
        self._records.clear()
        self._last_recorded_timestep = None
        self._observation_update_count = 0
        self._episode_step_count = 0

    def reset(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        self._flush_automatic_metadata()
        self._reset_tracking()
        return self._env.reset(*args, **kwargs)

    def set_init_state(self, init_state: Any) -> dict[str, Any]:
        self._flush_automatic_metadata()
        self._reset_tracking()
        return self._env.set_init_state(init_state)

    def step(self, action: Any) -> tuple[dict[str, Any], Any, Any, Any]:
        result = self._env.step(action)
        self._episode_step_count += 1
        return result

    def validation_report(self) -> dict[str, Any]:
        current = self._model_invariants()
        return {
            "schema_version": "wam_syncbench.a3.blur.validation.v1",
            "suite": self.suite,
            "task_idx": self.task_idx,
            "spec_path": str(self.spec_path),
            "parts": list(self.parts),
            "views": list(self.views),
            "implementation": "world_anchored_projected_image_blur",
            "blur_operator": self.blur_operator,
            "privacy_cell_radius_ratio": self.privacy_cell_radius_ratio,
            "privacy_soften_radius_ratio": self.privacy_soften_radius_ratio,
            "privacy_blur_mode": self.privacy_blur_mode,
            "observation_image_convention": self.observation_image_convention,
            "mask_to_source_body_visible_pixels": self.mask_to_source_body_visible_pixels,
            "protect_robot_visible_pixels": self.protect_robot_visible_pixels,
            "source_geom_names": [
                self.sim.model.geom_id2name(geom_id)
                for geom_id in self._source_geom_ids
            ],
            "model_invariants": {
                "before": self.model_invariants,
                "after": current,
                "unchanged": current == self.model_invariants,
            },
            "task_region_manifest": (
                None if self._manifest_path is None else str(self._manifest_path)
            ),
            "task_config": self._task_config,
            "template": self._template,
            "scene_observation_count": len(self._records),
            "internal_blur_update_count": self._observation_update_count,
            "last_record": self._records[-1] if self._records else None,
            "simulator_geometry_added": 0,
            "simulator_state_mutations": 0,
            "source_rgba_mutations": 0,
            "auxiliary_segmentation_renders_per_observation": (
                len(self.views) if self.mask_to_source_body_visible_pixels else 0
            ),
            "returned_image_replacements": self._observation_update_count,
        }

    def write_a3_metadata(self, path: str | os.PathLike[str]) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": "wam_syncbench.a3.blur.parts.v1",
            "suite": self.suite,
            "task_idx": self.task_idx,
            "spec_path": str(self.spec_path),
            "parts": list(self.parts),
            "spec": self.spec,
            "summary": self.validation_report(),
            "frames": self._records,
        }
        temporary = output.with_suffix(output.suffix + ".tmp")
        temporary.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        temporary.replace(output)
        return output

    def close(self) -> Any:
        self._flush_automatic_metadata()
        close_fn = getattr(self._env, "close", None)
        return close_fn() if callable(close_fn) else None


def spec_uses_world_blur(path: str | os.PathLike[str]) -> bool:
    try:
        payload = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError):
        return False
    return payload.get("occlusion_space") == "world_anchored_camera_blur"
