"""Process-local environment hooks for an unmodified upstream LIBERO install."""

from __future__ import annotations

import functools
import inspect
import json
import os
from pathlib import Path


def install_object_xml_overrides() -> None:
    """Apply explicit object XML overrides at the standard robosuite constructor.

    This replaces the benchmark-specific base_object.py modification formerly
    required in the external LIBERO checkout. With no map the constructor is
    unchanged; all hook state is local to the evaluation process.
    """
    from robosuite.models.objects import MujocoXMLObject

    original = MujocoXMLObject.__init__
    if getattr(original, "_libero_vpro_xml_overrides", False):
        return
    signature = inspect.signature(original)

    @functools.wraps(original)
    def initialize(self, *args, **kwargs):
        raw = os.environ.get("LIBERO_OBJECT_XML_OVERRIDE_MAP", "")
        if raw:
            mapping = json.loads(raw)
            if not isinstance(mapping, dict):
                raise ValueError("LIBERO_OBJECT_XML_OVERRIDE_MAP must be an object")
            bound = signature.bind(self, *args, **kwargs)
            name = str(bound.arguments.get("name", ""))
            replacement = mapping.get(name)
            if replacement is None:
                candidates = [value for key, value in mapping.items() if str(key).endswith(":" + name)]
                if len(candidates) > 1:
                    raise ValueError(f"ambiguous XML override for {name}")
                replacement = candidates[0] if candidates else None
            if isinstance(replacement, dict):
                replacement = replacement.get("xml_path")
            if replacement:
                path = Path(replacement).expanduser()
                if not path.is_file():
                    raise FileNotFoundError(f"object XML override does not exist: {path}")
                bound.arguments["fname"] = str(path)
                return original(*bound.args, **bound.kwargs)
        return original(self, *args, **kwargs)

    initialize._libero_vpro_xml_overrides = True
    MujocoXMLObject.__init__ = initialize
    install_benchmark_objects()


def install_benchmark_objects() -> None:
    """Register the two custom BDDL categories with their original XML assets."""
    from robosuite.models.objects import MujocoXMLObject
    from libero.libero.envs.base_object import OBJECTS_DICT

    asset_root = Path(__file__).resolve().parents[2] / "assets/D1_task_relevant_distractors/assets/default_objects"

    class BenchmarkObject(MujocoXMLObject):
        category = ""

        def __init__(self, name=None, obj_name=None, joints=None):
            category = self.category
            super().__init__(str(asset_root / f"{category}.xml"), name=name or category,
                             joints=[dict(type="free", damping="0.0005")] if joints is None else joints,
                             obj_type="all", duplicate_collision_geoms=False)
            self.category_name = category
            self.rotation = (0.0, 0.0)
            self.rotation_axis = "z"
            self.object_properties = {"vis_site_names": {}}

    class PaperRoll(BenchmarkObject):
        category = "paper_roll"

    class TrajectoryBlock(BenchmarkObject):
        category = "trajectory_block"

    OBJECTS_DICT.update(paper_roll=PaperRoll, trajectory_block=TrajectoryBlock)
