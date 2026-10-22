"""Canonical perturbation-asset contracts for the public release.

The benchmark has two kinds of inputs: assets that are shipped with this
release and the unmodified LIBERO/model installation that users provide.  A
setting records which side it uses through this table instead of making each
runner invent its own path convention.
    """

from __future__ import annotations

from typing import Any


ASSET_CONTRACT_SCHEMA = "libero_vpro.asset_contract.v1"

_COMMON_EXTERNAL = [
    "unmodified LIBERO bddl_files and init_files",
    "MuJoCo and the selected model backend/checkpoint",
]


ASSET_CONTRACTS: dict[str, dict[str, Any]] = {
    "A1": {
        "asset_mode": "bundled_environment_assets",
        "asset_roots": ["assets/A1_target_object/precompiled_variants_full_surface_v3"],
        "external_dependencies": _COMMON_EXTERNAL,
    },
    "A2": {
        "asset_mode": "bundled_environment_assets",
        "asset_roots": ["assets/A2_target_receptacle/precompiled_variants_full_surface_v3"],
        "external_dependencies": _COMMON_EXTERNAL,
    },
    "A3": {
        "asset_mode": "bundled_protocol_specs",
        "asset_roots": ["source/A3_interaction_part_occlusion/specs"],
        "external_dependencies": _COMMON_EXTERNAL,
    },
    "A4": {
        "asset_mode": "runtime_observation_transform",
        "asset_roots": ["source/A4_temporary_observation_corruption/configs"],
        "external_dependencies": _COMMON_EXTERNAL,
    },
    "A5": {
        "asset_mode": "runtime_observation_transform",
        "asset_roots": [
            "source/A5_single_view",
            "source/A5_view_copy",
            "source/A5_blank_view_control",
            "source/A5_white_view_control",
        ],
        "external_dependencies": _COMMON_EXTERNAL,
    },
    "B1": {
        "asset_mode": "runtime_observation_transform",
        "asset_roots": ["source/B1_camera_staleness/configs"],
        "external_dependencies": _COMMON_EXTERNAL,
    },
    "C1": {
        "asset_mode": "runtime_observation_transform",
        "asset_roots": [
            "source/C1_cross_view_20260825/cases",
            "source/C1_cross_view_20260825/episode_manifests",
        ],
        "external_dependencies": _COMMON_EXTERNAL,
    },
    "C2": {
        "asset_mode": "runtime_feedback_transform",
        "asset_roots": ["paper_facing_configs/C2"],
        "external_dependencies": _COMMON_EXTERNAL,
    },
    "D1": {
        "asset_mode": "bundled_environment_assets_and_bddl",
        "asset_roots": [
            "assets/D1_task_relevant_distractors/assets",
            "assets/D1_task_relevant_distractors/bddl_overrides",
        ],
        "external_dependencies": _COMMON_EXTERNAL,
    },
    "D2": {
        "asset_mode": "bundled_environment_assets_and_bddl",
        "asset_roots": [
            "assets/D2_D3_state_variation/assets",
            "assets/D2_D3_state_variation/bddl_overrides",
        ],
        "external_dependencies": _COMMON_EXTERNAL,
    },
    "D3": {
        "asset_mode": "bundled_environment_assets_and_bddl",
        "asset_roots": [
            "assets/D2_D3_state_variation/assets",
            "assets/D2_D3_state_variation/bddl_overrides",
        ],
        "external_dependencies": _COMMON_EXTERNAL,
    },
    "D4": {
        "asset_mode": "bundled_environment_assets",
        "asset_roots": ["assets/D4_target_geometry/assets"],
        "external_dependencies": _COMMON_EXTERNAL,
    },
}


def contract_for(category: str) -> dict[str, Any]:
    """Return a copy so callers cannot mutate the release-wide contract."""

    try:
        contract = ASSET_CONTRACTS[category]
    except KeyError as exc:
        raise KeyError(f"no asset contract for category {category!r}") from exc
    return {
        "asset_contract_schema": ASSET_CONTRACT_SCHEMA,
        **contract,
        "asset_roots": list(contract["asset_roots"]),
        "external_dependencies": list(contract["external_dependencies"]),
    }
