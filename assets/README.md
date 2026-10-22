# Canonical perturbation assets

This directory is the single source of truth for modified LIBERO environment
assets shipped by LIBERO-VPro.  The setting-level contract is recorded in each
`runnable_settings/**/setting.json` through `asset_mode` and `asset_roots`.

| Directory | Paper categories | Contents |
| --- | --- | --- |
| `A1_target_object/` | A1 | precompiled target-object XML variants |
| `A2_target_receptacle/` | A2 | precompiled target-receptacle XML variants |
| `D1_task_relevant_distractors/` | D1 | scene-variation BDDL overrides and custom XML/mesh/texture assets |
| `D2_D3_state_variation/` | D2, D3 | state-variation meshes/XML and BDDL overrides |
| `D4_target_geometry/` | D4 | contact-geometry XML assets |

The A4, A5, B1, C1, and C2 conditions are observation/runtime transformations.  Their
protocol files and case manifests remain under `source/` and are referenced by
the same contract; they do not require a second physical-asset tree.

Manifests reference these canonical directories directly. Relative symlinks
within the asset tree share identical meshes and textures; preserve them when
copying or archiving the release. No second asset copy under `source/` is needed.
