# Canonical perturbation assets

This directory contains the LIBERO-VPro environment assets referenced by the
`asset_mode` and `asset_roots` fields in each setting configuration.

| Directory | Paper categories | Contents |
| --- | --- | --- |
| `A1_target_object/` | A1 | precompiled target-object XML variants |
| `A2_target_receptacle/` | A2 | precompiled target-receptacle XML variants |
| `D1_task_relevant_distractors/` | D1 | scene-variation BDDL overrides and custom XML/mesh/texture assets |
| `D2_D3_state_variation/` | D2, D3 | state-variation meshes/XML and BDDL overrides |
| `D4_target_geometry/` | D4 | contact-geometry XML assets |

The A4, A5, B1, C1, and C2 observation transforms have protocol files and case
manifests under `source/` and `paper_facing_configs/`.

Manifests reference these canonical directories directly. Relative symlinks
within the asset tree share identical meshes and textures; preserve them when
copying or archiving the release.
