# Setting configurations

`PAPER_FACING_SETTINGS.json` indexes the 96 settings. This directory contains
the A1/A2 case inventories and configuration files for A4 and C2. Runtime
implementations are under `source/`.

The A4 `target_region_noise.json` entry applies Gaussian noise only to the
target-object segmentation region in both views. It uses the fixed discrete
replan schedule `[2,5,8]`; it is separate from the
full-frame black/noise implementation under `source/A4_temporary_observation_corruption/configs/`.

Explicit per-task target overrides take precedence over `obj_of_interest`.
LIBERO-Goal task 0 selects `wooden_cabinet_1`; symbolic BDDL regions are not
automatically interpreted as complete parent-object segmentation masks.

D2 covers articulation and object-support variation. D3 covers truth-A
obfuscated articulation and object-support ambiguity.
