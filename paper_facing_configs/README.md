# Paper-facing configuration contract

`PAPER_FACING_SETTINGS.json` is the single index for the 96 unique leaf settings in
the paper-facing statistical hierarchy. The executable implementation remains
in `source/`; this directory contains only configuration entries that need an
explicit paper-scope note.

The A4 `target_region_noise.json` entry applies Gaussian noise only to the
target-object segmentation region in both views. It uses the fixed discrete
replan schedule `[2,5,8]`; it is separate from the
full-frame black/noise implementation under `source/A4_temporary_observation_corruption/configs/`.

Explicit per-task target overrides take precedence over `obj_of_interest`.
LIBERO-Goal task 0 selects `wooden_cabinet_1`; symbolic BDDL regions are not
automatically interpreted as complete parent-object segmentation masks.

The paper-facing D2/D3 subset excludes occupancy cases in the broader source
manifests. Aggregate statistics use the selected subset.
