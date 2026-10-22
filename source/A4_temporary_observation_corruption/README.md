# A4: temporary observation corruption

A4 changes both policy camera inputs without editing simulator physics.
The 21 settings comprise ten full-frame black/noise schedules, ten
phase-triggered black/noise schedules, and target-region Gaussian noise.

## Fixed schedules

All indices are zero-based policy replans, not simulator frames.

| Suffix | Active replan indices |
| --- | --- |
| `N1` | `[2]` |
| `N2` | `[2,5]` |
| `N3` | `[2,5,8]` |
| `D2` | `[2,3]` |
| `D3` | `[2,3,4]` |

Each suffix is available as `A4.black_<suffix>` and
`A4.noise_<suffix>`. N counts discrete events; D specifies a contiguous
duration. Black sets all selected RGB pixels to zero. Full-frame noise uses
the Gaussian distribution and deterministic seed in its observation config.

## Phase-triggered schedules

The five target phases are `reach`, `pre_grasp`, `grasp`, `transport`,
and `pre_place`, each with black and noise variants named
`A4.<black|noise>_<phase>_3replan_once`.

The FastWAM gate starts when a **replan boundary** first observes the requested
phase. It then corrupts that replan and the next two, even if the phase changes,
and does not retrigger. Phase detection updates after each control step; a
short phase entirely between replans can therefore be missed. Episode
termination can truncate the three-replan window or its clean recovery input.
The schedule does not extend the episode.

## Target-region noise

`A4.target_region_noise` uses
`paper_facing_configs/A4/target_region_noise.json` at indices `[2,5,8]`.
It adds Gaussian noise only inside resolved target-instance segmentation masks;
pixels outside them remain unchanged. Both views are selected. The config's
explicit per-task target override takes precedence over `obj_of_interest`;
LIBERO-Goal task 0 selects `wooden_cabinet_1`. Symbolic BDDL regions are not
automatically expanded to an entire parent object.

From the release root:

```bash
python3 scripts/run_eval.py --setting A4.black_N3 --model fastwam --check
python3 scripts/run_eval.py --setting A4.noise_transport_3replan_once --model fastwam --plan
```

Current runtime hooks are in `source/_shared_core/` and the bundled FastWAM
reference evaluator.
