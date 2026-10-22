# D3 V5: paired-state visual evidence

D3 combines visual evidence from two plausible simulator endpoints while
physics, rewards, success predicates, instructions, proprioception and actions
remain tied to the active truth state. The public release selects the truth-A
obfuscated articulation and object-support settings described in [README.md](README.md).

## Observation operator

1. Calibrate both endpoints from the same official initial-state index.
2. Build one state-invariant world-space evidence volume from both endpoints.
3. At each replan, snapshot the complete simulator integration state and
   wrapper clocks; install the alternate endpoint only for rendering.
4. Render both official cameras at the same robot/camera pose without advancing
   physics, then restore the active truth state and observable cache.
5. Inside the projected volume, use the same integer-symmetric 50/50 RGB average.
   Outside it, retain active-truth RGB. Preserve every non-RGB observation.

The alternate endpoint never supplies control state or success evaluation.
Both camera masks derive from the same world-space volume.

## Endpoint constraints

Articulation endpoints stay within the task's calibrated mechanical range.
Object-support endpoints depend on morphology:

| Morphology | Alternate support state |
| --- | --- |
| Axial container | Lying on its side |
| Handled vessel | Side-tipped; never inverted base-up |
| Open vessel | Rim-down |
| Box | Laid on a side |
| Planar object | Face-down |

Endpoints must be stable, avoid penetration and reset-time success, and
preserve task feasibility. Source occupancy definitions additionally use
receiver-specific neutral objects, calibrated to fit without blocking every
valid placement; those definitions are outside the canonical D3 subset.

## Required invariants

- Swapping endpoint order leaves the in-volume mixture unchanged.
- Pixels outside the evidence volume and non-RGB fields remain unchanged.
- Simulator arrays and wrapper clocks match exactly after private rendering.
- The camera pose and robot state used by both renders are identical.
- Visible state evidence must actually change; a valid configuration alone
  does not prove that a cue is visible in a particular frame.

The D3 setting manifests and `scripts/run_eval.py` provide the corresponding
case lists and evaluation entry point.
