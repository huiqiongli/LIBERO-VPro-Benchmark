# A1: physical target-object surface occlusion

The settings `A1.ratio_25pct`, `ratio_50pct`, `ratio_75pct`, and
`ratio_100pct` blacken the corresponding fraction of the manipulated object's
**complete 3-D rendering surface area**, including initially hidden, rear,
inner, and already dark surfaces. They cover 40 tasks per setting.

For partial ratios, precompiled XML adds massless, non-colliding black surface
overlays while preserving original materials and visual geometry. The 100%
setting blackens all target visual geometry. The selection follows an
agentview-aligned surface partition, but its denominator is 3-D area, not
visible image pixels. Each view therefore sees the projection of the same
physical surface, which moves with the object.

Targets are task-specific; multi-target cases apply the ratio to every declared
target. Collision geometry, mass, pose, joints, sites, instructions and success
conditions are unchanged. This is not an additional image-space mask.

Canonical XML and area metadata are under
`assets/A1_target_object/precompiled_variants_full_surface_v3/`.
The same local surface partition is reused across initial-state trials.

From the release root:

```bash
python3 scripts/run_eval.py --setting A1.ratio_25pct --model fastwam --check
```
