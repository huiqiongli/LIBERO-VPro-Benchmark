# A2: physical target-receptacle surface occlusion

The settings `A2.ratio_25pct`, `ratio_50pct`, `ratio_75pct`, and
`ratio_100pct` blacken the corresponding fraction of the target receptacle's
**complete 3-D rendering surface area**, including initially hidden, rear,
inner, and already dark surfaces. They cover 40 tasks per setting.

Partial ratios use precompiled, massless, non-colliding black surface overlays.
The partition is aligned with agentview, but the ratio is solved from complete
3-D surface area rather than initial visible pixels. Original materials and
visual geometry stay intact outside the overlay. The 100% setting blackens all
target visual geometry.

Both cameras render the same modified physical surface. Collision geometry,
mass, pose, joints, sites, instructions and success conditions are unchanged;
no second image-space mask is applied.

Canonical XML and area metadata are under
`assets/A2_target_receptacle/precompiled_variants_full_surface_v3/`.
Runtime checks verify that the expected overlay is loaded.

From the release root:

```bash
python3 scripts/run_eval.py --setting A2.ratio_25pct --model fastwam --check
```
