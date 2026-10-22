# A1: target-object occlusion timing

These four settings mask the task's manipulated-object segmentation region in
both camera observations. They are distinct from A1's physical surface-ratio
settings.

| Setting | Current schedule |
| --- | --- |
| `A1.initially_visible_then_occluded` | Clean before timestep 80; masked from timestep 80 |
| `A1.initially_occluded_then_revealed` | Masked before timestep 80; clean from timestep 80 |
| `A1.pre_grasp` | Mask while the detector reports pre-grasp, before the grasp latch |
| `A1.post_grasp` | Mask during the 40-control-step window after the first grasp latch |

The FastWAM adapter evaluates observation gates at replan boundaries. Timestep
windows use the rollout control-step counter, including its initial wait.
Phase detection runs after control steps using simulator state, independently
of the corrupted images. A phase may occur between replans without a policy
input sampling it; an episode may finish before a window or recovery frame.

Use the setting contract to select the schedule and task-specific targets.
Use the configured detection targets and thresholds consistently across policy
comparisons.

From the release root:

```bash
python3 scripts/run_eval.py --setting A1.pre_grasp --model fastwam --check
```

The shared detector and gate are in `source/_shared_core/`; the public runner
is `scripts/run_eval.py`.
