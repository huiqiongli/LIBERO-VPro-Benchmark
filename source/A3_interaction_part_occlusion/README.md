# A3: Interaction-Cue Masking

The settings are `gripper`, `target_contact`, `fixture_contact`, and
`bounded_receiver`. Their configurations specify applicable tasks and fixed
region definitions.

The canonical `A3_blur_*_candidate.json` specs apply world-anchored image blur.
They alter observations without adding collision geometry or changing the
simulator state. Region position, orientation and extent are defined in
`specs/regions/`; the gripper spec uses its existing geometry reference.
The runtime loads these region definitions.

From the release root:

```bash
python3 scripts/run_eval.py --setting A3.gripper --model fastwam --check
python3 scripts/run_eval.py --setting A3.gripper --model fastwam \
  --suite libero_10 --task 2 --trials 1 --gpu 0
```

Task filters select from the applicable tasks in the setting manifest.
