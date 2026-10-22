# A5: direct single-view inputs

These settings supply exactly one camera stream to the policy:

- `A5.agent_only_direct`: third-person agentview only.
- `A5.wrist_only_direct`: wrist / eye-in-hand only.

No image is duplicated or replaced with a blank tensor. The FastWAM reference
adapter selects one camera input and one output camera instead of the normal
two-camera layout. Other adapters must accept one camera stream. Two-slot copy
and fill inputs are separate settings.

Both settings apply to all 40 tasks. From the release root:

```bash
python3 scripts/run_eval.py --setting A5.agent_only_direct --model fastwam --check
```

The eight A5 settings also include duplicated-view, black-fill, and white-fill
controls, documented in the adjacent A5 directories.
