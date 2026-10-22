# A5: black-filled missing view

These settings retain the two-camera input structure and replace one complete
policy camera input with RGB `[0,0,0]` at every decision:

- `A5.agent_plus_black_wrist`: real agentview, black wrist.
- `A5.wrist_plus_black_agent`: black agentview, real wrist.

Simulator observations and physics remain unchanged. Unlike direct single-view
evaluation, the filled second input is still supplied. Both settings apply to
all 40 tasks and are included in the 96-setting release.

From the release root:

```bash
python3 scripts/run_eval.py --setting A5.agent_plus_black_wrist --model fastwam --check
```
