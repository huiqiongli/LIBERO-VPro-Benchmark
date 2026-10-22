# A5: white-filled missing view

These settings retain the two-camera input structure and replace one complete
policy camera input with RGB `[255,255,255]` at every decision:

- `A5.agent_plus_white_wrist`: real agentview, white wrist.
- `A5.wrist_plus_white_agent`: white agentview, real wrist.

Simulator observations and physics remain unchanged. Both settings apply to
all 40 tasks and are included in the 96-setting release.

From the release root:

```bash
python3 scripts/run_eval.py --setting A5.agent_plus_white_wrist --model fastwam --check
```
