# B1: camera staleness

B1 replaces one policy camera stream with its own earlier observations while
the other stream stays current. The ten settings combine five modes with
agentview or wrist: `B1.delay1_agent`, `B1.delay3_agent`,
`B1.delay5_agent`, `B1.freeze_agent`, `B1.replay_agent`, and their
`_wrist` counterparts.

Staleness is measured in zero-based policy decisions/replans, not simulator
steps. Each adapter must advance its history once per replan.

| Mode | Source observation |
| --- | --- |
| Delay 1/3/5 | Replan index minus the lag, clamped to index 0 during warm-up |
| Freeze | Current through index 2; index 2 thereafter |
| Replay | Record indices 1–3, then loop that fixed segment from index 4 |

For replay the source sequence is `0,1,2,3,1,2,3,1,2,…`. For every mode,
the unselected view uses the current observation. Simulator state is unchanged.

Source-decision indices in rollout logs record the applied lag.

From the release root:

```bash
python3 scripts/run_eval.py --setting B1.delay3_agent --model fastwam --check
```
