# A5: duplicated-view inputs

These settings retain the two-camera input structure but copy one real view
into both policy slots:

- `A5.agentview_copy_to_dual`: `(agentview, agentview)`.
- `A5.wrist_copy_to_dual`: `(wrist, wrist)`.

They differ from direct single-view evaluation: two input slots remain.
Simulator observations and physics are not changed. Both settings apply to all
40 tasks and are included in the 96-setting release.

From the release root:

```bash
python3 scripts/run_eval.py --setting A5.agentview_copy_to_dual --model fastwam --check
```
