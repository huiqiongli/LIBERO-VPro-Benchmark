# A2: target-receptacle occlusion timing

These settings mask the target receptacle in both camera observations.
Placement phase detection follows the manipulated object, not the receptacle;
the canonical schedules provide `phase_targets_by_task` separately from the
observation targets.

| Setting | Current schedule |
| --- | --- |
| `A2.initially_visible_then_occluded` | Clean at replan indices 0–3; masked from index 4 |
| `A2.initially_occluded_then_revealed` | Masked at replan indices 0–3; clean from index 4 |
| `A2.pre_place` | Mask at replans whose detected phase is `pre_place` |
| `A2.post_place` | After a detected pre-place phase, mask during the first 40 control steps following release |

Replan indices are zero-based: index 4 is the fifth policy decision. The
canonical configurations are in `configs/replan5/`; the setting contracts
select them explicitly.

The FastWAM adapter updates phase state after each control step and evaluates
the observation gate only at replans. Post-place does not trigger on an
arbitrary release without a preceding pre-place phase. Success can terminate
an episode before release, before a full window, or before a recovery input;
no extra actions are implied by the schedule.

From the release root:

```bash
python3 scripts/run_eval.py --setting A2.post_place --model fastwam --check
```
