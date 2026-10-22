# C1: cross-view contradiction

This implementation creates existence, position, state, and count
contradictions in one camera stream while preserving the other camera's
physical observation. The canonical settings and runnable entry
points are indexed under `runnable_settings/settings/C1/`.

## Case inventory

- Existence contradiction: 38 semantic cases, 76 camera-direction specs.
- Position contradiction: 38 semantic cases, 76 camera-direction specs.
- State contradiction: 7 semantic cases, 14 camera-direction specs.
- Count contradiction: 12 semantic cases, 24 camera-direction specs.
- Total: 95 semantic cases, 190 camera-direction specs.

Each direction edits exactly one camera in a temporary simulator branch and
leaves the other camera as physical truth. The branch is restored before the
real rollout advances. Each case specifies its `corrupted_view` and `truth_view`.
The case definition describes the intended scene edit; visual effectiveness
depends on the edited cue being visible in the corrupted view. Initial
visibility can vary with camera pose.

Articulated-state contradictions normally mirror the current joint value,
while the microwave case selects the farther joint limit to create a state
difference.

From the release root:

```bash
python3 scripts/run_eval.py --setting C1.count.agent --model fastwam --check
```

