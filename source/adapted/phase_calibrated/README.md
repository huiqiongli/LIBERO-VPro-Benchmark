# Model-specific phase calibration

These files derive decision windows from successful clean trajectories.
Indices are zero-based model inference/decision indices, not simulator steps.

- `grasp_*` uses first target finger-pad contact, with the grasp latch as fallback.
- Goal tasks 0, 5 and 7 use `interaction_*` for drawer, pushing and stove contact.
- `completion_before_*` ends before LIBERO reports success.
- No `completion_after_*` window is emitted: success terminates the episode.
- A window requires exactly N distinct decisions; unavailable windows are
  recorded in `events.json`.

Model-specific schedules account for different action-chunk lengths. The
`decision_indices` and `random_replans.indices` fields describe the same
selected decisions for adapters accepting either form.

These are calibration data, not public launchers. Use the current setting
contracts and `scripts/run_eval.py` to select an evaluation schedule.
