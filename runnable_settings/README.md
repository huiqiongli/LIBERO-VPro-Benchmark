# Setting contracts and reference adapter

`index.json` lists the 96 settings from `PAPER_FACING_SETTINGS.json`.
Each `settings/<category>/<setting>/` directory contains a model-independent
`setting.json` and a convenience `run.sh` for the FastWAM reference adapter.
Task manifests are included where a setting has a restricted task inventory.
Backend-specific batch manifests adapt cases to the reference evaluator;
they do not change the dataset's task-condition case definition.

From this directory:

```bash
./settings/A1/ratio_25pct/run.sh --check
./settings/C1/count.agent/run.sh --suite libero_10 --plan
./settings/A3/gripper/run.sh --suite libero_10 --task 2 --trials 1 --trial-start 0 --gpu 0
```

Default evaluation uses ten trials on every applicable task. `--suite`, `--task`,
`--trials`, and `--trial-start` narrow a run without editing the dataset.
`--output-root` selects a separate result directory. Checks and plans are
read-only setup inspections; see the release README for their validation scope.

A4 `target_region_noise` corrupts the target-object segmentation region in both
views at replans `[2,5,8]`. N1/N2/N3 count discrete corruption events; D2/D3
specify contiguous durations. The five phase-targeted conditions retain their
approach-to-pre-placement definitions for both blackout and noise. A phase
pulse starts only when a replan boundary observes the target phase, then lasts
three consecutive replans regardless of later phase changes. A phase between
replans may be missed, and episode termination may truncate the window.

`build_runnable_settings.py` regenerates derived files from source definitions.
It is a write operation, not a preflight check. Keep experimental outputs
outside this directory.
