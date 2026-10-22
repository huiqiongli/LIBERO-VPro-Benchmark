# Settings and trial manifests

`index.json` lists the 96 settings from `PAPER_FACING_SETTINGS.json`.
Each `settings/<category>/<setting>/` directory contains a `setting.json` with
the perturbation configuration. Task and trial manifests list the applicable
cases. The FastWAM example uses the central evaluation script.

From the release root:

```bash
python3 scripts/run_eval.py --setting A1.ratio_25pct --model fastwam --check
python3 scripts/run_eval.py --setting C1.count.agent --model fastwam --suite libero_10 --plan
python3 scripts/run_eval.py --setting A3.gripper --model fastwam --suite libero_10 --task 2 --trials 1 --trial-start 0 --gpu 0
```

Default evaluation uses ten trials on every applicable task. `--suite`, `--task`,
`--trials`, and `--trial-start` select a subset; `--output-root` selects the
result directory.

A4 `target_region_noise` corrupts the target-object segmentation region in both
views at replans `[2,5,8]`. N1/N2/N3 count discrete corruption events; D2/D3
specify contiguous durations. The five phase-targeted conditions retain their
approach-to-pre-placement definitions for both blackout and noise. A phase
pulse starts only when a replan boundary observes the target phase, then lasts
three consecutive replans regardless of later phase changes. A phase between
replans may be missed, and episode termination may truncate the window.

`build_runnable_settings.py` regenerates the contracts and manifests from the
source definitions.
