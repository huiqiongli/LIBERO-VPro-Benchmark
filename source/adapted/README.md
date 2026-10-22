# Task-specific observation and schedule configurations

This directory contains task-specific target mappings, observation specs and
decision schedules used by the release. The authoritative selection is each
`runnable_settings/settings/<category>/<setting>/setting.json`, not a filename
prefix in this directory.

A decision is one policy inference/action-chunk prediction. Decision indices
are zero-based; their control-step spacing depends on the adapter. The
FastWAM adapter applies observation corruption immediately before inference.

Segmentation-mask coverage in these observation specs is image-space
coverage. Do not confuse it with the complete 3-D surface-area ratios used by
the canonical physical A1/A2 ratio settings. Likewise, source filenames using
older category labels do not rename the current B1 camera-staleness or C1
cross-view settings.

The `phase_calibrated/` files describe model-specific clean-trajectory
calibration. They are not independent evaluation entry points. Use
`scripts/run_eval.py` to resolve the current setting contract.
