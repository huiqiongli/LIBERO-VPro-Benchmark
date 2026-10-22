# Timing manifests and task mappings

These manifests associate suite/task cases with observation targets and timing
specifications. Current public setting IDs and selected configurations are
defined by `runnable_settings/settings/`; do not select a schedule solely by
its source filename.

Decision/replan indices are zero-based policy-inference indices, not simulator
steps. A1's time windows use control timesteps, while the canonical A2
initial-visibility settings switch at replan index 4 (the fifth decision).
A2 post-place requires pre-place followed by release and counts a 40-control-step
window. See the A1/A2 timing READMEs and the exact `config_files` selected by
each setting.

Manipulated-object phase targets and masked receptacle targets are distinct.
A task's presence in a manifest does not guarantee that a successful trajectory
will encounter every phase, provide a full corruption window, or contain a
post-window recovery input.
