# Timing manifests and task mappings

These manifests associate suite/task cases with observation targets and timing
specifications. `runnable_settings/settings/` lists each setting's selected
observation and timing configurations.

Decision/replan indices are zero-based policy-inference indices, not simulator
steps. A1's time windows use control timesteps, while the canonical A2
initial-visibility settings switch at replan index 4 (the fifth decision).
A2 post-place requires pre-place followed by release and counts a 40-control-step
window. See the A1/A2 timing READMEs and the exact `config_files` selected by
each setting.

Manipulated-object phases and receptacle phases use separate triggers. A
phase-triggered window begins when the rollout reaches that phase at a replan.
