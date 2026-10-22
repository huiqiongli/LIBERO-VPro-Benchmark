# Clean LIBERO baseline

The baseline covers the four suites `libero_object`, `libero_spatial`,
`libero_goal` and `libero_10`: ten tasks per suite and ten official
initial-state trials per task.

It uses official clean initialization and both agentview and wrist cameras,
without scene, camera, observation or timing perturbations. Baseline trials
do not add task-condition cases to the 3,296-case perturbation inventory.

The case definitions and FastWAM reference worker are bundled here.
Model code, checkpoints and evaluation outputs are external. The perturbation
settings and their public entry point are documented in the release README.
