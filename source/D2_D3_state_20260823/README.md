# D2/D3: task-relevant state variation

D2 changes the real initial scene state, visible in both cameras. D3 keeps the
active physical truth while presenting equal visual evidence for two plausible
states through the V5 observation operator.

## Canonical scope

The release contains **96 settings and 3,296 task-condition cases** overall.
Within that inventory:

- D2 has eight settings and 80 cases: the applicable opening fractions
  (`open_5pct`, `open_12pct`, `open_20pct`, `open_50pct`,
  `open_85pct`, `open_90pct`) and adjacent/opposite support faces.
- D3 has two settings and 45 cases: articulation-state and object-support-state
  ambiguity, both `truth_a_obfuscated`.

Applicability is task-specific. The public setting manifests select these
cases from the broader source definitions. Occupancy cases and other paired
conditions in the source manifests are outside the canonical totals. Runtime
preflight also resolves these additional source definitions.

## State and observation contract

D2 uses calibrated articulation or stable object-support endpoints.
Instructions and success predicates remain unchanged.

For D3, truth A is the official initial state. A private render branch installs
truth B at the same robot/camera pose, renders both cameras without advancing
an environment control step, and restores the complete simulator state and
wrapper clocks. Inside a shared world-space evidence volume, the two RGB
renders receive integer-symmetric 50/50 exposure. Outside the volume and in
non-RGB observations, active truth is preserved. Physics, reward, success,
proprioception and control always use the active truth.

Endpoint and rendering requirements are detailed in [D3_REDESIGN.md](D3_REDESIGN.md).
The common implementation is `experiment_env.PreparedEnvironment`, with
state helpers in `state_runtime.py`.

From the release root:

```bash
python3 scripts/run_eval.py --setting D2.adjacent_support_face --model fastwam --check
python3 scripts/run_eval.py --setting D3.articulation_state_ambiguity.truth_a_obfuscated --model fastwam --plan
```
