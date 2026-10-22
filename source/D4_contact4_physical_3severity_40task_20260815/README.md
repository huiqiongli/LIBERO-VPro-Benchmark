# D4 paper-facing configuration

The paper-facing D4 expansion is 4 deformation families (`taper`, `multipeak`,
`twist`, `asymmetric`) times 3 severity levels (`low`, `medium`, `high`). The
case manifest contains 36 eligible LIBERO tasks per leaf setting. Three
LIBERO-Goal tasks have no grasped target; a fourth is excluded because outward
contact-band deformation makes the initial bowl pose intersect its drawer.
This gives `12 x 36 = 432` unique task-condition cases.

`manifest.json` records one definition per case. Evaluation expands each case
over ten initial-state trials.
