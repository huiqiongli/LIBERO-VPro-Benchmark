# D4 paper-facing configuration

The paper-facing D4 expansion is 4 deformation families (`taper`, `multipeak`,
`twist`, `asymmetric`) times 3 severity levels (`low`, `medium`, `high`). The
case manifest contains 36 eligible LIBERO tasks per leaf setting. Three
LIBERO-Goal tasks have no grasped target; a fourth is excluded because outward
contact-band deformation makes the initial bowl pose intersect its drawer.
This gives `12 x 36 = 432` unique task-condition cases.

`manifest.json` is a case-definition manifest and records one generator entry
per case. The formal result archive evaluates each accepted case over the ten
initial-state trials specified by the paper protocol; the one-entry generator
field must not be interpreted as a one-trial paper result.
