# Reference evaluator provenance

`eval_libero_single.py`, `libero_utils.py`, and `action_ensembler.py` originate
from the FastWAM evaluation implementation used for this benchmark. Its MIT
license is preserved in `LICENSE`. The model, processor, Hydra configurations,
and weights remain in the separately installed FastWAM checkout.

The release adaptations load perturbation and recording code from this package,
resolve runtime paths from environment variables, and install the object XML
hook in the evaluation process.

`../_shared_core/robust_libero_perturbations.py` implements shared visual
perturbations; `../_shared_core/perturbation_schedule.py` implements their
temporal schedules. Project licenses are listed in the release README.

The `paper_roll` and `trajectory_block` default XML files under
`assets/D1_task_relevant_distractors/assets/default_objects/` are exact copies of the
benchmark-specific objects formerly installed in LIBERO's
`assets/a4_distractor_objects/`. Their constructor attributes are preserved in
`source/_shared_core/libero_hooks.py`.
