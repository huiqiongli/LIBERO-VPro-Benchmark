# Reference evaluator provenance

`eval_libero_single.py`, `libero_utils.py`, and `action_ensembler.py` originate
from the FastWAM evaluation implementation used for this benchmark. Its MIT
license is preserved in `LICENSE`. The model, processor, Hydra configurations,
and weights remain in the separately installed FastWAM checkout.

The release adaptations load perturbation and recording code from this package,
resolve external paths from explicit runtime variables, and install the object
XML hook in the current process. No external repository files are modified.

`../_shared_core/robust_libero_perturbations.py` is the benchmark's shared
perturbation implementation, previously stored at the simulation workspace
root. `../_shared_core/perturbation_schedule.py` is the benchmark temporal
schedule implementation previously colocated with LingBot-VA's LIBERO client;
the existing upstream Apache-2.0 notice is preserved as
`../_shared_core/LICENSE.lingbot-va`. These notices do not select a license for
the benchmark authors' original modifications or for benchmark assets.

The `paper_roll` and `trajectory_block` default XML files under
`assets/D1_task_relevant_distractors/assets/default_objects/` are exact copies of the
benchmark-specific objects formerly installed in LIBERO's
`assets/a4_distractor_objects/`. Their constructor attributes are preserved in
`source/_shared_core/libero_hooks.py`; no new geometry or perturbation values
were introduced by packaging them.
