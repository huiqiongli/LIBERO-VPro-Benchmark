# LIBERO-VPro

LIBERO-VPro provides model-independent visual perturbation definitions and
environment assets for LIBERO: four dimensions, 12 categories, 96 settings,
and 3,296 task-condition cases. Trial repetitions do not create new cases.

| Dimension | Categories |
| --- | --- |
| Visual Evidence Degradation | A1–A5 |
| Camera Staleness | B1 |
| Visual Source Consistency | C1–C2 |
| Task-Relevant Scene Variation | D1–D4 |

`PAPER_FACING_SETTINGS.json` defines the taxonomy;
`runnable_settings/settings/` contains setting contracts and task manifests.
`assets/` holds modified XML, meshes, textures and BDDL files. `source/`
contains observation transforms, timing logic and reference evaluation code.

Setting definitions are model-independent. The public runner provides a
FastWAM reference adapter; other policies require adapters implementing the
same observation and environment contracts.

## Installation

Install the small release interface from the full checkout using Python 3.9 or
newer. Keep the checkout available: configs and assets are not bundled into a
standalone Python wheel.

```bash
python3 -m pip install -e /path/to/release
```

LIBERO, MuJoCo, FastWAM's model package, model checkpoints and their compatible
runtime environment are external prerequisites. Benchmark-specific evaluation
and perturbation code is bundled in this checkout.

Configure external paths for the FastWAM reference adapter:

```bash
export LIBERO_VPRO_LIBERO_ROOT=/path/to/LIBERO
export LIBERO_VPRO_DATA_ROOT=/path/to/LIBERO/libero/libero
export LIBERO_VPRO_LIBERO_ASSETS_ROOT=/path/to/LIBERO/libero/libero/assets
export FASTWAM_ROOT=/path/to/FastWAM
export FASTWAM_PYTHON=/path/to/fastwam-environment/bin/python
export FASTWAM_CHECKPOINT=/path/to/libero_uncond_2cam224.pt
export FASTWAM_DATASET_STATS=/path/to/libero_uncond_2cam224_dataset_stats.json
export LIBERO_VPRO_OUTPUT_ROOT=/path/to/results
```

The data root contains `bddl_files/` and `init_files/`. This release does not
distribute the complete upstream LIBERO environment; it does include derived
assets and some shared upstream meshes. For an installation with `Repos/FastWAM`, `Repos/LIBERO`
and `.venvs/fastwam` under one directory, `LIBERO_VPRO_SIM_ROOT` supplies
defaults; explicit paths take priority. Paths can also be set in
`evaluation_config.yaml` for LIBERO paths; FastWAM paths are supplied through
the exported adapter environment variables above.

After copying the release, rebase manifest and XML paths:

```bash
cd /path/to/release
python3 scripts/relocate_paths.py --write
python3 scripts/check_assets.py --strict
```

For a subsequent move from a custom directory name, pass
`--old-root /previous/package/path` to the relocator. Without `--write`,
it only previews changes. Preserve relative symlinks when copying assets.

An external LIBERO asset archive can optionally be extracted with
`scripts/download_assets.sh` by setting `LIBERO_VPRO_ASSET_ARCHIVE` or
`LIBERO_VPRO_ASSET_URL`, together with `LIBERO_VPRO_DATA_ROOT`.

## Inspect and evaluate

These checks do not execute a policy:

```bash
libero-vpro list
libero-vpro check
python3 scripts/check_assets.py --strict
python3 scripts/run_eval.py --setting A4.black_N1 --model fastwam --check
python3 scripts/run_eval.py --setting C1.count.agent --model fastwam --plan
```

Preflight checks inspect configuration, manifests and referenced files. They
do not demonstrate that a perturbation is visually effective in a rollout.

Run a setting with ten trials per applicable task:

```bash
python3 scripts/run_eval.py --setting A4.black_N1 --model fastwam --gpu 0
```

For a small validation run, choose a task belonging to the setting:

```bash
python3 scripts/run_eval.py --setting A4.black_N1 --model fastwam \
  --suite libero_10 --task 2 --trials 1 --trial-start 0 --gpu 0
```

Filters narrow evaluation only; they do not rewrite the dataset or add
unsupported tasks. `--trial-start` selects the first initial-state trial index.
Use `--output-root` for a separate run directory. No suite filter means all
suites applicable to the selected setting.

## Regenerate configuration files

Regeneration is not an installation or verification step. Developers changing
source definitions can regenerate derived contracts and manifests with:

```bash
python3 runnable_settings/build_runnable_settings.py
python3 verify_paper_alignment.py
```

This updates generated files. Back up manual changes first. Validation outputs,
videos, caches and private audit reports belong outside the release directory.

## Licenses

Project-authored code is [MIT licensed](LICENSE). The authors' original setting
definitions, case manifests, annotations, original assets, and original
modifications to upstream assets are [CC BY 4.0 licensed](LICENSE-DATA.md). This follows the separate
code/data licensing used by LIBERO and LIBERO-X.

Bundled third-party files retain their applicable upstream notices and are not
relicensed by the project licenses; see [third-party notices](THIRD_PARTY_NOTICES.md).
External LIBERO assets and model checkpoints remain subject to their own terms.
