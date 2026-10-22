# LIBERO-VPro

LIBERO-VPro is a visual-perturbation benchmark for LIBERO with four dimensions,
12 categories, 96 settings, and 3,296 task-condition cases. It includes camera
observation transforms, scene and object assets, task manifests, and evaluation
scripts.

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

The included FastWAM example shows how to evaluate a policy with these settings.

## Installation

Install the interface from the checkout using Python 3.9 or newer:

```bash
python3 -m pip install -e /path/to/release
```

The FastWAM example uses separate LIBERO, MuJoCo, FastWAM, checkpoint, and
dataset-statistics installations.

Configure the example evaluator:

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

The LIBERO data root contains `bddl_files/` and `init_files/`. For an
installation with `Repos/FastWAM`, `Repos/LIBERO`, and `.venvs/fastwam` under
one directory, `LIBERO_VPRO_SIM_ROOT` supplies default paths. LIBERO paths can
also be set in `evaluation_config.yaml`.

After copying the release, rebase manifest and XML paths:

```bash
cd /path/to/release
python3 scripts/relocate_paths.py --write
python3 scripts/check_assets.py --strict
```

If the previous checkout used a different directory name, add
`--old-root /previous/package/path`. Preserve relative symlinks when copying
assets.

To extract a LIBERO asset archive, use
`scripts/download_assets.sh` by setting `LIBERO_VPRO_ASSET_ARCHIVE` or
`LIBERO_VPRO_ASSET_URL`, together with `LIBERO_VPRO_DATA_ROOT`.

## Inspect and evaluate

List settings and check referenced files:

```bash
libero-vpro list
libero-vpro check
python3 scripts/check_assets.py --strict
python3 scripts/run_eval.py --setting A4.black_N1 --model fastwam --check
python3 scripts/run_eval.py --setting C1.count.agent --model fastwam --plan
```

Run a setting with ten trials per applicable task:

```bash
python3 scripts/run_eval.py --setting A4.black_N1 --model fastwam --gpu 0
```

For a small validation run, choose a task belonging to the setting:

```bash
python3 scripts/run_eval.py --setting A4.black_N1 --model fastwam \
  --suite libero_10 --task 2 --trials 1 --trial-start 0 --gpu 0
```

`--trial-start` selects the first initial-state trial index. `--output-root`
selects the result directory. Omitting `--suite` evaluates every applicable
suite.

## Regenerate configuration files

After changing source definitions, regenerate the setting contracts and
manifests with:

```bash
python3 runnable_settings/build_runnable_settings.py
python3 verify_paper_alignment.py
```

Store evaluation outputs separately from the release directory.

## Licenses

Project-authored code is [MIT licensed](LICENSE). The authors' original setting
definitions, case manifests, annotations, original assets, and original
modifications to upstream assets are [CC BY 4.0 licensed](LICENSE-DATA.md).

Upstream components are listed in the [third-party notices](THIRD_PARTY_NOTICES.md).
