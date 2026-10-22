# LIBERO-VPro

**A benchmark for closed-loop visual robustness in robotic manipulation.**

[Paper](https://arxiv.org/abs/2609.24350) · [Project page](https://huiqiongli.github.io/LIBERO-VPro/)

LIBERO-VPro extends 40 tasks from LIBERO-Long (`libero_10`), LIBERO-Object,
LIBERO-Spatial, and LIBERO-Goal with controlled visual perturbations during
execution. It examines whether a policy can act when visual evidence is
incomplete, stale, inconsistent across sources, or changed by the scene itself.
The benchmark contains **four dimensions, 12 challenge categories, 96 settings,
and 3,296 task-condition cases**.

![Overview of the four LIBERO-VPro dimensions](https://huiqiongli.github.io/LIBERO-VPro/assets/images/benchmark-overview-pdf.png)

## Benchmark design

| Dimension | Category | Settings | Cases | What changes |
| --- | --- | ---: | ---: | --- |
| Visual Evidence Degradation | A1 Target-Object Masking | 8 | 320 | Target-object visibility: four masking ratios and four temporal schedules |
| | A2 Target-Receptacle Masking | 8 | 320 | Receptacle visibility: four masking ratios and four temporal schedules |
| | A3 Interaction-Cue Masking | 4 | 69 | Gripper and contact-region visual cues |
| | A4 Temporary Observation Corruption | 21 | 840 | Transient blackout or noise, scheduled by occurrence or manipulation phase |
| | A5 View Unavailability | 8 | 320 | Third-person or wrist view removed, replaced, or copied from the other view |
| Camera Staleness | B1 Camera Staleness | 10 | 400 | One camera delayed, frozen, or replayed while the other remains current |
| Visual Source Consistency | C1 Cross-View Contradiction | 8 | 190 | Quantity, existence, position, or state differs between camera views |
| | C2 Prediction–Reality Divergence | 1 | 40 | Predicted frames recursively replace later visual inputs |
| Task-Relevant Scene Variation | D1 Task-Relevant Distractors | 6 | 240 | Distractors share target attributes, occupy its familiar position, or obstruct its trajectory |
| | D2 Task-Precondition Variation | 8 | 80 | Receptacle openness or object-support pose changes |
| | D3 Visual State Ambiguity | 2 | 45 | Visual evidence of openness or object pose becomes ambiguous |
| | D4 Target Geometry Variation | 12 | 432 | Taper, Multipeak, Twist, or Asymmetric deformation at three severity levels |

A **case** is one LIBERO suite–task–setting combination. Each case uses ten
official initial-state trials. The Normal condition uses the same tasks and
initial states with standard observations. C2 uses predicted future frames and
is intended for policies that generate them.

## What is included

| Path | Contents |
| --- | --- |
| [`PAPER_FACING_SETTINGS.json`](PAPER_FACING_SETTINGS.json) | Dimension, category, and setting index |
| [`runnable_settings/settings/`](runnable_settings/settings/) | Configuration and case/trial manifests for each setting |
| [`assets/`](assets/) | Modified XML, meshes, textures, and BDDL scene assets |
| [`source/`](source/) | Observation transforms, temporal schedules, and evaluation code |
| [`paper_facing_configs/`](paper_facing_configs/) | Shared case inventories and selected setting configurations |
| [`scripts/`](scripts/) | Asset checks, path relocation, and the FastWAM evaluation example |

The setting ID follows `<category>.<setting>`, for example
`B1.delay3_agent` or `D1.same_position`. The complete list is in
[`runnable_settings/index.json`](runnable_settings/index.json).

## Installation and inspection

Install [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) and clone
this repository:

```bash
git clone https://github.com/huiqiongli/LIBERO-VPro-Dataset.git
cd LIBERO-VPro-Dataset
python3 -m pip install -e .
```

Set the LIBERO paths and update paths in the bundled manifests after cloning:

```bash
export LIBERO_VPRO_LIBERO_ROOT=/path/to/LIBERO
export LIBERO_VPRO_DATA_ROOT=/path/to/LIBERO/libero/libero
export LIBERO_VPRO_LIBERO_ASSETS_ROOT=/path/to/LIBERO/libero/libero/assets

python3 scripts/relocate_paths.py --write
libero-vpro list
libero-vpro check
python3 scripts/check_assets.py --strict
```

`LIBERO_VPRO_DATA_ROOT` contains LIBERO's `bddl_files/` and `init_files/`.
Preserve relative symlinks when copying the asset directory. The path updater
also accepts `--old-root /previous/checkout/path` after moving an existing
checkout.

## FastWAM evaluation example

The repository includes a FastWAM integration for running the released settings.
Set the model paths in addition to the LIBERO paths above:

```bash
export FASTWAM_ROOT=/path/to/FastWAM
export FASTWAM_PYTHON=/path/to/fastwam-environment/bin/python
export FASTWAM_CHECKPOINT=/path/to/libero_uncond_2cam224.pt
export FASTWAM_DATASET_STATS=/path/to/libero_uncond_2cam224_dataset_stats.json
export LIBERO_VPRO_OUTPUT_ROOT=/path/to/results
```

Inspect a setting, then run one trial or the full ten-trial protocol:

```bash
python3 scripts/run_eval.py --setting B1.delay3_agent --model fastwam --check
python3 scripts/run_eval.py --setting B1.delay3_agent --model fastwam \
  --suite libero_10 --task 2 --trials 1 --trial-start 0 --gpu 0
python3 scripts/run_eval.py --setting B1.delay3_agent --model fastwam --gpu 0
```

`--suite`, `--task`, `--trials`, and `--trial-start` select evaluation cases;
`--output-root` selects the result directory. Omitting `--suite` evaluates all
tasks applicable to the selected setting.

## Citation

If you use LIBERO-VPro, please cite the accompanying paper:

```bibtex
@article{li2026liberovpro,
  title   = {LIBERO-VPro: Benchmarking Closed-Loop Visual Robustness of Robotic Foundation Models},
  author  = {Li, Huiqiong and Mei, Zhiting and Majumdar, Anirudha and Chen, Jingjing and Jiang, Yu-Gang and Zhu, Bin},
  journal = {arXiv preprint arXiv:2609.24350},
  year    = {2026}
}
```

## License

Project code: [MIT](LICENSE). Original benchmark settings and assets:
[CC BY 4.0](LICENSE-DATA.md). Upstream components are listed in
[third-party notices](THIRD_PARTY_NOTICES.md).
