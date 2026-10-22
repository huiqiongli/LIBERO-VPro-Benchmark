"""Portable paths for the bundled FastWAM evaluation adapter.

Only upstream model code, weights, the Python environment, and LIBERO are
external. Benchmark evaluation and perturbation code always come from release.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import shlex
import sys
import tempfile
from typing import Mapping


RELEASE = Path(__file__).resolve().parents[1]
_LIBERO_CONFIG_DIRECTORY = None

# Inputs that select an experiment, not runtime resources. Clear these only
# when entering a new setting; prepare_env() is also used *after* a runner has
# injected its chosen perturbation and must not clear them again.
SETTING_ENV_KEYS = frozenset({
    "WAM_SINGLE_VIEW", "FASTWAM_SINGLE_VIEW", "LIBERO_SINGLE_VIEW",
    "WAM_VIEW_COPY", "FASTWAM_VIEW_COPY", "LIBERO_VIEW_COPY",
    "FASTWAM_OBS_OCCLUSION_SPEC", "LIBERO_OBS_OCCLUSION_SPEC",
    "FASTWAM_DECISION_SCHEDULE_SPEC", "LIBERO_SCHEDULE_SPEC",
    "FASTWAM_ENV_EDIT_SPEC", "LIBERO_ENV_EDIT_SPEC", "WAM_PERTURBATION_SPEC",
    "FASTWAM_CROSS_TASK_VIEW_SWAP_SPEC", "LIBERO_CROSS_TASK_VIEW_SWAP_SPEC",
    "WAM_RECURSIVE_FEEDBACK", "FASTWAM_RECURSIVE_FEEDBACK", "LIBERO_RECURSIVE_FEEDBACK",
    "LIBERO_OBJECT_XML_OVERRIDE_MAP", "LIBERO_USE_SEGMENTATION_ENV",
    "WAM_A1_OBS_SPEC", "WAM_A1_SCHEDULE_SPEC", "WAM_A1_TRACE_PATH",
    "WAM_A3_SPEC", "WAM_A3_SUITE", "WAM_A3_TASK_ID", "WAM_A3_CONDITION",
    "WAM_A3_METADATA_DIR", "WAM_A3_TRIAL_START",
    "WAM_A5_VIEW_FILL", "WAM_A5_KEEP_VIEW", "WAM_STATE_CASE_JSON",
    "FASTWAM_TASK_DESCRIPTIONS_FILE", "LIBERO_TASK_DESCRIPTIONS_FILE",
    "FASTWAM_USE_TASK_DESCRIPTIONS", "LIBERO_USE_TASK_DESCRIPTIONS",
    "A4_SCHEDULE_SPEC", "A4_SMOKE", "A4_SMOKE_SUITE", "A4_SMOKE_TASK",
    "A4_SMOKE_TRIALS", "A4_KIND", "A4_TARGET_PHASE", "B1_MODE", "B1_TARGET_VIEW",
    "WAM_TRACK", "WAM_SUBTASK", "WAM_CONDITION", "WAM_CASE_ID", "WAM_EPISODE_ID",
    "WAM_PAIR_ID", "WAM_CLEAN_EPISODE_ID", "WAM_PERTURBED_EPISODE_ID",
    "WAM_AGENT_REFERENCE_EPISODE_ID", "WAM_WRIST_REFERENCE_EPISODE_ID",
})


@dataclass(frozen=True)
class RuntimePaths:
    release: Path
    sim: Path
    fastwam: Path
    python: Path
    libero: Path
    checkpoint: Path
    dataset_stats: Path
    runner: Path


def resolve_runtime(require: bool = True, environ: Mapping[str, str] | None = None) -> RuntimePaths:
    env = os.environ if environ is None else environ

    def path(name: str, default: Path) -> Path:
        return Path(env.get(name) or default).expanduser().resolve()

    sim = path("LIBERO_VPRO_SIM_ROOT", RELEASE.parent)
    fastwam = path("FASTWAM_ROOT", sim / "Repos/FastWAM")
    paths = RuntimePaths(
        release=RELEASE,
        sim=sim,
        fastwam=fastwam,
        python=Path(env.get("FASTWAM_PYTHON") or sim / ".venvs/fastwam/bin/python").expanduser().absolute(),
        libero=path("LIBERO_VPRO_LIBERO_ROOT", sim / "Repos/LIBERO"),
        checkpoint=path("FASTWAM_CHECKPOINT", fastwam / "checkpoints/fastwam_release/libero_uncond_2cam224.pt"),
        dataset_stats=path("FASTWAM_DATASET_STATS", fastwam / "checkpoints/fastwam_release/libero_uncond_2cam224_dataset_stats.json"),
        runner=RELEASE / "source/fastwam_reference/eval_libero_single.py",
    )
    if require:
        required = {
            "FASTWAM_ROOT (upstream model package)": (paths.fastwam / "src/fastwam"
                if (paths.fastwam / "src/fastwam").is_dir() else paths.fastwam / "fastwam"),
            "FASTWAM_ROOT (Hydra configs)": paths.fastwam / "configs/sim_libero.yaml",
            "LIBERO_VPRO_LIBERO_ROOT (upstream simulator package)": paths.libero / "libero/libero/__init__.py",
            "FASTWAM_PYTHON": paths.python,
            "FASTWAM_CHECKPOINT": paths.checkpoint,
            "FASTWAM_DATASET_STATS": paths.dataset_stats,
            "bundled evaluation runner": paths.runner,
        }
        missing = [f"{name}: {value}" for name, value in required.items() if not value.exists()]
        if missing:
            raise FileNotFoundError("Missing FastWAM runtime inputs:\n" + "\n".join(missing))
    return paths


def prepare_env(base_env: Mapping[str, str] | None = None) -> dict[str, str]:
    """Resolve runtime paths without changing an already-selected experiment."""
    env = dict(os.environ if base_env is None else base_env)
    paths = resolve_runtime(require=False, environ=env)
    env.update({
        "LIBERO_VPRO_RELEASE_ROOT": str(paths.release),
        "FASTWAM_ROOT": str(paths.fastwam),
        "FASTWAM_PYTHON": str(paths.python),
        "FASTWAM_CHECKPOINT": str(paths.checkpoint),
        "FASTWAM_DATASET_STATS": str(paths.dataset_stats),
        "LIBERO_VPRO_FASTWAM_RUNNER": str(paths.runner),
        "LIBERO_VPRO_LIBERO_ROOT": str(paths.libero),
    })
    search = [paths.release, paths.release / "source", paths.release / "source/_shared_core",
              paths.release / "source/fastwam_reference",
              paths.release / "source/D2_D3_state_20260823",
              paths.fastwam / "src", paths.fastwam, paths.libero]
    existing = [value for value in env.get("PYTHONPATH", "").split(os.pathsep) if value]
    env["PYTHONPATH"] = os.pathsep.join(dict.fromkeys([str(value) for value in search] + existing))
    for key, value in {
        "MUJOCO_GL": "egl", "PYOPENGL_PLATFORM": "egl", "TOKENIZERS_PARALLELISM": "false",
        "HYDRA_FULL_ERROR": "1", "DIFFSYNTH_MODEL_BASE_PATH": str(paths.fastwam / "checkpoints/diffsyn"),
    }.items():
        env.setdefault(key, value)
    return env


def prepare_setting_env(
    base_env: Mapping[str, str] | None = None,
    *,
    overrides: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Start an independent setting, then apply its explicit local overrides.

    Use at a dispatcher/standalone runner boundary, before adding that setting's
    observation or environment edits. Do not use in evaluator bootstrapping or
    shared shell runtime setup. GPU, model/weight paths, dataset/asset paths,
    output/recording options, and canonical suite/task/trial selection survive.
    Neither this function nor prepare_env mutates the caller's environment.
    """
    env = dict(os.environ if base_env is None else base_env)
    for key in SETTING_ENV_KEYS:
        env.pop(key, None)
    env = prepare_env(env)
    if overrides:
        env.update(overrides)
    return env


def configure_libero() -> None:
    """Create a process-local LIBERO path config without touching user settings.

    Called only by the actual evaluator, before importing LIBERO. Planning and
    runtime path validation remain read-only.
    """
    global _LIBERO_CONFIG_DIRECTORY
    if _LIBERO_CONFIG_DIRECTORY is not None:
        return
    import yaml

    runtime = resolve_runtime(require=False)
    configured = os.environ.get("LIBERO_CONFIG_PATH", "")
    data = os.environ.get("LIBERO_VPRO_DATA_ROOT", "")
    assets = os.environ.get("LIBERO_VPRO_LIBERO_ASSETS_ROOT", "")
    config = {}
    if configured:
        config_path = Path(configured).expanduser() / "config.yaml"
        if not config_path.is_file():
            raise FileNotFoundError(f"LIBERO_CONFIG_PATH has no config.yaml: {config_path}")
        if not data and not assets:
            return
        config = yaml.safe_load(config_path.read_text()) or {}
    package = runtime.libero / "libero/libero"
    defaults = {
        "benchmark_root": str(package), "bddl_files": str(package / "bddl_files"),
        "init_states": str(package / "init_files"), "datasets": str(runtime.libero / "datasets"),
        "assets": str(package / "assets"),
    }
    config = {**defaults, **config}
    if data:
        data_root = Path(data).expanduser().resolve()
        config.update({"bddl_files": str(data_root / "bddl_files"),
                       "init_states": str(data_root / "init_files")})
    if assets:
        config["assets"] = str(Path(assets).expanduser().resolve())
    _LIBERO_CONFIG_DIRECTORY = tempfile.TemporaryDirectory(prefix="libero-vpro-config-")
    config_path = Path(_LIBERO_CONFIG_DIRECTORY.name) / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    os.environ["LIBERO_CONFIG_PATH"] = str(config_path.parent)


def main() -> int:
    if sys.argv[1:] != ["--shell"]:
        raise SystemExit("usage: python -m libero_vpro.runtime --shell")
    resolve_runtime(require=True)
    prepared = prepare_env()
    keys = ("LIBERO_VPRO_RELEASE_ROOT", "FASTWAM_ROOT", "FASTWAM_PYTHON", "FASTWAM_CHECKPOINT",
            "FASTWAM_DATASET_STATS", "LIBERO_VPRO_FASTWAM_RUNNER", "LIBERO_VPRO_LIBERO_ROOT",
            "PYTHONPATH", "MUJOCO_GL", "PYOPENGL_PLATFORM", "TOKENIZERS_PARALLELISM",
            "HYDRA_FULL_ERROR", "DIFFSYNTH_MODEL_BASE_PATH")
    for key in keys:
        print(f"export {key}={shlex.quote(prepared[key])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
