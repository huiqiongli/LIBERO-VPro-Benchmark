from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import re
from typing import Any

import yaml


_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-(.*?))?\}")


def _expand(value: Any) -> Any:
    if isinstance(value, str):
        def replace(match: re.Match[str]) -> str:
            name, fallback = match.group(1), match.group(2)
            return os.environ.get(name) or fallback or ""

        return _ENV_PATTERN.sub(replace, value)
    if isinstance(value, list):
        return [_expand(item) for item in value]
    if isinstance(value, dict):
        return {key: _expand(item) for key, item in value.items()}
    return value


@dataclass(frozen=True)
class ReleaseConfig:
    path: Path
    data: dict[str, Any]

    @property
    def root(self) -> Path:
        value = self.data.get("release", {}).get("root")
        if value:
            root = Path(str(value)).expanduser()
            return root.resolve() if root.is_absolute() else (self.path.parent / root).resolve()
        return self.path.parent

    def environment(self) -> dict[str, str]:
        """Pass explicit YAML paths to subprocesses, preserving env overrides."""
        env = dict(os.environ)
        names = {
            "sim_root": "LIBERO_VPRO_SIM_ROOT",
            "backend_root": "LIBERO_VPRO_BACKEND_ROOT",
            "data_root": "LIBERO_VPRO_DATA_ROOT",
            "libero_assets_root": "LIBERO_VPRO_LIBERO_ASSETS_ROOT",
            "libero_root": "LIBERO_VPRO_LIBERO_ROOT",
            "fastwam_root": "FASTWAM_ROOT",
            "fastwam_python": "FASTWAM_PYTHON",
            "fastwam_checkpoint": "FASTWAM_CHECKPOINT",
            "fastwam_stats": "FASTWAM_DATASET_STATS",
        }
        for key, name in names.items():
            value = str(self.data.get("external", {}).get(key, "")).strip()
            if value and not env.get(name):
                path = Path(value).expanduser()
                env[name] = str(path if path.is_absolute() else (self.path.parent / path).resolve())
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        return env

    @property
    def release_manifest(self) -> Path:
        return self.root / self.data["release"]["paper_manifest"]

    @property
    def statistics_manifest(self) -> Path:
        return self.root / self.data["release"]["statistics_manifest"]

    @property
    def settings_index(self) -> Path:
        return self.root / self.data["release"]["settings_index"]

    @property
    def data_root(self) -> Path | None:
        value = str(self.data.get("external", {}).get("data_root", "")).strip()
        return Path(value).expanduser() if value else None

    @property
    def sim_root(self) -> Path | None:
        value = str(self.data.get("external", {}).get("sim_root", "")).strip()
        return Path(value).expanduser() if value else None

    @property
    def backend_root(self) -> Path | None:
        value = str(self.data.get("external", {}).get("backend_root", "")).strip()
        return Path(value).expanduser() if value else None

    @property
    def output_root(self) -> Path:
        value = str(self.data["evaluation"]["output_root"])
        path = Path(value).expanduser()
        return path if path.is_absolute() else (self.root / path).resolve()

    @classmethod
    def load(cls, path: str | Path) -> "ReleaseConfig":
        config_path = Path(path).expanduser().resolve()
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        return cls(config_path, _expand(payload))
