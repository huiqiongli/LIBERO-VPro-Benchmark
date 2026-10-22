from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import ReleaseConfig


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def setting_rows(config: ReleaseConfig) -> list[dict[str, Any]]:
    return load_json(config.settings_index).get("settings", [])


def setting_path(config: ReleaseConfig, setting_id: str) -> Path:
    for row in setting_rows(config):
        if row.get("id") == setting_id:
            return (config.settings_index.parent / row["path"] / "setting.json").resolve()
    raise KeyError(f"unknown LIBERO-VPro setting: {setting_id}")
