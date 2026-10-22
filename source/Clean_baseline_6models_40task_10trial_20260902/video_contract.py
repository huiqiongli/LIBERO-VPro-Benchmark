"""Compatibility import for the shared portable video validator."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from source._shared_core.video_contract import video_probe

__all__ = ["video_probe"]
