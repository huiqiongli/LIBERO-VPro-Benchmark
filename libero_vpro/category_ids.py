"""Canonical paper-aligned category identifiers used by the release.

    A: Visual Evidence Degradation
    B: Camera Staleness
    C: Visual Source Consistency
    D: Task-Relevant Scene Variation
"""

from __future__ import annotations


CATEGORY_IDS = (
    "A1", "A2", "A3", "A4", "A5",
    "B1", "C1", "C2", "D1", "D2", "D3", "D4",
)

DIMENSIONS = {
    "A": "Visual Evidence Degradation",
    "B": "Camera Staleness",
    "C": "Visual Source Consistency",
    "D": "Task-Relevant Scene Variation",
}


def canonical_category(category: str) -> str:
    """Validate and return one of the 12 canonical paper category IDs."""
    if category not in CATEGORY_IDS:
        raise KeyError(f"unknown category {category!r}")
    return category

