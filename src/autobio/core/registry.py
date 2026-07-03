"""Tool category taxonomy."""

from __future__ import annotations

from enum import StrEnum


class ToolCategory(StrEnum):
    """Functional categories for computational biology tools."""

    STRUCTURE_PREDICTION = "structure-prediction"
    EMBEDDING = "embedding"
    INVERSE_FOLDING = "inverse-folding"
    SCORING = "scoring"
    STRUCTURE_DESIGN = "structure-design"
    SIMULATION = "simulation"
