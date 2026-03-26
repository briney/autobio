"""Input/output schemas for structure scoring tools."""

from __future__ import annotations

from pathlib import Path  # noqa: TC003 - Pydantic needs at runtime
from typing import Any

from pydantic import BaseModel, Field

from autobio.schemas.base import BaseInput, BaseOutput


class ScoringInput(BaseInput):
    """Input schema for structure scoring tools (e.g., Rosetta, OpenMM energy)."""

    structure_path: Path = Field(description="Path to the structure to score (PDB or mmCIF).")
    sequences: dict[str, str] | None = Field(
        default=None,
        description=(
            "Optional mapping of chain ID to amino acid sequence. "
            "If provided, scores the structure with these sequences threaded onto the backbone."
        ),
    )


class ScoredStructure(BaseModel):
    """Scoring results for a single structure."""

    total_score: float = Field(description="Total energy score for the structure.")
    per_residue_scores: list[float] | None = Field(
        default=None,
        description="Per-residue energy scores in residue order.",
    )
    score_breakdown: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Breakdown of the total score by energy term "
            "(e.g., {'van_der_waals': -120.5, 'electrostatics': -45.2})."
        ),
    )
    units: str | None = Field(
        default=None,
        description="Units of the score (e.g., 'REU' for Rosetta energy units, 'kcal/mol').",
    )
    structure_path: Path | None = Field(
        default=None,
        description="Path to the scored/refined structure file, if the tool produces one.",
    )
    ddg: float | None = Field(
        default=None,
        description=(
            "Delta-delta-G: change in binding free energy upon mutation "
            "(kcal/mol or REU). Positive values indicate destabilization."
        ),
    )
    mutations: list[str] | None = Field(
        default=None,
        description="Mutations scored, e.g., ['A42F', 'L55W'].",
    )


class ScoringOutput(BaseOutput):
    """Output schema for structure scoring tools."""

    scores: list[ScoredStructure] = Field(
        description="Scoring results for each evaluated structure."
    )
