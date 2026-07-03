"""Input/output schemas for structure scoring tools."""

from __future__ import annotations

from pathlib import Path  # noqa: TC003 - Pydantic needs at runtime
from typing import Any, Literal

from pydantic import BaseModel, Field

from autobio.schemas.base import BaseInput, BaseOutput
from autobio.schemas.hints import Tier, Widget, ui


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


class FreeSASABaseInput(BaseInput):
    """Shared input for FreeSASA modes (SASA and BSA)."""

    structure_path: Path = Field(
        description="Path to the input PDB structure.",
        json_schema_extra=ui(widget=Widget.FILE, tier=Tier.PRIMARY, order=0),
    )
    algorithm: Literal["LeeRichards", "ShrakeRupley"] = Field(
        default="LeeRichards",
        description="SASA computation algorithm.",
        json_schema_extra=ui(widget=Widget.SELECT, tier=Tier.ADVANCED, order=10),
    )
    probe_radius: float = Field(
        default=1.4,
        gt=0,
        description="Solvent probe radius.",
        json_schema_extra=ui(
            widget=Widget.NUMBER, tier=Tier.ADVANCED, unit="Å", step=0.1, order=11
        ),
    )
    per_residue: bool = Field(
        default=False,
        description="Return per-residue values in addition to totals.",
        json_schema_extra=ui(widget=Widget.TOGGLE, tier=Tier.ADVANCED, order=12),
    )


class FreeSASASASAInput(FreeSASABaseInput):
    """Input for the FreeSASA ``sasa`` mode (solvent-accessible surface area)."""


class FreeSASABSAInput(FreeSASABaseInput):
    """Input for the FreeSASA ``bsa`` mode (buried surface area at an interface)."""

    partner1: str = Field(
        description="Comma-separated chain IDs for interface partner 1 (e.g. 'A,B').",
        json_schema_extra=ui(widget=Widget.TEXT, tier=Tier.PRIMARY, order=1),
    )
    partner2: str = Field(
        description="Comma-separated chain IDs for interface partner 2 (e.g. 'C').",
        json_schema_extra=ui(widget=Widget.TEXT, tier=Tier.PRIMARY, order=2),
    )


class BAddGInput(BaseInput):
    """Input for BA-ddG binding-ddG prediction (single ``predict`` mode)."""

    structure_path: Path = Field(
        description="Path to the protein complex PDB structure.",
        json_schema_extra=ui(widget=Widget.FILE, tier=Tier.PRIMARY, order=0),
    )
    mutations: list[str] = Field(
        description=(
            "Mutations to score, format [WT_AA][ChainID][Resnum][Mut_AA] "
            "(e.g. ['YH103H', 'QD30V']); combined effect is predicted."
        ),
        json_schema_extra=ui(widget=Widget.TEXT, tier=Tier.PRIMARY, order=1),
    )
    chains: str = Field(
        description="Binding interface as 'binder1_binder2' (e.g. 'ABC_DE' = A,B,C vs D,E).",
        json_schema_extra=ui(widget=Widget.TEXT, tier=Tier.PRIMARY, order=2),
    )
    n_folds: int = Field(
        default=3,
        ge=1,
        le=3,
        description="Cross-validation folds to average (1-3).",
        json_schema_extra=ui(widget=Widget.NUMBER, tier=Tier.ADVANCED, order=10),
    )
    seed: int = Field(
        default=0,
        description="Random seed.",
        json_schema_extra=ui(widget=Widget.NUMBER, tier=Tier.ADVANCED, order=11),
    )
    device: str = Field(
        default="auto",
        description="Compute device ('auto', 'cpu', or 'cuda').",
        json_schema_extra=ui(widget=Widget.TEXT, tier=Tier.ADVANCED, order=12),
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
