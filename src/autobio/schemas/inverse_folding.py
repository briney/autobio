"""Input/output schemas for inverse folding tools."""

from __future__ import annotations

from pathlib import Path  # noqa: TC003 - Pydantic needs at runtime

from pydantic import BaseModel, Field

from autobio.schemas.base import BaseInput, BaseOutput


class InverseFoldingInput(BaseInput):
    """Input schema for inverse folding tools (e.g., ProteinMPNN, ESM-IF1)."""

    structure_path: Path = Field(
        description="Path to the input backbone structure (PDB or mmCIF)."
    )
    chains_to_design: list[str] | None = Field(
        default=None,
        description="Chain IDs to redesign. None designs all chains.",
    )
    num_sequences: int = Field(
        default=1,
        description="Number of designed sequences to generate.",
    )
    temperature: float = Field(
        default=0.1,
        description="Sampling temperature. Lower values produce more conserved designs.",
    )
    fixed_positions: dict[str, list[int]] | None = Field(
        default=None,
        description=(
            "Positions to keep fixed (not redesigned), "
            "as a mapping of chain ID to 1-based residue indices."
        ),
    )


class DesignedSequence(BaseModel):
    """A single sequence designed by inverse folding."""

    rank: int = Field(description="Rank by design score (1 = best).")
    sequence: dict[str, str] = Field(
        description="Mapping of chain ID to designed amino acid sequence."
    )
    score: float | None = Field(
        default=None,
        description="Negative log-likelihood score (lower is better).",
    )
    recovery: float | None = Field(
        default=None,
        description="Sequence recovery rate vs. native sequence (0-1 scale).",
    )


class InverseFoldingOutput(BaseOutput):
    """Output schema for inverse folding tools."""

    designed_sequences: list[DesignedSequence] = Field(
        description="Designed sequences ranked by score."
    )
    native_sequence: dict[str, str] | None = Field(
        default=None,
        description="Native sequence extracted from the input structure, per chain.",
    )
