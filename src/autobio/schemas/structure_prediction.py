"""Input/output schemas for structure prediction tools."""

from __future__ import annotations

from pathlib import Path  # noqa: TC003 - Pydantic needs at runtime

from pydantic import BaseModel, Field

from autobio.schemas.base import BaseInput, BaseOutput


class StructurePredictionInput(BaseInput):
    """Input schema for structure prediction tools (e.g., AlphaFold, ESMFold)."""

    sequences: dict[str, str] = Field(
        description=(
            "Mapping of chain ID to amino acid sequence (e.g., {'A': 'MKLL...', 'B': 'GVSE...'})."
        ),
    )
    num_models: int = Field(
        default=1,
        description="Number of structure models to generate.",
    )
    templates: list[Path] | None = Field(
        default=None,
        description="Paths to template structures (PDB or mmCIF) for template-based prediction.",
    )


class PredictedStructure(BaseModel):
    """A single predicted structure model with confidence metrics."""

    model_rank: int = Field(description="Rank by predicted quality (1 = best).")
    structure_path: Path = Field(
        description="Path to the predicted structure in outputs/standardized/."
    )
    plddt_per_residue: list[float] | None = Field(
        default=None,
        description="Per-residue pLDDT confidence scores (0-100 scale).",
    )
    plddt_mean: float | None = Field(
        default=None,
        description="Mean pLDDT score across all residues (0-100 scale).",
    )
    ptm: float | None = Field(
        default=None,
        description="Predicted TM-score (0-1 scale, global structure confidence).",
    )
    iptm: float | None = Field(
        default=None,
        description="Interface predicted TM-score (0-1 scale, multimer interface confidence).",
    )
    chain_mapping: dict[str, str] | None = Field(
        default=None,
        description="Mapping from input chain IDs to output chain IDs.",
    )


class ConfidenceMetrics(BaseModel):
    """Summary confidence metrics across all predicted models."""

    best_plddt_mean: float | None = Field(
        default=None,
        description="Highest mean pLDDT across all models (0-100 scale).",
    )
    best_ptm: float | None = Field(
        default=None,
        description="Highest predicted TM-score across all models (0-1 scale).",
    )
    best_iptm: float | None = Field(
        default=None,
        description="Highest interface predicted TM-score across all models (0-1 scale).",
    )


class StructurePredictionOutput(BaseOutput):
    """Output schema for structure prediction tools."""

    structures: list[PredictedStructure] = Field(
        description="Predicted structures ranked by quality."
    )
    confidence: ConfidenceMetrics = Field(
        description="Summary confidence metrics across all predicted models."
    )
