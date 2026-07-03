"""Input/output schemas for structure prediction tools."""

from __future__ import annotations

from pathlib import Path  # noqa: TC003 - Pydantic needs at runtime

from pydantic import BaseModel, Field

from autobio.schemas.base import BaseInput, BaseOutput
from autobio.schemas.hints import Tier, Widget, ui
from autobio.schemas.sequences import GenericSequenceSet  # noqa: TC001 - needed at runtime


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
    affinity_probability: float | None = Field(
        default=None,
        description="Predicted probability of binding (0-1 scale).",
    )
    affinity_value: float | None = Field(
        default=None,
        description="Predicted binding affinity as log10(IC50) in uM.",
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


class ESMFoldInput(BaseInput):
    """Input for ESMFold single-sequence structure prediction (single ``predict`` mode)."""

    sequences: GenericSequenceSet = Field(
        description=(
            "A single protein sequence: a dict of id→sequence (one chain), "
            "FASTA text, or a FASTA file path."
        ),
        json_schema_extra=ui(widget=Widget.SEQUENCE, flavor="generic", tier=Tier.PRIMARY, order=0),
    )
    num_models: int = Field(
        default=1,
        description="Number of models. ESMFold is deterministic; must be 1.",
        json_schema_extra=ui(widget=Widget.NUMBER, tier=Tier.ADVANCED, order=10),
    )
    templates: list[Path] | None = Field(
        default=None,
        description="Template structures. ESMFold does not use templates; must be None/empty.",
        json_schema_extra=ui(widget=Widget.FILE, tier=Tier.ADVANCED, order=11),
    )
