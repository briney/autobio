"""Input/output schemas for general protein-protein binding affinity prediction tools."""

from __future__ import annotations

from pathlib import Path  # noqa: TC003 - Pydantic needs at runtime
from typing import Any

from pydantic import BaseModel, Field

from autobio.schemas.base import BaseInput, BaseOutput
from autobio.schemas.hints import Tier, Widget, ui


class ProteinBindingAffinityInput(BaseInput):
    """Input schema for general protein-protein binding affinity prediction tools.

    Unlike :class:`~autobio.schemas.binding_affinity.BindingAffinityInput`, this
    schema is not antibody-specific — it works with arbitrary protein-protein
    complexes and uses flexible chain selection instead of requiring heavy/light
    chain assignments.
    """

    structure_path: Path = Field(
        description="Path to the protein-protein complex structure (PDB or mmCIF).",
        json_schema_extra=ui(widget=Widget.FILE, tier=Tier.PRIMARY, order=0),
    )
    chain_selection: str | None = Field(
        default=None,
        description=(
            "Chain grouping for binding affinity calculation. Format follows "
            "PRODIGY convention: e.g., 'A B' for chains A vs B, or 'A,B C' to "
            "treat chains A+B as one partner against chain C. When None, all "
            "inter-chain contacts are used."
        ),
        json_schema_extra=ui(widget=Widget.TEXT, tier=Tier.PRIMARY, order=1),
    )
    temperature: float = Field(
        default=25.0,
        description=(
            "Temperature in Celsius for dissociation constant (Kd) calculation. "
            "Must be above absolute zero (-273.15)."
        ),
        json_schema_extra=ui(widget=Widget.NUMBER, tier=Tier.ADVANCED, unit="°C", order=10),
    )


class ProdigyInput(ProteinBindingAffinityInput):
    """Input for PRODIGY protein-protein affinity prediction."""

    distance_cutoff: float = Field(
        default=5.5,
        gt=0,
        description="Interatomic contact distance cutoff.",
        json_schema_extra=ui(
            widget=Widget.NUMBER, tier=Tier.ADVANCED, unit="Å", step=0.1, order=11
        ),
    )
    contact_list: bool = Field(
        default=False,
        description="Include the detailed interface contact list in the output breakdown.",
        json_schema_extra=ui(widget=Widget.TOGGLE, tier=Tier.ADVANCED, order=12),
    )


class ProteinBindingAffinityPrediction(BaseModel):
    """Binding affinity prediction for a single protein-protein complex."""

    delta_g_kcal_mol: float = Field(
        description=(
            "Predicted binding free energy in kcal/mol. "
            "More negative values indicate tighter binding."
        ),
    )
    kd_molar: float | None = Field(
        default=None,
        description="Derived dissociation constant in molar, provided as a convenience.",
    )
    units: str | None = Field(
        default=None,
        description="Units description (e.g., 'kcal/mol').",
    )
    score_breakdown: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Tool-specific metadata (e.g., contact counts by type, "
            "non-interacting surface percentages, chain assignments)."
        ),
    )


class ProteinBindingAffinityOutput(BaseOutput):
    """Output schema for general protein-protein binding affinity prediction tools."""

    predictions: list[ProteinBindingAffinityPrediction] = Field(
        description="Binding affinity predictions for each input structure."
    )
