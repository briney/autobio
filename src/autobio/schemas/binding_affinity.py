"""Input/output schemas for antibody binding affinity prediction tools."""

from __future__ import annotations

from pathlib import Path  # noqa: TC003 - Pydantic needs at runtime
from typing import Any

from pydantic import BaseModel, Field

from autobio.schemas.base import BaseInput, BaseOutput
from autobio.schemas.hints import Tier, Widget, ui


class BindingAffinityInput(BaseInput):
    """Input schema for antibody binding affinity prediction tools (e.g., ANTIPASTI).

    Requires a 3D structure of an antibody-antigen complex with explicit chain
    identification for the heavy chain, light chain, and antigen chain(s).
    """

    structure_path: Path = Field(
        description="Path to the antibody-antigen complex structure (PDB or mmCIF).",
        json_schema_extra=ui(widget=Widget.FILE, tier=Tier.PRIMARY, order=0),
    )
    heavy_chain: str = Field(
        description="Chain ID of the antibody heavy chain in the structure (e.g., 'H').",
        json_schema_extra=ui(widget=Widget.TEXT, tier=Tier.PRIMARY, order=1),
    )
    light_chain: str = Field(
        description="Chain ID of the antibody light chain in the structure (e.g., 'L').",
        json_schema_extra=ui(widget=Widget.TEXT, tier=Tier.PRIMARY, order=2),
    )
    antigen_chains: list[str] = Field(
        description=(
            "Chain ID(s) of the antigen in the structure (e.g., ['A'] or ['A', 'B'] "
            "for multi-chain antigens)."
        ),
        json_schema_extra=ui(widget=Widget.TEXT, tier=Tier.PRIMARY, order=3),
    )


class AntipastiInput(BindingAffinityInput):
    """Input for ANTIPASTI antibody-antigen affinity prediction."""

    modes: str | int = Field(
        default="all",
        description=(
            "Normal modes for the DCCM calculation: 'all', or an integer count. "
            "This is the ANTIPASTI-specific 'modes' config key (number of normal "
            "modes used in the Dynamic Cross-Correlation Matrix), distinct from "
            "the Tool-level notion of 'modes' (e.g. Tool.modes / predict mode)."
        ),
        json_schema_extra=ui(widget=Widget.TEXT, tier=Tier.ADVANCED, order=10),
    )


class BindingAffinityPrediction(BaseModel):
    """Binding affinity prediction for a single antibody-antigen complex."""

    log10_kd: float = Field(
        description="Predicted log10(Kd) in molar. More negative values indicate tighter binding."
    )
    kd_molar: float | None = Field(
        default=None,
        description="Derived Kd in molar (10^log10_kd), provided as a convenience.",
    )
    units: str | None = Field(
        default=None,
        description="Units description (e.g., 'log10(Kd) [M]').",
    )
    score_breakdown: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Model-specific metadata (e.g., checkpoint name, normal modes used, chain assignments)."
        ),
    )


class BindingAffinityOutput(BaseOutput):
    """Output schema for antibody binding affinity prediction tools."""

    predictions: list[BindingAffinityPrediction] = Field(
        description="Binding affinity predictions for each input structure."
    )
