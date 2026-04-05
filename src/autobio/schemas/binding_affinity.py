"""Input/output schemas for antibody binding affinity prediction tools."""

from __future__ import annotations

from pathlib import Path  # noqa: TC003 - Pydantic needs at runtime
from typing import Any

from pydantic import BaseModel, Field

from autobio.schemas.base import BaseInput, BaseOutput


class BindingAffinityInput(BaseInput):
    """Input schema for antibody binding affinity prediction tools (e.g., ANTIPASTI).

    Requires a 3D structure of an antibody-antigen complex with explicit chain
    identification for the heavy chain, light chain, and antigen chain(s).
    """

    structure_path: Path = Field(
        description="Path to the antibody-antigen complex structure (PDB or mmCIF)."
    )
    heavy_chain: str = Field(
        description="Chain ID of the antibody heavy chain in the structure (e.g., 'H')."
    )
    light_chain: str = Field(
        description="Chain ID of the antibody light chain in the structure (e.g., 'L')."
    )
    antigen_chains: list[str] = Field(
        description=(
            "Chain ID(s) of the antigen in the structure (e.g., ['A'] or ['A', 'B'] "
            "for multi-chain antigens)."
        ),
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
