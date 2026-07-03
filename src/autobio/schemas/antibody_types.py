"""Pydantic-only leaf types shared by antibody sequence utilities and schemas.

This module exists to break an import cycle: both ``autobio.utils.sequences``
and ``autobio.schemas.sequences`` need :class:`AntibodySequence`, and
``autobio.schemas.antibody`` needs ``AntibodySequenceSet`` (defined in
``autobio.schemas.sequences``). Keeping :class:`AntibodySequence` in its own
leaf module (depending only on ``pydantic``) lets all three modules import it
without cycling.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator


class AntibodySequence(BaseModel):
    """A single antibody sequence entry with optional heavy and light chains.

    At least one of ``heavy_chain`` or ``light_chain`` must be provided.
    """

    id: str = Field(description="Unique identifier for this antibody sequence.")
    heavy_chain: str | None = Field(
        default=None,
        description="Variable heavy (VH) amino acid sequence.",
    )
    light_chain: str | None = Field(
        default=None,
        description="Variable light (VL) amino acid sequence.",
    )

    @model_validator(mode="after")
    def _at_least_one_chain(self) -> AntibodySequence:
        if self.heavy_chain is None and self.light_chain is None:
            msg = f"Sequence '{self.id}': at least one of heavy_chain or light_chain is required."
            raise ValueError(msg)
        return self
