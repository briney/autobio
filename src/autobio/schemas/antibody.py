"""Input/output schemas for antibody language model tools.

Shared by CurrAb, BALM-paired, BALM-unpaired, ft-ESM, AbLang2, and
AntiBERTa2.  Each model has two tool variants (embedding and pseudo
log-likelihood), all using ``AntibodyInput`` as input.  Embedding tools
reuse ``EmbeddingOutput`` from :mod:`autobio.schemas.embedding`; PLL
tools return ``AntibodyPLLOutput``.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator

from autobio.schemas.base import BaseInput, BaseOutput


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


class AntibodyInput(BaseInput):
    """Input schema for antibody language model tools.

    Used by all antibody LM tools (CurrAb, BALM-paired, BALM-unpaired,
    ft-ESM, AbLang2, AntiBERTa2) for both embedding extraction and pseudo
    log-likelihood.
    """

    sequences: list[AntibodySequence] = Field(
        description="One or more antibody sequences to process.",
    )
    layer: int | None = Field(
        default=None,
        description=(
            "Model layer from which to extract embeddings. "
            "None uses the final layer. Only used in embedding mode."
        ),
    )
    pooling: str | None = Field(
        default=None,
        description=(
            "Pooling strategy for per-residue embeddings "
            "(e.g., 'mean', 'cls', 'per_residue'). Only used in embedding mode."
        ),
    )


class SequencePLL(BaseModel):
    """Pseudo log-likelihood result for a single antibody sequence."""

    sequence_id: str = Field(
        description="Identifier matching a sequence ID in the input.",
    )
    pll: float = Field(
        description="Total pseudo log-likelihood (sum of per-position log-probabilities).",
    )
    per_position_pll: list[float] | None = Field(
        default=None,
        description=(
            "Per-residue log-probabilities. Only populated when extra['per_position'] is True."
        ),
    )
    sequence_length: int = Field(
        description="Total number of non-special tokens scored.",
    )


class AntibodyPLLOutput(BaseOutput):
    """Output schema for antibody pseudo log-likelihood tools."""

    scores: list[SequencePLL] = Field(
        description="PLL results for each input sequence.",
    )
    model_name: str = Field(
        description="Name of the antibody language model used.",
    )
