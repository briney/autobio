"""Input/output schemas for antibody language model tools.

Shared by CurrAb, BALM-paired, BALM-unpaired, ft-ESM, AbLang2, and
AntiBERTa2.  Each model is one catalog Tool with two modes (embedding and
pseudo log-likelihood).  The embedding mode uses ``AntibodyEmbeddingInput``
and the PLL mode uses ``AntibodyPLLInput`` (both extend ``AntibodyBaseInput``).
The embedding mode reuses ``EmbeddingOutput`` from
:mod:`autobio.schemas.embedding`; the PLL mode returns ``AntibodyPLLOutput``.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from autobio.schemas.antibody_types import AntibodySequence  # re-export
from autobio.schemas.base import BaseInput, BaseOutput
from autobio.schemas.hints import Tier, Widget, ui
from autobio.schemas.sequences import AntibodySequenceSet  # noqa: TC001 - runtime field type

__all__ = [
    "AntibodyBaseInput",
    "AntibodyEmbeddingInput",
    "AntibodyPLLInput",
    "AntibodyPLLOutput",
    "AntibodySequence",
    "SequencePLL",
]


class AntibodyBaseInput(BaseInput):
    """Shared antibody-LM input: the sequence set (plus inherited ``extra``)."""

    sequences: AntibodySequenceSet = Field(
        description=(
            "One or more antibody sequences: a list of AntibodySequence/dicts, "
            "FASTA text, or a path to a .fasta/.fa file."
        ),
        json_schema_extra=ui(widget=Widget.SEQUENCE, flavor="antibody", tier=Tier.PRIMARY, order=0),
    )


class AntibodyEmbeddingInput(AntibodyBaseInput):
    """Input for the ``embedding`` mode."""

    layer: int | None = Field(
        default=None,
        description="Model layer from which to extract embeddings. None uses the final layer.",
        json_schema_extra=ui(widget=Widget.NUMBER, tier=Tier.ADVANCED, order=10),
    )
    pooling: str | None = Field(
        default=None,
        description=("Pooling strategy for per-residue embeddings ('mean', 'cls', 'per_residue')."),
        json_schema_extra=ui(
            widget=Widget.SELECT,
            tier=Tier.PRIMARY,
            order=1,
            enum_labels={"mean": "Mean pool", "cls": "CLS token", "per_residue": "Per-residue"},
        ),
    )


class AntibodyPLLInput(AntibodyBaseInput):
    """Input for the ``pll`` (pseudo log-likelihood) mode."""

    per_position: bool = Field(
        default=False,
        description="Return per-position PLL scores. Slower.",
        json_schema_extra=ui(widget=Widget.TOGGLE, tier=Tier.ADVANCED, order=11),
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
        description=("Per-residue log-probabilities. Only populated when per_position is True."),
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
