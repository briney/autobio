"""Input/output schemas for sequence embedding tools."""

from __future__ import annotations

from pathlib import Path  # noqa: TC003 - Pydantic needs at runtime
from typing import Literal

from pydantic import BaseModel, Field

from autobio.schemas.base import BaseInput, BaseOutput
from autobio.schemas.hints import Tier, Widget, ui
from autobio.schemas.sequences import GenericSequenceSet  # noqa: TC001 - needed at runtime


class ESMEmbedInput(BaseInput):
    """Input for ESM embedding (esm1b): sequences + layer/pooling."""

    sequences: GenericSequenceSet = Field(
        description="Protein sequences: a dict of id→sequence, FASTA text, or a FASTA file path.",
        json_schema_extra=ui(widget=Widget.SEQUENCE, flavor="generic", tier=Tier.PRIMARY, order=0),
    )
    layer: int | None = Field(
        default=None,
        description="Model layer to extract embeddings from (None = final layer).",
        json_schema_extra=ui(widget=Widget.NUMBER, tier=Tier.ADVANCED, order=10),
    )
    pooling: Literal["mean", "cls", "per_residue"] = Field(
        default="mean",
        description="Pooling strategy for per-residue embeddings.",
        json_schema_extra=ui(widget=Widget.SELECT, tier=Tier.PRIMARY, order=1),
    )


class ESM2Input(ESMEmbedInput):
    """Input for ESM-2 (adds checkpoint size selection)."""

    checkpoint: Literal["8M", "35M", "150M", "650M", "3B", "15B"] = Field(
        default="650M",
        description="ESM-2 checkpoint size.",
        json_schema_extra=ui(
            widget=Widget.SELECT,
            tier=Tier.PRIMARY,
            order=2,
            enum_labels={
                "8M": "8M (t6)",
                "35M": "35M (t12)",
                "150M": "150M (t30)",
                "650M": "650M (t33, default)",
                "3B": "3B (t36)",
                "15B": "15B (t48)",
            },
        ),
    )


class SequenceEmbedding(BaseModel):
    """Embedding result for a single sequence."""

    sequence_id: str = Field(description="Identifier matching a key in the input sequences.")
    embedding_path: Path = Field(
        description="Path to the embedding file in outputs/standardized/ (e.g., .npy or .pt)."
    )
    dimension: int = Field(description="Dimensionality of the embedding vector.")
    layer: int | None = Field(
        default=None,
        description="Model layer from which the embedding was extracted.",
    )
    pooling: str | None = Field(
        default=None,
        description="Pooling strategy applied to produce this embedding.",
    )


class EmbeddingOutput(BaseOutput):
    """Output schema for sequence embedding tools."""

    embeddings: list[SequenceEmbedding] = Field(
        description="Embedding results for each input sequence."
    )
    model_name: str = Field(
        description="Name of the embedding model used (e.g., 'esm2_t33_650M_UR50D')."
    )
    embedding_dimension: int = Field(description="Dimensionality of all output embeddings.")
