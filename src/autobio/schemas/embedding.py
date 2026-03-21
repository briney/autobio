"""Input/output schemas for sequence embedding tools."""

from __future__ import annotations

from pathlib import Path  # noqa: TC003 - Pydantic needs at runtime

from pydantic import BaseModel, Field

from autobio.schemas.base import BaseInput, BaseOutput


class EmbeddingInput(BaseInput):
    """Input schema for sequence embedding tools (e.g., ESM-2, ProtTrans)."""

    sequences: dict[str, str] = Field(
        description="Mapping of sequence ID to amino acid sequence (e.g., {'seq1': 'MKLL...'})."
    )
    layer: int | None = Field(
        default=None,
        description="Model layer from which to extract embeddings. None uses the final layer.",
    )
    pooling: str | None = Field(
        default=None,
        description=(
            "Pooling strategy for per-residue embeddings "
            "(e.g., 'mean', 'cls', 'per_residue')."
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
    embedding_dimension: int = Field(
        description="Dimensionality of all output embeddings."
    )
