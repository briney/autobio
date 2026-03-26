"""Standardised I/O schemas for autobio tool categories."""

from __future__ import annotations

from autobio.schemas.base import BaseInput, BaseOutput, RunMetadata
from autobio.schemas.embedding import EmbeddingInput, EmbeddingOutput, SequenceEmbedding
from autobio.schemas.inverse_folding import (
    DesignedSequence,
    InverseFoldingInput,
    InverseFoldingOutput,
)
from autobio.schemas.scoring import ScoredStructure, ScoringInput, ScoringOutput
from autobio.schemas.structure_design import (
    DesignedStructure,
    StructureDesignInput,
    StructureDesignOutput,
)
from autobio.schemas.structure_prediction import (
    ConfidenceMetrics,
    PredictedStructure,
    StructurePredictionInput,
    StructurePredictionOutput,
)

__all__ = [
    # base
    "BaseInput",
    "BaseOutput",
    "RunMetadata",
    # structure prediction
    "ConfidenceMetrics",
    "PredictedStructure",
    "StructurePredictionInput",
    "StructurePredictionOutput",
    # embedding
    "EmbeddingInput",
    "EmbeddingOutput",
    "SequenceEmbedding",
    # inverse folding
    "DesignedSequence",
    "InverseFoldingInput",
    "InverseFoldingOutput",
    # scoring
    "ScoredStructure",
    "ScoringInput",
    "ScoringOutput",
    # structure design
    "DesignedStructure",
    "StructureDesignInput",
    "StructureDesignOutput",
]
