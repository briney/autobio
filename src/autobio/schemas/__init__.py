"""Standardised I/O schemas for autobio tool categories."""

from __future__ import annotations

from autobio.schemas.antibody import (
    AntibodyInput,
    AntibodyPLLOutput,
    AntibodySequence,
    SequencePLL,
)
from autobio.schemas.base import BaseInput, BaseOutput, RunMetadata
from autobio.schemas.binding_affinity import (
    BindingAffinityInput,
    BindingAffinityOutput,
    BindingAffinityPrediction,
)
from autobio.schemas.embedding import EmbeddingOutput, SequenceEmbedding
from autobio.schemas.inverse_folding import (
    DesignedSequence,
    InverseFoldingInput,
    InverseFoldingOutput,
)
from autobio.schemas.protein_binding_affinity import (
    ProteinBindingAffinityInput,
    ProteinBindingAffinityOutput,
    ProteinBindingAffinityPrediction,
)
from autobio.schemas.scoring import ScoredStructure, ScoringInput, ScoringOutput
from autobio.schemas.simulation import (
    EnergyRecord,
    SimulationInput,
    SimulationOutput,
    SimulationSummary,
)
from autobio.schemas.structure_design import (
    DesignedStructure,
    StructureDesignOutput,
)
from autobio.schemas.structure_prediction import (
    ConfidenceMetrics,
    PredictedStructure,
    StructurePredictionOutput,
)

__all__ = [
    # binding affinity
    "BindingAffinityInput",
    "BindingAffinityOutput",
    "BindingAffinityPrediction",
    # antibody
    "AntibodyInput",
    "AntibodyPLLOutput",
    "AntibodySequence",
    "SequencePLL",
    # base
    "BaseInput",
    "BaseOutput",
    "RunMetadata",
    # structure prediction
    "ConfidenceMetrics",
    "PredictedStructure",
    "StructurePredictionOutput",
    # embedding
    "EmbeddingOutput",
    "SequenceEmbedding",
    # inverse folding
    "DesignedSequence",
    "InverseFoldingInput",
    "InverseFoldingOutput",
    # protein binding affinity
    "ProteinBindingAffinityInput",
    "ProteinBindingAffinityOutput",
    "ProteinBindingAffinityPrediction",
    # scoring
    "ScoredStructure",
    "ScoringInput",
    "ScoringOutput",
    # simulation
    "EnergyRecord",
    "SimulationInput",
    "SimulationOutput",
    "SimulationSummary",
    # structure design
    "DesignedStructure",
    "StructureDesignOutput",
]
