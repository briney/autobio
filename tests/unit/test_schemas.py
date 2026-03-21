"""Tests for category schemas (structure_prediction, embedding, inverse_folding, scoring)."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from autobio.schemas.embedding import EmbeddingInput, EmbeddingOutput, SequenceEmbedding
from autobio.schemas.inverse_folding import (
    DesignedSequence,
    InverseFoldingInput,
    InverseFoldingOutput,
)
from autobio.schemas.scoring import ScoredStructure, ScoringInput, ScoringOutput
from autobio.schemas.structure_prediction import (
    ConfidenceMetrics,
    PredictedStructure,
    StructurePredictionInput,
    StructurePredictionOutput,
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_METADATA: dict[str, Any] = {
    "tool_name": "test-tool",
    "tool_version": "1.0.0",
    "image_uri": "ghcr.io/briney/autobio-test:1.0.0",
    "wall_time_seconds": 10.0,
    "workspace_path": "/tmp/ws",
    "timestamp": datetime(2025, 1, 1, tzinfo=UTC).isoformat(),
}


# ---------------------------------------------------------------------------
# StructurePrediction
# ---------------------------------------------------------------------------


class TestStructurePredictionInput:
    def test_required_sequences(self) -> None:
        inp = StructurePredictionInput(sequences={"A": "MKLLVVF"})
        assert inp.sequences == {"A": "MKLLVVF"}
        assert inp.num_models == 1
        assert inp.templates is None

    def test_optional_fields(self) -> None:
        inp = StructurePredictionInput(
            sequences={"A": "MKLLVVF"},
            num_models=5,
            templates=[Path("/data/template.pdb")],
        )
        assert inp.num_models == 5
        assert inp.templates == [Path("/data/template.pdb")]

    def test_missing_sequences_raises(self) -> None:
        with pytest.raises(ValidationError):
            StructurePredictionInput()  # type: ignore[call-arg]

    def test_extra_dict_passthrough(self) -> None:
        inp = StructurePredictionInput(
            sequences={"A": "MK"},
            extra={"recycles": 3},
        )
        assert inp.extra["recycles"] == 3

    def test_round_trip(self) -> None:
        inp = StructurePredictionInput(
            sequences={"A": "MKLLVVF", "B": "GVSEK"},
            num_models=3,
        )
        dumped = inp.model_dump()
        restored = StructurePredictionInput.model_validate(dumped)
        assert restored.sequences == inp.sequences
        assert restored.num_models == inp.num_models


class TestPredictedStructure:
    def test_required_fields(self) -> None:
        ps = PredictedStructure(
            model_rank=1,
            structure_path=Path("outputs/standardized/model_1.pdb"),
        )
        assert ps.model_rank == 1
        assert ps.plddt_mean is None
        assert ps.ptm is None
        assert ps.iptm is None
        assert ps.chain_mapping is None

    def test_all_fields(self) -> None:
        ps = PredictedStructure(
            model_rank=1,
            structure_path=Path("outputs/standardized/model_1.pdb"),
            plddt_per_residue=[92.1, 88.4, 91.7],
            plddt_mean=90.7,
            ptm=0.89,
            iptm=0.85,
            chain_mapping={"A": "A", "B": "B"},
        )
        assert ps.plddt_mean == 90.7
        assert ps.chain_mapping == {"A": "A", "B": "B"}


class TestConfidenceMetrics:
    def test_all_optional(self) -> None:
        cm = ConfidenceMetrics()
        assert cm.best_plddt_mean is None
        assert cm.best_ptm is None
        assert cm.best_iptm is None

    def test_with_values(self) -> None:
        cm = ConfidenceMetrics(best_plddt_mean=90.7, best_ptm=0.89, best_iptm=0.85)
        assert cm.best_plddt_mean == 90.7


class TestStructurePredictionOutput:
    def test_round_trip(self) -> None:
        out = StructurePredictionOutput(
            metadata=_METADATA,  # type: ignore[arg-type]
            raw_output_path=Path("/tmp/ws/outputs/raw"),
            structures=[
                PredictedStructure(
                    model_rank=1,
                    structure_path=Path("outputs/standardized/model_1.pdb"),
                    plddt_mean=90.7,
                    ptm=0.89,
                ),
            ],
            confidence=ConfidenceMetrics(best_plddt_mean=90.7, best_ptm=0.89),
        )
        dumped = out.model_dump()
        restored = StructurePredictionOutput.model_validate(dumped)
        assert len(restored.structures) == 1
        assert restored.structures[0].plddt_mean == 90.7
        assert restored.confidence.best_ptm == 0.89
        assert restored.metadata.tool_name == "test-tool"

    def test_missing_required_raises(self) -> None:
        with pytest.raises(ValidationError):
            StructurePredictionOutput(
                metadata=_METADATA,  # type: ignore[arg-type]
                raw_output_path=Path("/tmp"),
                # missing structures and confidence
            )  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------


class TestEmbeddingInput:
    def test_required_sequences(self) -> None:
        inp = EmbeddingInput(sequences={"seq1": "MKLLVVF"})
        assert inp.sequences == {"seq1": "MKLLVVF"}
        assert inp.layer is None
        assert inp.pooling is None

    def test_optional_fields(self) -> None:
        inp = EmbeddingInput(
            sequences={"seq1": "MK"},
            layer=33,
            pooling="mean",
        )
        assert inp.layer == 33
        assert inp.pooling == "mean"

    def test_missing_sequences_raises(self) -> None:
        with pytest.raises(ValidationError):
            EmbeddingInput()  # type: ignore[call-arg]

    def test_extra_dict_passthrough(self) -> None:
        inp = EmbeddingInput(
            sequences={"seq1": "MK"},
            extra={"batch_size": 32},
        )
        assert inp.extra["batch_size"] == 32

    def test_round_trip(self) -> None:
        inp = EmbeddingInput(
            sequences={"seq1": "MKLLVVF", "seq2": "GVSEK"},
            layer=33,
            pooling="mean",
        )
        dumped = inp.model_dump()
        restored = EmbeddingInput.model_validate(dumped)
        assert restored.sequences == inp.sequences
        assert restored.layer == 33


class TestSequenceEmbedding:
    def test_required_fields(self) -> None:
        se = SequenceEmbedding(
            sequence_id="seq1",
            embedding_path=Path("outputs/standardized/seq1.npy"),
            dimension=1280,
        )
        assert se.dimension == 1280
        assert se.layer is None
        assert se.pooling is None

    def test_all_fields(self) -> None:
        se = SequenceEmbedding(
            sequence_id="seq1",
            embedding_path=Path("outputs/standardized/seq1.npy"),
            dimension=1280,
            layer=33,
            pooling="mean",
        )
        assert se.layer == 33
        assert se.pooling == "mean"


class TestEmbeddingOutput:
    def test_round_trip(self) -> None:
        out = EmbeddingOutput(
            metadata=_METADATA,  # type: ignore[arg-type]
            raw_output_path=Path("/tmp/ws/outputs/raw"),
            embeddings=[
                SequenceEmbedding(
                    sequence_id="seq1",
                    embedding_path=Path("outputs/standardized/seq1.npy"),
                    dimension=1280,
                    layer=33,
                    pooling="mean",
                ),
            ],
            model_name="esm2_t33_650M_UR50D",
            embedding_dimension=1280,
        )
        dumped = out.model_dump()
        restored = EmbeddingOutput.model_validate(dumped)
        assert len(restored.embeddings) == 1
        assert restored.embeddings[0].sequence_id == "seq1"
        assert restored.model_name == "esm2_t33_650M_UR50D"
        assert restored.embedding_dimension == 1280

    def test_missing_required_raises(self) -> None:
        with pytest.raises(ValidationError):
            EmbeddingOutput(
                metadata=_METADATA,  # type: ignore[arg-type]
                raw_output_path=Path("/tmp"),
                # missing embeddings, model_name, embedding_dimension
            )  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# InverseFolding
# ---------------------------------------------------------------------------


class TestInverseFoldingInput:
    def test_required_structure_path(self) -> None:
        inp = InverseFoldingInput(structure_path=Path("/data/backbone.pdb"))
        assert inp.structure_path == Path("/data/backbone.pdb")
        assert inp.chains_to_design is None
        assert inp.num_sequences == 1
        assert inp.temperature == 0.1
        assert inp.fixed_positions is None

    def test_optional_fields(self) -> None:
        inp = InverseFoldingInput(
            structure_path=Path("/data/backbone.pdb"),
            chains_to_design=["A"],
            num_sequences=10,
            temperature=0.2,
            fixed_positions={"A": [1, 5, 10]},
        )
        assert inp.chains_to_design == ["A"]
        assert inp.num_sequences == 10
        assert inp.temperature == 0.2
        assert inp.fixed_positions == {"A": [1, 5, 10]}

    def test_missing_structure_path_raises(self) -> None:
        with pytest.raises(ValidationError):
            InverseFoldingInput()  # type: ignore[call-arg]

    def test_extra_dict_passthrough(self) -> None:
        inp = InverseFoldingInput(
            structure_path=Path("/data/backbone.pdb"),
            extra={"tied_positions": [[1, 2], [3, 4]]},
        )
        assert inp.extra["tied_positions"] == [[1, 2], [3, 4]]

    def test_round_trip(self) -> None:
        inp = InverseFoldingInput(
            structure_path=Path("/data/backbone.pdb"),
            chains_to_design=["A", "B"],
            num_sequences=5,
            temperature=0.15,
            fixed_positions={"A": [1, 2, 3]},
        )
        dumped = inp.model_dump()
        restored = InverseFoldingInput.model_validate(dumped)
        assert restored.structure_path == inp.structure_path
        assert restored.fixed_positions == inp.fixed_positions


class TestDesignedSequence:
    def test_required_fields(self) -> None:
        ds = DesignedSequence(rank=1, sequence={"A": "MKLLVVF"})
        assert ds.rank == 1
        assert ds.score is None
        assert ds.recovery is None

    def test_all_fields(self) -> None:
        ds = DesignedSequence(
            rank=1,
            sequence={"A": "MKLLVVF", "B": "GVSEK"},
            score=-2.34,
            recovery=0.75,
        )
        assert ds.score == -2.34
        assert ds.recovery == 0.75


class TestInverseFoldingOutput:
    def test_round_trip(self) -> None:
        out = InverseFoldingOutput(
            metadata=_METADATA,  # type: ignore[arg-type]
            raw_output_path=Path("/tmp/ws/outputs/raw"),
            designed_sequences=[
                DesignedSequence(
                    rank=1,
                    sequence={"A": "MKLLVVF"},
                    score=-2.34,
                    recovery=0.75,
                ),
                DesignedSequence(
                    rank=2,
                    sequence={"A": "MKLLVVG"},
                    score=-2.10,
                ),
            ],
            native_sequence={"A": "MKLLVVF"},
        )
        dumped = out.model_dump()
        restored = InverseFoldingOutput.model_validate(dumped)
        assert len(restored.designed_sequences) == 2
        assert restored.designed_sequences[0].score == -2.34
        assert restored.native_sequence == {"A": "MKLLVVF"}

    def test_native_sequence_optional(self) -> None:
        out = InverseFoldingOutput(
            metadata=_METADATA,  # type: ignore[arg-type]
            raw_output_path=Path("/tmp/ws/outputs/raw"),
            designed_sequences=[
                DesignedSequence(rank=1, sequence={"A": "MK"}),
            ],
        )
        assert out.native_sequence is None

    def test_missing_required_raises(self) -> None:
        with pytest.raises(ValidationError):
            InverseFoldingOutput(
                metadata=_METADATA,  # type: ignore[arg-type]
                raw_output_path=Path("/tmp"),
                # missing designed_sequences
            )  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


class TestScoringInput:
    def test_required_structure_path(self) -> None:
        inp = ScoringInput(structure_path=Path("/data/structure.pdb"))
        assert inp.structure_path == Path("/data/structure.pdb")
        assert inp.sequences is None

    def test_optional_sequences(self) -> None:
        inp = ScoringInput(
            structure_path=Path("/data/structure.pdb"),
            sequences={"A": "MKLLVVF"},
        )
        assert inp.sequences == {"A": "MKLLVVF"}

    def test_missing_structure_path_raises(self) -> None:
        with pytest.raises(ValidationError):
            ScoringInput()  # type: ignore[call-arg]

    def test_extra_dict_passthrough(self) -> None:
        inp = ScoringInput(
            structure_path=Path("/data/structure.pdb"),
            extra={"relax": True},
        )
        assert inp.extra["relax"] is True

    def test_round_trip(self) -> None:
        inp = ScoringInput(
            structure_path=Path("/data/structure.pdb"),
            sequences={"A": "MK", "B": "GV"},
        )
        dumped = inp.model_dump()
        restored = ScoringInput.model_validate(dumped)
        assert restored.structure_path == inp.structure_path
        assert restored.sequences == inp.sequences


class TestScoredStructure:
    def test_required_total_score(self) -> None:
        ss = ScoredStructure(total_score=-250.5)
        assert ss.total_score == -250.5
        assert ss.per_residue_scores is None
        assert ss.score_breakdown is None
        assert ss.units is None

    def test_all_fields(self) -> None:
        ss = ScoredStructure(
            total_score=-250.5,
            per_residue_scores=[-3.2, -1.5, -4.8],
            score_breakdown={"van_der_waals": -120.5, "electrostatics": -45.2},
            units="REU",
        )
        assert ss.per_residue_scores == [-3.2, -1.5, -4.8]
        assert ss.score_breakdown["van_der_waals"] == -120.5
        assert ss.units == "REU"


class TestScoringOutput:
    def test_round_trip(self) -> None:
        out = ScoringOutput(
            metadata=_METADATA,  # type: ignore[arg-type]
            raw_output_path=Path("/tmp/ws/outputs/raw"),
            scores=[
                ScoredStructure(
                    total_score=-250.5,
                    per_residue_scores=[-3.2, -1.5],
                    score_breakdown={"vdw": -120.5, "elec": -45.2},
                    units="REU",
                ),
            ],
        )
        dumped = out.model_dump()
        restored = ScoringOutput.model_validate(dumped)
        assert len(restored.scores) == 1
        assert restored.scores[0].total_score == -250.5
        assert restored.scores[0].units == "REU"

    def test_missing_required_raises(self) -> None:
        with pytest.raises(ValidationError):
            ScoringOutput(
                metadata=_METADATA,  # type: ignore[arg-type]
                raw_output_path=Path("/tmp"),
                # missing scores
            )  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# Cross-cutting: inheritance from BaseInput/BaseOutput
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "input_cls,kwargs",
    [
        (StructurePredictionInput, {"sequences": {"A": "MK"}}),
        (EmbeddingInput, {"sequences": {"seq1": "MK"}}),
        (InverseFoldingInput, {"structure_path": Path("/x.pdb")}),
        (ScoringInput, {"structure_path": Path("/x.pdb")}),
    ],
    ids=["structure_prediction", "embedding", "inverse_folding", "scoring"],
)
class TestInputInheritance:
    def test_has_extra_field(self, input_cls: type, kwargs: dict[str, Any]) -> None:
        inp = input_cls(**kwargs)
        assert hasattr(inp, "extra")
        assert inp.extra == {}

    def test_extra_passthrough(self, input_cls: type, kwargs: dict[str, Any]) -> None:
        inp = input_cls(**kwargs, extra={"custom": 42})
        assert inp.extra["custom"] == 42


@pytest.mark.parametrize(
    "output_cls,kwargs",
    [
        (
            StructurePredictionOutput,
            {
                "structures": [
                    PredictedStructure(
                        model_rank=1,
                        structure_path=Path("model.pdb"),
                    ),
                ],
                "confidence": ConfidenceMetrics(),
            },
        ),
        (
            EmbeddingOutput,
            {
                "embeddings": [
                    SequenceEmbedding(
                        sequence_id="s1",
                        embedding_path=Path("s1.npy"),
                        dimension=1280,
                    ),
                ],
                "model_name": "esm2",
                "embedding_dimension": 1280,
            },
        ),
        (
            InverseFoldingOutput,
            {
                "designed_sequences": [
                    DesignedSequence(rank=1, sequence={"A": "MK"}),
                ],
            },
        ),
        (
            ScoringOutput,
            {
                "scores": [ScoredStructure(total_score=-100.0)],
            },
        ),
    ],
    ids=["structure_prediction", "embedding", "inverse_folding", "scoring"],
)
class TestOutputInheritance:
    def test_has_metadata_and_raw_output_path(
        self, output_cls: type, kwargs: dict[str, Any]
    ) -> None:
        out = output_cls(
            metadata=_METADATA,  # type: ignore[arg-type]
            raw_output_path=Path("/tmp/ws/outputs/raw"),
            **kwargs,
        )
        assert out.metadata.tool_name == "test-tool"
        assert out.raw_output_path == Path("/tmp/ws/outputs/raw")

    def test_missing_metadata_raises(self, output_cls: type, kwargs: dict[str, Any]) -> None:
        with pytest.raises(ValidationError):
            output_cls(
                raw_output_path=Path("/tmp"),
                **kwargs,
            )
