"""Tests for category schemas (structure_prediction, embedding, inverse_folding, scoring)."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from autobio.schemas.antibody import (
    AntibodyInput,
    AntibodyPLLOutput,
    AntibodySequence,
    SequencePLL,
)
from autobio.schemas.embedding import EmbeddingInput, EmbeddingOutput, SequenceEmbedding
from autobio.schemas.inverse_folding import (
    DesignedSequence,
    InverseFoldingInput,
    InverseFoldingOutput,
)
from autobio.schemas.scoring import ScoredStructure, ScoringInput, ScoringOutput
from autobio.schemas.structure_design import (
    DesignedStructure as DesignDesignedStructure,
)
from autobio.schemas.structure_design import (
    StructureDesignInput,
    StructureDesignOutput,
)
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
# Antibody
# ---------------------------------------------------------------------------


class TestAntibodySequence:
    def test_paired_sequence(self) -> None:
        seq = AntibodySequence(id="ab1", heavy_chain="EVQLV", light_chain="DIQMT")
        assert seq.heavy_chain == "EVQLV"
        assert seq.light_chain == "DIQMT"

    def test_heavy_only(self) -> None:
        seq = AntibodySequence(id="ab1", heavy_chain="EVQLV")
        assert seq.heavy_chain == "EVQLV"
        assert seq.light_chain is None

    def test_light_only(self) -> None:
        seq = AntibodySequence(id="ab1", light_chain="DIQMT")
        assert seq.heavy_chain is None
        assert seq.light_chain == "DIQMT"

    def test_neither_chain_raises(self) -> None:
        with pytest.raises(ValidationError, match="at least one"):
            AntibodySequence(id="ab1")

    def test_round_trip(self) -> None:
        seq = AntibodySequence(id="ab1", heavy_chain="EVQLV", light_chain="DIQMT")
        dumped = seq.model_dump()
        restored = AntibodySequence.model_validate(dumped)
        assert restored == seq


class TestAntibodyInput:
    def test_required_sequences(self) -> None:
        inp = AntibodyInput(sequences=[AntibodySequence(id="ab1", heavy_chain="EVQLV")])
        assert len(inp.sequences) == 1
        assert inp.layer is None
        assert inp.pooling is None

    def test_optional_fields(self) -> None:
        inp = AntibodyInput(
            sequences=[AntibodySequence(id="ab1", heavy_chain="EVQLV")],
            layer=10,
            pooling="mean",
        )
        assert inp.layer == 10
        assert inp.pooling == "mean"

    def test_extra_passthrough(self) -> None:
        inp = AntibodyInput(
            sequences=[AntibodySequence(id="ab1", heavy_chain="EVQLV")],
            extra={"per_position": True, "batch_size": 8},
        )
        assert inp.extra["per_position"] is True
        assert inp.extra["batch_size"] == 8

    def test_round_trip(self) -> None:
        inp = AntibodyInput(
            sequences=[
                AntibodySequence(id="ab1", heavy_chain="EVQLV", light_chain="DIQMT"),
                AntibodySequence(id="ab2", heavy_chain="QVQLV"),
            ],
            layer=20,
            pooling="per_residue",
        )
        dumped = inp.model_dump()
        restored = AntibodyInput.model_validate(dumped)
        assert len(restored.sequences) == 2
        assert restored.layer == 20

    def test_accepts_fasta_text(self) -> None:
        fasta = ">ab1|heavy\nEVQLVESGG\n>ab1|light\nDIQMTQSPS\n"
        inp = AntibodyInput(sequences=fasta)
        assert len(inp.sequences) == 1
        assert inp.sequences[0].id == "ab1"
        assert inp.sequences[0].heavy_chain == "EVQLVESGG"
        assert inp.sequences[0].light_chain == "DIQMTQSPS"


class TestSequencePLL:
    def test_required_fields(self) -> None:
        s = SequencePLL(sequence_id="ab1", pll=-45.23, sequence_length=100)
        assert s.pll == -45.23
        assert s.per_position_pll is None

    def test_optional_per_position(self) -> None:
        s = SequencePLL(
            sequence_id="ab1",
            pll=-3.5,
            per_position_pll=[-1.2, -0.8, -1.5],
            sequence_length=3,
        )
        assert s.per_position_pll == [-1.2, -0.8, -1.5]

    def test_round_trip(self) -> None:
        s = SequencePLL(sequence_id="ab1", pll=-45.0, sequence_length=100)
        dumped = s.model_dump()
        restored = SequencePLL.model_validate(dumped)
        assert restored == s


class TestAntibodyPLLOutput:
    def test_round_trip(self) -> None:
        out = AntibodyPLLOutput(
            metadata=_METADATA,  # type: ignore[arg-type]
            raw_output_path=Path("/tmp/ws/outputs/raw"),
            scores=[SequencePLL(sequence_id="ab1", pll=-45.0, sequence_length=100)],
            model_name="CurrAb",
        )
        dumped = out.model_dump()
        restored = AntibodyPLLOutput.model_validate(dumped)
        assert len(restored.scores) == 1
        assert restored.model_name == "CurrAb"

    def test_missing_required_raises(self) -> None:
        with pytest.raises(ValidationError):
            AntibodyPLLOutput(
                metadata=_METADATA,  # type: ignore[arg-type]
                raw_output_path=Path("/tmp"),
                # missing scores and model_name
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
# StructureDesign
# ---------------------------------------------------------------------------


class TestStructureDesignInput:
    def test_required_design_specs(self) -> None:
        inp = StructureDesignInput(design_specs={"test": {"length": "50"}})
        assert "test" in inp.design_specs
        assert inp.n_batches == 1
        assert inp.input_structures == []

    def test_optional_fields(self) -> None:
        inp = StructureDesignInput(
            design_specs={"test": {"length": "50"}},
            input_structures=[Path("/data/target.pdb")],
            n_batches=3,
        )
        assert inp.n_batches == 3
        assert inp.input_structures == [Path("/data/target.pdb")]

    def test_missing_design_specs_raises(self) -> None:
        with pytest.raises(ValidationError):
            StructureDesignInput()  # type: ignore[call-arg]

    def test_extra_dict_passthrough(self) -> None:
        inp = StructureDesignInput(
            design_specs={"test": {"length": "50"}},
            extra={"step_scale": 3.0, "gamma_0": 0.2},
        )
        assert inp.extra["step_scale"] == 3.0
        assert inp.extra["gamma_0"] == 0.2

    def test_round_trip(self) -> None:
        inp = StructureDesignInput(
            design_specs={
                "binder": {"input": "target.pdb", "contig": "40-80"},
                "uncond": {"length": "100"},
            },
            input_structures=[Path("/data/target.pdb")],
            n_batches=2,
        )
        dumped = inp.model_dump()
        restored = StructureDesignInput.model_validate(dumped)
        assert restored.design_specs == inp.design_specs
        assert restored.n_batches == inp.n_batches


class TestDesignDesignedStructure:
    def test_required_fields(self) -> None:
        ds = DesignDesignedStructure(
            spec_name="test",
            batch_index=0,
            design_index=0,
            structure_path=Path("outputs/standardized/test_b0_d0.cif"),
        )
        assert ds.spec_name == "test"
        assert ds.diffusion_metadata is None

    def test_all_fields(self) -> None:
        ds = DesignDesignedStructure(
            spec_name="binder",
            batch_index=1,
            design_index=3,
            structure_path=Path("outputs/standardized/binder_b1_d3.cif"),
            diffusion_metadata={"timing": {"total_seconds": 42.5}},
        )
        assert ds.diffusion_metadata is not None
        assert ds.diffusion_metadata["timing"]["total_seconds"] == 42.5


class TestStructureDesignOutput:
    def test_round_trip(self) -> None:
        out = StructureDesignOutput(
            metadata=_METADATA,  # type: ignore[arg-type]
            raw_output_path=Path("/tmp/ws/outputs/raw"),
            designs=[
                DesignDesignedStructure(
                    spec_name="test",
                    batch_index=0,
                    design_index=0,
                    structure_path=Path("outputs/standardized/test_b0_d0.cif"),
                    diffusion_metadata={"timing": {"total_seconds": 10.0}},
                ),
            ],
            spec_summary={"test": 1},
        )
        dumped = out.model_dump()
        restored = StructureDesignOutput.model_validate(dumped)
        assert len(restored.designs) == 1
        assert restored.designs[0].spec_name == "test"
        assert restored.spec_summary == {"test": 1}

    def test_missing_required_raises(self) -> None:
        with pytest.raises(ValidationError):
            StructureDesignOutput(
                metadata=_METADATA,  # type: ignore[arg-type]
                raw_output_path=Path("/tmp"),
                # missing designs and spec_summary
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
        (StructureDesignInput, {"design_specs": {"t": {"length": "50"}}}),
        (
            AntibodyInput,
            {"sequences": [AntibodySequence(id="ab1", heavy_chain="EVQLV")]},
        ),
    ],
    ids=[
        "structure_prediction",
        "embedding",
        "inverse_folding",
        "scoring",
        "structure_design",
        "antibody",
    ],
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
        (
            StructureDesignOutput,
            {
                "designs": [
                    DesignDesignedStructure(
                        spec_name="t",
                        batch_index=0,
                        design_index=0,
                        structure_path=Path("t_b0_d0.cif"),
                    ),
                ],
                "spec_summary": {"t": 1},
            },
        ),
        (
            AntibodyPLLOutput,
            {
                "scores": [SequencePLL(sequence_id="ab1", pll=-45.0, sequence_length=100)],
                "model_name": "CurrAb",
            },
        ),
    ],
    ids=[
        "structure_prediction",
        "embedding",
        "inverse_folding",
        "scoring",
        "structure_design",
        "antibody_pll",
    ],
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
