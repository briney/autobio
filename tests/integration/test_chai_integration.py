"""Integration tests for Chai-1 structure prediction runner.

These tests require Docker and a GPU. They run the autobio-chai container
and verify end-to-end structure prediction output.

Run with:
    pytest tests/integration/test_chai_integration.py -v -m docker
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from autobio.core.config import AutobioConfig
from autobio.schemas.structure_prediction import Chai1Input, StructurePredictionOutput
from autobio.tools import get_runner

if TYPE_CHECKING:
    from pathlib import Path

# All tests in this module require Docker, GPU, and are slow
pytestmark = [pytest.mark.docker, pytest.mark.gpu, pytest.mark.slow]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

# Short test sequences — keep predictions fast on GPU
# Crambin (PDB: 1CRN) — 46 residues, single chain, well-folded small protein
_CRAMBIN_SEQ = "TTCCPSIVARSNFNVCRLPGTPEAICATYTGCIIIPGATCPGDYAN"

# Insulin A and B chains — small, well-characterized complex
_INSULIN_A = "GIVEQCCTSICSLYQLENYCN"
_INSULIN_B = "FVNQHLCGSHLVEALYLVCGERGFFYTPKT"


@pytest.fixture(scope="session")
def autobio_config() -> AutobioConfig:
    return AutobioConfig.resolve()


# ---------------------------------------------------------------------------
# Chai-1 — single protein
# ---------------------------------------------------------------------------


class TestChai1SingleProtein:
    """Chai-1 structure prediction on a single small protein."""

    def test_full_pipeline(self, autobio_config: AutobioConfig, tmp_path: Path) -> None:
        """Run Chai-1 end-to-end and verify output structure."""
        input_data = Chai1Input(
            sequences={"A": _CRAMBIN_SEQ},
            num_models=1,
            extra={
                "num_trunk_recycles": 1,
                "num_diffn_timesteps": 50,
            },
        )
        runner = get_runner("chai1", autobio_config)
        output = runner.run(input_data, gpu="auto", output_dir=tmp_path / "ws")

        assert isinstance(output, StructurePredictionOutput)
        assert len(output.structures) == 1
        assert output.metadata.tool_name == "chai1"
        assert output.metadata.wall_time_seconds > 0

        s = output.structures[0]
        assert s.model_rank == 1
        assert s.structure_path.exists()

    def test_confidence_scores(self, autobio_config: AutobioConfig, tmp_path: Path) -> None:
        """Verify confidence metrics are populated."""
        input_data = Chai1Input(
            sequences={"A": _CRAMBIN_SEQ},
            num_models=1,
            extra={
                "num_trunk_recycles": 1,
                "num_diffn_timesteps": 50,
            },
        )
        runner = get_runner("chai1", autobio_config)
        output = runner.run(input_data, gpu="auto", output_dir=tmp_path / "ws")

        s = output.structures[0]
        # Confidence scores should be populated
        assert s.ptm is not None
        assert 0.0 <= s.ptm <= 1.0
        assert s.plddt_mean is not None
        assert s.plddt_mean > 0

        # Summary metrics should match the single model
        assert output.confidence.best_ptm is not None
        assert output.confidence.best_plddt_mean is not None

    def test_multiple_models(self, autobio_config: AutobioConfig, tmp_path: Path) -> None:
        """Request 3 models and verify count and ranking."""
        input_data = Chai1Input(
            sequences={"A": _CRAMBIN_SEQ},
            num_models=3,
            extra={
                "num_trunk_recycles": 1,
                "num_diffn_timesteps": 50,
            },
        )
        runner = get_runner("chai1", autobio_config)
        output = runner.run(input_data, gpu="auto", output_dir=tmp_path / "ws")

        assert len(output.structures) == 3
        ranks = [s.model_rank for s in output.structures]
        assert ranks == [1, 2, 3]

        # All structures should have files
        for s in output.structures:
            assert s.structure_path.exists()

        # Best confidence metrics should come from rank 1
        assert output.confidence.best_ptm is not None


# ---------------------------------------------------------------------------
# Chai-1 — multi-chain complex
# ---------------------------------------------------------------------------


class TestChai1MultiChain:
    """Chai-1 prediction of a multi-chain complex (insulin A+B)."""

    def test_two_chain_complex(self, autobio_config: AutobioConfig, tmp_path: Path) -> None:
        """Predict a 2-chain insulin complex."""
        input_data = Chai1Input(
            sequences={"A": _INSULIN_A, "B": _INSULIN_B},
            num_models=1,
            extra={
                "num_trunk_recycles": 1,
                "num_diffn_timesteps": 50,
            },
        )
        runner = get_runner("chai1", autobio_config)
        output = runner.run(input_data, gpu="auto", output_dir=tmp_path / "ws")

        assert isinstance(output, StructurePredictionOutput)
        assert len(output.structures) == 1

        s = output.structures[0]
        assert s.structure_path.exists()
        assert s.ptm is not None


# ---------------------------------------------------------------------------
# Chai-1 — protein + ligand
# ---------------------------------------------------------------------------


class TestChai1ProteinLigand:
    """Chai-1 prediction with a protein and small molecule ligand."""

    def test_protein_ligand_complex(self, autobio_config: AutobioConfig, tmp_path: Path) -> None:
        """Predict a protein-ligand complex using SMILES notation."""
        # Ibuprofen SMILES
        ibuprofen = "CC(C)Cc1ccc(cc1)C(C)C(O)=O"
        input_data = Chai1Input(
            sequences={"A": _CRAMBIN_SEQ, "L": ibuprofen},
            num_models=1,
            entity_types={"L": "ligand"},
            extra={
                "num_trunk_recycles": 1,
                "num_diffn_timesteps": 50,
            },
        )
        runner = get_runner("chai1", autobio_config)
        output = runner.run(input_data, gpu="auto", output_dir=tmp_path / "ws")

        assert isinstance(output, StructurePredictionOutput)
        assert len(output.structures) == 1
        assert output.structures[0].structure_path.exists()


# ---------------------------------------------------------------------------
# Chai-1 — with contact restraints
# ---------------------------------------------------------------------------


class TestChai1Restraints:
    """Chai-1 prediction with contact restraints."""

    def test_with_contact_restraint(self, autobio_config: AutobioConfig, tmp_path: Path) -> None:
        """Predict a 2-chain complex with a contact restraint."""
        csv_content = (
            "chainA,res_idxA,chainB,res_idxB,connection_type,confidence,"
            "min_distance_angstrom,max_distance_angstrom,comment,restraint_id\n"
            "A,C3,B,C7,contact,1.0,0.0,5.5,test-restraint,r1"
        )
        input_data = Chai1Input(
            sequences={"A": _INSULIN_A, "B": _INSULIN_B},
            num_models=1,
            constraints=csv_content,
            extra={
                "num_trunk_recycles": 1,
                "num_diffn_timesteps": 50,
            },
        )
        runner = get_runner("chai1", autobio_config)
        output = runner.run(input_data, gpu="auto", output_dir=tmp_path / "ws")

        assert isinstance(output, StructurePredictionOutput)
        assert len(output.structures) == 1
        assert output.structures[0].structure_path.exists()


# ---------------------------------------------------------------------------
# Chai-1 — raw FASTA passthrough
# ---------------------------------------------------------------------------


class TestChai1RawFASTA:
    """Verify that raw FASTA passthrough works end-to-end."""

    def test_raw_fasta_prediction(self, autobio_config: AutobioConfig, tmp_path: Path) -> None:
        """Provide a raw Chai-1 FASTA and verify prediction."""
        fasta = f">protein|name=A\n{_CRAMBIN_SEQ}\n"
        input_data = Chai1Input(
            sequences={},
            chai_fasta=fasta,
            extra={
                "num_trunk_recycles": 1,
                "num_diffn_timesteps": 50,
            },
        )
        runner = get_runner("chai1", autobio_config)
        output = runner.run(input_data, gpu="auto", output_dir=tmp_path / "ws")

        assert isinstance(output, StructurePredictionOutput)
        assert len(output.structures) >= 1
        assert output.structures[0].structure_path.exists()
