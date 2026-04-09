"""Integration tests for ProteinMPNN and LigandMPNN scoring runners.

These tests require Docker and a GPU. They download PDB structures from RCSB,
build and run the mpnn-score container, and verify end-to-end output.

Run with:
    pytest tests/integration/test_mpnn_score_integration.py -v -m docker
"""

from __future__ import annotations

import urllib.request
from typing import TYPE_CHECKING

import pytest

from autobio.core.config import AutobioConfig
from autobio.schemas.scoring import ScoringInput, ScoringOutput
from autobio.tools import get_runner

if TYPE_CHECKING:
    from pathlib import Path

# All tests in this module require Docker and GPU
pytestmark = [pytest.mark.docker, pytest.mark.gpu]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def rcsb_pdb(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Download 1CRN (crambin, 46 residues, single chain) from RCSB."""
    cache = tmp_path_factory.mktemp("pdb_cache")
    pdb_path = cache / "1crn.pdb"
    urllib.request.urlretrieve(  # noqa: S310
        "https://files.rcsb.org/download/1CRN.pdb",
        pdb_path,
    )
    return pdb_path


@pytest.fixture(scope="session")
def rcsb_fab_pdb(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Download 7FAB (Fab fragment, 2 chains: H ~209 res, L ~204 res) from RCSB."""
    cache = tmp_path_factory.mktemp("pdb_cache")
    pdb_path = cache / "7fab.pdb"
    urllib.request.urlretrieve(  # noqa: S310
        "https://files.rcsb.org/download/7FAB.pdb",
        pdb_path,
    )
    return pdb_path


@pytest.fixture(scope="session")
def rcsb_ligand_pdb(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Download 3PTB (trypsin + benzamidine ligand, chain A) from RCSB."""
    cache = tmp_path_factory.mktemp("pdb_cache")
    pdb_path = cache / "3ptb.pdb"
    urllib.request.urlretrieve(  # noqa: S310
        "https://files.rcsb.org/download/3PTB.pdb",
        pdb_path,
    )
    return pdb_path


@pytest.fixture(scope="session")
def autobio_config() -> AutobioConfig:
    return AutobioConfig.resolve()


# ---------------------------------------------------------------------------
# ProteinMPNN Score — single chain (1CRN)
# ---------------------------------------------------------------------------


class TestProteinMPNNScore:
    """Full pipeline tests for ProteinMPNN scoring."""

    def test_score_native_sequence(
        self, rcsb_pdb: Path, autobio_config: AutobioConfig, tmp_path: Path
    ) -> None:
        """Score native sequence of crambin (sequences=None)."""
        input_data = ScoringInput(structure_path=rcsb_pdb, sequences=None)
        runner = get_runner("proteinmpnn_score", autobio_config)
        output = runner.run(input_data, gpu="auto", output_dir=tmp_path / "ws")

        assert isinstance(output, ScoringOutput)
        assert len(output.scores) == 1
        score = output.scores[0]
        assert isinstance(score.total_score, float)
        assert score.units == "avg_nll"
        assert score.score_breakdown is not None
        assert "perplexity" in score.score_breakdown
        assert output.metadata.tool_name == "proteinmpnn_score"
        assert output.metadata.wall_time_seconds > 0

    def test_per_residue_scores(
        self, rcsb_pdb: Path, autobio_config: AutobioConfig, tmp_path: Path
    ) -> None:
        """Verify per-residue scores are returned with correct count."""
        input_data = ScoringInput(structure_path=rcsb_pdb, sequences=None)
        runner = get_runner("proteinmpnn_score", autobio_config)
        output = runner.run(input_data, gpu="auto", output_dir=tmp_path / "ws")

        score = output.scores[0]
        assert score.per_residue_scores is not None
        # Crambin is 46 residues; allow some parser flexibility
        assert len(score.per_residue_scores) >= 40
        # All NLL values should be finite positive numbers
        assert all(v > 0 for v in score.per_residue_scores)

    def test_score_explicit_sequences(
        self, rcsb_pdb: Path, autobio_config: AutobioConfig, tmp_path: Path
    ) -> None:
        """Score an explicit (non-native) sequence against the backbone."""
        # First run with native to get the sequence length
        native_input = ScoringInput(structure_path=rcsb_pdb, sequences=None)
        runner = get_runner("proteinmpnn_score", autobio_config)
        native_output = runner.run(native_input, gpu="auto", output_dir=tmp_path / "native_ws")
        native_score = native_output.scores[0].total_score

        # Build a mutated sequence (all alanine) of the correct length
        n_res = len(native_output.scores[0].per_residue_scores)
        mutant_seq = {"A": "A" * n_res}
        mutant_input = ScoringInput(structure_path=rcsb_pdb, sequences=mutant_seq)
        mutant_output = runner.run(mutant_input, gpu="auto", output_dir=tmp_path / "mutant_ws")

        mutant_score = mutant_output.scores[0].total_score
        # All-alanine should score worse (higher NLL) than native
        assert mutant_score > native_score


# ---------------------------------------------------------------------------
# ProteinMPNN Score — multi-chain Fab (7FAB)
# ---------------------------------------------------------------------------

_7FAB_CHAINS = ("H", "L")


class TestProteinMPNNScoreMultiChain:
    """ProteinMPNN scoring on a multi-chain Fab (7FAB)."""

    def test_score_multi_chain_native(
        self, rcsb_fab_pdb: Path, autobio_config: AutobioConfig, tmp_path: Path
    ) -> None:
        """Score native sequences of both chains."""
        input_data = ScoringInput(structure_path=rcsb_fab_pdb, sequences=None)
        runner = get_runner("proteinmpnn_score", autobio_config)
        output = runner.run(input_data, gpu="auto", output_dir=tmp_path / "ws")

        assert isinstance(output, ScoringOutput)
        score = output.scores[0]
        assert isinstance(score.total_score, float)
        assert score.score_breakdown is not None

        # Should have per-chain breakdown
        for chain_id in _7FAB_CHAINS:
            assert f"{chain_id}_mean_nll" in score.score_breakdown
            assert f"{chain_id}_perplexity" in score.score_breakdown

        # Per-residue scores should span both chains
        assert score.per_residue_scores is not None
        assert len(score.per_residue_scores) >= 380  # H + L combined


# ---------------------------------------------------------------------------
# LigandMPNN Score — single chain, no ligand (1CRN)
# ---------------------------------------------------------------------------


class TestLigandMPNNScore:
    """Full pipeline tests for LigandMPNN scoring."""

    def test_score_native_sequence(
        self, rcsb_pdb: Path, autobio_config: AutobioConfig, tmp_path: Path
    ) -> None:
        """Score native crambin sequence with LigandMPNN."""
        input_data = ScoringInput(structure_path=rcsb_pdb, sequences=None)
        runner = get_runner("ligandmpnn_score", autobio_config)
        output = runner.run(input_data, gpu="auto", output_dir=tmp_path / "ws")

        assert isinstance(output, ScoringOutput)
        assert len(output.scores) == 1
        score = output.scores[0]
        assert isinstance(score.total_score, float)
        assert score.units == "avg_nll"
        assert output.metadata.tool_name == "ligandmpnn_score"


# ---------------------------------------------------------------------------
# LigandMPNN Score — protein-ligand complex (3PTB)
# ---------------------------------------------------------------------------


class TestLigandMPNNScoreWithLigand:
    """LigandMPNN scoring on a protein-ligand complex (trypsin + benzamidine)."""

    def test_score_protein_ligand_complex(
        self, rcsb_ligand_pdb: Path, autobio_config: AutobioConfig, tmp_path: Path
    ) -> None:
        """Score trypsin sequence with benzamidine ligand context."""
        input_data = ScoringInput(structure_path=rcsb_ligand_pdb, sequences=None)
        runner = get_runner("ligandmpnn_score", autobio_config)
        output = runner.run(input_data, gpu="auto", output_dir=tmp_path / "ws")

        assert isinstance(output, ScoringOutput)
        score = output.scores[0]
        assert isinstance(score.total_score, float)
        assert score.per_residue_scores is not None
        # Trypsin is ~223 residues
        assert len(score.per_residue_scores) >= 200
        assert score.score_breakdown is not None
