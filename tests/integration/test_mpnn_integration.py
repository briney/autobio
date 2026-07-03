"""Integration tests for ProteinMPNN and LigandMPNN runners.

These tests require Docker and a GPU. They download PDB structures from RCSB,
build and run the autobio-mpnn container, and verify end-to-end output.

Run with:
    pytest tests/integration/test_mpnn_integration.py -v -m docker
"""

from __future__ import annotations

import urllib.request
from typing import TYPE_CHECKING

import pytest

from autobio.core.config import AutobioConfig
from autobio.schemas.inverse_folding import InverseFoldingOutput, MPNNInput
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
# ProteinMPNN — single chain (1CRN)
# ---------------------------------------------------------------------------


class TestProteinMPNNIntegration:
    """Full pipeline tests for ProteinMPNN."""

    def test_full_pipeline(
        self, rcsb_pdb: Path, autobio_config: AutobioConfig, tmp_path: Path
    ) -> None:
        """Run ProteinMPNN end-to-end and verify output structure."""
        input_data = MPNNInput(
            structure_path=rcsb_pdb,
            num_sequences=2,
            temperature=0.1,
        )
        runner = get_runner("proteinmpnn", autobio_config)
        output = runner.run(input_data, gpu="auto", output_dir=tmp_path / "ws")

        assert isinstance(output, InverseFoldingOutput)
        assert len(output.designed_sequences) == 2
        assert output.metadata.tool_name == "proteinmpnn"
        assert output.metadata.wall_time_seconds > 0

        # Each designed sequence should have the right chain
        for seq in output.designed_sequences:
            assert "A" in seq.sequence
            # Crambin is 46 residues
            assert len(seq.sequence["A"]) == 46
            assert seq.recovery is not None
            assert 0.0 <= seq.recovery <= 1.0

    def test_native_sequence_extracted(
        self, rcsb_pdb: Path, autobio_config: AutobioConfig, tmp_path: Path
    ) -> None:
        """Verify native sequence is extracted from the input structure."""
        input_data = MPNNInput(structure_path=rcsb_pdb)
        runner = get_runner("proteinmpnn", autobio_config)
        output = runner.run(input_data, gpu="auto", output_dir=tmp_path / "ws")

        assert output.native_sequence is not None
        assert "A" in output.native_sequence
        # Crambin native sequence starts with TT
        assert output.native_sequence["A"].startswith("T")

    def test_multiple_sequences(
        self, rcsb_pdb: Path, autobio_config: AutobioConfig, tmp_path: Path
    ) -> None:
        """Request multiple designs and verify count."""
        input_data = MPNNInput(
            structure_path=rcsb_pdb,
            num_sequences=5,
        )
        runner = get_runner("proteinmpnn", autobio_config)
        output = runner.run(input_data, gpu="auto", output_dir=tmp_path / "ws")

        assert len(output.designed_sequences) == 5
        # Ranks should be 1-5
        ranks = [s.rank for s in output.designed_sequences]
        assert ranks == [1, 2, 3, 4, 5]


# ---------------------------------------------------------------------------
# LigandMPNN — single chain, no ligand (1CRN)
# ---------------------------------------------------------------------------


class TestLigandMPNNIntegration:
    """Full pipeline tests for LigandMPNN."""

    def test_full_pipeline(
        self, rcsb_pdb: Path, autobio_config: AutobioConfig, tmp_path: Path
    ) -> None:
        """Run LigandMPNN end-to-end (on a plain protein — no ligand needed)."""
        input_data = MPNNInput(
            structure_path=rcsb_pdb,
            num_sequences=2,
            temperature=0.1,
        )
        runner = get_runner("ligandmpnn", autobio_config)
        output = runner.run(input_data, gpu="auto", output_dir=tmp_path / "ws")

        assert isinstance(output, InverseFoldingOutput)
        assert len(output.designed_sequences) == 2
        assert output.metadata.tool_name == "ligandmpnn"

        for seq in output.designed_sequences:
            assert len(seq.sequence["A"]) == 46


# ---------------------------------------------------------------------------
# ProteinMPNN — multi-chain Fab (7FAB)
# ---------------------------------------------------------------------------

# 7FAB is a mouse IgG1 Fab fragment with heavy (H) and light (L) chains.
# PDB CA atom counts: H ~209, L ~204. The foundry parser may resolve
# slightly fewer residues, so tests use minimum thresholds.
_7FAB_CHAINS = ("H", "L")
_7FAB_MIN_RESIDUES = {"H": 190, "L": 190}


class TestMultiChainIntegration:
    """ProteinMPNN on a multi-chain Fab antibody fragment (7FAB)."""

    def test_full_pipeline_multi_chain(
        self, rcsb_fab_pdb: Path, autobio_config: AutobioConfig, tmp_path: Path
    ) -> None:
        """Run ProteinMPNN on a 2-chain Fab and verify per-chain output."""
        input_data = MPNNInput(
            structure_path=rcsb_fab_pdb,
            num_sequences=2,
            temperature=0.1,
        )
        runner = get_runner("proteinmpnn", autobio_config)
        output = runner.run(input_data, gpu="auto", output_dir=tmp_path / "ws")

        assert isinstance(output, InverseFoldingOutput)
        assert len(output.designed_sequences) == 2
        assert output.metadata.tool_name == "proteinmpnn"

        for seq in output.designed_sequences:
            # Both chains should be present with reasonable lengths
            for chain_id in _7FAB_CHAINS:
                assert chain_id in seq.sequence, f"Missing chain {chain_id}"
                assert len(seq.sequence[chain_id]) >= _7FAB_MIN_RESIDUES[chain_id], (
                    f"Chain {chain_id}: expected >= {_7FAB_MIN_RESIDUES[chain_id]}, "
                    f"got {len(seq.sequence[chain_id])}"
                )
            assert seq.recovery is not None
            assert 0.0 <= seq.recovery <= 1.0

    def test_native_sequence_multi_chain(
        self, rcsb_fab_pdb: Path, autobio_config: AutobioConfig, tmp_path: Path
    ) -> None:
        """Verify native sequences are extracted for both chains."""
        input_data = MPNNInput(structure_path=rcsb_fab_pdb)
        runner = get_runner("proteinmpnn", autobio_config)
        output = runner.run(input_data, gpu="auto", output_dir=tmp_path / "ws")

        assert output.native_sequence is not None
        for chain_id in _7FAB_CHAINS:
            assert chain_id in output.native_sequence, f"Missing native chain {chain_id}"
            assert len(output.native_sequence[chain_id]) >= _7FAB_MIN_RESIDUES[chain_id]

    def test_chains_to_design(
        self, rcsb_fab_pdb: Path, autobio_config: AutobioConfig, tmp_path: Path
    ) -> None:
        """Design only the heavy chain H, keeping light chain L at native sequence."""
        input_data = MPNNInput(
            structure_path=rcsb_fab_pdb,
            chains_to_design=["H"],
            num_sequences=1,
            temperature=0.1,
        )
        runner = get_runner("proteinmpnn", autobio_config)
        output = runner.run(input_data, gpu="auto", output_dir=tmp_path / "ws")

        assert len(output.designed_sequences) == 1
        seq = output.designed_sequences[0]

        # Both chains should still be present in the output
        for chain_id in _7FAB_CHAINS:
            assert chain_id in seq.sequence, f"Missing chain {chain_id}"

        # Light chain should match the native sequence exactly
        assert output.native_sequence is not None
        assert seq.sequence["L"] == output.native_sequence["L"], (
            "Light chain L should be native (not redesigned)"
        )


# ---------------------------------------------------------------------------
# LigandMPNN — protein-ligand complex (3PTB: trypsin + benzamidine)
# ---------------------------------------------------------------------------


class TestLigandMPNNWithLigand:
    """LigandMPNN on a protein-ligand complex (trypsin + benzamidine)."""

    def test_protein_ligand_complex(
        self, rcsb_ligand_pdb: Path, autobio_config: AutobioConfig, tmp_path: Path
    ) -> None:
        """Run LigandMPNN on trypsin-benzamidine and verify ligand-aware design."""
        input_data = MPNNInput(
            structure_path=rcsb_ligand_pdb,
            num_sequences=2,
            temperature=0.1,
        )
        runner = get_runner("ligandmpnn", autobio_config)
        output = runner.run(input_data, gpu="auto", output_dir=tmp_path / "ws")

        assert isinstance(output, InverseFoldingOutput)
        assert len(output.designed_sequences) == 2
        assert output.metadata.tool_name == "ligandmpnn"

        for seq in output.designed_sequences:
            assert "A" in seq.sequence
            # 3PTB has 223 CA atoms in the PDB, but the foundry parser may
            # resolve fewer residues. Trypsin is ~223 residues; verify we get
            # a reasonable protein-length sequence.
            assert len(seq.sequence["A"]) >= 200, (
                f"Expected >= 200 residues for trypsin, got {len(seq.sequence['A'])}"
            )
            assert seq.recovery is not None
            assert 0.0 <= seq.recovery <= 1.0

    def test_native_sequence_with_ligand(
        self, rcsb_ligand_pdb: Path, autobio_config: AutobioConfig, tmp_path: Path
    ) -> None:
        """Verify native sequence extraction works on a ligand-containing PDB."""
        input_data = MPNNInput(structure_path=rcsb_ligand_pdb)
        runner = get_runner("ligandmpnn", autobio_config)
        output = runner.run(input_data, gpu="auto", output_dir=tmp_path / "ws")

        assert output.native_sequence is not None
        assert "A" in output.native_sequence
        # Native sequence should be purely protein (no ligand residues)
        valid_aa = set("ACDEFGHIKLMNPQRSTVWXY")
        assert all(c in valid_aa for c in output.native_sequence["A"])
