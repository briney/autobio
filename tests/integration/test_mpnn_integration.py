"""Integration tests for ProteinMPNN and LigandMPNN runners.

These tests require Docker and a GPU. They download a small PDB from RCSB,
build and run the autobio-mpnn container, and verify end-to-end output.

Run with:
    pytest tests/integration/test_mpnn_integration.py -v -m docker
"""

from __future__ import annotations

import urllib.request
from typing import TYPE_CHECKING

import pytest

from autobio.core.config import AutobioConfig
from autobio.schemas.inverse_folding import InverseFoldingInput, InverseFoldingOutput
from autobio.tools import get_runner

if TYPE_CHECKING:
    from pathlib import Path

# All tests in this module require Docker and GPU
pytestmark = [pytest.mark.docker, pytest.mark.gpu]


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
def autobio_config() -> AutobioConfig:
    return AutobioConfig.resolve()


class TestProteinMPNNIntegration:
    """Full pipeline tests for ProteinMPNN."""

    def test_full_pipeline(
        self, rcsb_pdb: Path, autobio_config: AutobioConfig, tmp_path: Path
    ) -> None:
        """Run ProteinMPNN end-to-end and verify output structure."""
        input_data = InverseFoldingInput(
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
        input_data = InverseFoldingInput(structure_path=rcsb_pdb)
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
        input_data = InverseFoldingInput(
            structure_path=rcsb_pdb,
            num_sequences=5,
        )
        runner = get_runner("proteinmpnn", autobio_config)
        output = runner.run(input_data, gpu="auto", output_dir=tmp_path / "ws")

        assert len(output.designed_sequences) == 5
        # Ranks should be 1-5
        ranks = [s.rank for s in output.designed_sequences]
        assert ranks == [1, 2, 3, 4, 5]


class TestLigandMPNNIntegration:
    """Full pipeline tests for LigandMPNN."""

    def test_full_pipeline(
        self, rcsb_pdb: Path, autobio_config: AutobioConfig, tmp_path: Path
    ) -> None:
        """Run LigandMPNN end-to-end (on a plain protein — no ligand needed)."""
        input_data = InverseFoldingInput(
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
