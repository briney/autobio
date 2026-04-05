"""Integration tests for ligandmpnn_build_mutant (sidechain packing).

These tests require Docker and a GPU. They download a protein PDB from RCSB,
run the ligandmpnn-packer container, and verify end-to-end output.

Run with:
    pytest tests/integration/test_ligandmpnn_packer_integration.py -v -m "docker and gpu"
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

# All tests require Docker and GPU
pytestmark = [pytest.mark.docker, pytest.mark.gpu, pytest.mark.slow]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def autobio_config() -> AutobioConfig:
    return AutobioConfig.resolve()


@pytest.fixture(scope="session")
def rcsb_1brs(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Download 1BRS (barnase-barstar complex) from RCSB."""
    cache = tmp_path_factory.mktemp("pdb_cache")
    pdb_path = cache / "1brs.pdb"
    urllib.request.urlretrieve(  # noqa: S310
        "https://files.rcsb.org/download/1BRS.pdb",
        pdb_path,
    )
    return pdb_path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestLigandMPNNBuildMutant:
    """Build mutant structures with LigandMPNN sidechain packing."""

    def test_single_mutation(
        self, rcsb_1brs: Path, autobio_config: AutobioConfig, tmp_path: Path
    ) -> None:
        """Build a single-point mutant of barnase and verify packed output."""
        input_data = ScoringInput(
            structure_path=rcsb_1brs,
            extra={
                "mutations": ["AA11G"],
                "num_packs": 2,
                "num_denoising_steps": 2,
            },
        )
        runner = get_runner("ligandmpnn_build_mutant", autobio_config)
        output = runner.run(input_data, output_dir=tmp_path / "ws")

        assert isinstance(output, ScoringOutput)
        assert len(output.scores) == 2  # num_packs=2
        assert output.metadata.tool_name == "ligandmpnn_build_mutant"
        assert output.metadata.wall_time_seconds > 0

        for s in output.scores:
            assert s.structure_path is not None
            assert s.structure_path.exists()
            assert s.structure_path.suffix == ".pdb"
            content = s.structure_path.read_text()
            assert "ATOM" in content
            assert s.mutations == ["AA11G"]
            assert s.units == "LigandMPNN_SC_logprob"
            assert isinstance(s.total_score, float)

    def test_multiple_mutations(
        self, rcsb_1brs: Path, autobio_config: AutobioConfig, tmp_path: Path
    ) -> None:
        """Build a double mutant."""
        input_data = ScoringInput(
            structure_path=rcsb_1brs,
            extra={
                "mutations": ["AA11G", "KA19A"],
                "num_packs": 1,
                "num_denoising_steps": 2,
            },
        )
        runner = get_runner("ligandmpnn_build_mutant", autobio_config)
        output = runner.run(input_data, output_dir=tmp_path / "ws")

        assert isinstance(output, ScoringOutput)
        assert len(output.scores) >= 1

        s = output.scores[0]
        assert s.mutations == ["AA11G", "KA19A"]
        assert s.structure_path is not None
        assert s.structure_path.exists()

    def test_per_residue_scores(
        self, rcsb_1brs: Path, autobio_config: AutobioConfig, tmp_path: Path
    ) -> None:
        """Verify per-residue scores are populated."""
        input_data = ScoringInput(
            structure_path=rcsb_1brs,
            extra={
                "mutations": ["AA11G"],
                "num_packs": 1,
                "num_denoising_steps": 2,
            },
        )
        runner = get_runner("ligandmpnn_build_mutant", autobio_config)
        output = runner.run(input_data, output_dir=tmp_path / "ws")

        s = output.scores[0]
        assert s.per_residue_scores is not None
        assert len(s.per_residue_scores) > 0
        assert all(isinstance(v, float) for v in s.per_residue_scores)
