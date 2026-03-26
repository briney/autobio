"""Integration tests for Rosetta tools.

These tests require Docker (no GPU — Rosetta is CPU-only). They download PDB
structures from RCSB, run the autobio-rosetta containers, and verify
end-to-end output.

Run with:
    pytest tests/integration/test_rosetta_integration.py -v -m docker
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

# All tests in this module require Docker and are slow (Rosetta is CPU-only — no GPU)
pytestmark = [pytest.mark.docker, pytest.mark.slow]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def autobio_config() -> AutobioConfig:
    return AutobioConfig.resolve()


@pytest.fixture(scope="session")
def rcsb_1ubq(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Download 1UBQ (ubiquitin, 76 residues, single chain) from RCSB."""
    cache = tmp_path_factory.mktemp("pdb_cache")
    pdb_path = cache / "1ubq.pdb"
    urllib.request.urlretrieve(  # noqa: S310
        "https://files.rcsb.org/download/1UBQ.pdb",
        pdb_path,
    )
    return pdb_path


@pytest.fixture(scope="session")
def rcsb_1brs(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Download 1BRS (barnase-barstar complex) from RCSB for DDG tests."""
    cache = tmp_path_factory.mktemp("pdb_cache")
    pdb_path = cache / "1brs.pdb"
    urllib.request.urlretrieve(  # noqa: S310
        "https://files.rcsb.org/download/1BRS.pdb",
        pdb_path,
    )
    return pdb_path


# ---------------------------------------------------------------------------
# rosetta_score
# ---------------------------------------------------------------------------


class TestRosettaScore:
    """Score a structure with Rosetta energy function."""

    def test_score_1ubq(
        self, rcsb_1ubq: Path, autobio_config: AutobioConfig, tmp_path: Path
    ) -> None:
        """Score ubiquitin and verify energy breakdown."""
        input_data = ScoringInput(structure_path=rcsb_1ubq)
        runner = get_runner("rosetta_score", autobio_config)
        output = runner.run(input_data, gpu="none", output_dir=tmp_path / "ws")

        assert isinstance(output, ScoringOutput)
        assert len(output.scores) >= 1
        assert output.metadata.tool_name == "rosetta_score"
        assert output.metadata.wall_time_seconds > 0

        s = output.scores[0]
        assert s.units == "REU"
        assert isinstance(s.total_score, float)
        assert s.score_breakdown is not None
        assert "fa_atr" in s.score_breakdown
        assert "fa_rep" in s.score_breakdown
        assert s.structure_path is None


# ---------------------------------------------------------------------------
# rosetta_relax
# ---------------------------------------------------------------------------


class TestRosettaRelax:
    """Relax a structure with FastRelax."""

    def test_relax_1ubq(
        self, rcsb_1ubq: Path, autobio_config: AutobioConfig, tmp_path: Path
    ) -> None:
        """Relax ubiquitin (1 structure) and verify output PDB."""
        input_data = ScoringInput(
            structure_path=rcsb_1ubq,
            extra={"nstruct": 1},
        )
        runner = get_runner("rosetta_relax", autobio_config)
        output = runner.run(input_data, gpu="none", output_dir=tmp_path / "ws")

        assert isinstance(output, ScoringOutput)
        assert len(output.scores) >= 1
        assert output.metadata.tool_name == "rosetta_relax"

        s = output.scores[0]
        assert s.total_score < 0
        assert s.structure_path is not None
        assert s.structure_path.exists()
        assert s.structure_path.suffix == ".pdb"
        content = s.structure_path.read_text()
        assert "ATOM" in content


# ---------------------------------------------------------------------------
# rosetta_minimize
# ---------------------------------------------------------------------------


class TestRosettaMinimize:
    """Minimize a structure."""

    def test_minimize_1ubq(
        self, rcsb_1ubq: Path, autobio_config: AutobioConfig, tmp_path: Path
    ) -> None:
        """Minimize ubiquitin and verify output."""
        input_data = ScoringInput(structure_path=rcsb_1ubq)
        runner = get_runner("rosetta_minimize", autobio_config)
        output = runner.run(input_data, gpu="none", output_dir=tmp_path / "ws")

        assert isinstance(output, ScoringOutput)
        assert len(output.scores) >= 1
        assert output.metadata.tool_name == "rosetta_minimize"

        s = output.scores[0]
        assert s.total_score < 0
        assert s.structure_path is not None
        assert s.structure_path.exists()


# ---------------------------------------------------------------------------
# rosetta_flexddg
# ---------------------------------------------------------------------------


class TestRosettaFlexddG:
    """Ensemble DDG prediction at protein-protein interface."""

    def test_flexddg_1brs(
        self, rcsb_1brs: Path, autobio_config: AutobioConfig, tmp_path: Path
    ) -> None:
        """Predict binding DDG for barnase-barstar (quick: 3 samples)."""
        input_data = ScoringInput(
            structure_path=rcsb_1brs,
            extra={
                "mutations": ["A42G"],
                "chains_to_move": "D",
                "nstruct": 3,
                "backrub_trials": 1000,
                "max_minimization_iter": 100,
            },
        )
        runner = get_runner("rosetta_flexddg", autobio_config)
        output = runner.run(input_data, gpu="none", output_dir=tmp_path / "ws")

        assert isinstance(output, ScoringOutput)
        assert len(output.scores) >= 1
        assert output.metadata.tool_name == "rosetta_flexddg"

        s = output.scores[0]
        assert s.mutations == ["A42G"]
        assert s.units == "REU"
        assert s.score_breakdown is not None
        assert isinstance(s.total_score, float)
