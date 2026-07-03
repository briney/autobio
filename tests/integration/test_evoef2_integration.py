"""Integration tests for EvoEF2 tools.

These tests require Docker (no GPU — EvoEF2 is CPU-only). They download a
protein complex PDB from RCSB, run the autobio-evoef2 container, and verify
end-to-end output.

Run with:
    pytest tests/integration/test_evoef2_integration.py -v -m docker
"""

from __future__ import annotations

import urllib.request
from typing import TYPE_CHECKING

import pytest

from autobio.core.config import AutobioConfig
from autobio.schemas.scoring import (
    EvoEF2BindingInput,
    EvoEF2BuildMutantInput,
    EvoEF2RepairInput,
    ScoringOutput,
)
from autobio.tools import get_runner

if TYPE_CHECKING:
    from pathlib import Path

# All tests in this module require Docker and are slow (CPU-only, no GPU needed)
pytestmark = [pytest.mark.docker, pytest.mark.slow]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def autobio_config() -> AutobioConfig:
    return AutobioConfig.resolve()


@pytest.fixture(scope="session")
def rcsb_1brs(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Download 1BRS (barnase-barstar complex, chains A-D) from RCSB.

    A well-characterised protein–protein complex suitable for binding
    energy and mutation tests.
    """
    cache = tmp_path_factory.mktemp("pdb_cache")
    pdb_path = cache / "1brs.pdb"
    urllib.request.urlretrieve(  # noqa: S310
        "https://files.rcsb.org/download/1BRS.pdb",
        pdb_path,
    )
    return pdb_path


# ---------------------------------------------------------------------------
# evoef2_repair
# ---------------------------------------------------------------------------


class TestEvoEF2Repair:
    """Repair a structure with EvoEF2."""

    def test_repair_1brs(
        self, rcsb_1brs: Path, autobio_config: AutobioConfig, tmp_path: Path
    ) -> None:
        """Repair barnase-barstar and verify output structure."""
        input_data = EvoEF2RepairInput(structure_path=rcsb_1brs)
        runner = get_runner("evoef2", autobio_config)
        output = runner.run(input_data, gpu="none", mode="repair", output_dir=tmp_path / "ws")

        assert isinstance(output, ScoringOutput)
        assert len(output.scores) >= 1
        assert output.metadata.tool_name == "evoef2"
        assert output.metadata.mode == "repair"
        assert output.metadata.wall_time_seconds > 0

        s = output.scores[0]
        assert s.units == "EvoEF2"
        assert isinstance(s.total_score, float)
        assert s.structure_path is not None
        assert s.structure_path.exists()
        assert s.structure_path.suffix == ".pdb"
        content = s.structure_path.read_text()
        assert "ATOM" in content


# ---------------------------------------------------------------------------
# evoef2_binding
# ---------------------------------------------------------------------------


class TestEvoEF2Binding:
    """Compute binding energy with EvoEF2."""

    def test_binding_1brs_with_repair(
        self, rcsb_1brs: Path, autobio_config: AutobioConfig, tmp_path: Path
    ) -> None:
        """Compute binding energy for barnase-barstar with auto-repair."""
        input_data = EvoEF2BindingInput(
            structure_path=rcsb_1brs,
            split_chains="A,D",
        )
        runner = get_runner("evoef2", autobio_config)
        output = runner.run(input_data, gpu="none", mode="binding", output_dir=tmp_path / "ws")

        assert isinstance(output, ScoringOutput)
        assert len(output.scores) >= 1
        assert output.metadata.tool_name == "evoef2"
        assert output.metadata.mode == "binding"
        assert output.metadata.wall_time_seconds > 0

        s = output.scores[0]
        assert s.units == "EvoEF2"
        assert isinstance(s.total_score, float)
        # Binding energy should be negative for a real complex
        assert s.total_score < 0
        assert s.score_breakdown is not None
        assert len(s.score_breakdown) > 0
        # Auto-repair should produce a repaired PDB
        assert s.structure_path is not None
        assert s.structure_path.exists()

    def test_binding_1brs_no_repair(
        self, rcsb_1brs: Path, autobio_config: AutobioConfig, tmp_path: Path
    ) -> None:
        """Compute binding energy without auto-repair."""
        input_data = EvoEF2BindingInput(
            structure_path=rcsb_1brs,
            repair=False,
            split_chains="A,D",
        )
        runner = get_runner("evoef2", autobio_config)
        output = runner.run(input_data, gpu="none", mode="binding", output_dir=tmp_path / "ws")

        assert isinstance(output, ScoringOutput)
        s = output.scores[0]
        assert isinstance(s.total_score, float)
        assert s.total_score < 0
        # No repair means no structure output
        assert s.structure_path is None


# ---------------------------------------------------------------------------
# evoef2_build_mutant
# ---------------------------------------------------------------------------


class TestEvoEF2BuildMutant:
    """Build mutant structures with EvoEF2."""

    def test_build_mutant_1brs(
        self, rcsb_1brs: Path, autobio_config: AutobioConfig, tmp_path: Path
    ) -> None:
        """Build a single-point mutant of barnase."""
        input_data = EvoEF2BuildMutantInput(
            structure_path=rcsb_1brs,
            mutations=["AA42G"],
        )
        runner = get_runner("evoef2", autobio_config)
        output = runner.run(input_data, gpu="none", mode="build_mutant", output_dir=tmp_path / "ws")

        assert isinstance(output, ScoringOutput)
        assert len(output.scores) >= 1
        assert output.metadata.tool_name == "evoef2"
        assert output.metadata.mode == "build_mutant"

        s = output.scores[0]
        assert s.structure_path is not None
        assert s.structure_path.exists()
        assert s.structure_path.suffix == ".pdb"
        content = s.structure_path.read_text()
        assert "ATOM" in content
        assert s.mutations == ["AA42G"]
