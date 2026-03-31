"""Integration tests for StaB-ddG binding ddG prediction.

These tests require Docker and a GPU. They download a protein complex PDB
from RCSB, run the autobio-stabddg container, and verify end-to-end output.

Run with:
    pytest tests/integration/test_stabddg_integration.py -v -m docker
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
pytestmark = [pytest.mark.docker, pytest.mark.gpu, pytest.mark.slow]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def autobio_config() -> AutobioConfig:
    return AutobioConfig.resolve()


@pytest.fixture(scope="session")
def rcsb_1ao7(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Download 1AO7 (hGH receptor complex, chains A-E) from RCSB.

    This is the default example structure used by StaB-ddG. Chain spec: ABC_DE.
    """
    cache = tmp_path_factory.mktemp("pdb_cache")
    pdb_path = cache / "1AO7.pdb"
    urllib.request.urlretrieve(  # noqa: S310
        "https://files.rcsb.org/download/1AO7.pdb",
        pdb_path,
    )
    return pdb_path


# ---------------------------------------------------------------------------
# StaB-ddG — single mutation
# ---------------------------------------------------------------------------


class TestStaBddGSingleMutation:
    """Predict binding ddG for a single mutation."""

    def test_single_mutation_1ao7(
        self, rcsb_1ao7: Path, autobio_config: AutobioConfig, tmp_path: Path
    ) -> None:
        """Predict ddG for EA63Q on 1AO7 (default StaB-ddG example).

        Uses reduced mc_samples for faster execution.
        """
        input_data = ScoringInput(
            structure_path=rcsb_1ao7,
            extra={
                "mutations": ["EA63Q"],
                "chains": "ABC_DE",
                "mc_samples": 5,
                "seed": 0,
            },
        )
        runner = get_runner("stabddg", autobio_config)
        output = runner.run(input_data, gpu="auto", output_dir=tmp_path / "ws")

        assert isinstance(output, ScoringOutput)
        assert len(output.scores) >= 1
        assert output.metadata.tool_name == "stabddg"
        assert output.metadata.wall_time_seconds > 0

        s = output.scores[0]
        assert s.units == "kcal/mol"
        assert isinstance(s.ddg, float)
        assert isinstance(s.total_score, float)
        assert s.ddg == pytest.approx(s.total_score)
        assert s.structure_path is None
        assert s.mutations is not None
        assert "EA63Q" in s.mutations

        # Score breakdown should include chain info
        assert s.score_breakdown is not None
        assert s.score_breakdown["chains"] == "ABC_DE"


# ---------------------------------------------------------------------------
# StaB-ddG — multiple mutations (combined effect)
# ---------------------------------------------------------------------------


class TestStaBddGMultipleMutations:
    """Predict combined binding ddG for multiple mutations."""

    def test_multi_mutation_1ao7(
        self, rcsb_1ao7: Path, autobio_config: AutobioConfig, tmp_path: Path
    ) -> None:
        """Predict combined ddG for three mutations on 1AO7.

        This matches the default StaB-ddG example: EA63Q, QD30V, KA66A.
        """
        input_data = ScoringInput(
            structure_path=rcsb_1ao7,
            extra={
                "mutations": ["EA63Q", "QD30V", "KA66A"],
                "chains": "ABC_DE",
                "mc_samples": 5,
                "seed": 0,
            },
        )
        runner = get_runner("stabddg", autobio_config)
        output = runner.run(input_data, gpu="auto", output_dir=tmp_path / "ws")

        assert isinstance(output, ScoringOutput)
        assert len(output.scores) >= 1

        s = output.scores[0]
        assert s.units == "kcal/mol"
        assert isinstance(s.ddg, float)
        assert s.mutations is not None
        assert len(s.mutations) >= 1
