"""Integration tests for AntiFold antibody inverse folding and scoring.

These tests require Docker and a GPU. They download a real antibody-antigen
complex PDB from RCSB and run the full AntiFold pipeline end-to-end,
exercising ANARCI IMGT renumbering, sequence design, and scoring.

Run with:
    pytest tests/integration/test_antifold_integration.py -v -m docker
"""

from __future__ import annotations

import urllib.request
from typing import TYPE_CHECKING

import pytest

from autobio.core.config import AutobioConfig
from autobio.schemas.inverse_folding import InverseFoldingInput, InverseFoldingOutput
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
def rcsb_ab_ag_pdb(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Download 1YQV (anti-lysozyme Fab-antigen complex) from RCSB.

    1YQV has:
    - Chain H: antibody heavy chain (~220 residues)
    - Chain L: antibody light chain (~214 residues)
    - Chain C: hen egg-white lysozyme antigen (~129 residues)

    Uses author (non-IMGT) numbering, so ANARCI renumbering is exercised.
    """
    cache = tmp_path_factory.mktemp("pdb_cache")
    pdb_path = cache / "1yqv.pdb"
    urllib.request.urlretrieve(  # noqa: S310
        "https://files.rcsb.org/download/1YQV.pdb",
        pdb_path,
    )
    return pdb_path


@pytest.fixture(scope="session")
def autobio_config() -> AutobioConfig:
    return AutobioConfig.resolve()


# ---------------------------------------------------------------------------
# AntiFold Design
# ---------------------------------------------------------------------------


class TestAntiFoldDesign:
    """Full pipeline tests for AntiFold antibody inverse folding."""

    def test_full_pipeline(
        self, rcsb_ab_ag_pdb: Path, autobio_config: AutobioConfig, tmp_path: Path
    ) -> None:
        """Run AntiFold end-to-end with H/L chains and antigen context."""
        input_data = InverseFoldingInput(
            structure_path=rcsb_ab_ag_pdb,
            num_sequences=2,
            temperature=0.2,
            extra={
                "heavy_chain": "H",
                "light_chain": "L",
                "antigen_chain": "C",
            },
        )
        runner = get_runner("antifold", autobio_config)
        output = runner.run(input_data, gpu="auto", mode="design", output_dir=tmp_path / "ws")

        assert isinstance(output, InverseFoldingOutput)
        assert len(output.designed_sequences) == 2
        assert output.metadata.tool_name == "antifold"
        assert output.metadata.mode == "design"
        assert output.metadata.wall_time_seconds > 0

        for seq in output.designed_sequences:
            assert "H" in seq.sequence, "Missing heavy chain"
            assert "L" in seq.sequence, "Missing light chain"
            # Variable domain sequences should be reasonable length
            assert len(seq.sequence["H"]) >= 100
            assert len(seq.sequence["L"]) >= 90
            # AntiFold populates scores (unlike ESM-IF1)
            assert seq.score is not None
            assert seq.recovery is not None
            assert 0.0 <= seq.recovery <= 1.0

    def test_cdr_region_design(
        self, rcsb_ab_ag_pdb: Path, autobio_config: AutobioConfig, tmp_path: Path
    ) -> None:
        """Target only CDRH3 for redesign; other positions should match native."""
        input_data = InverseFoldingInput(
            structure_path=rcsb_ab_ag_pdb,
            num_sequences=1,
            temperature=0.2,
            extra={
                "heavy_chain": "H",
                "light_chain": "L",
                "regions": ["CDRH3"],
            },
        )
        runner = get_runner("antifold", autobio_config)
        output = runner.run(input_data, gpu="auto", mode="design", output_dir=tmp_path / "ws")

        assert len(output.designed_sequences) == 1
        assert output.native_sequence is not None

        # Light chain should be entirely native (no CDRL regions targeted)
        designed_L = output.designed_sequences[0].sequence["L"]
        native_L = output.native_sequence["L"]
        assert designed_L == native_L, "Light chain should be unchanged when only CDRH3 is targeted"

    def test_multiple_sequences(
        self, rcsb_ab_ag_pdb: Path, autobio_config: AutobioConfig, tmp_path: Path
    ) -> None:
        """Request multiple designs and verify count and ranking."""
        input_data = InverseFoldingInput(
            structure_path=rcsb_ab_ag_pdb,
            num_sequences=5,
            temperature=0.2,
            extra={
                "heavy_chain": "H",
                "light_chain": "L",
            },
        )
        runner = get_runner("antifold", autobio_config)
        output = runner.run(input_data, gpu="auto", mode="design", output_dir=tmp_path / "ws")

        assert len(output.designed_sequences) == 5
        ranks = [s.rank for s in output.designed_sequences]
        assert ranks == [1, 2, 3, 4, 5]

    def test_native_sequence_extracted(
        self, rcsb_ab_ag_pdb: Path, autobio_config: AutobioConfig, tmp_path: Path
    ) -> None:
        """Verify native sequence is extracted for both H and L chains."""
        input_data = InverseFoldingInput(
            structure_path=rcsb_ab_ag_pdb,
            extra={
                "heavy_chain": "H",
                "light_chain": "L",
            },
        )
        runner = get_runner("antifold", autobio_config)
        output = runner.run(input_data, gpu="auto", mode="design", output_dir=tmp_path / "ws")

        assert output.native_sequence is not None
        assert "H" in output.native_sequence
        assert "L" in output.native_sequence
        assert len(output.native_sequence["H"]) >= 100
        assert len(output.native_sequence["L"]) >= 90


# ---------------------------------------------------------------------------
# AntiFold Score
# ---------------------------------------------------------------------------


class TestAntiFoldScore:
    """Full pipeline tests for AntiFold sequence scoring."""

    def test_score_native_sequence(
        self, rcsb_ab_ag_pdb: Path, autobio_config: AutobioConfig, tmp_path: Path
    ) -> None:
        """Score the native sequence (sequences=None)."""
        score_input = ScoringInput(
            structure_path=rcsb_ab_ag_pdb,
            sequences=None,
            extra={
                "heavy_chain": "H",
                "light_chain": "L",
            },
        )
        runner = get_runner("antifold", autobio_config)
        output = runner.run(score_input, gpu="auto", mode="score", output_dir=tmp_path / "ws")

        assert isinstance(output, ScoringOutput)
        assert len(output.scores) == 1
        score = output.scores[0]
        assert score.total_score < 0  # NLL should be negative
        assert score.units == "avg_nll"
        assert score.score_breakdown is not None
        assert "perplexity" in score.score_breakdown
        assert score.score_breakdown["perplexity"] > 0
        assert output.metadata.tool_name == "antifold"
        assert output.metadata.mode == "score"

    def test_score_custom_sequences(
        self, rcsb_ab_ag_pdb: Path, autobio_config: AutobioConfig, tmp_path: Path
    ) -> None:
        """Score user-provided sequences against the structure."""
        # First get native sequences via a design run
        design_input = InverseFoldingInput(
            structure_path=rcsb_ab_ag_pdb,
            extra={"heavy_chain": "H", "light_chain": "L"},
        )
        design_runner = get_runner("antifold", autobio_config)
        design_output = design_runner.run(
            design_input, gpu="auto", mode="design", output_dir=tmp_path / "design_ws"
        )
        assert design_output.native_sequence is not None

        # Score the native sequences explicitly
        score_input = ScoringInput(
            structure_path=rcsb_ab_ag_pdb,
            sequences=design_output.native_sequence,
            extra={"heavy_chain": "H", "light_chain": "L"},
        )
        score_runner = get_runner("antifold", autobio_config)
        output = score_runner.run(
            score_input, gpu="auto", mode="score", output_dir=tmp_path / "score_ws"
        )

        assert isinstance(output, ScoringOutput)
        assert len(output.scores) == 1
        score = output.scores[0]
        assert score.total_score < 0

    def test_score_with_per_residue(
        self, rcsb_ab_ag_pdb: Path, autobio_config: AutobioConfig, tmp_path: Path
    ) -> None:
        """Verify per-residue scores are populated."""
        score_input = ScoringInput(
            structure_path=rcsb_ab_ag_pdb,
            sequences=None,
            extra={"heavy_chain": "H", "light_chain": "L"},
        )
        runner = get_runner("antifold", autobio_config)
        output = runner.run(score_input, gpu="auto", mode="score", output_dir=tmp_path / "ws")

        score = output.scores[0]
        assert score.per_residue_scores is not None
        assert len(score.per_residue_scores) > 0
        # All per-residue scores should be finite
        assert all(isinstance(s, float) for s in score.per_residue_scores)
