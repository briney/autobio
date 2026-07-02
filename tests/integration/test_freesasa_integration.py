"""Integration tests for the FreeSASA Tool (modes: sasa, bsa).

These tests require Docker (no GPU needed — FreeSASA is CPU-only). They
download a protein-protein complex PDB from RCSB, run the autobio-freesasa
container, and verify end-to-end output.

Run with:
    pytest tests/integration/test_freesasa_integration.py -v -m docker
"""

from __future__ import annotations

import urllib.request
from typing import TYPE_CHECKING

import pytest

from autobio.core.config import AutobioConfig
from autobio.schemas.scoring import FreeSASABSAInput, FreeSASASASAInput, ScoringOutput
from autobio.tools import get_runner

if TYPE_CHECKING:
    from pathlib import Path

# All tests in this module require Docker (no GPU needed)
pytestmark = [pytest.mark.docker, pytest.mark.slow]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def autobio_config() -> AutobioConfig:
    return AutobioConfig.resolve()


@pytest.fixture(scope="session")
def rcsb_1ppe(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Download 1PPE (trypsin/BPTI complex) from RCSB.

    PDB 1PPE is a classic protein-protein complex with:
    - Chain E: trypsin
    - Chain I: bovine pancreatic trypsin inhibitor (BPTI)

    Well-studied complex with a known interface (~1500-1800 Å² BSA).
    """
    cache = tmp_path_factory.mktemp("pdb_cache")
    pdb_path = cache / "1ppe.pdb"
    urllib.request.urlretrieve(  # noqa: S310
        "https://files.rcsb.org/download/1PPE.pdb",
        pdb_path,
    )
    return pdb_path


# ---------------------------------------------------------------------------
# freesasa (mode="bsa") — buried surface area
# ---------------------------------------------------------------------------


class TestFreeSASABSA:
    """Calculate BSA for a protein-protein complex."""

    def test_bsa_1ppe(self, rcsb_1ppe: Path, autobio_config: AutobioConfig, tmp_path: Path) -> None:
        """Calculate BSA for 1PPE (trypsin chain E vs BPTI chain I).

        Verifies the full pipeline: PDB input -> FreeSASA calculation -> BSA output.
        The expected BSA for 1PPE is approximately 1500-1800 Å².
        """
        input_data = FreeSASABSAInput(structure_path=rcsb_1ppe, partner1="E", partner2="I")
        runner = get_runner("freesasa", autobio_config)
        output = runner.run(input_data, gpu="none", output_dir=tmp_path / "ws_bsa", mode="bsa")

        assert isinstance(output, ScoringOutput)
        assert len(output.scores) >= 1
        assert output.metadata.tool_name == "freesasa"
        assert output.metadata.wall_time_seconds > 0

        s = output.scores[0]
        assert s.units == "angstrom^2"
        assert isinstance(s.total_score, float)

        # BSA should be positive and in a reasonable range for this complex
        assert 1000.0 < s.total_score < 2500.0

        # Score breakdown should contain all expected keys
        bd = s.score_breakdown
        assert bd is not None
        assert isinstance(bd["polar_bsa"], float)
        assert isinstance(bd["apolar_bsa"], float)
        assert bd["polar_bsa"] > 0
        assert bd["apolar_bsa"] > 0
        assert bd["polar_bsa"] + bd["apolar_bsa"] == pytest.approx(s.total_score, rel=0.01)

        assert isinstance(bd["complex_sasa"], float)
        assert isinstance(bd["partner1_sasa"], float)
        assert isinstance(bd["partner2_sasa"], float)
        assert bd["complex_sasa"] > 0
        assert bd["partner1_sasa"] > 0
        assert bd["partner2_sasa"] > 0

        # BSA = partner1 + partner2 - complex
        expected_bsa = bd["partner1_sasa"] + bd["partner2_sasa"] - bd["complex_sasa"]
        assert s.total_score == pytest.approx(expected_bsa, rel=1e-6)

        assert bd["partner1_chains"] == "E"
        assert bd["partner2_chains"] == "I"
        assert bd["algorithm"] == "LeeRichards"
        assert bd["probe_radius"] == pytest.approx(1.4)

        # Per-chain SASA should be present
        per_chain = bd["per_chain_sasa"]
        assert isinstance(per_chain, dict)
        assert "E" in per_chain
        assert "I" in per_chain

    def test_bsa_with_shrake_rupley(
        self, rcsb_1ppe: Path, autobio_config: AutobioConfig, tmp_path: Path
    ) -> None:
        """BSA with Shrake & Rupley algorithm should give a similar result."""
        input_data = FreeSASABSAInput(
            structure_path=rcsb_1ppe,
            partner1="E",
            partner2="I",
            algorithm="ShrakeRupley",
        )
        runner = get_runner("freesasa", autobio_config)
        output = runner.run(input_data, gpu="none", output_dir=tmp_path / "ws_bsa_sr", mode="bsa")

        assert isinstance(output, ScoringOutput)
        s = output.scores[0]
        assert s.units == "angstrom^2"

        # Should be in the same ballpark as Lee & Richards
        assert 1000.0 < s.total_score < 2500.0
        assert s.score_breakdown["algorithm"] == "ShrakeRupley"

    def test_bsa_with_per_residue(
        self, rcsb_1ppe: Path, autobio_config: AutobioConfig, tmp_path: Path
    ) -> None:
        """BSA with per-residue output enabled."""
        input_data = FreeSASABSAInput(
            structure_path=rcsb_1ppe,
            partner1="E",
            partner2="I",
            per_residue=True,
        )
        runner = get_runner("freesasa", autobio_config)
        output = runner.run(input_data, gpu="none", output_dir=tmp_path / "ws_bsa_res", mode="bsa")

        assert isinstance(output, ScoringOutput)
        s = output.scores[0]
        assert s.per_residue_scores is not None
        assert len(s.per_residue_scores) > 0

        # Some residues should have positive BSA (at the interface)
        assert any(v > 0 for v in s.per_residue_scores)

        # Per-residue detail should be in breakdown
        assert "per_residue_detail" in s.score_breakdown


# ---------------------------------------------------------------------------
# freesasa (mode="sasa") — solvent-accessible surface area
# ---------------------------------------------------------------------------


class TestFreeSASASASA:
    """Calculate SASA for a protein structure."""

    def test_sasa_1ppe(
        self, rcsb_1ppe: Path, autobio_config: AutobioConfig, tmp_path: Path
    ) -> None:
        """Calculate SASA for 1PPE.

        Verifies the full pipeline: PDB input -> FreeSASA calculation -> SASA output.
        """
        input_data = FreeSASASASAInput(structure_path=rcsb_1ppe)
        runner = get_runner("freesasa", autobio_config)
        output = runner.run(input_data, gpu="none", output_dir=tmp_path / "ws_sasa", mode="sasa")

        assert isinstance(output, ScoringOutput)
        assert len(output.scores) >= 1
        assert output.metadata.tool_name == "freesasa"
        assert output.metadata.wall_time_seconds > 0

        s = output.scores[0]
        assert s.units == "angstrom^2"
        assert isinstance(s.total_score, float)

        # SASA for a ~300 residue complex should be roughly 8000-15000 Å²
        assert 5000.0 < s.total_score < 20000.0

        # Score breakdown
        bd = s.score_breakdown
        assert bd is not None
        assert isinstance(bd["polar_sasa"], float)
        assert isinstance(bd["apolar_sasa"], float)
        assert bd["polar_sasa"] > 0
        assert bd["apolar_sasa"] > 0
        assert bd["polar_sasa"] + bd["apolar_sasa"] == pytest.approx(s.total_score, rel=0.01)

        assert bd["algorithm"] == "LeeRichards"
        assert bd["probe_radius"] == pytest.approx(1.4)

        # Per-chain SASA
        per_chain = bd["per_chain_sasa"]
        assert isinstance(per_chain, dict)
        assert len(per_chain) >= 2  # At least chains E and I

    def test_sasa_with_per_residue(
        self, rcsb_1ppe: Path, autobio_config: AutobioConfig, tmp_path: Path
    ) -> None:
        """SASA with per-residue output enabled."""
        input_data = FreeSASASASAInput(structure_path=rcsb_1ppe, per_residue=True)
        runner = get_runner("freesasa", autobio_config)
        output = runner.run(
            input_data, gpu="none", output_dir=tmp_path / "ws_sasa_res", mode="sasa"
        )

        assert isinstance(output, ScoringOutput)
        s = output.scores[0]
        assert s.per_residue_scores is not None
        assert len(s.per_residue_scores) > 0

        # All per-residue SASA values should be non-negative
        assert all(v >= 0 for v in s.per_residue_scores)

    def test_sasa_custom_probe_radius(
        self, rcsb_1ppe: Path, autobio_config: AutobioConfig, tmp_path: Path
    ) -> None:
        """SASA with a different probe radius should give a different area."""
        input_default = FreeSASASASAInput(structure_path=rcsb_1ppe)
        input_large = FreeSASASASAInput(structure_path=rcsb_1ppe, probe_radius=2.0)
        runner = get_runner("freesasa", autobio_config)

        output_default = runner.run(
            input_default, gpu="none", output_dir=tmp_path / "ws_sasa_d", mode="sasa"
        )
        output_large = runner.run(
            input_large, gpu="none", output_dir=tmp_path / "ws_sasa_l", mode="sasa"
        )

        sasa_default = output_default.scores[0].total_score
        sasa_large = output_large.scores[0].total_score

        # Larger probe smooths over crevices → different (typically smaller) SASA
        assert sasa_large != pytest.approx(sasa_default, rel=0.01)
        assert output_large.scores[0].score_breakdown["probe_radius"] == pytest.approx(2.0)
