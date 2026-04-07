"""Integration tests for PRODIGY protein-protein binding affinity prediction.

These tests require Docker (no GPU needed — PRODIGY is CPU-only). They
download a protein-protein complex PDB from RCSB, run the autobio-prodigy
container, and verify end-to-end output.

Run with:
    pytest tests/integration/test_prodigy_integration.py -v -m docker
"""

from __future__ import annotations

import urllib.request
from typing import TYPE_CHECKING

import pytest

from autobio.core.config import AutobioConfig
from autobio.schemas.protein_binding_affinity import (
    ProteinBindingAffinityInput,
    ProteinBindingAffinityOutput,
)
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

    This is a well-studied benchmark complex for PRODIGY.
    """
    cache = tmp_path_factory.mktemp("pdb_cache")
    pdb_path = cache / "1ppe.pdb"
    urllib.request.urlretrieve(  # noqa: S310
        "https://files.rcsb.org/download/1PPE.pdb",
        pdb_path,
    )
    return pdb_path


# ---------------------------------------------------------------------------
# PRODIGY — basic prediction
# ---------------------------------------------------------------------------


class TestProdigyPrediction:
    """Predict binding affinity for a protein-protein complex."""

    def test_basic_prediction_1ppe(
        self, rcsb_1ppe: Path, autobio_config: AutobioConfig, tmp_path: Path
    ) -> None:
        """Predict binding affinity for 1PPE (trypsin/BPTI).

        Verifies the full pipeline: PDB input -> contact counting -> prediction.
        """
        input_data = ProteinBindingAffinityInput(
            structure_path=rcsb_1ppe,
            chain_selection="E I",
        )
        runner = get_runner("prodigy", autobio_config)
        output = runner.run(input_data, gpu="none", output_dir=tmp_path / "ws")

        assert isinstance(output, ProteinBindingAffinityOutput)
        assert len(output.predictions) >= 1
        assert output.metadata.tool_name == "prodigy"
        assert output.metadata.wall_time_seconds > 0

        p = output.predictions[0]
        assert p.units == "kcal/mol"
        assert isinstance(p.delta_g_kcal_mol, float)
        # Binding affinity should be in a reasonable range for
        # protein-protein complexes: roughly -15 to 0 kcal/mol
        assert -15.0 < p.delta_g_kcal_mol < 0.0

        # Kd in molar should be positive and consistent with delta-G
        assert isinstance(p.kd_molar, float)
        assert p.kd_molar > 0.0

        # Score breakdown should include contact counts and NIS percentages
        assert p.score_breakdown is not None
        assert p.score_breakdown["intermolecular_contacts"] > 0
        assert "charged_charged_contacts" in p.score_breakdown
        assert "charged_polar_contacts" in p.score_breakdown
        assert "charged_apolar_contacts" in p.score_breakdown
        assert "polar_polar_contacts" in p.score_breakdown
        assert "polar_apolar_contacts" in p.score_breakdown
        assert "apolar_apolar_contacts" in p.score_breakdown
        assert 0.0 <= p.score_breakdown["pct_apolar_nis"] <= 100.0
        assert 0.0 <= p.score_breakdown["pct_charged_nis"] <= 100.0
        assert 0.0 <= p.score_breakdown["pct_polar_nis"] <= 100.0

    def test_prediction_without_chain_selection(
        self, rcsb_1ppe: Path, autobio_config: AutobioConfig, tmp_path: Path
    ) -> None:
        """Predict binding affinity using all inter-chain contacts (no selection)."""
        input_data = ProteinBindingAffinityInput(
            structure_path=rcsb_1ppe,
        )
        runner = get_runner("prodigy", autobio_config)
        output = runner.run(input_data, gpu="none", output_dir=tmp_path / "ws")

        assert isinstance(output, ProteinBindingAffinityOutput)
        assert len(output.predictions) >= 1
        p = output.predictions[0]
        assert isinstance(p.delta_g_kcal_mol, float)
        assert -15.0 < p.delta_g_kcal_mol < 0.0
