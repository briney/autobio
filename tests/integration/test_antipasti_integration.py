"""Integration tests for ANTIPASTI antibody binding affinity prediction.

These tests require Docker (no GPU needed — ANTIPASTI is CPU-only). They
download an antibody-antigen complex PDB from RCSB, run the autobio-antipasti
container, and verify end-to-end output.

Run with:
    pytest tests/integration/test_antipasti_integration.py -v -m docker
"""

from __future__ import annotations

import urllib.request
from typing import TYPE_CHECKING

import pytest

from autobio.core.config import AutobioConfig
from autobio.schemas.binding_affinity import AntipastiInput, BindingAffinityOutput
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
def rcsb_1ahw(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Download 1AHW (anti-tissue factor Fab complex) from RCSB.

    PDB 1AHW is an antibody-antigen complex with:
    - Heavy chain: A (VH + CH1)
    - Light chain: B (VL + CL)
    - Antigen chain: C (tissue factor)

    Note: 1AHW uses sequential numbering, not Chothia.
    """
    cache = tmp_path_factory.mktemp("pdb_cache")
    pdb_path = cache / "1ahw.pdb"
    urllib.request.urlretrieve(  # noqa: S310
        "https://files.rcsb.org/download/1AHW.pdb",
        pdb_path,
    )
    return pdb_path


# ---------------------------------------------------------------------------
# ANTIPASTI — basic prediction
# ---------------------------------------------------------------------------


class TestAntipastiPrediction:
    """Predict binding affinity for an antibody-antigen complex."""

    def test_basic_prediction_1ahw(
        self, rcsb_1ahw: Path, autobio_config: AutobioConfig, tmp_path: Path
    ) -> None:
        """Predict binding affinity for 1AHW (anti-tissue factor Fab).

        Verifies the full pipeline: PDB input -> NMA/DCCM -> CNN prediction.
        """
        input_data = AntipastiInput(
            structure_path=rcsb_1ahw,
            heavy_chain="A",
            light_chain="B",
            antigen_chains=["C"],
        )
        runner = get_runner("antipasti", autobio_config)
        output = runner.run(input_data, gpu="none", output_dir=tmp_path / "ws")

        assert isinstance(output, BindingAffinityOutput)
        assert len(output.predictions) >= 1
        assert output.metadata.tool_name == "antipasti"
        assert output.metadata.wall_time_seconds > 0

        p = output.predictions[0]
        assert p.units == "log10(Kd) [M]"
        assert isinstance(p.log10_kd, float)
        # Binding affinity should be in a reasonable range:
        # very tight binders: ~-12, very weak: ~-3
        assert -15.0 < p.log10_kd < 0.0

        # Kd in molar should be consistent with log10_kd
        assert isinstance(p.kd_molar, float)
        assert p.kd_molar > 0.0
        assert p.kd_molar == pytest.approx(10**p.log10_kd, rel=1e-2)

        # Score breakdown should include chain info
        assert p.score_breakdown is not None
        assert p.score_breakdown["heavy_chain"] == "A"
        assert p.score_breakdown["light_chain"] == "B"
        assert p.score_breakdown["antigen_chains"] == ["C"]
        assert p.score_breakdown["modes"] == "all"
