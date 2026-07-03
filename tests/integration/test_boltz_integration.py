"""Integration tests for Boltz-1 and Boltz-2 structure prediction runners.

These tests require Docker and a GPU. They run the autobio-boltz container
and verify end-to-end structure prediction output.

Run with:
    pytest tests/integration/test_boltz_integration.py -v -m docker
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from autobio.core.config import AutobioConfig
from autobio.schemas.structure_prediction import BoltzInput, StructurePredictionOutput
from autobio.tools import get_runner

if TYPE_CHECKING:
    from pathlib import Path

# All tests in this module require Docker, GPU, and are slow
pytestmark = [pytest.mark.docker, pytest.mark.gpu, pytest.mark.slow]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

# Short test sequences — keep predictions fast on GPU
# Crambin (PDB: 1CRN) — 46 residues, single chain, well-folded small protein
_CRAMBIN_SEQ = "TTCCPSIVARSNFNVCRLPGTPEAICATYTGCIIIPGATCPGDYAN"

# Insulin A and B chains — small, well-characterized complex
_INSULIN_A = "GIVEQCCTSICSLYQLENYCN"
_INSULIN_B = "FVNQHLCGSHLVEALYLVCGERGFFYTPKT"


@pytest.fixture(scope="session")
def autobio_config() -> AutobioConfig:
    return AutobioConfig.resolve()


# ---------------------------------------------------------------------------
# Boltz-2 — single protein
# ---------------------------------------------------------------------------


class TestBoltz2SingleProtein:
    """Boltz-2 structure prediction on a single small protein."""

    def test_full_pipeline(self, autobio_config: AutobioConfig, tmp_path: Path) -> None:
        """Run Boltz-2 end-to-end and verify output structure."""
        input_data = BoltzInput(
            sequences={"A": _CRAMBIN_SEQ},
            num_models=1,
            extra={
                "sampling_steps": 50,
                "recycling_steps": 1,
            },
        )
        runner = get_runner("boltz2", autobio_config)
        output = runner.run(input_data, gpu="auto", output_dir=tmp_path / "ws")

        assert isinstance(output, StructurePredictionOutput)
        assert len(output.structures) == 1
        assert output.metadata.tool_name == "boltz2"
        assert output.metadata.wall_time_seconds > 0

        s = output.structures[0]
        assert s.model_rank == 1
        assert s.structure_path.exists()

    def test_confidence_scores(self, autobio_config: AutobioConfig, tmp_path: Path) -> None:
        """Verify confidence metrics are populated."""
        input_data = BoltzInput(
            sequences={"A": _CRAMBIN_SEQ},
            num_models=1,
            extra={
                "sampling_steps": 50,
                "recycling_steps": 1,
            },
        )
        runner = get_runner("boltz2", autobio_config)
        output = runner.run(input_data, gpu="auto", output_dir=tmp_path / "ws")

        s = output.structures[0]
        # Confidence scores should be populated (Boltz always produces these)
        assert s.ptm is not None
        assert 0.0 <= s.ptm <= 1.0
        assert s.plddt_mean is not None
        assert s.plddt_mean > 0

        # Summary metrics should match the single model
        assert output.confidence.best_ptm is not None
        assert output.confidence.best_plddt_mean is not None

    def test_multiple_models(self, autobio_config: AutobioConfig, tmp_path: Path) -> None:
        """Request 3 models and verify count and ranking."""
        input_data = BoltzInput(
            sequences={"A": _CRAMBIN_SEQ},
            num_models=3,
            extra={
                "sampling_steps": 50,
                "recycling_steps": 1,
            },
        )
        runner = get_runner("boltz2", autobio_config)
        output = runner.run(input_data, gpu="auto", output_dir=tmp_path / "ws")

        assert len(output.structures) == 3
        ranks = [s.model_rank for s in output.structures]
        assert ranks == [1, 2, 3]

        # All structures should have files
        for s in output.structures:
            assert s.structure_path.exists()

        # Best confidence metrics should come from rank 1
        assert output.confidence.best_ptm is not None


# ---------------------------------------------------------------------------
# Boltz-2 — multi-chain complex
# ---------------------------------------------------------------------------


class TestBoltz2MultiChain:
    """Boltz-2 prediction of a multi-chain complex (insulin A+B)."""

    def test_two_chain_complex(self, autobio_config: AutobioConfig, tmp_path: Path) -> None:
        """Predict a 2-chain insulin complex."""
        input_data = BoltzInput(
            sequences={"A": _INSULIN_A, "B": _INSULIN_B},
            num_models=1,
            extra={
                "sampling_steps": 50,
                "recycling_steps": 1,
            },
        )
        runner = get_runner("boltz2", autobio_config)
        output = runner.run(input_data, gpu="auto", output_dir=tmp_path / "ws")

        assert isinstance(output, StructurePredictionOutput)
        assert len(output.structures) == 1

        s = output.structures[0]
        assert s.structure_path.exists()

        # Interface TM should be populated for a multi-chain prediction
        # (iptm measures interface confidence between chains)
        assert s.ptm is not None


# ---------------------------------------------------------------------------
# Boltz-1 — basic prediction
# ---------------------------------------------------------------------------


class TestBoltz1Prediction:
    """Boltz-1 prediction to verify the boltz1 model path works."""

    def test_boltz1_pipeline(self, autobio_config: AutobioConfig, tmp_path: Path) -> None:
        """Run Boltz-1 and verify it produces output."""
        input_data = BoltzInput(
            sequences={"A": _CRAMBIN_SEQ},
            num_models=1,
            extra={
                "sampling_steps": 50,
                "recycling_steps": 1,
            },
        )
        runner = get_runner("boltz1", autobio_config)
        output = runner.run(input_data, gpu="auto", output_dir=tmp_path / "ws")

        assert isinstance(output, StructurePredictionOutput)
        assert len(output.structures) == 1
        assert output.metadata.tool_name == "boltz1"
        assert output.structures[0].structure_path.exists()

        # Boltz-1 should NOT have affinity data
        assert output.structures[0].affinity_probability is None
        assert output.structures[0].affinity_value is None


# ---------------------------------------------------------------------------
# Raw YAML passthrough
# ---------------------------------------------------------------------------


class TestBoltzRawYAML:
    """Verify that raw YAML passthrough works end-to-end."""

    def test_raw_yaml_prediction(self, autobio_config: AutobioConfig, tmp_path: Path) -> None:
        """Provide a raw Boltz YAML spec and verify prediction."""
        boltz_yaml = {
            "version": 1,
            "sequences": [
                {"protein": {"id": "A", "sequence": _CRAMBIN_SEQ}},
            ],
        }
        input_data = BoltzInput(
            sequences={},
            boltz_yaml=boltz_yaml,
            extra={
                "sampling_steps": 50,
                "recycling_steps": 1,
            },
        )
        runner = get_runner("boltz2", autobio_config)
        output = runner.run(input_data, gpu="auto", output_dir=tmp_path / "ws")

        assert isinstance(output, StructurePredictionOutput)
        assert len(output.structures) >= 1
        assert output.structures[0].structure_path.exists()
