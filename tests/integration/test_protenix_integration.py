"""Integration tests for Protenix-v2 structure prediction runner.

These tests require Docker and a GPU. They run the autobio-protenix container
and verify end-to-end structure prediction output.

Run with:
    pytest tests/integration/test_protenix_integration.py -v -m docker
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from autobio.core.config import AutobioConfig
from autobio.schemas.structure_prediction import (
    StructurePredictionInput,
    StructurePredictionOutput,
)
from autobio.tools import get_runner

if TYPE_CHECKING:
    from pathlib import Path

# All tests in this module require Docker, GPU, and are slow
pytestmark = [pytest.mark.docker, pytest.mark.gpu, pytest.mark.slow]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

# Crambin (PDB: 1CRN) — 46 residues, single chain, well-folded small protein
_CRAMBIN_SEQ = "TTCCPSIVARSNFNVCRLPGTPEAICATYTGCIIIPGATCPGDYAN"

# Insulin A and B chains — small, well-characterized complex
_INSULIN_A = "GIVEQCCTSICSLYQLENYCN"
_INSULIN_B = "FVNQHLCGSHLVEALYLVCGERGFFYTPKT"


@pytest.fixture(scope="session")
def autobio_config() -> AutobioConfig:
    return AutobioConfig.resolve()


# ---------------------------------------------------------------------------
# Protenix-v2 — single protein
# ---------------------------------------------------------------------------


class TestProtenixSingleProtein:
    """Protenix-v2 structure prediction on a single small protein."""

    def test_full_pipeline(self, autobio_config: AutobioConfig, tmp_path: Path) -> None:
        """Run Protenix-v2 end-to-end and verify output structure."""
        input_data = StructurePredictionInput(
            sequences={"A": _CRAMBIN_SEQ},
            num_models=1,
        )
        runner = get_runner("protenix_v2", autobio_config)
        output = runner.run(input_data, gpu="auto", output_dir=tmp_path / "ws")

        assert isinstance(output, StructurePredictionOutput)
        assert len(output.structures) == 1
        assert output.metadata.tool_name == "protenix_v2"
        assert output.metadata.wall_time_seconds > 0

        s = output.structures[0]
        assert s.model_rank == 1
        assert s.structure_path.exists()

    def test_confidence_scores(self, autobio_config: AutobioConfig, tmp_path: Path) -> None:
        """Verify confidence metrics are populated."""
        input_data = StructurePredictionInput(
            sequences={"A": _CRAMBIN_SEQ},
            num_models=1,
        )
        runner = get_runner("protenix_v2", autobio_config)
        output = runner.run(input_data, gpu="auto", output_dir=tmp_path / "ws")

        s = output.structures[0]
        assert s.ptm is not None
        assert 0.0 <= s.ptm <= 1.0
        assert s.plddt_mean is not None
        assert s.plddt_mean > 0

        assert output.confidence.best_ptm is not None
        assert output.confidence.best_plddt_mean is not None

    def test_multiple_models(self, autobio_config: AutobioConfig, tmp_path: Path) -> None:
        """Request 3 diffusion samples and verify count and ranking."""
        input_data = StructurePredictionInput(
            sequences={"A": _CRAMBIN_SEQ},
            num_models=3,
        )
        runner = get_runner("protenix_v2", autobio_config)
        output = runner.run(input_data, gpu="auto", output_dir=tmp_path / "ws")

        assert len(output.structures) == 3
        ranks = [s.model_rank for s in output.structures]
        assert ranks == [1, 2, 3]

        for s in output.structures:
            assert s.structure_path.exists()

        assert output.confidence.best_ptm is not None


# ---------------------------------------------------------------------------
# Protenix-v2 — multi-chain complex
# ---------------------------------------------------------------------------


class TestProtenixMultiChain:
    """Protenix-v2 prediction of a multi-chain complex (insulin A+B)."""

    def test_two_chain_complex(self, autobio_config: AutobioConfig, tmp_path: Path) -> None:
        """Predict a 2-chain insulin complex."""
        input_data = StructurePredictionInput(
            sequences={"A": _INSULIN_A, "B": _INSULIN_B},
            num_models=1,
        )
        runner = get_runner("protenix_v2", autobio_config)
        output = runner.run(input_data, gpu="auto", output_dir=tmp_path / "ws")

        assert isinstance(output, StructurePredictionOutput)
        assert len(output.structures) == 1

        s = output.structures[0]
        assert s.structure_path.exists()
        assert s.ptm is not None


# ---------------------------------------------------------------------------
# Protenix-v2 — protein + ligand
# ---------------------------------------------------------------------------


class TestProtenixProteinLigand:
    """Protenix-v2 prediction with a protein and small molecule ligand."""

    def test_protein_ligand_complex(self, autobio_config: AutobioConfig, tmp_path: Path) -> None:
        """Predict a protein-ligand complex using CCD code."""
        input_data = StructurePredictionInput(
            sequences={"A": _CRAMBIN_SEQ, "L": ""},
            num_models=1,
            extra={
                "entity_types": {"L": {"ccd": "ATP"}},
            },
        )
        runner = get_runner("protenix_v2", autobio_config)
        output = runner.run(input_data, gpu="auto", output_dir=tmp_path / "ws")

        assert isinstance(output, StructurePredictionOutput)
        assert len(output.structures) == 1
        assert output.structures[0].structure_path.exists()


# ---------------------------------------------------------------------------
# Protenix-v2 — protein + ion (unique to Protenix)
# ---------------------------------------------------------------------------


class TestProtenixProteinIon:
    """Protenix-v2 prediction with protein and ion entity."""

    def test_protein_ion_complex(self, autobio_config: AutobioConfig, tmp_path: Path) -> None:
        """Predict a protein with a magnesium ion."""
        input_data = StructurePredictionInput(
            sequences={"A": _CRAMBIN_SEQ, "I": "MG"},
            num_models=1,
            extra={
                "entity_types": {"I": "ion"},
            },
        )
        runner = get_runner("protenix_v2", autobio_config)
        output = runner.run(input_data, gpu="auto", output_dir=tmp_path / "ws")

        assert isinstance(output, StructurePredictionOutput)
        assert len(output.structures) == 1
        assert output.structures[0].structure_path.exists()


# ---------------------------------------------------------------------------
# Protenix-v2 — raw query JSON passthrough
# ---------------------------------------------------------------------------


class TestProtenixRawQueryJSON:
    """Verify that raw query JSON passthrough works end-to-end."""

    def test_raw_query_json_prediction(self, autobio_config: AutobioConfig, tmp_path: Path) -> None:
        """Provide a raw Protenix input JSON and verify prediction."""
        query = [
            {
                "name": "crambin",
                "sequences": [
                    {
                        "proteinChain": {
                            "sequence": _CRAMBIN_SEQ,
                            "count": 1,
                            "id": ["A"],
                        }
                    }
                ],
                "covalent_bonds": [],
                "constraint": {},
            }
        ]
        input_data = StructurePredictionInput(
            sequences={},
            extra={"query_json": query},
        )
        runner = get_runner("protenix_v2", autobio_config)
        output = runner.run(input_data, gpu="auto", output_dir=tmp_path / "ws")

        assert isinstance(output, StructurePredictionOutput)
        assert len(output.structures) >= 1
        assert output.structures[0].structure_path.exists()
