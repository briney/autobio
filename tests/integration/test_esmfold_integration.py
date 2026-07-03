"""Integration tests for ESMFold structure prediction runner.

These tests require Docker and a GPU. They run the autobio-esmfold container
and verify end-to-end structure prediction output.

Run with:
    pytest tests/integration/test_esmfold_integration.py -v -m docker
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from autobio.core.config import AutobioConfig
from autobio.schemas.structure_prediction import ESMFoldInput, StructurePredictionOutput
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


@pytest.fixture(scope="session")
def autobio_config() -> AutobioConfig:
    return AutobioConfig.resolve()


# ---------------------------------------------------------------------------
# ESMFold — single-chain prediction
# ---------------------------------------------------------------------------


class TestESMFoldSingleChain:
    """ESMFold structure prediction on a single small protein."""

    def test_full_pipeline(self, autobio_config: AutobioConfig, tmp_path: Path) -> None:
        """Run ESMFold end-to-end and verify output structure."""
        input_data = ESMFoldInput(
            sequences={"A": _CRAMBIN_SEQ},
            num_models=1,
        )
        runner = get_runner("esmfold", autobio_config)
        output = runner.run(input_data, gpu="auto", output_dir=tmp_path / "ws")

        assert isinstance(output, StructurePredictionOutput)
        assert len(output.structures) == 1
        assert output.metadata.tool_name == "esmfold"
        assert output.metadata.wall_time_seconds > 0

        s = output.structures[0]
        assert s.model_rank == 1
        assert s.structure_path.exists()
        assert s.structure_path.suffix == ".pdb"

    def test_confidence_scores(self, autobio_config: AutobioConfig, tmp_path: Path) -> None:
        """Verify pLDDT and pTM are populated."""
        input_data = ESMFoldInput(
            sequences={"A": _CRAMBIN_SEQ},
            num_models=1,
        )
        runner = get_runner("esmfold", autobio_config)
        output = runner.run(input_data, gpu="auto", output_dir=tmp_path / "ws")

        s = output.structures[0]
        assert s.plddt_mean is not None
        assert 0 < s.plddt_mean <= 100
        assert s.ptm is not None
        assert 0 < s.ptm <= 1
        assert s.iptm is None  # single-chain — no interface score

        assert s.plddt_per_residue is not None
        assert len(s.plddt_per_residue) == len(_CRAMBIN_SEQ)

    def test_summary_confidence(self, autobio_config: AutobioConfig, tmp_path: Path) -> None:
        """Summary confidence metrics match the single structure."""
        input_data = ESMFoldInput(
            sequences={"A": _CRAMBIN_SEQ},
            num_models=1,
        )
        runner = get_runner("esmfold", autobio_config)
        output = runner.run(input_data, gpu="auto", output_dir=tmp_path / "ws")

        assert output.confidence.best_plddt_mean == output.structures[0].plddt_mean
        assert output.confidence.best_ptm == output.structures[0].ptm
        assert output.confidence.best_iptm is None
