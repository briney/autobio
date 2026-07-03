"""Integration tests for RFDiffusion3 runner.

These tests require Docker and a GPU. They run the autobio-rfd3 container
and verify end-to-end output for various design use cases.

Run with:
    pytest tests/integration/test_rfd3_integration.py -v -m docker
"""

from __future__ import annotations

import urllib.request
from typing import TYPE_CHECKING

import pytest

from autobio.core.config import AutobioConfig
from autobio.schemas.structure_design import RFD3Input, StructureDesignOutput
from autobio.tools import get_runner

if TYPE_CHECKING:
    from pathlib import Path

# All tests in this module require Docker, GPU, and are slow
pytestmark = [pytest.mark.docker, pytest.mark.gpu, pytest.mark.slow]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def autobio_config() -> AutobioConfig:
    return AutobioConfig.resolve()


@pytest.fixture(scope="session")
def rcsb_target_pdb(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Download 1CRN (crambin, 46 residues) as a small target for binder design."""
    cache = tmp_path_factory.mktemp("pdb_cache")
    pdb_path = cache / "1crn.pdb"
    urllib.request.urlretrieve(  # noqa: S310
        "https://files.rcsb.org/download/1CRN.pdb",
        pdb_path,
    )
    return pdb_path


# ---------------------------------------------------------------------------
# Unconditioned design
# ---------------------------------------------------------------------------


class TestUnconditionalDesign:
    """Simplest RFD3 use case: design a small protein from scratch."""

    def test_unconditioned_design(self, autobio_config: AutobioConfig, tmp_path: Path) -> None:
        input_data = RFD3Input(
            design_specs={
                "small_protein": {
                    "length": "50",
                    "is_non_loopy": True,
                    "plddt_enhanced": True,
                }
            },
            n_batches=1,
            extra={"diffusion_batch_size": 2, "num_timesteps": 50},
        )
        runner = get_runner("rfd3", autobio_config)
        output = runner.run(input_data, gpu="auto", output_dir=tmp_path / "ws")

        assert isinstance(output, StructureDesignOutput)
        assert len(output.designs) == 2  # diffusion_batch_size=2
        assert output.spec_summary["small_protein"] == 2
        assert output.metadata.tool_name == "rfd3"
        assert output.metadata.wall_time_seconds > 0

        # Each design should have a valid structure path
        for d in output.designs:
            assert d.spec_name == "small_protein"
            assert d.structure_path.exists()
            assert d.structure_path.suffix == ".cif"


# ---------------------------------------------------------------------------
# Binder design
# ---------------------------------------------------------------------------


class TestBinderDesign:
    """Design a protein binder against a target structure."""

    def test_binder_design(
        self, rcsb_target_pdb: Path, autobio_config: AutobioConfig, tmp_path: Path
    ) -> None:
        input_data = RFD3Input(
            input_structures=[rcsb_target_pdb],
            design_specs={
                "crambin_binder": {
                    "input": str(rcsb_target_pdb),
                    "contig": "40-60,/0,A1-46",
                    "length": "86-106",
                    "is_non_loopy": True,
                    "plddt_enhanced": True,
                }
            },
            n_batches=1,
            extra={"diffusion_batch_size": 2, "num_timesteps": 50},
        )
        runner = get_runner("rfd3", autobio_config)
        output = runner.run(input_data, gpu="auto", output_dir=tmp_path / "ws")

        assert isinstance(output, StructureDesignOutput)
        assert len(output.designs) >= 1
        assert output.spec_summary["crambin_binder"] >= 1

        for d in output.designs:
            assert d.structure_path.exists()


# ---------------------------------------------------------------------------
# Multi-spec config
# ---------------------------------------------------------------------------


class TestMultiSpecConfig:
    """Multiple design specifications in a single run."""

    def test_multi_spec_produces_designs_for_each(
        self, autobio_config: AutobioConfig, tmp_path: Path
    ) -> None:
        input_data = RFD3Input(
            design_specs={
                "short_protein": {
                    "length": "50",
                    "is_non_loopy": True,
                },
                "medium_protein": {
                    "length": "80",
                    "is_non_loopy": True,
                },
            },
            n_batches=1,
            extra={"diffusion_batch_size": 2, "num_timesteps": 50},
        )
        runner = get_runner("rfd3", autobio_config)
        output = runner.run(input_data, gpu="auto", output_dir=tmp_path / "ws")

        assert isinstance(output, StructureDesignOutput)
        # Both specs should produce output
        assert "short_protein" in output.spec_summary
        assert "medium_protein" in output.spec_summary
        assert output.spec_summary["short_protein"] >= 1
        assert output.spec_summary["medium_protein"] >= 1
