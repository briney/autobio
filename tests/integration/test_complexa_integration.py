"""Integration tests for Proteina-Complexa binder design runner.

These tests require Docker and a GPU. They run the autobio-complexa container
and verify end-to-end output for binder design use cases.

Run with:
    pytest tests/integration/test_complexa_integration.py -v -m docker
"""

from __future__ import annotations

import urllib.request
from typing import TYPE_CHECKING

import pytest

from autobio.core.config import AutobioConfig
from autobio.schemas.structure_design import StructureDesignInput, StructureDesignOutput
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
# Protein binder design
# ---------------------------------------------------------------------------


class TestProteinBinderDesign:
    """Design a small protein binder against crambin (1CRN)."""

    def test_protein_binder_design(
        self, rcsb_target_pdb: Path, autobio_config: AutobioConfig, tmp_path: Path
    ) -> None:
        input_data = StructureDesignInput(
            input_structures=[rcsb_target_pdb],
            design_specs={
                "crambin_binder": {
                    "input": str(rcsb_target_pdb),
                    "target_input": "A1-46",
                    "hotspot_residues": ["A10", "A15", "A25"],
                    "binder_length": [40, 60],
                },
            },
            n_batches=1,
            extra={
                "batch_size": 2,
                "n_samples_per_length": 1,
                "search_algorithm": "single-pass",
                "seed": 42,
            },
        )
        runner = get_runner("complexa", autobio_config)
        output = runner.run(input_data, gpu="auto", output_dir=tmp_path / "ws")

        assert isinstance(output, StructureDesignOutput)
        assert len(output.designs) >= 1
        assert output.spec_summary["crambin_binder"] >= 1
        assert output.metadata.tool_name == "complexa"
        assert output.metadata.wall_time_seconds > 0

        # Each design should have a valid PDB path
        for d in output.designs:
            assert d.spec_name == "crambin_binder"
            assert d.structure_path.exists()
            assert d.structure_path.suffix == ".pdb"

            # PDB should have content
            content = d.structure_path.read_text()
            assert "ATOM" in content


# ---------------------------------------------------------------------------
# Verify output metadata
# ---------------------------------------------------------------------------


class TestOutputMetadata:
    """Verify that outputs include expected metadata and structure."""

    def test_diffusion_metadata_present(
        self, rcsb_target_pdb: Path, autobio_config: AutobioConfig, tmp_path: Path
    ) -> None:
        input_data = StructureDesignInput(
            input_structures=[rcsb_target_pdb],
            design_specs={
                "metadata_test": {
                    "input": str(rcsb_target_pdb),
                    "target_input": "A1-46",
                    "hotspot_residues": ["A10"],
                    "binder_length": [40, 50],
                },
            },
            n_batches=1,
            extra={
                "batch_size": 1,
                "n_samples_per_length": 1,
                "search_algorithm": "single-pass",
                "seed": 42,
            },
        )
        runner = get_runner("complexa", autobio_config)
        output = runner.run(input_data, gpu="auto", output_dir=tmp_path / "ws")

        assert len(output.designs) >= 1
        d = output.designs[0]
        # diffusion_metadata should include binder_length at minimum
        assert d.diffusion_metadata is not None
        assert "binder_length" in d.diffusion_metadata

        # Run metadata should be populated
        assert output.metadata.gpu_ids is not None
        assert len(output.metadata.gpu_ids) >= 1
        assert output.metadata.image_uri.endswith("complexa:1.0.0")
