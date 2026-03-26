"""Integration tests for Proteina-Complexa binder design runner.

These tests require Docker and a GPU. They run the autobio-complexa container
and verify end-to-end output for binder design use cases.

Run with:
    pytest tests/integration/test_complexa_integration.py -v -m docker

Design mode tests (full pipeline with AF2/RF3/MPNN evaluation) require the
full container image (``docker build -t autobio-complexa:2.0.0 .``, NOT the
``--target generate`` variant) and take significantly longer to complete.
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


@pytest.fixture(scope="session")
def rcsb_ligand_pdb(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Download 3PTB (trypsin + benzamidine, ~223 residues, chain A) from RCSB."""
    cache = tmp_path_factory.mktemp("pdb_cache")
    pdb_path = cache / "3ptb.pdb"
    urllib.request.urlretrieve(  # noqa: S310
        "https://files.rcsb.org/download/3PTB.pdb",
        pdb_path,
    )
    return pdb_path


# Minimal generation extra params shared across generate-mode tests
_GENERATE_EXTRA = {
    "batch_size": 2,
    "n_samples_per_length": 1,
    "search_algorithm": "single-pass",
    "seed": 42,
}


# ---------------------------------------------------------------------------
# Protein binder design — generate mode
# ---------------------------------------------------------------------------


class TestProteinBinderGenerate:
    """Generate protein binders against crambin (1CRN) in generate-only mode."""

    def test_protein_binder_design(
        self, rcsb_target_pdb: Path, autobio_config: AutobioConfig, tmp_path: Path
    ) -> None:
        """Basic end-to-end generate run with a single spec."""
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
            extra=_GENERATE_EXTRA,
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

            # Generate mode should NOT have evaluation metrics
            assert d.evaluation_metrics is None

    def test_explicit_generate_mode(
        self, rcsb_target_pdb: Path, autobio_config: AutobioConfig, tmp_path: Path
    ) -> None:
        """Explicit mode='generate' in extra dict behaves identically to default."""
        input_data = StructureDesignInput(
            input_structures=[rcsb_target_pdb],
            design_specs={
                "explicit_gen": {
                    "input": str(rcsb_target_pdb),
                    "target_input": "A1-46",
                    "hotspot_residues": ["A10"],
                    "binder_length": [40, 50],
                },
            },
            n_batches=1,
            extra={**_GENERATE_EXTRA, "mode": "generate"},
        )
        runner = get_runner("complexa", autobio_config)
        output = runner.run(input_data, gpu="auto", output_dir=tmp_path / "ws")

        assert isinstance(output, StructureDesignOutput)
        assert len(output.designs) >= 1
        for d in output.designs:
            assert d.structure_path.exists()
            assert d.evaluation_metrics is None


# ---------------------------------------------------------------------------
# Multi-spec generate
# ---------------------------------------------------------------------------


class TestMultiSpecGenerate:
    """Multiple design specs in a single generate run."""

    def test_two_specs_produce_separate_outputs(
        self, rcsb_target_pdb: Path, autobio_config: AutobioConfig, tmp_path: Path
    ) -> None:
        """Two specs against the same target produce designs for each."""
        input_data = StructureDesignInput(
            input_structures=[rcsb_target_pdb],
            design_specs={
                "short_binder": {
                    "input": str(rcsb_target_pdb),
                    "target_input": "A1-46",
                    "hotspot_residues": ["A10"],
                    "binder_length": [30, 40],
                },
                "long_binder": {
                    "input": str(rcsb_target_pdb),
                    "target_input": "A1-46",
                    "hotspot_residues": ["A25", "A30"],
                    "binder_length": [60, 80],
                },
            },
            n_batches=1,
            extra=_GENERATE_EXTRA,
        )
        runner = get_runner("complexa", autobio_config)
        output = runner.run(input_data, gpu="auto", output_dir=tmp_path / "ws")

        assert isinstance(output, StructureDesignOutput)
        # Both specs should produce output
        assert "short_binder" in output.spec_summary
        assert "long_binder" in output.spec_summary
        assert output.spec_summary["short_binder"] >= 1
        assert output.spec_summary["long_binder"] >= 1


# ---------------------------------------------------------------------------
# Ligand binder variant — generate mode
# ---------------------------------------------------------------------------


class TestLigandBinderGenerate:
    """Ligand binder variant using 3PTB (trypsin + benzamidine).

    This exercises the ``complexa_ligand`` checkpoint and
    ``search_ligand_binder_local_pipeline`` config. Benzamidine (BEN) in
    3PTB is a HETATM on chain A. The ``ligand`` field is the 3-letter
    residue name of the ligand in the PDB.
    """

    def test_ligand_binder_design(
        self, rcsb_ligand_pdb: Path, autobio_config: AutobioConfig, tmp_path: Path
    ) -> None:
        input_data = StructureDesignInput(
            input_structures=[rcsb_ligand_pdb],
            design_specs={
                "trypsin_binder": {
                    "input": str(rcsb_ligand_pdb),
                    "target_input": "A1-223",
                    "binder_length": [40, 60],
                    "ligand": "BEN",
                    "smiles": "c1ccc(cc1)C(=N)N",
                },
            },
            n_batches=1,
            extra=_GENERATE_EXTRA,
        )
        runner = get_runner("complexa_ligand", autobio_config)
        output = runner.run(input_data, gpu="auto", output_dir=tmp_path / "ws")

        assert isinstance(output, StructureDesignOutput)
        assert len(output.designs) >= 1
        assert output.metadata.tool_name == "complexa_ligand"

        for d in output.designs:
            assert d.spec_name == "trypsin_binder"
            assert d.structure_path.exists()
            content = d.structure_path.read_text()
            assert "ATOM" in content


# ---------------------------------------------------------------------------
# AME variant — generate mode
# ---------------------------------------------------------------------------


class TestAMEGenerate:
    """AME (motif scaffolding) variant using crambin as a motif.

    This exercises the ``complexa_ame`` checkpoint and
    ``search_ame_local_pipeline`` config. Crambin residues 5-15 (chain A)
    are used as the motif, with per-residue atom specs in ``contig_atoms``.
    """

    def test_ame_motif_scaffolding(
        self, rcsb_target_pdb: Path, autobio_config: AutobioConfig, tmp_path: Path
    ) -> None:
        input_data = StructureDesignInput(
            input_structures=[rcsb_target_pdb],
            design_specs={
                "crambin_motif": {
                    "input": str(rcsb_target_pdb),
                    "contig_atoms": (
                        "A5: [N, CA, C, O, CB], A10: [N, CA, C, O, CB], A15: [N, CA, C, O, CB]"
                    ),
                    "binder_length": [50, 80],
                },
            },
            n_batches=1,
            extra=_GENERATE_EXTRA,
        )
        runner = get_runner("complexa_ame", autobio_config)
        output = runner.run(input_data, gpu="auto", output_dir=tmp_path / "ws")

        assert isinstance(output, StructureDesignOutput)
        assert len(output.designs) >= 1
        assert output.metadata.tool_name == "complexa_ame"

        for d in output.designs:
            assert d.spec_name == "crambin_motif"
            assert d.structure_path.exists()
            content = d.structure_path.read_text()
            assert "ATOM" in content


# ---------------------------------------------------------------------------
# Design mode — full pipeline (generate + filter + evaluate + analyze)
# ---------------------------------------------------------------------------


class TestDesignPipeline:
    """Full design pipeline (mode='design') with AF2/RF3/MPNN evaluation.

    These tests require the full container image (not the ``--target generate``
    build) and take significantly longer than generate-only tests due to
    AF2/RF3 structure prediction evaluation.
    """

    def test_protein_binder_design_pipeline(
        self, rcsb_target_pdb: Path, autobio_config: AutobioConfig, tmp_path: Path
    ) -> None:
        """Full pipeline: generate, filter, evaluate, analyze."""
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
                "mode": "design",
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

        # Structure files should exist
        for d in output.designs:
            assert d.spec_name == "crambin_binder"
            assert d.structure_path.exists()
            assert d.structure_path.suffix == ".pdb"
            content = d.structure_path.read_text()
            assert "ATOM" in content

        # At least some designs should have evaluation metrics from the
        # evaluate step (AF2/RF3/MPNN). Designs that were filtered out
        # before evaluation may have evaluation_metrics=None.
        designs_with_metrics = [d for d in output.designs if d.evaluation_metrics is not None]
        assert len(designs_with_metrics) >= 1, (
            "Expected at least one design with evaluation_metrics from "
            "the full pipeline, but all were None."
        )

        # Verify evaluation metrics have expected structure
        for d in designs_with_metrics:
            assert isinstance(d.evaluation_metrics, dict)
            assert len(d.evaluation_metrics) >= 1


# ---------------------------------------------------------------------------
# Output metadata
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
            extra=_GENERATE_EXTRA,
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
        assert output.metadata.image_uri.endswith("complexa:2.0.0")
