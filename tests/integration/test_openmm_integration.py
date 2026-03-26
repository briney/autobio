"""Integration tests for OpenMM tools.

These tests require Docker (no GPU — amber minimize is CPU-only). They
download PDB structures from RCSB, run the autobio-openmm containers, and
verify end-to-end output.

Run with:
    pytest tests/integration/test_openmm_integration.py -v -m docker
"""

from __future__ import annotations

import urllib.request
from typing import TYPE_CHECKING

import pytest

from autobio.core.config import AutobioConfig
from autobio.schemas.scoring import ScoringInput, ScoringOutput
from autobio.schemas.simulation import SimulationInput, SimulationOutput
from autobio.tools import get_runner

if TYPE_CHECKING:
    from pathlib import Path

# All tests in this module require Docker and are slow
pytestmark = [pytest.mark.docker, pytest.mark.slow]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def autobio_config() -> AutobioConfig:
    """Config with local image prefix for locally-built containers."""
    return AutobioConfig.resolve(image_prefix="autobio-")


@pytest.fixture(scope="session")
def rcsb_1ubq(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Download 1UBQ (ubiquitin, 76 residues, single chain) from RCSB."""
    cache = tmp_path_factory.mktemp("pdb_cache")
    pdb_path = cache / "1ubq.pdb"
    urllib.request.urlretrieve(  # noqa: S310
        "https://files.rcsb.org/download/1UBQ.pdb",
        pdb_path,
    )
    return pdb_path


# ---------------------------------------------------------------------------
# openmm_amber_minimize
# ---------------------------------------------------------------------------


class TestOpenMMAmberMinimize:
    """Minimize a structure with OpenMM Amber force field."""

    def test_minimize_1ubq_defaults(
        self, rcsb_1ubq: Path, autobio_config: AutobioConfig, tmp_path: Path
    ) -> None:
        """Minimize ubiquitin with default settings (implicit solvent, no restraints)."""
        input_data = ScoringInput(structure_path=rcsb_1ubq)
        runner = get_runner("openmm_amber_minimize", autobio_config)
        output = runner.run(input_data, gpu="none", output_dir=tmp_path / "ws_default")

        assert isinstance(output, ScoringOutput)
        assert len(output.scores) == 1
        assert output.metadata.tool_name == "openmm_amber_minimize"
        assert output.metadata.wall_time_seconds > 0

        s = output.scores[0]
        assert s.units == "kJ/mol"
        assert isinstance(s.total_score, float)
        assert s.total_score < 0  # minimized energy should be negative

        # Score breakdown should have force-type terms
        assert s.score_breakdown is not None
        assert "initial_energy" in s.score_breakdown
        assert "num_minimization_rounds" in s.score_breakdown
        assert "remaining_violations" in s.score_breakdown
        assert s.score_breakdown["num_minimization_rounds"] >= 1

        # Energy should have decreased (or stayed same) from initial
        assert s.total_score <= s.score_breakdown["initial_energy"]

        # Should produce a minimized PDB
        assert s.structure_path is not None
        assert s.structure_path.exists()
        assert s.structure_path.suffix == ".pdb"
        content = s.structure_path.read_text()
        assert "ATOM" in content

    def test_minimize_1ubq_with_ca_restraints(
        self, rcsb_1ubq: Path, autobio_config: AutobioConfig, tmp_path: Path
    ) -> None:
        """Minimize ubiquitin with CA restraints."""
        input_data = ScoringInput(
            structure_path=rcsb_1ubq,
            extra={
                "restraint_set": "ca",
                "restraint_stiffness": 10.0,
                "max_outer_iterations": 5,
            },
        )
        runner = get_runner("openmm_amber_minimize", autobio_config)
        output = runner.run(input_data, gpu="none", output_dir=tmp_path / "ws_ca")

        assert isinstance(output, ScoringOutput)
        assert len(output.scores) == 1

        s = output.scores[0]
        assert s.units == "kJ/mol"
        assert isinstance(s.total_score, float)
        assert s.structure_path is not None
        assert s.structure_path.exists()

    def test_minimize_1ubq_vacuum(
        self, rcsb_1ubq: Path, autobio_config: AutobioConfig, tmp_path: Path
    ) -> None:
        """Minimize ubiquitin in vacuum (implicit_solvent=False)."""
        input_data = ScoringInput(
            structure_path=rcsb_1ubq,
            extra={
                "implicit_solvent": False,
                "max_iterations": 100,
            },
        )
        runner = get_runner("openmm_amber_minimize", autobio_config)
        output = runner.run(input_data, gpu="none", output_dir=tmp_path / "ws_vacuum")

        assert isinstance(output, ScoringOutput)
        assert len(output.scores) == 1

        s = output.scores[0]
        assert s.units == "kJ/mol"
        assert isinstance(s.total_score, float)
        assert s.structure_path is not None
        assert s.structure_path.exists()

    def test_minimize_1ubq_heavy_atom_restraints(
        self, rcsb_1ubq: Path, autobio_config: AutobioConfig, tmp_path: Path
    ) -> None:
        """Minimize ubiquitin with heavy atom restraints (AlphaFold-style)."""
        input_data = ScoringInput(
            structure_path=rcsb_1ubq,
            extra={
                "restraint_set": "heavy_atoms",
                "restraint_stiffness": 10.0,
                "max_outer_iterations": 3,
            },
        )
        runner = get_runner("openmm_amber_minimize", autobio_config)
        output = runner.run(input_data, gpu="none", output_dir=tmp_path / "ws_heavy")

        assert isinstance(output, ScoringOutput)
        s = output.scores[0]
        assert s.units == "kJ/mol"
        assert s.structure_path is not None
        assert s.structure_path.exists()
        assert s.score_breakdown is not None


# ---------------------------------------------------------------------------
# openmm_amber_relax
# ---------------------------------------------------------------------------


class TestOpenMMAmberRelax:
    """Relax a structure with OpenMM explicit/implicit solvent."""

    def test_relax_1ubq_explicit_solvent(
        self, rcsb_1ubq: Path, autobio_config: AutobioConfig, tmp_path: Path
    ) -> None:
        """Relax ubiquitin with default explicit solvent settings."""
        input_data = ScoringInput(structure_path=rcsb_1ubq)
        runner = get_runner("openmm_amber_relax", autobio_config)
        output = runner.run(input_data, gpu="none", output_dir=tmp_path / "ws_relax")

        assert isinstance(output, ScoringOutput)
        assert len(output.scores) == 1

        s = output.scores[0]
        assert s.units == "kJ/mol"
        assert isinstance(s.total_score, float)
        assert s.structure_path is not None
        assert s.structure_path.exists()
        assert "ATOM" in s.structure_path.read_text()

    def test_relax_1ubq_implicit_solvent(
        self, rcsb_1ubq: Path, autobio_config: AutobioConfig, tmp_path: Path
    ) -> None:
        """Relax ubiquitin with implicit solvent (OBC2)."""
        input_data = ScoringInput(
            structure_path=rcsb_1ubq,
            extra={"implicit_solvent": True},
        )
        runner = get_runner("openmm_amber_relax", autobio_config)
        output = runner.run(input_data, gpu="none", output_dir=tmp_path / "ws_relax_implicit")

        assert isinstance(output, ScoringOutput)
        assert output.scores[0].structure_path is not None

    def test_relax_custom_solvation(
        self, rcsb_1ubq: Path, autobio_config: AutobioConfig, tmp_path: Path
    ) -> None:
        """Relax ubiquitin with custom solvation parameters."""
        input_data = ScoringInput(
            structure_path=rcsb_1ubq,
            extra={
                "water_model": "tip4pew",
                "ion_type": "KCl",
                "ion_concentration": 0.1,
            },
        )
        runner = get_runner("openmm_amber_relax", autobio_config)
        output = runner.run(input_data, gpu="none", output_dir=tmp_path / "ws_relax_custom")

        assert isinstance(output, ScoringOutput)


# ---------------------------------------------------------------------------
# openmm_md_simulate
# ---------------------------------------------------------------------------


class TestOpenMMSimulate:
    """Run production MD simulation with OpenMM."""

    pytestmark = pytest.mark.gpu

    def test_simulate_1ubq_short(
        self, rcsb_1ubq: Path, autobio_config: AutobioConfig, tmp_path: Path
    ) -> None:
        """Short MD simulation of ubiquitin (0.1 ns)."""
        input_data = SimulationInput(
            structure_path=rcsb_1ubq,
            extra={
                "total_time_ns": 0.1,
                "reporting_interval_steps": 500,
            },
        )
        runner = get_runner("openmm_md_simulate", autobio_config)
        output = runner.run(input_data, output_dir=tmp_path / "ws_simulate")

        assert isinstance(output, SimulationOutput)
        assert output.trajectory_path.exists()
        assert len(output.energy_timeseries) > 0
        assert output.summary.total_time_ns == pytest.approx(0.1, abs=0.01)
        assert output.final_structure_path is not None
        assert output.final_structure_path.exists()
