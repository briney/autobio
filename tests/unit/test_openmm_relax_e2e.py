"""End-to-end tests for OpenMM amber relax.

Each test exercises the full pipeline:
    input construction → validation → prepare_workspace →
    (simulated raw output) → standardize.py → parse_output → verify

The only thing not tested is the actual OpenMM relaxation.
The standardize script is imported and run directly against realistic
OpenMM output data.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from autobio.core.catalog import get_tool
from autobio.core.config import AutobioConfig
from autobio.core.result import AutobioError
from autobio.core.workspace import Workspace
from autobio.schemas.scoring import ScoringInput, ScoringOutput
from autobio.tools.openmm import OpenMMRunner

# ---------------------------------------------------------------------------
# Realistic OpenMM output data
# ---------------------------------------------------------------------------

# Minimal but valid PDB content for testing
_MINIMAL_PDB = (
    "HEADER    TEST STRUCTURE\n"
    "ATOM      1  N   ALA A   1       1.000   2.000   3.000  1.00 10.00           N\n"
    "ATOM      2  CA  ALA A   1       2.000   3.000   4.000  1.00 10.00           C\n"
    "ATOM      3  C   ALA A   1       3.000   4.000   5.000  1.00 10.00           C\n"
    "ATOM      4  O   ALA A   1       3.500   4.500   5.500  1.00 10.00           O\n"
    "ATOM      5  N   GLY A   2       4.000   5.000   6.000  1.00 12.00           N\n"
    "ATOM      6  CA  GLY A   2       5.000   6.000   7.000  1.00 12.00           C\n"
    "ATOM      7  C   GLY A   2       6.000   7.000   8.000  1.00 12.00           C\n"
    "ATOM      8  O   GLY A   2       6.500   7.500   8.500  1.00 12.00           O\n"
    "END\n"
)

# Realistic energy.json from OpenMM relaxation (explicit solvent protocol)
_ENERGY_JSON = json.dumps(
    {
        "initial_energy_kj_mol": -45000.0,
        "final_energy_kj_mol": -52000.3,
        "energy_terms": {
            "HarmonicBondForce": -14000.1,
            "HarmonicAngleForce": -6500.2,
            "PeriodicTorsionForce": -3200.0,
            "NonbondedForce": -28300.0,
        },
        "protocol": {
            "phases_completed": ["minimize", "heating", "nvt", "npt", "production"],
            "temperature_kelvin": 300.0,
            "pressure_atm": 1.01,
        },
    }
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def config() -> AutobioConfig:
    return AutobioConfig.resolve()


@pytest.fixture()
def sample_pdb(tmp_path: Path) -> Path:
    """Write a minimal but valid PDB file."""
    pdb_path = tmp_path / "structure.pdb"
    pdb_path.write_text(_MINIMAL_PDB)
    return pdb_path


def _make_runner(mode_name: str, config: AutobioConfig) -> OpenMMRunner:
    """Create a runner with mocked container/GPU (we simulate container output)."""
    with (
        patch("autobio.tools.base.ContainerManager"),
        patch("autobio.tools.base.GPUManager"),
    ):
        runner = OpenMMRunner("openmm", config)
    runner.current_mode = get_tool("openmm").modes[mode_name]
    return runner


def _import_standardize():
    """Import the container's standardize module."""
    container_dir = str(
        Path(__file__).resolve().parent.parent.parent / "containers" / "openmm-amber-relax"
    )
    if container_dir not in sys.path:
        sys.path.insert(0, container_dir)
    import importlib

    mod = importlib.import_module("standardize")
    importlib.reload(mod)  # ensure fresh import
    return mod


def _run_e2e(
    config: AutobioConfig,
    input_data: ScoringInput,
    raw_output_files: dict[str, str],
    tmp_path: Path,
) -> ScoringOutput:
    """Full end-to-end pipeline without Docker.

    1. prepare_workspace
    2. Write simulated raw output files
    3. Run the container's standardize.py
    4. parse_output
    """
    runner = _make_runner("amber_relax", config)
    workspace = Workspace.create(tmp_path / "ws")

    # Step 1: prepare workspace (host-side validation + config writing)
    runner.prepare_workspace(input_data, workspace)

    # Verify config.json was written
    cfg = json.loads(workspace.config_path.read_text())
    assert cfg["protocol"] == "amber_relax"

    # Step 2: write simulated raw output (what the container would produce)
    for filename, content in raw_output_files.items():
        (workspace.raw_output_dir / filename).write_text(content)

    # Step 3: run the actual standardize.py script
    std_mod = _import_standardize()
    std_mod.standardize(workspace.root)

    # Verify result_data.json was produced
    result_data_path = workspace.std_output_dir / "result_data.json"
    assert result_data_path.exists(), "standardize.py did not produce result_data.json"

    # Step 4: parse output
    output = runner.parse_output(workspace)
    assert isinstance(output, ScoringOutput)
    return output


# ---------------------------------------------------------------------------
# TestOpenMMAmberRelaxE2E
# ---------------------------------------------------------------------------


class TestOpenMMAmberRelaxE2E:
    """End-to-end test for openmm_amber_relax: full lifecycle."""

    def test_relax_full_pipeline(
        self, config: AutobioConfig, sample_pdb: Path, tmp_path: Path
    ) -> None:
        """Relax a structure and verify energy breakdown."""
        input_data = ScoringInput(structure_path=sample_pdb)
        output = _run_e2e(
            config=config,
            input_data=input_data,
            raw_output_files={
                "energy.json": _ENERGY_JSON,
                "relaxed.pdb": _MINIMAL_PDB,
            },
            tmp_path=tmp_path,
        )

        assert len(output.scores) == 1
        s = output.scores[0]
        assert s.total_score == pytest.approx(-52000.3)
        assert s.units == "kJ/mol"
        assert s.ddg is None
        assert s.mutations is None

        # Verify score breakdown has energy force types
        assert s.score_breakdown is not None
        assert "HarmonicBondForce" in s.score_breakdown
        assert "NonbondedForce" in s.score_breakdown
        assert s.score_breakdown["HarmonicBondForce"] == pytest.approx(-14000.1)

    def test_relax_output_structure(
        self, config: AutobioConfig, sample_pdb: Path, tmp_path: Path
    ) -> None:
        """Relaxed PDB exists and is accessible."""
        input_data = ScoringInput(structure_path=sample_pdb)
        output = _run_e2e(
            config=config,
            input_data=input_data,
            raw_output_files={
                "energy.json": _ENERGY_JSON,
                "relaxed.pdb": _MINIMAL_PDB,
            },
            tmp_path=tmp_path,
        )

        s = output.scores[0]
        assert s.structure_path is not None
        assert s.structure_path.exists()
        assert s.structure_path.suffix == ".pdb"
        content = s.structure_path.read_text()
        assert "ATOM" in content

    def test_relax_energy_breakdown(
        self, config: AutobioConfig, sample_pdb: Path, tmp_path: Path
    ) -> None:
        """Score breakdown includes force types and protocol metadata."""
        input_data = ScoringInput(structure_path=sample_pdb)
        output = _run_e2e(
            config=config,
            input_data=input_data,
            raw_output_files={
                "energy.json": _ENERGY_JSON,
                "relaxed.pdb": _MINIMAL_PDB,
            },
            tmp_path=tmp_path,
        )

        breakdown = output.scores[0].score_breakdown
        assert breakdown is not None
        assert "initial_energy" in breakdown
        assert breakdown["initial_energy"] == pytest.approx(-45000.0)
        assert breakdown["protocol"] == "amber_relax"
        assert "phases_completed" in breakdown
        assert "minimize" in breakdown["phases_completed"]
        assert "production" in breakdown["phases_completed"]

    def test_relax_units(self, config: AutobioConfig, sample_pdb: Path, tmp_path: Path) -> None:
        """Units are kJ/mol."""
        input_data = ScoringInput(structure_path=sample_pdb)
        output = _run_e2e(
            config=config,
            input_data=input_data,
            raw_output_files={
                "energy.json": _ENERGY_JSON,
                "relaxed.pdb": _MINIMAL_PDB,
            },
            tmp_path=tmp_path,
        )
        assert output.scores[0].units == "kJ/mol"

    def test_relax_config_defaults(
        self, config: AutobioConfig, sample_pdb: Path, tmp_path: Path
    ) -> None:
        """Verify default config values written to config.json."""
        runner = _make_runner("amber_relax", config)
        workspace = Workspace.create(tmp_path / "ws")
        input_data = ScoringInput(structure_path=sample_pdb)
        runner.prepare_workspace(input_data, workspace)

        cfg = json.loads(workspace.config_path.read_text())
        assert cfg["force_field"] == "amber14-all.xml"
        assert cfg["implicit_solvent"] is False
        assert cfg["water_model"] == "tip3p"
        assert cfg["box_shape"] == "cubic"
        assert cfg["box_padding"] == pytest.approx(1.0)
        assert cfg["ion_type"] == "NaCl"
        assert cfg["ion_concentration"] == pytest.approx(0.15)
        assert cfg["temperature"] == pytest.approx(300.0)
        assert cfg["pressure"] == pytest.approx(1.0)
        assert cfg["restraint_set"] == "heavy_atoms"
        assert cfg["restraint_stiffness"] == pytest.approx(10.0)
        assert cfg["timestep"] == pytest.approx(2.0)
        assert cfg["minimize_max_iterations"] == 0
        assert cfg["heating_steps"] == 25000
        assert cfg["nvt_steps"] == 25000
        assert cfg["npt_steps"] == 50000
        assert cfg["production_steps"] == 25000

    def test_relax_custom_solvation_params(
        self, config: AutobioConfig, sample_pdb: Path, tmp_path: Path
    ) -> None:
        """Custom solvation parameters are written to config.json."""
        runner = _make_runner("amber_relax", config)
        workspace = Workspace.create(tmp_path / "ws")
        input_data = ScoringInput(
            structure_path=sample_pdb,
            extra={
                "water_model": "tip4pew",
                "ion_type": "KCl",
                "ion_concentration": 0.1,
                "box_shape": "dodecahedron",
                "box_padding": 1.5,
                "temperature": 310.0,
                "pressure": 1.5,
            },
        )
        runner.prepare_workspace(input_data, workspace)

        cfg = json.loads(workspace.config_path.read_text())
        assert cfg["water_model"] == "tip4pew"
        assert cfg["ion_type"] == "KCl"
        assert cfg["ion_concentration"] == pytest.approx(0.1)
        assert cfg["box_shape"] == "dodecahedron"
        assert cfg["box_padding"] == pytest.approx(1.5)
        assert cfg["temperature"] == pytest.approx(310.0)
        assert cfg["pressure"] == pytest.approx(1.5)


# ---------------------------------------------------------------------------
# TestOpenMMRelaxInputValidation
# ---------------------------------------------------------------------------


class TestOpenMMRelaxInputValidation:
    """Validation tests for openmm_amber_relax."""

    def test_nonexistent_structure_fails(self, config: AutobioConfig, tmp_path: Path) -> None:
        """Rejects nonexistent input structures."""
        runner = _make_runner("amber_relax", config)
        workspace = Workspace.create(tmp_path / "ws")
        input_data = ScoringInput(structure_path=tmp_path / "nonexistent.pdb")
        with pytest.raises(AutobioError, match="does not exist"):
            runner.prepare_workspace(input_data, workspace)

    def test_structure_copied_correctly(
        self, config: AutobioConfig, sample_pdb: Path, tmp_path: Path
    ) -> None:
        """Input structure is copied to workspace inputs/."""
        runner = _make_runner("amber_relax", config)
        workspace = Workspace.create(tmp_path / "ws")
        input_data = ScoringInput(structure_path=sample_pdb)
        runner.prepare_workspace(input_data, workspace)

        copied = workspace.inputs_dir / sample_pdb.name
        assert copied.exists()
        assert copied.read_text() == sample_pdb.read_text()
