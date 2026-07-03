"""End-to-end tests for OpenMM MD simulation.

Each test exercises the full pipeline:
    input construction → validation → prepare_workspace →
    (simulated raw output) → standardize.py → parse_output → verify

The only thing not tested is the actual OpenMM simulation.
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
from autobio.schemas.simulation import SimulationInput, SimulationOutput
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

# Realistic energy timeseries from OpenMM MD production
_ENERGY_TIMESERIES_JSON = json.dumps(
    [
        {
            "step": 5000,
            "time_ps": 10.0,
            "potential_energy_kj_mol": -120000.5,
            "kinetic_energy_kj_mol": 25000.3,
            "total_energy_kj_mol": -95000.2,
            "temperature_K": 298.5,
            "volume_nm3": 215.3,
            "density_kg_m3": 997.2,
        },
        {
            "step": 10000,
            "time_ps": 20.0,
            "potential_energy_kj_mol": -120500.1,
            "kinetic_energy_kj_mol": 25100.8,
            "total_energy_kj_mol": -95399.3,
            "temperature_K": 300.1,
            "volume_nm3": 215.1,
            "density_kg_m3": 997.4,
        },
        {
            "step": 15000,
            "time_ps": 30.0,
            "potential_energy_kj_mol": -120200.7,
            "kinetic_energy_kj_mol": 24900.2,
            "total_energy_kj_mol": -95300.5,
            "temperature_K": 299.8,
            "volume_nm3": 215.2,
            "density_kg_m3": 997.3,
        },
    ]
)

# Realistic simulation summary from OpenMM MD
_SIMULATION_SUMMARY_JSON = json.dumps(
    {
        "n_steps_completed": 50000,
        "total_time_ns": 0.1,
        "initial_potential_energy_kj_mol": -120000.5,
        "final_potential_energy_kj_mol": -120200.7,
        "mean_temperature_K": 299.5,
        "mean_potential_energy_kj_mol": -120233.8,
        "platform_used": "CPU",
        "force_field": "amber14-all.xml",
        "water_model": "tip3p",
        "box_shape": "cubic",
        "ion_concentration_M": 0.15,
        "equilibration_protocol": {
            "nvt_steps": 50000,
            "npt_steps": 100000,
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
        Path(__file__).resolve().parent.parent.parent / "containers" / "openmm-md-simulate"
    )
    if container_dir not in sys.path:
        sys.path.insert(0, container_dir)
    import importlib

    mod = importlib.import_module("standardize")
    importlib.reload(mod)  # ensure fresh import
    return mod


def _run_e2e(
    config: AutobioConfig,
    input_data: SimulationInput,
    raw_output_files: dict[str, str | bytes],
    tmp_path: Path,
) -> SimulationOutput:
    """Full end-to-end pipeline without Docker.

    1. prepare_workspace
    2. Write simulated raw output files
    3. Run the container's standardize.py
    4. parse_output
    """
    runner = _make_runner("md_simulate", config)
    workspace = Workspace.create(tmp_path / "ws")

    # Step 1: prepare workspace (host-side validation + config writing)
    runner.prepare_workspace(input_data, workspace)

    # Verify config.json was written
    cfg = json.loads(workspace.config_path.read_text())
    assert cfg["protocol"] == "md_simulate"

    # Step 2: write simulated raw output (what the container would produce)
    for filename, content in raw_output_files.items():
        path = workspace.raw_output_dir / filename
        if isinstance(content, bytes):
            path.write_bytes(content)
        else:
            path.write_text(content)

    # Step 3: run the actual standardize.py script
    std_mod = _import_standardize()
    std_mod.standardize(workspace.root)

    # Verify result_data.json was produced
    result_data_path = workspace.std_output_dir / "result_data.json"
    assert result_data_path.exists(), "standardize.py did not produce result_data.json"

    # Step 4: parse output
    output = runner.parse_output(workspace)
    assert isinstance(output, SimulationOutput)
    return output


# ---------------------------------------------------------------------------
# TestOpenMMSimulateE2E
# ---------------------------------------------------------------------------


class TestOpenMMSimulateE2E:
    """End-to-end test for openmm_md_simulate: full lifecycle."""

    def test_simulate_full_pipeline(
        self, config: AutobioConfig, sample_pdb: Path, tmp_path: Path
    ) -> None:
        """Run simulation pipeline and verify trajectory and energy output."""
        input_data = SimulationInput(structure_path=sample_pdb)
        output = _run_e2e(
            config=config,
            input_data=input_data,
            raw_output_files={
                "energy_timeseries.json": _ENERGY_TIMESERIES_JSON,
                "simulation_summary.json": _SIMULATION_SUMMARY_JSON,
                "trajectory.dcd": b"",  # zero-byte stub (path resolution only)
                "final.pdb": _MINIMAL_PDB,
            },
            tmp_path=tmp_path,
        )

        # Trajectory path should exist (copied by standardize.py)
        assert output.trajectory_path is not None
        assert output.trajectory_path.exists()
        assert output.trajectory_path.suffix == ".dcd"

        # Energy timeseries should be parsed
        assert len(output.energy_timeseries) == 3
        first = output.energy_timeseries[0]
        assert first.step == 5000
        assert first.time_ps == pytest.approx(10.0)
        assert first.potential_energy_kj_mol == pytest.approx(-120000.5)
        assert first.temperature_K == pytest.approx(298.5)

        # Summary should be populated
        assert output.summary.n_steps_completed == 50000
        assert output.summary.total_time_ns == pytest.approx(0.1)
        assert output.summary.platform_used == "CPU"
        assert output.summary.force_field == "amber14-all.xml"

    def test_simulate_final_structure(
        self, config: AutobioConfig, sample_pdb: Path, tmp_path: Path
    ) -> None:
        """Final PDB exists and is accessible."""
        input_data = SimulationInput(structure_path=sample_pdb)
        output = _run_e2e(
            config=config,
            input_data=input_data,
            raw_output_files={
                "energy_timeseries.json": _ENERGY_TIMESERIES_JSON,
                "simulation_summary.json": _SIMULATION_SUMMARY_JSON,
                "trajectory.dcd": b"",
                "final.pdb": _MINIMAL_PDB,
            },
            tmp_path=tmp_path,
        )

        assert output.final_structure_path is not None
        assert output.final_structure_path.exists()
        assert output.final_structure_path.suffix == ".pdb"
        content = output.final_structure_path.read_text()
        assert "ATOM" in content

    def test_simulate_energy_timeseries(
        self, config: AutobioConfig, sample_pdb: Path, tmp_path: Path
    ) -> None:
        """Energy timeseries includes all expected fields."""
        input_data = SimulationInput(structure_path=sample_pdb)
        output = _run_e2e(
            config=config,
            input_data=input_data,
            raw_output_files={
                "energy_timeseries.json": _ENERGY_TIMESERIES_JSON,
                "simulation_summary.json": _SIMULATION_SUMMARY_JSON,
                "trajectory.dcd": b"",
                "final.pdb": _MINIMAL_PDB,
            },
            tmp_path=tmp_path,
        )

        # Verify all records are present with expected data
        assert len(output.energy_timeseries) == 3
        for record in output.energy_timeseries:
            assert record.step > 0
            assert record.time_ps > 0
            assert record.potential_energy_kj_mol < 0
            assert record.kinetic_energy_kj_mol > 0
            assert record.temperature_K > 0
            assert record.volume_nm3 is not None
            assert record.density_kg_m3 is not None

        # Verify ordering
        steps = [r.step for r in output.energy_timeseries]
        assert steps == sorted(steps)

    def test_simulate_summary_metadata(
        self, config: AutobioConfig, sample_pdb: Path, tmp_path: Path
    ) -> None:
        """Summary includes simulation metadata and equilibration protocol."""
        input_data = SimulationInput(structure_path=sample_pdb)
        output = _run_e2e(
            config=config,
            input_data=input_data,
            raw_output_files={
                "energy_timeseries.json": _ENERGY_TIMESERIES_JSON,
                "simulation_summary.json": _SIMULATION_SUMMARY_JSON,
                "trajectory.dcd": b"",
                "final.pdb": _MINIMAL_PDB,
            },
            tmp_path=tmp_path,
        )

        summary = output.summary
        assert summary.mean_temperature_K == pytest.approx(299.5)
        assert summary.mean_potential_energy_kj_mol == pytest.approx(-120233.8)
        assert summary.water_model == "tip3p"
        assert summary.box_shape == "cubic"
        assert summary.ion_concentration_M == pytest.approx(0.15)
        assert summary.equilibration_protocol is not None
        assert summary.equilibration_protocol["nvt_steps"] == 50000
        assert summary.equilibration_protocol["npt_steps"] == 100000

    def test_simulate_config_defaults(
        self, config: AutobioConfig, sample_pdb: Path, tmp_path: Path
    ) -> None:
        """Verify default config values written to config.json."""
        runner = _make_runner("md_simulate", config)
        workspace = Workspace.create(tmp_path / "ws")
        input_data = SimulationInput(structure_path=sample_pdb)
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
        assert cfg["timestep"] == pytest.approx(2.0)
        assert cfg["total_time_ns"] == pytest.approx(10.0)
        assert cfg["reporting_interval_steps"] == 5000
        assert cfg["trajectory_format"] == "dcd"
        assert cfg["restraint_set"] == "none"
        assert cfg["minimize_max_iterations"] == 0
        assert cfg["equilibration_nvt_steps"] == 50000
        assert cfg["equilibration_npt_steps"] == 100000
        assert cfg["platform"] == "CUDA"

    def test_simulate_without_final_pdb(
        self, config: AutobioConfig, sample_pdb: Path, tmp_path: Path
    ) -> None:
        """Pipeline works even when final.pdb is absent."""
        input_data = SimulationInput(structure_path=sample_pdb)
        output = _run_e2e(
            config=config,
            input_data=input_data,
            raw_output_files={
                "energy_timeseries.json": _ENERGY_TIMESERIES_JSON,
                "simulation_summary.json": _SIMULATION_SUMMARY_JSON,
                "trajectory.dcd": b"",
            },
            tmp_path=tmp_path,
        )

        assert output.trajectory_path is not None
        assert output.final_structure_path is None
        assert len(output.energy_timeseries) == 3


# ---------------------------------------------------------------------------
# TestOpenMMSimulateInputValidation
# ---------------------------------------------------------------------------


class TestOpenMMSimulateInputValidation:
    """Validation tests for openmm_md_simulate."""

    def test_nonexistent_structure_fails(self, config: AutobioConfig, tmp_path: Path) -> None:
        """Rejects nonexistent input structures."""
        runner = _make_runner("md_simulate", config)
        workspace = Workspace.create(tmp_path / "ws")
        input_data = SimulationInput(structure_path=tmp_path / "nonexistent.pdb")
        with pytest.raises(AutobioError, match="does not exist"):
            runner.prepare_workspace(input_data, workspace)

    def test_structure_copied_correctly(
        self, config: AutobioConfig, sample_pdb: Path, tmp_path: Path
    ) -> None:
        """Input structure is copied to workspace inputs/."""
        runner = _make_runner("md_simulate", config)
        workspace = Workspace.create(tmp_path / "ws")
        input_data = SimulationInput(structure_path=sample_pdb)
        runner.prepare_workspace(input_data, workspace)

        copied = workspace.inputs_dir / sample_pdb.name
        assert copied.exists()
        assert copied.read_text() == sample_pdb.read_text()
