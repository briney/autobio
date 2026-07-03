"""Tests for the migrated openmm Tool (modes: amber_minimize, amber_relax, md_simulate)."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

from autobio.core.catalog import get_tool, list_tools, tool_categories
from autobio.core.config import AutobioConfig
from autobio.core.registry import ToolCategory
from autobio.core.result import AutobioError
from autobio.core.workspace import Workspace
from autobio.schemas.scoring import ScoringInput, ScoringOutput
from autobio.schemas.simulation import (
    EnergyRecord,
    SimulationInput,
    SimulationOutput,
    SimulationSummary,
)
from autobio.tools import TOOL_RUNNERS, get_runner
from autobio.tools.openmm import (
    _ALLOWED_BOX_SHAPES,
    _ALLOWED_FORCE_FIELDS,
    _ALLOWED_RESTRAINT_SETS,
    _ALLOWED_TRAJECTORY_FORMATS,
    _ALLOWED_WATER_MODELS,
    _VARIANT_CONFIG,
    OpenMMRunner,
)

if TYPE_CHECKING:
    from pathlib import Path

_OLD_FLAT_NAMES = ("openmm_amber_minimize", "openmm_amber_relax", "openmm_md_simulate")


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


@pytest.fixture()
def config() -> AutobioConfig:
    return AutobioConfig.resolve()


def _make_runner(mode_name: str, config: AutobioConfig) -> OpenMMRunner:
    """Create an OpenMMRunner with mocked deps, current_mode pinned to *mode_name*."""
    with (
        patch("autobio.tools.base.ContainerManager"),
        patch("autobio.tools.base.GPUManager"),
    ):
        runner = OpenMMRunner("openmm", config)
    runner.current_mode = get_tool("openmm").modes[mode_name]
    return runner


@pytest.fixture()
def runner(config: AutobioConfig) -> OpenMMRunner:
    """Amber-minimize mode runner (the common case)."""
    return _make_runner("amber_minimize", config)


@pytest.fixture()
def relax_runner(config: AutobioConfig) -> OpenMMRunner:
    return _make_runner("amber_relax", config)


@pytest.fixture()
def simulate_runner(config: AutobioConfig) -> OpenMMRunner:
    return _make_runner("md_simulate", config)


@pytest.fixture()
def sample_pdb(tmp_path: Path) -> Path:
    """Write a minimal PDB file for testing."""
    pdb_path = tmp_path / "test.pdb"
    pdb_path.write_text("ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00\nEND\n")
    return pdb_path


# ---------------------------------------------------------------------------
# TestOpenMMPrepareWorkspace (amber_minimize)
# ---------------------------------------------------------------------------

_ALL_MODE_NAMES = list(_VARIANT_CONFIG.keys())


class TestOpenMMPrepareWorkspace:
    """Tests for OpenMMRunner.prepare_workspace (amber_minimize mode)."""

    def test_basic_config(self, runner: OpenMMRunner, tmp_path: Path, sample_pdb: Path) -> None:
        """Config contains correct protocol, force_field, and tolerance."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = ScoringInput(structure_path=sample_pdb)
        runner.prepare_workspace(input_data, workspace)

        cfg = json.loads(workspace.config_path.read_text())
        assert cfg["protocol"] == "amber_minimize"
        assert cfg["force_field"] == "amber14-all.xml"
        assert cfg["tolerance"] == pytest.approx(2.39)
        assert cfg["implicit_solvent"] is True

    def test_structure_file_copied(
        self, runner: OpenMMRunner, tmp_path: Path, sample_pdb: Path
    ) -> None:
        """Input structure is copied to inputs/ and config references container path."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = ScoringInput(structure_path=sample_pdb)
        runner.prepare_workspace(input_data, workspace)

        copied = workspace.inputs_dir / "test.pdb"
        assert copied.exists()
        assert copied.read_text() == sample_pdb.read_text()

        cfg = json.loads(workspace.config_path.read_text())
        assert cfg["structure_path"] == "/workspace/inputs/test.pdb"

    def test_force_field_override(
        self, runner: OpenMMRunner, tmp_path: Path, sample_pdb: Path
    ) -> None:
        """Custom force field from extra dict is used."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = ScoringInput(structure_path=sample_pdb, extra={"force_field": "amber99sb.xml"})
        runner.prepare_workspace(input_data, workspace)

        cfg = json.loads(workspace.config_path.read_text())
        assert cfg["force_field"] == "amber99sb.xml"

    def test_tolerance_override(
        self, runner: OpenMMRunner, tmp_path: Path, sample_pdb: Path
    ) -> None:
        """Custom tolerance from extra dict is used."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = ScoringInput(structure_path=sample_pdb, extra={"tolerance": 10.0})
        runner.prepare_workspace(input_data, workspace)

        cfg = json.loads(workspace.config_path.read_text())
        assert cfg["tolerance"] == pytest.approx(10.0)

    def test_max_iterations_override(
        self, runner: OpenMMRunner, tmp_path: Path, sample_pdb: Path
    ) -> None:
        """Custom max_iterations from extra dict is used."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = ScoringInput(structure_path=sample_pdb, extra={"max_iterations": 500})
        runner.prepare_workspace(input_data, workspace)

        cfg = json.loads(workspace.config_path.read_text())
        assert cfg["max_iterations"] == 500

    def test_restraint_config(self, runner: OpenMMRunner, tmp_path: Path, sample_pdb: Path) -> None:
        """Restraint set and stiffness from extra dict are written to config."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = ScoringInput(
            structure_path=sample_pdb,
            extra={"restraint_set": "ca", "restraint_stiffness": 20.0},
        )
        runner.prepare_workspace(input_data, workspace)

        cfg = json.loads(workspace.config_path.read_text())
        assert cfg["restraint_set"] == "ca"
        assert cfg["restraint_stiffness"] == pytest.approx(20.0)

    def test_implicit_solvent_disabled(
        self, runner: OpenMMRunner, tmp_path: Path, sample_pdb: Path
    ) -> None:
        """Setting implicit_solvent to False enables vacuum mode."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = ScoringInput(structure_path=sample_pdb, extra={"implicit_solvent": False})
        runner.prepare_workspace(input_data, workspace)

        cfg = json.loads(workspace.config_path.read_text())
        assert cfg["implicit_solvent"] is False

    def test_max_outer_iterations_override(
        self, runner: OpenMMRunner, tmp_path: Path, sample_pdb: Path
    ) -> None:
        """Custom max_outer_iterations from extra dict is used."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = ScoringInput(structure_path=sample_pdb, extra={"max_outer_iterations": 5})
        runner.prepare_workspace(input_data, workspace)

        cfg = json.loads(workspace.config_path.read_text())
        assert cfg["max_outer_iterations"] == 5

    def test_defaults(self, runner: OpenMMRunner, tmp_path: Path, sample_pdb: Path) -> None:
        """Default values match variant config."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = ScoringInput(structure_path=sample_pdb)
        runner.prepare_workspace(input_data, workspace)

        cfg = json.loads(workspace.config_path.read_text())
        variant = _VARIANT_CONFIG["amber_minimize"]
        assert cfg["force_field"] == variant["default_force_field"]
        assert cfg["tolerance"] == pytest.approx(variant["default_tolerance"])
        assert cfg["max_iterations"] == variant["default_max_iterations"]
        assert cfg["restraint_set"] == variant["default_restraint_set"]
        assert cfg["restraint_stiffness"] == pytest.approx(variant["default_restraint_stiffness"])
        assert cfg["implicit_solvent"] == variant["default_implicit_solvent"]
        assert cfg["max_outer_iterations"] == variant["default_max_outer_iterations"]

    def test_extra_dict_merged(
        self, runner: OpenMMRunner, tmp_path: Path, sample_pdb: Path
    ) -> None:
        """Extra dict keys (non-consumed) appear at top level of config.json."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = ScoringInput(
            structure_path=sample_pdb,
            extra={"custom_flag": "value", "debug": True},
        )
        runner.prepare_workspace(input_data, workspace)

        cfg = json.loads(workspace.config_path.read_text())
        assert cfg["custom_flag"] == "value"
        assert cfg["debug"] is True

    def test_consumed_keys_not_double_merged(
        self, runner: OpenMMRunner, tmp_path: Path, sample_pdb: Path
    ) -> None:
        """Consumed extra keys are placed explicitly, not leaked via flat merge."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = ScoringInput(
            structure_path=sample_pdb,
            extra={
                "tolerance": 5.0,
                "force_field": "amber99sb.xml",
                "custom_flag": "value",
            },
        )
        runner.prepare_workspace(input_data, workspace)

        cfg = json.loads(workspace.config_path.read_text())
        assert cfg["tolerance"] == pytest.approx(5.0)
        assert cfg["force_field"] == "amber99sb.xml"
        assert cfg["custom_flag"] == "value"


# ---------------------------------------------------------------------------
# TestOpenMMValidation (amber_minimize)
# ---------------------------------------------------------------------------


class TestOpenMMValidation:
    """Tests for host-side input validation (amber_minimize mode)."""

    def test_missing_structure_raises(self, runner: OpenMMRunner, tmp_path: Path) -> None:
        """Missing structure file raises AutobioError."""
        workspace = Workspace.create(tmp_path / "ws")
        fake_path = tmp_path / "nonexistent.pdb"
        input_data = ScoringInput(structure_path=fake_path)
        with pytest.raises(AutobioError, match="does not exist"):
            runner.prepare_workspace(input_data, workspace)

    def test_invalid_force_field_raises(
        self, runner: OpenMMRunner, tmp_path: Path, sample_pdb: Path
    ) -> None:
        """Invalid force_field raises AutobioError."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = ScoringInput(structure_path=sample_pdb, extra={"force_field": "invalid.xml"})
        with pytest.raises(AutobioError, match="Invalid force_field"):
            runner.prepare_workspace(input_data, workspace)

    def test_invalid_restraint_set_raises(
        self, runner: OpenMMRunner, tmp_path: Path, sample_pdb: Path
    ) -> None:
        """Invalid restraint_set raises AutobioError."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = ScoringInput(structure_path=sample_pdb, extra={"restraint_set": "backbone"})
        with pytest.raises(AutobioError, match="Invalid restraint_set"):
            runner.prepare_workspace(input_data, workspace)

    def test_valid_force_fields_accepted(
        self, runner: OpenMMRunner, tmp_path: Path, sample_pdb: Path
    ) -> None:
        """All allowed force field values pass validation."""
        for ff in _ALLOWED_FORCE_FIELDS:
            workspace = Workspace.create(tmp_path / f"ws_{ff}")
            input_data = ScoringInput(structure_path=sample_pdb, extra={"force_field": ff})
            runner.prepare_workspace(input_data, workspace)
            cfg = json.loads(workspace.config_path.read_text())
            assert cfg["force_field"] == ff

    def test_valid_restraint_sets_accepted(
        self, runner: OpenMMRunner, tmp_path: Path, sample_pdb: Path
    ) -> None:
        """All allowed restraint_set values pass validation."""
        for rs in _ALLOWED_RESTRAINT_SETS:
            workspace = Workspace.create(tmp_path / f"ws_{rs}")
            input_data = ScoringInput(structure_path=sample_pdb, extra={"restraint_set": rs})
            runner.prepare_workspace(input_data, workspace)
            cfg = json.loads(workspace.config_path.read_text())
            assert cfg["restraint_set"] == rs


# ---------------------------------------------------------------------------
# TestOpenMMParseOutput (amber_minimize)
# ---------------------------------------------------------------------------

_MINIMIZE_RESULT = {
    "scores": [
        {
            "total_score": -48500.5,
            "score_breakdown": {
                "HarmonicBondForce": -12345.6,
                "HarmonicAngleForce": -5678.9,
                "PeriodicTorsionForce": -2345.6,
                "NonbondedForce": -28130.4,
                "initial_energy": -45000.0,
                "num_minimization_rounds": 3,
                "remaining_violations": 0,
            },
            "units": "kJ/mol",
            "structure_path": "/workspace/outputs/standardized/minimized.pdb",
            "per_residue_scores": None,
            "ddg": None,
            "mutations": None,
        }
    ]
}


class TestOpenMMParseOutput:
    """Tests for OpenMMRunner.parse_output (amber_minimize mode)."""

    def test_parse_minimize_output(self, runner: OpenMMRunner, tmp_path: Path) -> None:
        """Reads result_data.json and returns correct ScoringOutput."""
        workspace = Workspace.create(tmp_path / "ws")
        pdb_dest = workspace.std_output_dir / "minimized.pdb"
        pdb_dest.write_text("ATOM  minimized\nEND\n")
        (workspace.std_output_dir / "result_data.json").write_text(json.dumps(_MINIMIZE_RESULT))

        output = runner.parse_output(workspace)
        assert isinstance(output, ScoringOutput)
        assert len(output.scores) == 1

        s = output.scores[0]
        assert s.total_score == pytest.approx(-48500.5)
        assert s.ddg is None
        assert s.mutations is None

    def test_parse_output_with_structure_path(self, runner: OpenMMRunner, tmp_path: Path) -> None:
        """Container path is resolved to host path."""
        workspace = Workspace.create(tmp_path / "ws")
        pdb_dest = workspace.std_output_dir / "minimized.pdb"
        pdb_dest.write_text("ATOM  minimized\nEND\n")
        (workspace.std_output_dir / "result_data.json").write_text(json.dumps(_MINIMIZE_RESULT))

        output = runner.parse_output(workspace)
        s = output.scores[0]
        assert s.structure_path is not None
        assert s.structure_path.name == "minimized.pdb"

    def test_units_kj_mol(self, runner: OpenMMRunner, tmp_path: Path) -> None:
        """Units field is kJ/mol."""
        workspace = Workspace.create(tmp_path / "ws")
        (workspace.std_output_dir / "result_data.json").write_text(json.dumps(_MINIMIZE_RESULT))

        output = runner.parse_output(workspace)
        assert output.scores[0].units == "kJ/mol"

    def test_score_breakdown(self, runner: OpenMMRunner, tmp_path: Path) -> None:
        """Score breakdown has force-type keys and metadata."""
        workspace = Workspace.create(tmp_path / "ws")
        (workspace.std_output_dir / "result_data.json").write_text(json.dumps(_MINIMIZE_RESULT))

        output = runner.parse_output(workspace)
        breakdown = output.scores[0].score_breakdown
        assert breakdown is not None
        assert "HarmonicBondForce" in breakdown
        assert "NonbondedForce" in breakdown
        assert "initial_energy" in breakdown
        assert "num_minimization_rounds" in breakdown
        assert "remaining_violations" in breakdown
        assert breakdown["num_minimization_rounds"] == 3
        assert breakdown["remaining_violations"] == 0

    def test_output_type(self, runner: OpenMMRunner, tmp_path: Path) -> None:
        workspace = Workspace.create(tmp_path / "ws")
        (workspace.std_output_dir / "result_data.json").write_text(json.dumps(_MINIMIZE_RESULT))
        output = runner.parse_output(workspace)
        assert isinstance(output, ScoringOutput)

    def test_raw_output_path(self, runner: OpenMMRunner, tmp_path: Path) -> None:
        workspace = Workspace.create(tmp_path / "ws")
        (workspace.std_output_dir / "result_data.json").write_text(json.dumps(_MINIMIZE_RESULT))
        output = runner.parse_output(workspace)
        assert output.raw_output_path == workspace.raw_output_dir


# ---------------------------------------------------------------------------
# TestOpenMMByteCompatConfig — full-dict config.json equality, per mode
# ---------------------------------------------------------------------------


class TestOpenMMByteCompatConfig:
    """Full-dict ``config.json`` equality tests, pinning key order per mode.

    Key orders are taken verbatim from ``.superpowers/sdd/recon/openmm.md``.
    """

    def test_amber_minimize_full_config_defaults(
        self, config: AutobioConfig, tmp_path: Path, sample_pdb: Path
    ) -> None:
        r = _make_runner("amber_minimize", config)
        workspace = Workspace.create(tmp_path / "ws")
        r.prepare_workspace(ScoringInput(structure_path=sample_pdb), workspace)

        cfg = json.loads(workspace.config_path.read_text())
        expected = {
            "protocol": "amber_minimize",
            "structure_path": f"/workspace/inputs/{sample_pdb.name}",
            "out_dir": "/workspace/outputs/raw",
            "force_field": "amber14-all.xml",
            "tolerance": 2.39,
            "max_iterations": 0,
            "restraint_set": "none",
            "restraint_stiffness": 10.0,
            "implicit_solvent": True,
            "max_outer_iterations": 20,
        }
        assert cfg == expected
        assert list(cfg.keys()) == list(expected.keys())

    def test_amber_relax_full_config_defaults(
        self, config: AutobioConfig, tmp_path: Path, sample_pdb: Path
    ) -> None:
        r = _make_runner("amber_relax", config)
        workspace = Workspace.create(tmp_path / "ws")
        r.prepare_workspace(ScoringInput(structure_path=sample_pdb), workspace)

        cfg = json.loads(workspace.config_path.read_text())
        expected = {
            "protocol": "amber_relax",
            "structure_path": f"/workspace/inputs/{sample_pdb.name}",
            "out_dir": "/workspace/outputs/raw",
            "force_field": "amber14-all.xml",
            "implicit_solvent": False,
            "water_model": "tip3p",
            "box_shape": "cubic",
            "box_padding": 1.0,
            "ion_type": "NaCl",
            "ion_concentration": 0.15,
            "temperature": 300.0,
            "pressure": 1.0,
            "restraint_set": "heavy_atoms",
            "restraint_stiffness": 10.0,
            "timestep": 2.0,
            "minimize_max_iterations": 0,
            "heating_steps": 25000,
            "nvt_steps": 25000,
            "npt_steps": 50000,
            "production_steps": 25000,
        }
        assert cfg == expected
        assert list(cfg.keys()) == list(expected.keys())

    def test_md_simulate_full_config_defaults(
        self, config: AutobioConfig, tmp_path: Path, sample_pdb: Path
    ) -> None:
        r = _make_runner("md_simulate", config)
        workspace = Workspace.create(tmp_path / "ws")
        r.prepare_workspace(SimulationInput(structure_path=sample_pdb), workspace)

        cfg = json.loads(workspace.config_path.read_text())
        expected = {
            "protocol": "md_simulate",
            "structure_path": f"/workspace/inputs/{sample_pdb.name}",
            "out_dir": "/workspace/outputs/raw",
            "force_field": "amber14-all.xml",
            "implicit_solvent": False,
            "water_model": "tip3p",
            "box_shape": "cubic",
            "box_padding": 1.0,
            "ion_type": "NaCl",
            "ion_concentration": 0.15,
            "temperature": 300.0,
            "pressure": 1.0,
            "timestep": 2.0,
            "total_time_ns": 10.0,
            "reporting_interval_steps": 5000,
            "trajectory_format": "dcd",
            "restraint_set": "none",
            "restraint_stiffness": 10.0,
            "minimize_max_iterations": 0,
            "equilibration_nvt_steps": 50000,
            "equilibration_npt_steps": 100000,
            "platform": "CUDA",
        }
        assert cfg == expected
        assert list(cfg.keys()) == list(expected.keys())

    def test_amber_relax_full_config_with_consumed_overrides(
        self, config: AutobioConfig, tmp_path: Path, sample_pdb: Path
    ) -> None:
        """Consumed extra keys (force_field, temperature) override in place — no duplicate key."""
        r = _make_runner("amber_relax", config)
        workspace = Workspace.create(tmp_path / "ws")
        r.prepare_workspace(
            ScoringInput(
                structure_path=sample_pdb,
                extra={"force_field": "charmm36.xml", "temperature": 310.0},
            ),
            workspace,
        )

        cfg = json.loads(workspace.config_path.read_text())
        expected = {
            "protocol": "amber_relax",
            "structure_path": f"/workspace/inputs/{sample_pdb.name}",
            "out_dir": "/workspace/outputs/raw",
            "force_field": "charmm36.xml",
            "implicit_solvent": False,
            "water_model": "tip3p",
            "box_shape": "cubic",
            "box_padding": 1.0,
            "ion_type": "NaCl",
            "ion_concentration": 0.15,
            "temperature": 310.0,
            "pressure": 1.0,
            "restraint_set": "heavy_atoms",
            "restraint_stiffness": 10.0,
            "timestep": 2.0,
            "minimize_max_iterations": 0,
            "heating_steps": 25000,
            "nvt_steps": 25000,
            "npt_steps": 50000,
            "production_steps": 25000,
        }
        assert cfg == expected
        assert list(cfg.keys()) == list(expected.keys())

    def test_amber_minimize_full_config_with_flat_extra(
        self, config: AutobioConfig, tmp_path: Path, sample_pdb: Path
    ) -> None:
        """A non-consumed extra key flat-merges after the fixed keys."""
        r = _make_runner("amber_minimize", config)
        workspace = Workspace.create(tmp_path / "ws")
        r.prepare_workspace(
            ScoringInput(structure_path=sample_pdb, extra={"custom_flag": "value"}),
            workspace,
        )

        cfg = json.loads(workspace.config_path.read_text())
        expected = {
            "protocol": "amber_minimize",
            "structure_path": f"/workspace/inputs/{sample_pdb.name}",
            "out_dir": "/workspace/outputs/raw",
            "force_field": "amber14-all.xml",
            "tolerance": 2.39,
            "max_iterations": 0,
            "restraint_set": "none",
            "restraint_stiffness": 10.0,
            "implicit_solvent": True,
            "max_outer_iterations": 20,
            "custom_flag": "value",
        }
        assert cfg == expected
        assert list(cfg.keys()) == list(expected.keys())

    def test_md_simulate_full_config_with_n_steps_and_flat_extra(
        self, config: AutobioConfig, tmp_path: Path, sample_pdb: Path
    ) -> None:
        """n_steps lands after the default loop and before flat-merged extra keys."""
        r = _make_runner("md_simulate", config)
        workspace = Workspace.create(tmp_path / "ws")
        r.prepare_workspace(
            SimulationInput(
                structure_path=sample_pdb,
                extra={"n_steps": 123456, "custom_flag": "value"},
            ),
            workspace,
        )

        cfg = json.loads(workspace.config_path.read_text())
        expected = {
            "protocol": "md_simulate",
            "structure_path": f"/workspace/inputs/{sample_pdb.name}",
            "out_dir": "/workspace/outputs/raw",
            "force_field": "amber14-all.xml",
            "implicit_solvent": False,
            "water_model": "tip3p",
            "box_shape": "cubic",
            "box_padding": 1.0,
            "ion_type": "NaCl",
            "ion_concentration": 0.15,
            "temperature": 300.0,
            "pressure": 1.0,
            "timestep": 2.0,
            "total_time_ns": 10.0,
            "reporting_interval_steps": 5000,
            "trajectory_format": "dcd",
            "restraint_set": "none",
            "restraint_stiffness": 10.0,
            "minimize_max_iterations": 0,
            "equilibration_nvt_steps": 50000,
            "equilibration_npt_steps": 100000,
            "platform": "CUDA",
            "n_steps": 123456,
            "custom_flag": "value",
        }
        assert cfg == expected
        assert list(cfg.keys()) == list(expected.keys())


# ---------------------------------------------------------------------------
# TestOpenMMRegistration
# ---------------------------------------------------------------------------


class TestOpenMMRegistration:
    """Tests for the catalog Tool + runner registration."""

    def test_openmm_registered_as_single_tool(self) -> None:
        tool = get_tool("openmm")
        assert sorted(tool.modes) == ["amber_minimize", "amber_relax", "md_simulate"]
        assert tool.default_mode == "amber_minimize"
        assert tool.category == ToolCategory.SCORING
        assert tool.requires_gpu is True
        assert tool.gpu_count == 1
        assert tool.image_tag == "openmm-amber-minimize:1.1.0"

    @pytest.mark.parametrize("flat_name", _OLD_FLAT_NAMES)
    def test_old_flat_names_absent_from_tool_runners(self, flat_name: str) -> None:
        assert flat_name not in TOOL_RUNNERS

    def test_openmm_in_tool_runners(self) -> None:
        assert "openmm" in TOOL_RUNNERS
        assert TOOL_RUNNERS["openmm"] is OpenMMRunner

    @pytest.mark.parametrize(
        ("mode_name", "timeout"),
        [("amber_minimize", 600), ("amber_relax", 3600), ("md_simulate", 86400)],
    )
    def test_modes_have_per_mode_timeout(self, mode_name: str, timeout: int) -> None:
        assert get_tool("openmm").modes[mode_name].default_timeout == timeout

    @pytest.mark.parametrize(
        ("mode_name", "image_tag"),
        [
            ("amber_minimize", "openmm-amber-minimize:1.1.0"),
            ("amber_relax", "openmm-amber-relax:1.1.0"),
            ("md_simulate", "openmm-md-simulate:1.1.0"),
        ],
    )
    def test_modes_have_per_mode_image_tag(self, mode_name: str, image_tag: str) -> None:
        assert get_tool("openmm").modes[mode_name].image_tag == image_tag

    def test_mode_schemas(self) -> None:
        tool = get_tool("openmm")
        assert tool.modes["amber_minimize"].input_schema is ScoringInput
        assert tool.modes["amber_minimize"].output_schema is ScoringOutput
        assert tool.modes["amber_relax"].input_schema is ScoringInput
        assert tool.modes["amber_relax"].output_schema is ScoringOutput
        assert tool.modes["md_simulate"].input_schema is SimulationInput
        assert tool.modes["md_simulate"].output_schema is SimulationOutput

    def test_get_runner_returns_openmm_runner(self, config: AutobioConfig) -> None:
        with (
            patch("autobio.tools.base.ContainerManager"),
            patch("autobio.tools.base.GPUManager"),
        ):
            r = get_runner("openmm", config)
        assert isinstance(r, OpenMMRunner)
        assert r.tool_name == "openmm"

    @pytest.mark.parametrize("flat_name", _OLD_FLAT_NAMES)
    def test_get_runner_removed_flat_name_raises(
        self, flat_name: str, config: AutobioConfig
    ) -> None:
        with pytest.raises(KeyError, match=flat_name):
            get_runner(flat_name, config)


# ---------------------------------------------------------------------------
# TestOpenMMRelaxPrepareWorkspace
# ---------------------------------------------------------------------------


class TestOpenMMRelaxPrepareWorkspace:
    """Tests for OpenMMRunner.prepare_workspace (amber_relax mode)."""

    def test_basic_config(
        self, relax_runner: OpenMMRunner, tmp_path: Path, sample_pdb: Path
    ) -> None:
        """Config contains correct protocol, force_field, water_model, box_shape."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = ScoringInput(structure_path=sample_pdb)
        relax_runner.prepare_workspace(input_data, workspace)

        cfg = json.loads(workspace.config_path.read_text())
        assert cfg["protocol"] == "amber_relax"
        assert cfg["force_field"] == "amber14-all.xml"
        assert cfg["water_model"] == "tip3p"
        assert cfg["box_shape"] == "cubic"
        assert cfg["temperature"] == pytest.approx(300.0)
        assert cfg["pressure"] == pytest.approx(1.0)
        assert cfg["timestep"] == pytest.approx(2.0)

    def test_structure_file_copied(
        self, relax_runner: OpenMMRunner, tmp_path: Path, sample_pdb: Path
    ) -> None:
        """Input structure is copied to inputs/ and config references container path."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = ScoringInput(structure_path=sample_pdb)
        relax_runner.prepare_workspace(input_data, workspace)

        copied = workspace.inputs_dir / "test.pdb"
        assert copied.exists()
        assert copied.read_text() == sample_pdb.read_text()

        cfg = json.loads(workspace.config_path.read_text())
        assert cfg["structure_path"] == "/workspace/inputs/test.pdb"

    def test_explicit_solvent_defaults(
        self, relax_runner: OpenMMRunner, tmp_path: Path, sample_pdb: Path
    ) -> None:
        """Verify explicit solvent defaults: implicit_solvent=False, tip3p, box_padding=1.0."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = ScoringInput(structure_path=sample_pdb)
        relax_runner.prepare_workspace(input_data, workspace)

        cfg = json.loads(workspace.config_path.read_text())
        assert cfg["implicit_solvent"] is False
        assert cfg["water_model"] == "tip3p"
        assert cfg["box_padding"] == pytest.approx(1.0)
        assert cfg["ion_type"] == "NaCl"
        assert cfg["ion_concentration"] == pytest.approx(0.15)

    def test_implicit_solvent_override(
        self, relax_runner: OpenMMRunner, tmp_path: Path, sample_pdb: Path
    ) -> None:
        """Setting implicit_solvent=True overrides explicit solvent default."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = ScoringInput(structure_path=sample_pdb, extra={"implicit_solvent": True})
        relax_runner.prepare_workspace(input_data, workspace)

        cfg = json.loads(workspace.config_path.read_text())
        assert cfg["implicit_solvent"] is True

    def test_water_model_override(
        self, relax_runner: OpenMMRunner, tmp_path: Path, sample_pdb: Path
    ) -> None:
        """Custom water_model from extra dict is used."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = ScoringInput(structure_path=sample_pdb, extra={"water_model": "tip4pew"})
        relax_runner.prepare_workspace(input_data, workspace)

        cfg = json.loads(workspace.config_path.read_text())
        assert cfg["water_model"] == "tip4pew"

    def test_solvation_overrides(
        self, relax_runner: OpenMMRunner, tmp_path: Path, sample_pdb: Path
    ) -> None:
        """Solvation parameters from extra dict override defaults."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = ScoringInput(
            structure_path=sample_pdb,
            extra={
                "box_shape": "dodecahedron",
                "box_padding": 1.5,
                "ion_type": "KCl",
                "ion_concentration": 0.20,
            },
        )
        relax_runner.prepare_workspace(input_data, workspace)

        cfg = json.loads(workspace.config_path.read_text())
        assert cfg["box_shape"] == "dodecahedron"
        assert cfg["box_padding"] == pytest.approx(1.5)
        assert cfg["ion_type"] == "KCl"
        assert cfg["ion_concentration"] == pytest.approx(0.20)

    def test_temperature_override(
        self, relax_runner: OpenMMRunner, tmp_path: Path, sample_pdb: Path
    ) -> None:
        """Custom temperature from extra dict is used."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = ScoringInput(structure_path=sample_pdb, extra={"temperature": 310.0})
        relax_runner.prepare_workspace(input_data, workspace)

        cfg = json.loads(workspace.config_path.read_text())
        assert cfg["temperature"] == pytest.approx(310.0)

    def test_equilibration_step_overrides(
        self, relax_runner: OpenMMRunner, tmp_path: Path, sample_pdb: Path
    ) -> None:
        """Custom heating_steps, nvt_steps, npt_steps, production_steps override defaults."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = ScoringInput(
            structure_path=sample_pdb,
            extra={
                "heating_steps": 50000,
                "nvt_steps": 50000,
                "npt_steps": 100000,
                "production_steps": 50000,
            },
        )
        relax_runner.prepare_workspace(input_data, workspace)

        cfg = json.loads(workspace.config_path.read_text())
        assert cfg["heating_steps"] == 50000
        assert cfg["nvt_steps"] == 50000
        assert cfg["npt_steps"] == 100000
        assert cfg["production_steps"] == 50000

    def test_extra_dict_merged(
        self, relax_runner: OpenMMRunner, tmp_path: Path, sample_pdb: Path
    ) -> None:
        """Non-consumed extra keys appear at top level of config.json."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = ScoringInput(
            structure_path=sample_pdb,
            extra={"custom_flag": "value", "debug": True},
        )
        relax_runner.prepare_workspace(input_data, workspace)

        cfg = json.loads(workspace.config_path.read_text())
        assert cfg["custom_flag"] == "value"
        assert cfg["debug"] is True


# ---------------------------------------------------------------------------
# TestOpenMMRelaxValidation
# ---------------------------------------------------------------------------


class TestOpenMMRelaxValidation:
    """Tests for host-side input validation (amber_relax mode)."""

    def test_invalid_water_model_raises(
        self, relax_runner: OpenMMRunner, tmp_path: Path, sample_pdb: Path
    ) -> None:
        """Invalid water_model raises AutobioError."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = ScoringInput(structure_path=sample_pdb, extra={"water_model": "tip5p"})
        with pytest.raises(AutobioError, match="Invalid water_model"):
            relax_runner.prepare_workspace(input_data, workspace)

    def test_invalid_box_shape_raises(
        self, relax_runner: OpenMMRunner, tmp_path: Path, sample_pdb: Path
    ) -> None:
        """Invalid box_shape raises AutobioError."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = ScoringInput(structure_path=sample_pdb, extra={"box_shape": "sphere"})
        with pytest.raises(AutobioError, match="Invalid box_shape"):
            relax_runner.prepare_workspace(input_data, workspace)

    def test_invalid_ion_type_raises(
        self, relax_runner: OpenMMRunner, tmp_path: Path, sample_pdb: Path
    ) -> None:
        """Invalid ion_type raises AutobioError."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = ScoringInput(structure_path=sample_pdb, extra={"ion_type": "MgCl2"})
        with pytest.raises(AutobioError, match="Invalid ion_type"):
            relax_runner.prepare_workspace(input_data, workspace)

    def test_negative_temperature_raises(
        self, relax_runner: OpenMMRunner, tmp_path: Path, sample_pdb: Path
    ) -> None:
        """Negative temperature raises AutobioError."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = ScoringInput(structure_path=sample_pdb, extra={"temperature": -10.0})
        with pytest.raises(AutobioError, match="temperature must be positive"):
            relax_runner.prepare_workspace(input_data, workspace)

    def test_negative_pressure_raises(
        self, relax_runner: OpenMMRunner, tmp_path: Path, sample_pdb: Path
    ) -> None:
        """Negative pressure raises AutobioError."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = ScoringInput(structure_path=sample_pdb, extra={"pressure": -1.0})
        with pytest.raises(AutobioError, match="pressure must be positive"):
            relax_runner.prepare_workspace(input_data, workspace)

    def test_negative_box_padding_raises(
        self, relax_runner: OpenMMRunner, tmp_path: Path, sample_pdb: Path
    ) -> None:
        """Negative box_padding raises AutobioError."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = ScoringInput(structure_path=sample_pdb, extra={"box_padding": -0.5})
        with pytest.raises(AutobioError, match="box_padding must be positive"):
            relax_runner.prepare_workspace(input_data, workspace)

    def test_negative_ion_concentration_raises(
        self, relax_runner: OpenMMRunner, tmp_path: Path, sample_pdb: Path
    ) -> None:
        """Negative ion_concentration raises AutobioError."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = ScoringInput(structure_path=sample_pdb, extra={"ion_concentration": -0.1})
        with pytest.raises(AutobioError, match="ion_concentration must be non-negative"):
            relax_runner.prepare_workspace(input_data, workspace)

    def test_invalid_timestep_raises(
        self, relax_runner: OpenMMRunner, tmp_path: Path, sample_pdb: Path
    ) -> None:
        """Timestep outside 0.5-4.0 fs range raises AutobioError."""
        workspace = Workspace.create(tmp_path / "ws")
        # Too low
        input_data = ScoringInput(structure_path=sample_pdb, extra={"timestep": 0.1})
        with pytest.raises(AutobioError, match="timestep must be between"):
            relax_runner.prepare_workspace(input_data, workspace)

        # Too high
        workspace2 = Workspace.create(tmp_path / "ws2")
        input_data2 = ScoringInput(structure_path=sample_pdb, extra={"timestep": 5.0})
        with pytest.raises(AutobioError, match="timestep must be between"):
            relax_runner.prepare_workspace(input_data2, workspace2)

    @pytest.mark.parametrize("water_model", sorted(_ALLOWED_WATER_MODELS))
    def test_valid_water_models_accepted(
        self,
        relax_runner: OpenMMRunner,
        tmp_path: Path,
        sample_pdb: Path,
        water_model: str,
    ) -> None:
        """All allowed water model values pass validation."""
        workspace = Workspace.create(tmp_path / f"ws_{water_model}")
        input_data = ScoringInput(structure_path=sample_pdb, extra={"water_model": water_model})
        relax_runner.prepare_workspace(input_data, workspace)
        cfg = json.loads(workspace.config_path.read_text())
        assert cfg["water_model"] == water_model

    @pytest.mark.parametrize("box_shape", sorted(_ALLOWED_BOX_SHAPES))
    def test_valid_box_shapes_accepted(
        self,
        relax_runner: OpenMMRunner,
        tmp_path: Path,
        sample_pdb: Path,
        box_shape: str,
    ) -> None:
        """All allowed box shape values pass validation."""
        workspace = Workspace.create(tmp_path / f"ws_{box_shape}")
        input_data = ScoringInput(structure_path=sample_pdb, extra={"box_shape": box_shape})
        relax_runner.prepare_workspace(input_data, workspace)
        cfg = json.loads(workspace.config_path.read_text())
        assert cfg["box_shape"] == box_shape


# ---------------------------------------------------------------------------
# TestOpenMMSimulatePrepareWorkspace
# ---------------------------------------------------------------------------


class TestOpenMMSimulatePrepareWorkspace:
    """Tests for OpenMMRunner.prepare_workspace (md_simulate mode)."""

    def test_basic_config(
        self, simulate_runner: OpenMMRunner, tmp_path: Path, sample_pdb: Path
    ) -> None:
        """Config contains correct protocol, trajectory_format, total_time_ns."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = SimulationInput(structure_path=sample_pdb)
        simulate_runner.prepare_workspace(input_data, workspace)

        cfg = json.loads(workspace.config_path.read_text())
        assert cfg["protocol"] == "md_simulate"
        assert cfg["trajectory_format"] == "dcd"
        assert cfg["total_time_ns"] == pytest.approx(10.0)
        assert cfg["force_field"] == "amber14-all.xml"
        assert cfg["temperature"] == pytest.approx(300.0)

    def test_structure_file_copied(
        self, simulate_runner: OpenMMRunner, tmp_path: Path, sample_pdb: Path
    ) -> None:
        """Input structure is copied to inputs/ and config references container path."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = SimulationInput(structure_path=sample_pdb)
        simulate_runner.prepare_workspace(input_data, workspace)

        copied = workspace.inputs_dir / "test.pdb"
        assert copied.exists()
        assert copied.read_text() == sample_pdb.read_text()

        cfg = json.loads(workspace.config_path.read_text())
        assert cfg["structure_path"] == "/workspace/inputs/test.pdb"

    def test_total_time_ns_override(
        self, simulate_runner: OpenMMRunner, tmp_path: Path, sample_pdb: Path
    ) -> None:
        """Custom total_time_ns from extra dict is used."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = SimulationInput(structure_path=sample_pdb, extra={"total_time_ns": 50.0})
        simulate_runner.prepare_workspace(input_data, workspace)

        cfg = json.loads(workspace.config_path.read_text())
        assert cfg["total_time_ns"] == pytest.approx(50.0)

    def test_n_steps_override(
        self, simulate_runner: OpenMMRunner, tmp_path: Path, sample_pdb: Path
    ) -> None:
        """Custom n_steps from extra dict is written to config."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = SimulationInput(structure_path=sample_pdb, extra={"n_steps": 100000})
        simulate_runner.prepare_workspace(input_data, workspace)

        cfg = json.loads(workspace.config_path.read_text())
        assert cfg["n_steps"] == 100000

    def test_reporting_interval_override(
        self, simulate_runner: OpenMMRunner, tmp_path: Path, sample_pdb: Path
    ) -> None:
        """Custom reporting_interval_steps from extra dict is used."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = SimulationInput(
            structure_path=sample_pdb, extra={"reporting_interval_steps": 10000}
        )
        simulate_runner.prepare_workspace(input_data, workspace)

        cfg = json.loads(workspace.config_path.read_text())
        assert cfg["reporting_interval_steps"] == 10000

    def test_trajectory_format_override(
        self, simulate_runner: OpenMMRunner, tmp_path: Path, sample_pdb: Path
    ) -> None:
        """Custom trajectory_format from extra dict is used."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = SimulationInput(structure_path=sample_pdb, extra={"trajectory_format": "pdb"})
        simulate_runner.prepare_workspace(input_data, workspace)

        cfg = json.loads(workspace.config_path.read_text())
        assert cfg["trajectory_format"] == "pdb"

    def test_solvation_defaults(
        self, simulate_runner: OpenMMRunner, tmp_path: Path, sample_pdb: Path
    ) -> None:
        """Default solvation parameters match variant config."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = SimulationInput(structure_path=sample_pdb)
        simulate_runner.prepare_workspace(input_data, workspace)

        cfg = json.loads(workspace.config_path.read_text())
        assert cfg["implicit_solvent"] is False
        assert cfg["water_model"] == "tip3p"
        assert cfg["box_shape"] == "cubic"
        assert cfg["box_padding"] == pytest.approx(1.0)
        assert cfg["ion_type"] == "NaCl"
        assert cfg["ion_concentration"] == pytest.approx(0.15)

    def test_extra_dict_merged(
        self, simulate_runner: OpenMMRunner, tmp_path: Path, sample_pdb: Path
    ) -> None:
        """Non-consumed extra keys appear at top level of config.json."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = SimulationInput(
            structure_path=sample_pdb,
            extra={"custom_flag": "value", "debug": True},
        )
        simulate_runner.prepare_workspace(input_data, workspace)

        cfg = json.loads(workspace.config_path.read_text())
        assert cfg["custom_flag"] == "value"
        assert cfg["debug"] is True


# ---------------------------------------------------------------------------
# TestOpenMMSimulateValidation
# ---------------------------------------------------------------------------


class TestOpenMMSimulateValidation:
    """Tests for host-side input validation (md_simulate mode)."""

    def test_negative_total_time_raises(
        self, simulate_runner: OpenMMRunner, tmp_path: Path, sample_pdb: Path
    ) -> None:
        """Negative total_time_ns raises AutobioError."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = SimulationInput(structure_path=sample_pdb, extra={"total_time_ns": -5.0})
        with pytest.raises(AutobioError, match="total_time_ns must be positive"):
            simulate_runner.prepare_workspace(input_data, workspace)

    def test_negative_n_steps_raises(
        self, simulate_runner: OpenMMRunner, tmp_path: Path, sample_pdb: Path
    ) -> None:
        """Negative n_steps raises AutobioError."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = SimulationInput(structure_path=sample_pdb, extra={"n_steps": -100})
        with pytest.raises(AutobioError, match="n_steps must be positive"):
            simulate_runner.prepare_workspace(input_data, workspace)

    def test_negative_reporting_interval_raises(
        self, simulate_runner: OpenMMRunner, tmp_path: Path, sample_pdb: Path
    ) -> None:
        """Negative reporting_interval_steps raises AutobioError."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = SimulationInput(
            structure_path=sample_pdb, extra={"reporting_interval_steps": -1}
        )
        with pytest.raises(AutobioError, match="reporting_interval_steps must be positive"):
            simulate_runner.prepare_workspace(input_data, workspace)

    def test_invalid_trajectory_format_raises(
        self, simulate_runner: OpenMMRunner, tmp_path: Path, sample_pdb: Path
    ) -> None:
        """Invalid trajectory_format raises AutobioError."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = SimulationInput(structure_path=sample_pdb, extra={"trajectory_format": "trr"})
        with pytest.raises(AutobioError, match="Invalid trajectory_format"):
            simulate_runner.prepare_workspace(input_data, workspace)

    def test_invalid_water_model_raises(
        self, simulate_runner: OpenMMRunner, tmp_path: Path, sample_pdb: Path
    ) -> None:
        """Invalid water_model raises AutobioError (shared validation)."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = SimulationInput(structure_path=sample_pdb, extra={"water_model": "tip5p"})
        with pytest.raises(AutobioError, match="Invalid water_model"):
            simulate_runner.prepare_workspace(input_data, workspace)

    @pytest.mark.parametrize("traj_format", sorted(_ALLOWED_TRAJECTORY_FORMATS))
    def test_valid_trajectory_formats_accepted(
        self,
        simulate_runner: OpenMMRunner,
        tmp_path: Path,
        sample_pdb: Path,
        traj_format: str,
    ) -> None:
        """All allowed trajectory format values pass validation."""
        workspace = Workspace.create(tmp_path / f"ws_{traj_format}")
        input_data = SimulationInput(
            structure_path=sample_pdb, extra={"trajectory_format": traj_format}
        )
        simulate_runner.prepare_workspace(input_data, workspace)
        cfg = json.loads(workspace.config_path.read_text())
        assert cfg["trajectory_format"] == traj_format


# ---------------------------------------------------------------------------
# TestOpenMMSimulateParseOutput
# ---------------------------------------------------------------------------

_SIMULATE_RESULT = {
    "trajectory_path": "/workspace/outputs/standardized/trajectory.dcd",
    "final_structure_path": "/workspace/outputs/standardized/final.pdb",
    "energy_timeseries": [
        {
            "step": 0,
            "time_ps": 0.0,
            "potential_energy_kj_mol": -48000.0,
            "kinetic_energy_kj_mol": 12000.0,
            "total_energy_kj_mol": -36000.0,
            "temperature_K": 300.1,
            "volume_nm3": 125.5,
            "density_kg_m3": 997.0,
        },
        {
            "step": 5000,
            "time_ps": 10.0,
            "potential_energy_kj_mol": -48200.0,
            "kinetic_energy_kj_mol": 12100.0,
            "total_energy_kj_mol": -36100.0,
            "temperature_K": 300.5,
            "volume_nm3": 125.3,
            "density_kg_m3": 997.2,
        },
    ],
    "summary": {
        "n_steps_completed": 5000000,
        "total_time_ns": 10.0,
        "initial_potential_energy_kj_mol": -48000.0,
        "final_potential_energy_kj_mol": -49200.0,
        "mean_temperature_K": 300.1,
        "mean_potential_energy_kj_mol": -48700.0,
        "platform_used": "CUDA",
        "force_field": "amber14-all.xml",
        "water_model": "tip3p",
        "box_shape": "cubic",
        "ion_concentration_M": 0.15,
        "equilibration_protocol": {"nvt_steps": 50000, "npt_steps": 100000},
    },
}


class TestOpenMMSimulateParseOutput:
    """Tests for OpenMMRunner.parse_output (md_simulate mode)."""

    def test_parse_simulation_output(self, simulate_runner: OpenMMRunner, tmp_path: Path) -> None:
        """Reads result_data.json and returns correct SimulationOutput."""
        workspace = Workspace.create(tmp_path / "ws")
        (workspace.std_output_dir / "trajectory.dcd").write_bytes(b"\x00")
        (workspace.std_output_dir / "final.pdb").write_text("ATOM  final\nEND\n")
        (workspace.std_output_dir / "result_data.json").write_text(json.dumps(_SIMULATE_RESULT))

        output = simulate_runner.parse_output(workspace)
        assert isinstance(output, SimulationOutput)

    def test_trajectory_path_resolved(self, simulate_runner: OpenMMRunner, tmp_path: Path) -> None:
        """Container trajectory path is resolved to host path."""
        workspace = Workspace.create(tmp_path / "ws")
        (workspace.std_output_dir / "trajectory.dcd").write_bytes(b"\x00")
        (workspace.std_output_dir / "final.pdb").write_text("ATOM  final\nEND\n")
        (workspace.std_output_dir / "result_data.json").write_text(json.dumps(_SIMULATE_RESULT))

        output = simulate_runner.parse_output(workspace)
        assert output.trajectory_path.name == "trajectory.dcd"

    def test_final_structure_path_resolved(
        self, simulate_runner: OpenMMRunner, tmp_path: Path
    ) -> None:
        """Container final_structure_path is resolved to host path."""
        workspace = Workspace.create(tmp_path / "ws")
        (workspace.std_output_dir / "trajectory.dcd").write_bytes(b"\x00")
        (workspace.std_output_dir / "final.pdb").write_text("ATOM  final\nEND\n")
        (workspace.std_output_dir / "result_data.json").write_text(json.dumps(_SIMULATE_RESULT))

        output = simulate_runner.parse_output(workspace)
        assert output.final_structure_path is not None
        assert output.final_structure_path.name == "final.pdb"

    def test_energy_timeseries(self, simulate_runner: OpenMMRunner, tmp_path: Path) -> None:
        """Energy timeseries is a list of EnergyRecord with correct values."""
        workspace = Workspace.create(tmp_path / "ws")
        (workspace.std_output_dir / "trajectory.dcd").write_bytes(b"\x00")
        (workspace.std_output_dir / "final.pdb").write_text("ATOM  final\nEND\n")
        (workspace.std_output_dir / "result_data.json").write_text(json.dumps(_SIMULATE_RESULT))

        output = simulate_runner.parse_output(workspace)
        assert len(output.energy_timeseries) == 2

        rec0 = output.energy_timeseries[0]
        assert isinstance(rec0, EnergyRecord)
        assert rec0.step == 0
        assert rec0.time_ps == pytest.approx(0.0)
        assert rec0.potential_energy_kj_mol == pytest.approx(-48000.0)
        assert rec0.kinetic_energy_kj_mol == pytest.approx(12000.0)
        assert rec0.total_energy_kj_mol == pytest.approx(-36000.0)
        assert rec0.temperature_K == pytest.approx(300.1)
        assert rec0.volume_nm3 == pytest.approx(125.5)
        assert rec0.density_kg_m3 == pytest.approx(997.0)

        rec1 = output.energy_timeseries[1]
        assert rec1.step == 5000
        assert rec1.time_ps == pytest.approx(10.0)
        assert rec1.potential_energy_kj_mol == pytest.approx(-48200.0)

    def test_summary_fields(self, simulate_runner: OpenMMRunner, tmp_path: Path) -> None:
        """Summary fields are parsed correctly."""
        workspace = Workspace.create(tmp_path / "ws")
        (workspace.std_output_dir / "trajectory.dcd").write_bytes(b"\x00")
        (workspace.std_output_dir / "final.pdb").write_text("ATOM  final\nEND\n")
        (workspace.std_output_dir / "result_data.json").write_text(json.dumps(_SIMULATE_RESULT))

        output = simulate_runner.parse_output(workspace)
        summary = output.summary
        assert isinstance(summary, SimulationSummary)
        assert summary.n_steps_completed == 5000000
        assert summary.total_time_ns == pytest.approx(10.0)
        assert summary.initial_potential_energy_kj_mol == pytest.approx(-48000.0)
        assert summary.final_potential_energy_kj_mol == pytest.approx(-49200.0)
        assert summary.mean_temperature_K == pytest.approx(300.1)
        assert summary.mean_potential_energy_kj_mol == pytest.approx(-48700.0)
        assert summary.platform_used == "CUDA"
        assert summary.force_field == "amber14-all.xml"
        assert summary.water_model == "tip3p"
        assert summary.box_shape == "cubic"
        assert summary.ion_concentration_M == pytest.approx(0.15)
        assert summary.equilibration_protocol == {"nvt_steps": 50000, "npt_steps": 100000}

    def test_raw_output_path(self, simulate_runner: OpenMMRunner, tmp_path: Path) -> None:
        """raw_output_path is set to the workspace raw output directory."""
        workspace = Workspace.create(tmp_path / "ws")
        (workspace.std_output_dir / "trajectory.dcd").write_bytes(b"\x00")
        (workspace.std_output_dir / "final.pdb").write_text("ATOM  final\nEND\n")
        (workspace.std_output_dir / "result_data.json").write_text(json.dumps(_SIMULATE_RESULT))

        output = simulate_runner.parse_output(workspace)
        assert output.raw_output_path == workspace.raw_output_dir


# ---------------------------------------------------------------------------
# TestOpenMMCrossCategory — modes span SCORING and SIMULATION
# ---------------------------------------------------------------------------


class TestOpenMMCrossCategory:
    """openmm's modes span SCORING (minimize, relax) and SIMULATION (md_simulate)."""

    def test_tool_categories_union(self) -> None:
        tool = get_tool("openmm")
        assert tool_categories(tool) == (ToolCategory.SCORING, ToolCategory.SIMULATION)

    def test_listed_under_scoring(self) -> None:
        assert "openmm" in list_tools(category=ToolCategory.SCORING)

    def test_listed_under_simulation(self) -> None:
        assert "openmm" in list_tools(category=ToolCategory.SIMULATION)


# ---------------------------------------------------------------------------
# TestOpenMMInfoSnapshot
# ---------------------------------------------------------------------------


class TestOpenMMInfoSnapshot:
    """``autobio info openmm`` output — per-mode notes, output_schema, category."""

    def test_info_snapshot(self) -> None:
        from autobio.cli.formatters import OutputFormat, format_tool_info_catalog

        parsed = json.loads(format_tool_info_catalog(get_tool("openmm"), OutputFormat.JSON))
        assert [m["name"] for m in parsed["modes"]] == [
            "amber_minimize",
            "amber_relax",
            "md_simulate",
        ]

        minimize_mode, relax_mode, simulate_mode = parsed["modes"]

        assert len(minimize_mode["notes"]) > 0
        assert "output_schema" in minimize_mode
        assert minimize_mode["category"] == "scoring"

        assert len(relax_mode["notes"]) > 0
        assert "output_schema" in relax_mode
        assert relax_mode["category"] == "scoring"

        assert len(simulate_mode["notes"]) > 0
        assert "output_schema" in simulate_mode
        assert simulate_mode["category"] == "simulation"


# ---------------------------------------------------------------------------
# TestOpenMMRunMetadataMode — full run() lifecycle threads mode into metadata
# ---------------------------------------------------------------------------


class TestOpenMMRunMetadataMode:
    """``run(...).metadata.mode`` reflects the selected mode for each mode."""

    def test_run_metadata_mode_amber_minimize(
        self,
        config: AutobioConfig,
        tmp_path: Path,
        sample_pdb: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        output_dir = tmp_path / "ws"
        std_dir = output_dir / "outputs" / "standardized"
        std_dir.mkdir(parents=True)
        (std_dir / "result_data.json").write_text(json.dumps(_MINIMIZE_RESULT))

        monkeypatch.setattr(
            "autobio.core.workspace.Workspace.read_result",
            lambda self: SimpleNamespace(
                status="success", phase="run", exit_code=0, error_message=None
            ),
        )

        with (
            patch("autobio.tools.base.ContainerManager"),
            patch("autobio.tools.base.GPUManager"),
        ):
            r = OpenMMRunner("openmm", config)

        input_data = ScoringInput(structure_path=sample_pdb)
        out = r.run(input_data, gpu="none", output_dir=output_dir, mode="amber_minimize")
        assert out.metadata.mode == "amber_minimize"
        assert out.metadata.tool_name == "openmm"
        assert isinstance(out, ScoringOutput)

    def test_run_metadata_mode_md_simulate(
        self,
        config: AutobioConfig,
        tmp_path: Path,
        sample_pdb: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        output_dir = tmp_path / "ws"
        std_dir = output_dir / "outputs" / "standardized"
        std_dir.mkdir(parents=True)
        (std_dir / "trajectory.dcd").write_bytes(b"\x00")
        (std_dir / "final.pdb").write_text("ATOM  final\nEND\n")
        (std_dir / "result_data.json").write_text(json.dumps(_SIMULATE_RESULT))

        monkeypatch.setattr(
            "autobio.core.workspace.Workspace.read_result",
            lambda self: SimpleNamespace(
                status="success", phase="run", exit_code=0, error_message=None
            ),
        )

        with (
            patch("autobio.tools.base.ContainerManager"),
            patch("autobio.tools.base.GPUManager"),
        ):
            r = OpenMMRunner("openmm", config)

        input_data = SimulationInput(structure_path=sample_pdb)
        out = r.run(input_data, gpu="none", output_dir=output_dir, mode="md_simulate")
        assert out.metadata.mode == "md_simulate"
        assert out.metadata.tool_name == "openmm"
        assert isinstance(out, SimulationOutput)
