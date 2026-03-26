"""Tests for OpenMMRunner — prepare_workspace, parse_output, and registration."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

from autobio.core.config import AutobioConfig
from autobio.core.registry import TOOL_REGISTRY, ToolCategory
from autobio.core.result import AutobioError
from autobio.core.workspace import Workspace
from autobio.schemas.scoring import ScoringInput, ScoringOutput
from autobio.tools import TOOL_RUNNERS, get_runner
from autobio.tools.openmm import (
    _ALLOWED_FORCE_FIELDS,
    _ALLOWED_RESTRAINT_SETS,
    _VARIANT_CONFIG,
    OpenMMRunner,
)

if TYPE_CHECKING:
    from pathlib import Path


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def config() -> AutobioConfig:
    return AutobioConfig.resolve()


@pytest.fixture()
def runner(config: AutobioConfig) -> OpenMMRunner:
    """Create an OpenMMRunner for openmm_amber_minimize with mocked deps."""
    with (
        patch("autobio.tools.base.ContainerManager"),
        patch("autobio.tools.base.GPUManager"),
    ):
        return OpenMMRunner("openmm_amber_minimize", config)


@pytest.fixture()
def sample_pdb(tmp_path: Path) -> Path:
    """Write a minimal PDB file for testing."""
    pdb_path = tmp_path / "test.pdb"
    pdb_path.write_text("ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00\nEND\n")
    return pdb_path


# ---------------------------------------------------------------------------
# TestOpenMMPrepareWorkspace
# ---------------------------------------------------------------------------

_ALL_TOOL_NAMES = list(_VARIANT_CONFIG.keys())


class TestOpenMMPrepareWorkspace:
    """Tests for OpenMMRunner.prepare_workspace."""

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
        variant = _VARIANT_CONFIG["openmm_amber_minimize"]
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
# TestOpenMMValidation
# ---------------------------------------------------------------------------


class TestOpenMMValidation:
    """Tests for host-side input validation."""

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
# TestOpenMMParseOutput
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
    """Tests for OpenMMRunner.parse_output."""

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
# TestOpenMMRegistration
# ---------------------------------------------------------------------------


class TestOpenMMRegistration:
    """Tests for tool and runner registration."""

    @pytest.mark.parametrize("tool_name", _ALL_TOOL_NAMES)
    def test_in_tool_registry(self, tool_name: str) -> None:
        assert tool_name in TOOL_REGISTRY

    @pytest.mark.parametrize("tool_name", _ALL_TOOL_NAMES)
    def test_in_tool_runners(self, tool_name: str) -> None:
        assert tool_name in TOOL_RUNNERS
        assert TOOL_RUNNERS[tool_name] is OpenMMRunner

    @pytest.mark.parametrize("tool_name", _ALL_TOOL_NAMES)
    def test_scoring_category(self, tool_name: str) -> None:
        assert TOOL_REGISTRY[tool_name].category == ToolCategory.SCORING

    @pytest.mark.parametrize("tool_name", _ALL_TOOL_NAMES)
    def test_no_gpu_required(self, tool_name: str) -> None:
        entry = TOOL_REGISTRY[tool_name]
        assert entry.requires_gpu is False
        assert entry.gpu_count == 0

    @pytest.mark.parametrize("tool_name", _ALL_TOOL_NAMES)
    def test_schema_types(self, tool_name: str) -> None:
        entry = TOOL_REGISTRY[tool_name]
        assert entry.input_schema is ScoringInput
        assert entry.output_schema is ScoringOutput

    def test_get_runner_returns_openmm_runner(self, config: AutobioConfig) -> None:
        with (
            patch("autobio.tools.base.ContainerManager"),
            patch("autobio.tools.base.GPUManager"),
        ):
            r = get_runner("openmm_amber_minimize", config)
        assert isinstance(r, OpenMMRunner)
        assert r.tool_name == "openmm_amber_minimize"
