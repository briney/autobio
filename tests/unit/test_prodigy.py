"""Tests for the migrated prodigy Tool (mode: predict)."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

from autobio.core.catalog import get_tool
from autobio.core.config import AutobioConfig
from autobio.core.registry import ToolCategory
from autobio.core.result import AutobioError
from autobio.core.workspace import Workspace
from autobio.schemas.protein_binding_affinity import ProdigyInput, ProteinBindingAffinityOutput
from autobio.tools import TOOL_RUNNERS, get_runner
from autobio.tools.prodigy import ProdigyRunner

if TYPE_CHECKING:
    from pathlib import Path


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def config() -> AutobioConfig:
    return AutobioConfig.resolve()


def _make_runner() -> ProdigyRunner:
    """Create a ProdigyRunner with mocked deps, current_mode set to 'predict'."""
    with (
        patch("autobio.tools.base.ContainerManager"),
        patch("autobio.tools.base.GPUManager"),
    ):
        runner = ProdigyRunner("prodigy", AutobioConfig.resolve())
    runner.current_mode = get_tool("prodigy").modes["predict"]
    return runner


@pytest.fixture()
def runner() -> ProdigyRunner:
    return _make_runner()


@pytest.fixture()
def sample_pdb(tmp_path: Path) -> Path:
    """Write a minimal two-chain PDB file for testing."""
    content = (
        "ATOM      1  N   ALA A   1       1.000   2.000   3.000  1.00 10.00           N\n"
        "ATOM      2  CA  ALA A   1       2.000   3.000   4.000  1.00 10.00           C\n"
        "ATOM      3  N   GLY B   1       4.000   5.000   6.000  1.00 12.00           N\n"
        "ATOM      4  CA  GLY B   1       5.000   6.000   7.000  1.00 12.00           C\n"
        "END\n"
    )
    pdb_path = tmp_path / "complex.pdb"
    pdb_path.write_text(content)
    return pdb_path


# ---------------------------------------------------------------------------
# TestProdigyPrepareWorkspace
# ---------------------------------------------------------------------------


class TestProdigyPrepareWorkspace:
    """Tests for ProdigyRunner.prepare_workspace."""

    def test_basic_config(self, runner: ProdigyRunner, tmp_path: Path, sample_pdb: Path) -> None:
        """Config contains correct fields for PRODIGY."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = ProdigyInput(
            structure_path=sample_pdb,
            chain_selection="A B",
        )
        runner.prepare_workspace(input_data, workspace)

        cfg = json.loads(workspace.config_path.read_text())
        assert cfg["structure_path"] == "/workspace/inputs/complex.pdb"
        assert cfg["selection"] == "A B"
        assert cfg["output_dir"] == "/workspace/outputs/raw"

    def test_full_config_byte_compat(
        self, runner: ProdigyRunner, tmp_path: Path, sample_pdb: Path
    ) -> None:
        """Full config.json equality — byte-compat contract, incl. the selection remap."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = ProdigyInput(
            structure_path=sample_pdb,
            chain_selection="A B",
        )
        runner.prepare_workspace(input_data, workspace)

        cfg = json.loads(workspace.config_path.read_text())
        assert cfg == {
            "structure_path": "/workspace/inputs/complex.pdb",
            "selection": "A B",
            "temperature": 25.0,
            "distance_cutoff": 5.5,
            "contact_list": False,
            "output_dir": "/workspace/outputs/raw",
        }

    def test_structure_file_copied(
        self, runner: ProdigyRunner, tmp_path: Path, sample_pdb: Path
    ) -> None:
        """Input structure is copied to inputs/ directory."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = ProdigyInput(structure_path=sample_pdb)
        runner.prepare_workspace(input_data, workspace)

        copied = workspace.inputs_dir / "complex.pdb"
        assert copied.exists()
        assert copied.read_text() == sample_pdb.read_text()

    def test_default_temperature(
        self, runner: ProdigyRunner, tmp_path: Path, sample_pdb: Path
    ) -> None:
        """Default temperature is 25.0."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = ProdigyInput(structure_path=sample_pdb)
        runner.prepare_workspace(input_data, workspace)

        cfg = json.loads(workspace.config_path.read_text())
        assert cfg["temperature"] == 25.0

    def test_custom_temperature(
        self, runner: ProdigyRunner, tmp_path: Path, sample_pdb: Path
    ) -> None:
        """Custom temperature is written to config."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = ProdigyInput(
            structure_path=sample_pdb,
            temperature=37.0,
        )
        runner.prepare_workspace(input_data, workspace)

        cfg = json.loads(workspace.config_path.read_text())
        assert cfg["temperature"] == 37.0

    def test_default_distance_cutoff(
        self, runner: ProdigyRunner, tmp_path: Path, sample_pdb: Path
    ) -> None:
        """Default distance_cutoff is 5.5."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = ProdigyInput(structure_path=sample_pdb)
        runner.prepare_workspace(input_data, workspace)

        cfg = json.loads(workspace.config_path.read_text())
        assert cfg["distance_cutoff"] == 5.5

    def test_custom_distance_cutoff(
        self, runner: ProdigyRunner, tmp_path: Path, sample_pdb: Path
    ) -> None:
        """Custom distance_cutoff typed field value."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = ProdigyInput(
            structure_path=sample_pdb,
            distance_cutoff=4.0,
        )
        runner.prepare_workspace(input_data, workspace)

        cfg = json.loads(workspace.config_path.read_text())
        assert cfg["distance_cutoff"] == 4.0

    def test_chain_selection_passthrough(
        self, runner: ProdigyRunner, tmp_path: Path, sample_pdb: Path
    ) -> None:
        """Chain selection string appears in config."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = ProdigyInput(
            structure_path=sample_pdb,
            chain_selection="A,B C",
        )
        runner.prepare_workspace(input_data, workspace)

        cfg = json.loads(workspace.config_path.read_text())
        assert cfg["selection"] == "A,B C"

    def test_chain_selection_none(
        self, runner: ProdigyRunner, tmp_path: Path, sample_pdb: Path
    ) -> None:
        """When None, selection is None in config."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = ProdigyInput(structure_path=sample_pdb)
        runner.prepare_workspace(input_data, workspace)

        cfg = json.loads(workspace.config_path.read_text())
        assert cfg["selection"] is None

    def test_contact_list_default_false(
        self, runner: ProdigyRunner, tmp_path: Path, sample_pdb: Path
    ) -> None:
        """Default contact_list is False."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = ProdigyInput(structure_path=sample_pdb)
        runner.prepare_workspace(input_data, workspace)

        cfg = json.loads(workspace.config_path.read_text())
        assert cfg["contact_list"] is False

    def test_contact_list_true(
        self, runner: ProdigyRunner, tmp_path: Path, sample_pdb: Path
    ) -> None:
        """contact_list=True typed field value."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = ProdigyInput(
            structure_path=sample_pdb,
            contact_list=True,
        )
        runner.prepare_workspace(input_data, workspace)

        cfg = json.loads(workspace.config_path.read_text())
        assert cfg["contact_list"] is True

    def test_extra_dict_merged(
        self, runner: ProdigyRunner, tmp_path: Path, sample_pdb: Path
    ) -> None:
        """Non-typed extra dict keys appear at top level of config.json."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = ProdigyInput(
            structure_path=sample_pdb,
            extra={"custom_flag": "value"},
        )
        runner.prepare_workspace(input_data, workspace)

        cfg = json.loads(workspace.config_path.read_text())
        assert cfg["custom_flag"] == "value"

    def test_extra_shadowing_typed_field_rejected(
        self, runner: ProdigyRunner, tmp_path: Path, sample_pdb: Path
    ) -> None:
        """extra keys colliding with typed fields are rejected fail-fast."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = ProdigyInput(
            structure_path=sample_pdb,
            extra={"distance_cutoff": 4.0},
        )
        with pytest.raises(AutobioError, match="collide with typed input fields"):
            runner.prepare_workspace(input_data, workspace)


# ---------------------------------------------------------------------------
# TestProdigyValidation
# ---------------------------------------------------------------------------


class TestProdigyValidation:
    """Tests for host-side input validation."""

    def test_missing_structure_raises(self, runner: ProdigyRunner, tmp_path: Path) -> None:
        """Missing structure file raises AutobioError."""
        workspace = Workspace.create(tmp_path / "ws")
        fake_path = tmp_path / "nonexistent.pdb"
        input_data = ProdigyInput(structure_path=fake_path)
        with pytest.raises(AutobioError, match="does not exist"):
            runner.prepare_workspace(input_data, workspace)

    def test_invalid_temperature_raises(
        self, runner: ProdigyRunner, tmp_path: Path, sample_pdb: Path
    ) -> None:
        """Temperature below absolute zero raises AutobioError."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = ProdigyInput(
            structure_path=sample_pdb,
            temperature=-300.0,
        )
        with pytest.raises(AutobioError, match="absolute zero"):
            runner.prepare_workspace(input_data, workspace)

    def test_empty_chain_selection_raises(
        self, runner: ProdigyRunner, tmp_path: Path, sample_pdb: Path
    ) -> None:
        """Empty chain_selection string raises AutobioError."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = ProdigyInput(
            structure_path=sample_pdb,
            chain_selection="   ",
        )
        with pytest.raises(AutobioError, match="chain_selection"):
            runner.prepare_workspace(input_data, workspace)


# ---------------------------------------------------------------------------
# TestProdigyParseOutput
# ---------------------------------------------------------------------------

_PRODIGY_RESULT = {
    "predictions": [
        {
            "delta_g_kcal_mol": -10.2,
            "kd_molar": 3.37e-08,
            "units": "kcal/mol",
            "score_breakdown": {
                "intermolecular_contacts": 42,
                "charged_charged_contacts": 3.0,
                "charged_polar_contacts": 5.0,
                "charged_apolar_contacts": 12.0,
                "polar_polar_contacts": 2.0,
                "polar_apolar_contacts": 8.0,
                "apolar_apolar_contacts": 12.0,
                "hydrophilic_hydrophilic_contacts": 6.0,
                "hydrophobic_hydrophilic_contacts": 14.0,
                "hydrophobic_hydrophobic_contacts": 22.0,
                "pct_apolar_nis": 42.31,
                "pct_charged_nis": 18.46,
                "pct_polar_nis": 39.23,
                "chain_selection": "A B",
                "temperature_celsius": 25.0,
                "distance_cutoff_angstrom": 5.5,
                "n_chains": 2,
                "n_residues": 350,
                "structure": "complex",
            },
        }
    ]
}


class TestProdigyParseOutput:
    """Tests for ProdigyRunner.parse_output."""

    def test_parse_binding_output(self, runner: ProdigyRunner, tmp_path: Path) -> None:
        """Standard result_data.json is deserialized correctly."""
        workspace = Workspace.create(tmp_path / "ws")
        (workspace.std_output_dir / "result_data.json").write_text(json.dumps(_PRODIGY_RESULT))

        output = runner.parse_output(workspace)
        assert isinstance(output, ProteinBindingAffinityOutput)
        assert len(output.predictions) == 1

    def test_delta_g_value(self, runner: ProdigyRunner, tmp_path: Path) -> None:
        """delta_g_kcal_mol value is correct."""
        workspace = Workspace.create(tmp_path / "ws")
        (workspace.std_output_dir / "result_data.json").write_text(json.dumps(_PRODIGY_RESULT))

        output = runner.parse_output(workspace)
        assert output.predictions[0].delta_g_kcal_mol == pytest.approx(-10.2)

    def test_kd_molar_value(self, runner: ProdigyRunner, tmp_path: Path) -> None:
        """Kd in molar is correct."""
        workspace = Workspace.create(tmp_path / "ws")
        (workspace.std_output_dir / "result_data.json").write_text(json.dumps(_PRODIGY_RESULT))

        output = runner.parse_output(workspace)
        assert output.predictions[0].kd_molar == pytest.approx(3.37e-08, rel=1e-2)

    def test_units_field(self, runner: ProdigyRunner, tmp_path: Path) -> None:
        """Units string is correct."""
        workspace = Workspace.create(tmp_path / "ws")
        (workspace.std_output_dir / "result_data.json").write_text(json.dumps(_PRODIGY_RESULT))

        output = runner.parse_output(workspace)
        assert output.predictions[0].units == "kcal/mol"

    def test_score_breakdown(self, runner: ProdigyRunner, tmp_path: Path) -> None:
        """Score breakdown contains contact counts and NIS percentages."""
        workspace = Workspace.create(tmp_path / "ws")
        (workspace.std_output_dir / "result_data.json").write_text(json.dumps(_PRODIGY_RESULT))

        output = runner.parse_output(workspace)
        breakdown = output.predictions[0].score_breakdown
        assert breakdown is not None
        assert breakdown["intermolecular_contacts"] == 42
        assert breakdown["pct_apolar_nis"] == pytest.approx(42.31)
        assert breakdown["pct_charged_nis"] == pytest.approx(18.46)
        assert breakdown["chain_selection"] == "A B"

    def test_output_type(self, runner: ProdigyRunner, tmp_path: Path) -> None:
        """Returns ProteinBindingAffinityOutput."""
        workspace = Workspace.create(tmp_path / "ws")
        (workspace.std_output_dir / "result_data.json").write_text(json.dumps(_PRODIGY_RESULT))
        output = runner.parse_output(workspace)
        assert isinstance(output, ProteinBindingAffinityOutput)

    def test_raw_output_path(self, runner: ProdigyRunner, tmp_path: Path) -> None:
        """raw_output_path points to the raw output directory."""
        workspace = Workspace.create(tmp_path / "ws")
        (workspace.std_output_dir / "result_data.json").write_text(json.dumps(_PRODIGY_RESULT))
        output = runner.parse_output(workspace)
        assert output.raw_output_path == workspace.raw_output_dir


# ---------------------------------------------------------------------------
# TestProdigyRegistration
# ---------------------------------------------------------------------------


class TestProdigyRegistration:
    """Tests for tool and runner registration in the catalog."""

    def test_registered_as_catalog_tool(self) -> None:
        import autobio.tools  # noqa: F401
        from autobio.core.catalog import CATALOG

        assert "prodigy" in CATALOG
        assert set(get_tool("prodigy").modes) == {"predict"}
        assert get_tool("prodigy").default_mode == "predict"

    def test_in_tool_runners(self) -> None:
        assert "prodigy" in TOOL_RUNNERS
        assert TOOL_RUNNERS["prodigy"] is ProdigyRunner

    def test_scoring_category(self) -> None:
        assert get_tool("prodigy").category == ToolCategory.SCORING

    def test_no_gpu_required(self) -> None:
        tool = get_tool("prodigy")
        assert tool.requires_gpu is False
        assert tool.gpu_count == 0

    def test_schema_types(self) -> None:
        mode = get_tool("prodigy").modes["predict"]
        assert mode.input_schema is ProdigyInput
        assert mode.output_schema is ProteinBindingAffinityOutput

    def test_image_tag(self) -> None:
        assert get_tool("prodigy").image_tag == "prodigy:2.4.0"

    def test_timeout(self) -> None:
        assert get_tool("prodigy").modes["predict"].default_timeout == 300

    def test_get_runner_returns_prodigy_runner(self, config: AutobioConfig) -> None:
        with (
            patch("autobio.tools.base.ContainerManager"),
            patch("autobio.tools.base.GPUManager"),
        ):
            r = get_runner("prodigy", config)
        assert isinstance(r, ProdigyRunner)
        assert r.tool_name == "prodigy"

    def test_tool_constant_registered(self) -> None:
        import autobio.tools  # noqa: F401
        from autobio.tools.prodigy import PRODIGY_TOOL

        assert PRODIGY_TOOL.name == "prodigy"
        assert get_tool("prodigy") is PRODIGY_TOOL


def test_info_snapshot_prodigy() -> None:
    import autobio.tools  # noqa: F401
    from autobio.cli.formatters import OutputFormat, format_tool_info_catalog

    parsed = json.loads(format_tool_info_catalog(get_tool("prodigy"), OutputFormat.JSON))
    assert [m["name"] for m in parsed["modes"]] == ["predict"]
    predict = parsed["modes"][0]
    struct = predict["input_schema"]["properties"]["structure_path"]
    assert struct["x-autobio"]["widget"] == "file"
    contact_list = predict["input_schema"]["properties"]["contact_list"]
    assert contact_list["x-autobio"]["widget"] == "toggle"
    assert "output_schema" in predict
