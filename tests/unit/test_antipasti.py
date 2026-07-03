"""Tests for the migrated antipasti Tool (mode: predict)."""

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
from autobio.schemas.binding_affinity import AntipastiInput, BindingAffinityOutput
from autobio.tools import TOOL_RUNNERS, get_runner
from autobio.tools.antipasti import AntipastiRunner

if TYPE_CHECKING:
    from pathlib import Path


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def config() -> AutobioConfig:
    return AutobioConfig.resolve()


def _make_runner() -> AntipastiRunner:
    """Create an AntipastiRunner with mocked deps, current_mode set to 'predict'."""
    with (
        patch("autobio.tools.base.ContainerManager"),
        patch("autobio.tools.base.GPUManager"),
    ):
        runner = AntipastiRunner("antipasti", AutobioConfig.resolve())
    runner.current_mode = get_tool("antipasti").modes["predict"]
    return runner


@pytest.fixture()
def runner() -> AntipastiRunner:
    return _make_runner()


@pytest.fixture()
def sample_pdb(tmp_path: Path) -> Path:
    """Write a minimal multi-chain PDB file for testing."""
    content = (
        "ATOM      1  N   ALA H   1       1.000   2.000   3.000  1.00 10.00           N\n"
        "ATOM      2  CA  ALA H   1       2.000   3.000   4.000  1.00 10.00           C\n"
        "ATOM      3  N   GLY L   1       4.000   5.000   6.000  1.00 12.00           N\n"
        "ATOM      4  CA  GLY L   1       5.000   6.000   7.000  1.00 12.00           C\n"
        "ATOM      5  N   LEU A   1       7.000   8.000   9.000  1.00 14.00           N\n"
        "ATOM      6  CA  LEU A   1       8.000   9.000  10.000  1.00 14.00           C\n"
        "END\n"
    )
    pdb_path = tmp_path / "complex.pdb"
    pdb_path.write_text(content)
    return pdb_path


# ---------------------------------------------------------------------------
# TestAntipastiPrepareWorkspace
# ---------------------------------------------------------------------------


class TestAntipastiPrepareWorkspace:
    """Tests for AntipastiRunner.prepare_workspace."""

    def test_basic_config(self, runner: AntipastiRunner, tmp_path: Path, sample_pdb: Path) -> None:
        """Config contains correct fields for ANTIPASTI."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = AntipastiInput(
            structure_path=sample_pdb,
            heavy_chain="H",
            light_chain="L",
            antigen_chains=["A"],
        )
        runner.prepare_workspace(input_data, workspace)

        cfg = json.loads(workspace.config_path.read_text())
        assert cfg["pdb_path"] == "/workspace/inputs/complex.pdb"
        assert cfg["heavy_chain"] == "H"
        assert cfg["light_chain"] == "L"
        assert cfg["antigen_chains"] == ["A"]
        assert cfg["output_dir"] == "/workspace/outputs/raw"
        assert cfg["antipasti_dir"] == "/app/antipasti"

    def test_full_config_byte_compat(
        self, runner: AntipastiRunner, tmp_path: Path, sample_pdb: Path
    ) -> None:
        """Full config.json equality — byte-compat contract, incl. hardcoded constants."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = AntipastiInput(
            structure_path=sample_pdb,
            heavy_chain="H",
            light_chain="L",
            antigen_chains=["A"],
        )
        runner.prepare_workspace(input_data, workspace)

        cfg = json.loads(workspace.config_path.read_text())
        assert cfg == {
            "pdb_path": "/workspace/inputs/complex.pdb",
            "heavy_chain": "H",
            "light_chain": "L",
            "antigen_chains": ["A"],
            "checkpoint_path": (
                "/app/antipasti/checkpoints/full_ags_all_modes/"
                "model_epochs_1044_modes_all_pool_1_filters_4_size_4.pt"
            ),
            "output_dir": "/workspace/outputs/raw",
            "antipasti_dir": "/app/antipasti",
            "modes": "all",
        }

    def test_structure_file_copied(
        self, runner: AntipastiRunner, tmp_path: Path, sample_pdb: Path
    ) -> None:
        """Input structure is copied to inputs/ directory."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = AntipastiInput(
            structure_path=sample_pdb,
            heavy_chain="H",
            light_chain="L",
            antigen_chains=["A"],
        )
        runner.prepare_workspace(input_data, workspace)

        copied = workspace.inputs_dir / "complex.pdb"
        assert copied.exists()
        assert copied.read_text() == sample_pdb.read_text()

    def test_chain_ids_passthrough(
        self, runner: AntipastiRunner, tmp_path: Path, sample_pdb: Path
    ) -> None:
        """Chain IDs are passed through to config."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = AntipastiInput(
            structure_path=sample_pdb,
            heavy_chain="A",
            light_chain="B",
            antigen_chains=["C", "D"],
        )
        runner.prepare_workspace(input_data, workspace)

        cfg = json.loads(workspace.config_path.read_text())
        assert cfg["heavy_chain"] == "A"
        assert cfg["light_chain"] == "B"
        assert cfg["antigen_chains"] == ["C", "D"]

    def test_default_modes(self, runner: AntipastiRunner, tmp_path: Path, sample_pdb: Path) -> None:
        """Default modes value is 'all'."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = AntipastiInput(
            structure_path=sample_pdb,
            heavy_chain="H",
            light_chain="L",
            antigen_chains=["A"],
        )
        runner.prepare_workspace(input_data, workspace)

        cfg = json.loads(workspace.config_path.read_text())
        assert cfg["modes"] == "all"

    def test_modes_int_override(
        self, runner: AntipastiRunner, tmp_path: Path, sample_pdb: Path
    ) -> None:
        """modes can be set to an integer count of normal modes."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = AntipastiInput(
            structure_path=sample_pdb,
            heavy_chain="H",
            light_chain="L",
            antigen_chains=["A"],
            modes=100,
        )
        runner.prepare_workspace(input_data, workspace)

        cfg = json.loads(workspace.config_path.read_text())
        assert cfg["modes"] == 100

    def test_checkpoint_path(
        self, runner: AntipastiRunner, tmp_path: Path, sample_pdb: Path
    ) -> None:
        """Config references baked-in checkpoint."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = AntipastiInput(
            structure_path=sample_pdb,
            heavy_chain="H",
            light_chain="L",
            antigen_chains=["A"],
        )
        runner.prepare_workspace(input_data, workspace)

        cfg = json.loads(workspace.config_path.read_text())
        assert "checkpoints/full_ags_all_modes" in cfg["checkpoint_path"]
        assert cfg["checkpoint_path"].endswith(".pt")

    def test_extra_dict_merged(
        self, runner: AntipastiRunner, tmp_path: Path, sample_pdb: Path
    ) -> None:
        """Non-consumed extra dict keys appear at top level of config.json."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = AntipastiInput(
            structure_path=sample_pdb,
            heavy_chain="H",
            light_chain="L",
            antigen_chains=["A"],
            extra={"custom_flag": "value"},
        )
        runner.prepare_workspace(input_data, workspace)

        cfg = json.loads(workspace.config_path.read_text())
        assert cfg["custom_flag"] == "value"

    def test_extra_shadowing_typed_field_rejected(
        self, runner: AntipastiRunner, tmp_path: Path, sample_pdb: Path
    ) -> None:
        """extra must not shadow the typed 'modes' field."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = AntipastiInput(
            structure_path=sample_pdb,
            heavy_chain="H",
            light_chain="L",
            antigen_chains=["A"],
            extra={"modes": 100},
        )
        with pytest.raises(AutobioError, match="collide with typed input fields"):
            runner.prepare_workspace(input_data, workspace)


# ---------------------------------------------------------------------------
# TestAntipastiValidation
# ---------------------------------------------------------------------------


class TestAntipastiValidation:
    """Tests for host-side input validation."""

    def test_missing_structure_raises(self, runner: AntipastiRunner, tmp_path: Path) -> None:
        """Missing structure file raises AutobioError."""
        workspace = Workspace.create(tmp_path / "ws")
        fake_path = tmp_path / "nonexistent.pdb"
        input_data = AntipastiInput(
            structure_path=fake_path,
            heavy_chain="H",
            light_chain="L",
            antigen_chains=["A"],
        )
        with pytest.raises(AutobioError, match="does not exist"):
            runner.prepare_workspace(input_data, workspace)

    def test_empty_heavy_chain_raises(
        self, runner: AntipastiRunner, tmp_path: Path, sample_pdb: Path
    ) -> None:
        """Empty heavy chain raises AutobioError."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = AntipastiInput(
            structure_path=sample_pdb,
            heavy_chain="",
            light_chain="L",
            antigen_chains=["A"],
        )
        with pytest.raises(AutobioError, match="heavy_chain"):
            runner.prepare_workspace(input_data, workspace)

    def test_empty_light_chain_raises(
        self, runner: AntipastiRunner, tmp_path: Path, sample_pdb: Path
    ) -> None:
        """Empty light chain raises AutobioError."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = AntipastiInput(
            structure_path=sample_pdb,
            heavy_chain="H",
            light_chain="",
            antigen_chains=["A"],
        )
        with pytest.raises(AutobioError, match="light_chain"):
            runner.prepare_workspace(input_data, workspace)

    def test_empty_antigen_chains_raises(
        self, runner: AntipastiRunner, tmp_path: Path, sample_pdb: Path
    ) -> None:
        """Empty antigen chains list raises AutobioError."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = AntipastiInput(
            structure_path=sample_pdb,
            heavy_chain="H",
            light_chain="L",
            antigen_chains=[],
        )
        with pytest.raises(AutobioError, match="antigen_chains"):
            runner.prepare_workspace(input_data, workspace)

    def test_empty_antigen_chain_id_raises(
        self, runner: AntipastiRunner, tmp_path: Path, sample_pdb: Path
    ) -> None:
        """Empty string in antigen chains raises AutobioError."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = AntipastiInput(
            structure_path=sample_pdb,
            heavy_chain="H",
            light_chain="L",
            antigen_chains=["A", ""],
        )
        with pytest.raises(AutobioError, match="non-empty string"):
            runner.prepare_workspace(input_data, workspace)

    def test_duplicate_chain_ids_raises(
        self, runner: AntipastiRunner, tmp_path: Path, sample_pdb: Path
    ) -> None:
        """Duplicate chain IDs across heavy/light/antigen raises AutobioError."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = AntipastiInput(
            structure_path=sample_pdb,
            heavy_chain="H",
            light_chain="H",
            antigen_chains=["A"],
        )
        with pytest.raises(AutobioError, match="Duplicate"):
            runner.prepare_workspace(input_data, workspace)

    def test_duplicate_antigen_chain_id_raises(
        self, runner: AntipastiRunner, tmp_path: Path, sample_pdb: Path
    ) -> None:
        """Duplicate chain ID between antigen and heavy raises AutobioError."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = AntipastiInput(
            structure_path=sample_pdb,
            heavy_chain="A",
            light_chain="B",
            antigen_chains=["A"],
        )
        with pytest.raises(AutobioError, match="Duplicate"):
            runner.prepare_workspace(input_data, workspace)


# ---------------------------------------------------------------------------
# TestAntipastiParseOutput
# ---------------------------------------------------------------------------

_AFFINITY_RESULT = {
    "predictions": [
        {
            "log10_kd": -8.542,
            "kd_molar": 2.87e-9,
            "units": "log10(Kd) [M]",
            "score_breakdown": {
                "heavy_chain": "H",
                "light_chain": "L",
                "antigen_chains": ["A"],
                "modes": "all",
                "checkpoint": "model_epochs_1044_modes_all_pool_1_filters_4_size_4",
                "pdb_id": "8hn6",
            },
        }
    ]
}


class TestAntipastiParseOutput:
    """Tests for AntipastiRunner.parse_output."""

    def test_parse_affinity_output(self, runner: AntipastiRunner, tmp_path: Path) -> None:
        """Standard result_data.json is deserialized correctly."""
        workspace = Workspace.create(tmp_path / "ws")
        (workspace.std_output_dir / "result_data.json").write_text(json.dumps(_AFFINITY_RESULT))

        output = runner.parse_output(workspace)
        assert isinstance(output, BindingAffinityOutput)
        assert len(output.predictions) == 1

    def test_log10_kd_value(self, runner: AntipastiRunner, tmp_path: Path) -> None:
        """log10_kd value is correct."""
        workspace = Workspace.create(tmp_path / "ws")
        (workspace.std_output_dir / "result_data.json").write_text(json.dumps(_AFFINITY_RESULT))

        output = runner.parse_output(workspace)
        assert output.predictions[0].log10_kd == pytest.approx(-8.542)

    def test_kd_molar_value(self, runner: AntipastiRunner, tmp_path: Path) -> None:
        """Derived Kd in molar is correct."""
        workspace = Workspace.create(tmp_path / "ws")
        (workspace.std_output_dir / "result_data.json").write_text(json.dumps(_AFFINITY_RESULT))

        output = runner.parse_output(workspace)
        assert output.predictions[0].kd_molar == pytest.approx(2.87e-9, rel=1e-2)

    def test_units_field(self, runner: AntipastiRunner, tmp_path: Path) -> None:
        """Units string is correct."""
        workspace = Workspace.create(tmp_path / "ws")
        (workspace.std_output_dir / "result_data.json").write_text(json.dumps(_AFFINITY_RESULT))

        output = runner.parse_output(workspace)
        assert output.predictions[0].units == "log10(Kd) [M]"

    def test_score_breakdown(self, runner: AntipastiRunner, tmp_path: Path) -> None:
        """Score breakdown contains chain info and model metadata."""
        workspace = Workspace.create(tmp_path / "ws")
        (workspace.std_output_dir / "result_data.json").write_text(json.dumps(_AFFINITY_RESULT))

        output = runner.parse_output(workspace)
        breakdown = output.predictions[0].score_breakdown
        assert breakdown is not None
        assert breakdown["heavy_chain"] == "H"
        assert breakdown["modes"] == "all"

    def test_output_type(self, runner: AntipastiRunner, tmp_path: Path) -> None:
        """Returns BindingAffinityOutput."""
        workspace = Workspace.create(tmp_path / "ws")
        (workspace.std_output_dir / "result_data.json").write_text(json.dumps(_AFFINITY_RESULT))
        output = runner.parse_output(workspace)
        assert isinstance(output, BindingAffinityOutput)

    def test_raw_output_path(self, runner: AntipastiRunner, tmp_path: Path) -> None:
        """raw_output_path points to the raw output directory."""
        workspace = Workspace.create(tmp_path / "ws")
        (workspace.std_output_dir / "result_data.json").write_text(json.dumps(_AFFINITY_RESULT))
        output = runner.parse_output(workspace)
        assert output.raw_output_path == workspace.raw_output_dir


# ---------------------------------------------------------------------------
# TestAntipastiRegistration
# ---------------------------------------------------------------------------


class TestAntipastiRegistration:
    """Tests for tool and runner registration in the catalog."""

    def test_registered_as_catalog_tool(self) -> None:
        from autobio.core.catalog import CATALOG

        assert "antipasti" in CATALOG
        assert set(get_tool("antipasti").modes) == {"predict"}
        assert get_tool("antipasti").default_mode == "predict"

    def test_in_tool_runners(self) -> None:
        assert "antipasti" in TOOL_RUNNERS
        assert TOOL_RUNNERS["antipasti"] is AntipastiRunner

    def test_scoring_category(self) -> None:
        assert get_tool("antipasti").category == ToolCategory.SCORING

    def test_no_gpu_required(self) -> None:
        tool = get_tool("antipasti")
        assert tool.requires_gpu is False
        assert tool.gpu_count == 0

    def test_schema_types(self) -> None:
        mode = get_tool("antipasti").modes["predict"]
        assert mode.input_schema is AntipastiInput
        assert mode.output_schema is BindingAffinityOutput

    def test_image_tag(self) -> None:
        assert get_tool("antipasti").image_tag == "antipasti:1.0.0"

    def test_timeout(self) -> None:
        assert get_tool("antipasti").modes["predict"].default_timeout == 1800

    def test_get_runner_returns_antipasti_runner(self, config: AutobioConfig) -> None:
        with (
            patch("autobio.tools.base.ContainerManager"),
            patch("autobio.tools.base.GPUManager"),
        ):
            r = get_runner("antipasti", config)
        assert isinstance(r, AntipastiRunner)
        assert r.tool_name == "antipasti"

    def test_tool_constant_registered(self) -> None:
        from autobio.tools.antipasti import ANTIPASTI_TOOL

        assert ANTIPASTI_TOOL.name == "antipasti"
        assert get_tool("antipasti") is ANTIPASTI_TOOL


def test_info_snapshot_antipasti() -> None:
    from autobio.cli.formatters import OutputFormat, format_tool_info_catalog

    parsed = json.loads(format_tool_info_catalog(get_tool("antipasti"), OutputFormat.JSON))
    assert [m["name"] for m in parsed["modes"]] == ["predict"]
    predict = parsed["modes"][0]
    struct = predict["input_schema"]["properties"]["structure_path"]
    assert struct["x-autobio"]["widget"] == "file"
    assert "output_schema" in predict
