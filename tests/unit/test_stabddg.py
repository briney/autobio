"""Unit tests for the migrated stabddg Tool (mode: predict)."""

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
from autobio.schemas.scoring import ScoringOutput, StaBddGInput
from autobio.tools import TOOL_RUNNERS, get_runner
from autobio.tools.stabddg import StaBddGRunner

if TYPE_CHECKING:
    from pathlib import Path


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def config() -> AutobioConfig:
    return AutobioConfig.resolve()


def _make_runner() -> StaBddGRunner:
    """Create a StaBddGRunner with mocked deps, current_mode set to 'predict'."""
    with (
        patch("autobio.tools.base.ContainerManager"),
        patch("autobio.tools.base.GPUManager"),
    ):
        runner = StaBddGRunner("stabddg", AutobioConfig.resolve())
    runner.current_mode = get_tool("stabddg").modes["predict"]
    return runner


@pytest.fixture()
def runner() -> StaBddGRunner:
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


def _written_config(runner: StaBddGRunner, input_data: StaBddGInput, tmp_path: Path) -> dict:
    ws = Workspace.create(tmp_path / "ws")
    try:
        runner.prepare_workspace(input_data, ws)
        return json.loads((ws.root / "config.json").read_text())
    finally:
        ws.cleanup()


# ---------------------------------------------------------------------------
# TestStaBddGPrepareWorkspace
# ---------------------------------------------------------------------------


class TestStaBddGPrepareWorkspace:
    """Tests for StaBddGRunner.prepare_workspace."""

    def test_basic_config(self, runner: StaBddGRunner, tmp_path: Path, sample_pdb: Path) -> None:
        """Config contains correct fields for StaB-ddG."""
        input_data = StaBddGInput(
            structure_path=sample_pdb,
            mutations=["EA63Q"],
            chains="A_B",
        )
        cfg = _written_config(runner, input_data, tmp_path)

        assert cfg["pdb_path"] == "/workspace/inputs/complex.pdb"
        assert cfg["mutations"] == "EA63Q"
        assert cfg["chains"] == "A_B"
        assert cfg["checkpoint_path"] == "/app/stabddg/model_ckpts/stabddg.pt"
        assert cfg["output_dir"] == "/workspace/outputs/raw"

    def test_full_config_dict(
        self, runner: StaBddGRunner, tmp_path: Path, sample_pdb: Path
    ) -> None:
        """The entire config.json dict matches the byte-compat contract exactly."""
        input_data = StaBddGInput(
            structure_path=sample_pdb,
            mutations=["YH103H", "QD30V"],
            chains="ABC_DE",
            mc_samples=50,
            noise_level=0.2,
            batch_size=5000,
            trials=3,
            seed=42,
            device="cpu",
        )
        cfg = _written_config(runner, input_data, tmp_path)

        assert cfg == {
            "pdb_path": "/workspace/inputs/complex.pdb",
            "mutations": "YH103H,QD30V",
            "chains": "ABC_DE",
            "checkpoint_path": "/app/stabddg/model_ckpts/stabddg.pt",
            "output_dir": "/workspace/outputs/raw",
            "mc_samples": 50,
            "noise_level": 0.2,
            "batch_size": 5000,
            "trials": 3,
            "seed": 42,
            "device": "cpu",
        }

    def test_structure_file_copied(
        self, runner: StaBddGRunner, tmp_path: Path, sample_pdb: Path
    ) -> None:
        """Input structure is copied to inputs/ and config references container path."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = StaBddGInput(
            structure_path=sample_pdb,
            mutations=["EA63Q"],
            chains="A_B",
        )
        runner.prepare_workspace(input_data, workspace)

        copied = workspace.inputs_dir / "complex.pdb"
        assert copied.exists()
        assert copied.read_text() == sample_pdb.read_text()

    def test_mutations_comma_joined(
        self, runner: StaBddGRunner, tmp_path: Path, sample_pdb: Path
    ) -> None:
        """Multiple mutations are joined into comma-separated string."""
        input_data = StaBddGInput(
            structure_path=sample_pdb,
            mutations=["YH103H", "QD30V", "KA66A"],
            chains="ABC_DE",
        )
        cfg = _written_config(runner, input_data, tmp_path)

        assert cfg["mutations"] == "YH103H,QD30V,KA66A"

    def test_chains_passthrough(
        self, runner: StaBddGRunner, tmp_path: Path, sample_pdb: Path
    ) -> None:
        """Chain specification is passed through as-is."""
        input_data = StaBddGInput(
            structure_path=sample_pdb,
            mutations=["EA63Q"],
            chains="ABC_DE",
        )
        cfg = _written_config(runner, input_data, tmp_path)

        assert cfg["chains"] == "ABC_DE"

    def test_default_params(self, runner: StaBddGRunner, tmp_path: Path, sample_pdb: Path) -> None:
        """Default parameter values are written to config."""
        input_data = StaBddGInput(
            structure_path=sample_pdb,
            mutations=["EA63Q"],
            chains="A_B",
        )
        cfg = _written_config(runner, input_data, tmp_path)

        assert cfg["mc_samples"] == 20
        assert cfg["noise_level"] == pytest.approx(0.1)
        assert cfg["batch_size"] == 10000
        assert cfg["trials"] == 1
        assert cfg["seed"] == 0
        assert cfg["device"] == "auto"

    def test_override_params(self, runner: StaBddGRunner, tmp_path: Path, sample_pdb: Path) -> None:
        """Non-default typed field values override defaults."""
        input_data = StaBddGInput(
            structure_path=sample_pdb,
            mutations=["EA63Q"],
            chains="A_B",
            mc_samples=50,
            noise_level=0.2,
            batch_size=5000,
            trials=3,
            seed=42,
            device="cpu",
        )
        cfg = _written_config(runner, input_data, tmp_path)

        assert cfg["mc_samples"] == 50
        assert cfg["noise_level"] == pytest.approx(0.2)
        assert cfg["batch_size"] == 5000
        assert cfg["trials"] == 3
        assert cfg["seed"] == 42
        assert cfg["device"] == "cpu"

    def test_extra_dict_merged(
        self, runner: StaBddGRunner, tmp_path: Path, sample_pdb: Path
    ) -> None:
        """Non-typed extra dict keys appear at top level of config.json."""
        input_data = StaBddGInput(
            structure_path=sample_pdb,
            mutations=["EA63Q"],
            chains="A_B",
            extra={"custom_flag": "value"},
        )
        cfg = _written_config(runner, input_data, tmp_path)

        assert cfg["custom_flag"] == "value"

    def test_extra_shadowing_typed_field_rejected(
        self, runner: StaBddGRunner, tmp_path: Path, sample_pdb: Path
    ) -> None:
        """extra keys colliding with typed fields are rejected fail-fast."""
        input_data = StaBddGInput(
            structure_path=sample_pdb,
            mutations=["EA63Q"],
            chains="A_B",
            extra={"mc_samples": 50},
        )
        with pytest.raises(AutobioError, match="collide with typed input fields"):
            _written_config(runner, input_data, tmp_path)


# ---------------------------------------------------------------------------
# TestStaBddGValidation
# ---------------------------------------------------------------------------


class TestStaBddGValidation:
    """Tests for host-side input validation."""

    def test_missing_structure_raises(self, runner: StaBddGRunner, tmp_path: Path) -> None:
        """Missing structure file raises AutobioError."""
        workspace = Workspace.create(tmp_path / "ws")
        fake_path = tmp_path / "nonexistent.pdb"
        input_data = StaBddGInput(
            structure_path=fake_path,
            mutations=["EA63Q"],
            chains="A_B",
        )
        with pytest.raises(AutobioError, match="does not exist"):
            runner.prepare_workspace(input_data, workspace)

    def test_empty_mutations_raises(
        self, runner: StaBddGRunner, tmp_path: Path, sample_pdb: Path
    ) -> None:
        """Empty mutations list raises AutobioError."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = StaBddGInput(
            structure_path=sample_pdb,
            mutations=[],
            chains="A_B",
        )
        with pytest.raises(AutobioError, match="requires at least one mutation"):
            runner.prepare_workspace(input_data, workspace)

    def test_missing_chains_raises(
        self, runner: StaBddGRunner, tmp_path: Path, sample_pdb: Path
    ) -> None:
        """Empty chains raises AutobioError."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = StaBddGInput(
            structure_path=sample_pdb,
            mutations=["EA63Q"],
            chains="",
        )
        with pytest.raises(AutobioError, match="requires a non-empty 'chains'"):
            runner.prepare_workspace(input_data, workspace)

    def test_invalid_chains_no_underscore_raises(
        self, runner: StaBddGRunner, tmp_path: Path, sample_pdb: Path
    ) -> None:
        """Chains without underscore raises AutobioError."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = StaBddGInput(
            structure_path=sample_pdb,
            mutations=["EA63Q"],
            chains="ABC",
        )
        with pytest.raises(AutobioError, match="exactly one underscore"):
            runner.prepare_workspace(input_data, workspace)

    def test_invalid_chains_multiple_underscores_raises(
        self, runner: StaBddGRunner, tmp_path: Path, sample_pdb: Path
    ) -> None:
        """Chains with multiple underscores raises AutobioError."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = StaBddGInput(
            structure_path=sample_pdb,
            mutations=["EA63Q"],
            chains="A_B_C",
        )
        with pytest.raises(AutobioError, match="exactly one underscore"):
            runner.prepare_workspace(input_data, workspace)


# ---------------------------------------------------------------------------
# TestStaBddGParseOutput
# ---------------------------------------------------------------------------

_DDG_RESULT = {
    "scores": [
        {
            "total_score": 1.23,
            "score_breakdown": {"chains": "A_B"},
            "units": "kcal/mol",
            "per_residue_scores": None,
            "structure_path": None,
            "ddg": 1.23,
            "mutations": ["EA63Q"],
        }
    ]
}

_MULTI_TRIAL_RESULT = {
    "scores": [
        {
            "total_score": 1.35,
            "score_breakdown": {
                "chains": "A_B",
                "trial_values": {"pred_1": 1.23, "pred_2": 1.47},
                "n_trials": 2,
            },
            "units": "kcal/mol",
            "per_residue_scores": None,
            "structure_path": None,
            "ddg": 1.35,
            "mutations": ["EA63Q"],
        }
    ]
}


class TestStaBddGParseOutput:
    """Tests for StaBddGRunner.parse_output."""

    def test_parse_ddg_output(self, runner: StaBddGRunner, tmp_path: Path) -> None:
        """Standard result_data.json is deserialized correctly."""
        workspace = Workspace.create(tmp_path / "ws")
        (workspace.std_output_dir / "result_data.json").write_text(json.dumps(_DDG_RESULT))

        output = runner.parse_output(workspace)
        assert isinstance(output, ScoringOutput)
        assert len(output.scores) == 1

    def test_ddg_value(self, runner: StaBddGRunner, tmp_path: Path) -> None:
        """DDG and total_score are correct."""
        workspace = Workspace.create(tmp_path / "ws")
        (workspace.std_output_dir / "result_data.json").write_text(json.dumps(_DDG_RESULT))

        output = runner.parse_output(workspace)
        s = output.scores[0]
        assert s.ddg == pytest.approx(1.23)
        assert s.total_score == pytest.approx(1.23)

    def test_units_kcal_mol(self, runner: StaBddGRunner, tmp_path: Path) -> None:
        """Units are kcal/mol."""
        workspace = Workspace.create(tmp_path / "ws")
        (workspace.std_output_dir / "result_data.json").write_text(json.dumps(_DDG_RESULT))

        output = runner.parse_output(workspace)
        assert output.scores[0].units == "kcal/mol"

    def test_mutations_in_output(self, runner: StaBddGRunner, tmp_path: Path) -> None:
        """Mutations list is preserved."""
        workspace = Workspace.create(tmp_path / "ws")
        (workspace.std_output_dir / "result_data.json").write_text(json.dumps(_DDG_RESULT))

        output = runner.parse_output(workspace)
        assert output.scores[0].mutations == ["EA63Q"]

    def test_no_structure_path(self, runner: StaBddGRunner, tmp_path: Path) -> None:
        """StaB-ddG does not produce output structures."""
        workspace = Workspace.create(tmp_path / "ws")
        (workspace.std_output_dir / "result_data.json").write_text(json.dumps(_DDG_RESULT))

        output = runner.parse_output(workspace)
        assert output.scores[0].structure_path is None

    def test_output_type(self, runner: StaBddGRunner, tmp_path: Path) -> None:
        """Returns ScoringOutput."""
        workspace = Workspace.create(tmp_path / "ws")
        (workspace.std_output_dir / "result_data.json").write_text(json.dumps(_DDG_RESULT))
        output = runner.parse_output(workspace)
        assert isinstance(output, ScoringOutput)

    def test_raw_output_path(self, runner: StaBddGRunner, tmp_path: Path) -> None:
        """raw_output_path points to the raw output directory."""
        workspace = Workspace.create(tmp_path / "ws")
        (workspace.std_output_dir / "result_data.json").write_text(json.dumps(_DDG_RESULT))
        output = runner.parse_output(workspace)
        assert output.raw_output_path == workspace.raw_output_dir

    def test_multi_trial_output(self, runner: StaBddGRunner, tmp_path: Path) -> None:
        """Multi-trial results include trial breakdown."""
        workspace = Workspace.create(tmp_path / "ws")
        (workspace.std_output_dir / "result_data.json").write_text(json.dumps(_MULTI_TRIAL_RESULT))

        output = runner.parse_output(workspace)
        s = output.scores[0]
        assert s.ddg == pytest.approx(1.35)
        assert s.score_breakdown is not None
        assert s.score_breakdown["n_trials"] == 2


# ---------------------------------------------------------------------------
# TestStaBddGRegistration
# ---------------------------------------------------------------------------


class TestStaBddGRegistration:
    """Tests for tool and runner registration in the catalog."""

    def test_registered_as_catalog_tool(self) -> None:
        import autobio.tools  # noqa: F401
        from autobio.core.catalog import CATALOG
        from autobio.core.registry import TOOL_REGISTRY

        assert "stabddg" in CATALOG
        assert set(get_tool("stabddg").modes) == {"predict"}
        assert get_tool("stabddg").default_mode == "predict"
        assert "stabddg" not in TOOL_REGISTRY

    def test_in_tool_runners(self) -> None:
        assert "stabddg" in TOOL_RUNNERS
        assert TOOL_RUNNERS["stabddg"] is StaBddGRunner

    def test_scoring_category(self) -> None:
        assert get_tool("stabddg").category == ToolCategory.SCORING

    def test_gpu_required(self) -> None:
        tool = get_tool("stabddg")
        assert tool.requires_gpu is True
        assert tool.gpu_count == 1

    def test_schema_types(self) -> None:
        mode = get_tool("stabddg").modes["predict"]
        assert mode.input_schema is StaBddGInput
        assert mode.output_schema is ScoringOutput

    def test_image_tag(self) -> None:
        assert get_tool("stabddg").image_tag == "stabddg:1.0.0"

    def test_timeout(self) -> None:
        assert get_tool("stabddg").modes["predict"].default_timeout == 600

    def test_get_runner_returns_stabddg_runner(self, config: AutobioConfig) -> None:
        with (
            patch("autobio.tools.base.ContainerManager"),
            patch("autobio.tools.base.GPUManager"),
        ):
            r = get_runner("stabddg", config)
        assert isinstance(r, StaBddGRunner)
        assert r.tool_name == "stabddg"

    def test_tool_constant_registered(self) -> None:
        import autobio.tools  # noqa: F401
        from autobio.tools.stabddg import STABDDG_TOOL

        assert STABDDG_TOOL.name == "stabddg"
        assert get_tool("stabddg") is STABDDG_TOOL


def test_info_snapshot_stabddg() -> None:
    import autobio.tools  # noqa: F401
    from autobio.cli.formatters import OutputFormat, format_tool_info_catalog

    parsed = json.loads(format_tool_info_catalog(get_tool("stabddg"), OutputFormat.JSON))
    assert [m["name"] for m in parsed["modes"]] == ["predict"]
    predict = parsed["modes"][0]
    struct = predict["input_schema"]["properties"]["structure_path"]
    assert struct["x-autobio"]["widget"] == "file"
    assert "output_schema" in predict
