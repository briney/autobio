"""Unit tests for the migrated baddg Tool (mode: predict)."""

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
from autobio.schemas.scoring import BAddGInput, ScoringOutput
from autobio.tools import TOOL_RUNNERS, get_runner
from autobio.tools.baddg import BAddGRunner

if TYPE_CHECKING:
    from pathlib import Path


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def config() -> AutobioConfig:
    return AutobioConfig.resolve()


def _make_runner() -> BAddGRunner:
    """Create a BAddGRunner with mocked deps, current_mode set to 'predict'."""
    with (
        patch("autobio.tools.base.ContainerManager"),
        patch("autobio.tools.base.GPUManager"),
    ):
        runner = BAddGRunner("baddg", AutobioConfig.resolve())
    runner.current_mode = get_tool("baddg").modes["predict"]
    return runner


@pytest.fixture()
def runner() -> BAddGRunner:
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


def _written_config(runner: BAddGRunner, input_data: BAddGInput, tmp_path: Path) -> dict:
    ws = Workspace.create(tmp_path / "ws")
    try:
        runner.prepare_workspace(input_data, ws)
        return json.loads((ws.root / "config.json").read_text())
    finally:
        ws.cleanup()


# ---------------------------------------------------------------------------
# TestBAddGPrepareWorkspace
# ---------------------------------------------------------------------------


class TestBAddGPrepareWorkspace:
    """Tests for BAddGRunner.prepare_workspace."""

    def test_basic_config(self, runner: BAddGRunner, tmp_path: Path, sample_pdb: Path) -> None:
        """Config contains correct fields for BA-ddG."""
        input_data = BAddGInput(
            structure_path=sample_pdb,
            mutations=["EA63Q"],
            chains="A_B",
        )
        cfg = _written_config(runner, input_data, tmp_path)

        assert cfg["pdb_path"] == "/workspace/inputs/complex.pdb"
        assert cfg["mutations"] == "EA63Q"
        assert cfg["chains"] == "A_B"
        assert cfg["mpnn_checkpoint_path"] == "/app/baddg/ckpt/soluble_model_weights/v_48_020.pt"
        assert cfg["ddg_checkpoint_path"] == "/app/baddg/ckpt/ddg_model.ckpt"
        assert cfg["output_dir"] == "/workspace/outputs/raw"

    def test_structure_file_copied(
        self, runner: BAddGRunner, tmp_path: Path, sample_pdb: Path
    ) -> None:
        """Input structure is copied to inputs/ and config references container path."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = BAddGInput(
            structure_path=sample_pdb,
            mutations=["EA63Q"],
            chains="A_B",
        )
        runner.prepare_workspace(input_data, workspace)

        copied = workspace.inputs_dir / "complex.pdb"
        assert copied.exists()
        assert copied.read_text() == sample_pdb.read_text()

    def test_mutations_comma_joined(
        self, runner: BAddGRunner, tmp_path: Path, sample_pdb: Path
    ) -> None:
        """Multiple mutations are joined into comma-separated string."""
        input_data = BAddGInput(
            structure_path=sample_pdb,
            mutations=["YH103H", "QD30V", "KA66A"],
            chains="ABC_DE",
        )
        cfg = _written_config(runner, input_data, tmp_path)

        assert cfg["mutations"] == "YH103H,QD30V,KA66A"

    def test_chains_passthrough(
        self, runner: BAddGRunner, tmp_path: Path, sample_pdb: Path
    ) -> None:
        """Chain specification is passed through as-is."""
        input_data = BAddGInput(
            structure_path=sample_pdb,
            mutations=["EA63Q"],
            chains="ABC_DE",
        )
        cfg = _written_config(runner, input_data, tmp_path)

        assert cfg["chains"] == "ABC_DE"

    def test_default_params(self, runner: BAddGRunner, tmp_path: Path, sample_pdb: Path) -> None:
        """Default parameter values are written to config."""
        input_data = BAddGInput(
            structure_path=sample_pdb,
            mutations=["EA63Q"],
            chains="A_B",
        )
        cfg = _written_config(runner, input_data, tmp_path)

        assert cfg["n_folds"] == 3
        assert cfg["seed"] == 0
        assert cfg["device"] == "auto"

    def test_override_params(self, runner: BAddGRunner, tmp_path: Path, sample_pdb: Path) -> None:
        """Non-default typed field values override defaults."""
        input_data = BAddGInput(
            structure_path=sample_pdb,
            mutations=["EA63Q"],
            chains="A_B",
            n_folds=1,
            seed=42,
            device="cpu",
        )
        cfg = _written_config(runner, input_data, tmp_path)

        assert cfg["n_folds"] == 1
        assert cfg["seed"] == 42
        assert cfg["device"] == "cpu"

    def test_extra_dict_merged(self, runner: BAddGRunner, tmp_path: Path, sample_pdb: Path) -> None:
        """Non-typed extra dict keys appear at top level of config.json."""
        input_data = BAddGInput(
            structure_path=sample_pdb,
            mutations=["EA63Q"],
            chains="A_B",
            extra={"custom_flag": "value"},
        )
        cfg = _written_config(runner, input_data, tmp_path)

        assert cfg["custom_flag"] == "value"

    def test_extra_shadowing_typed_field_rejected(
        self, runner: BAddGRunner, tmp_path: Path, sample_pdb: Path
    ) -> None:
        """extra keys colliding with typed fields are rejected fail-fast."""
        input_data = BAddGInput(
            structure_path=sample_pdb,
            mutations=["EA63Q"],
            chains="A_B",
            extra={"n_folds": 2},
        )
        with pytest.raises(AutobioError, match="collide with typed input fields"):
            _written_config(runner, input_data, tmp_path)

    def test_two_checkpoint_paths(
        self, runner: BAddGRunner, tmp_path: Path, sample_pdb: Path
    ) -> None:
        """Config has both MPNN backbone and fine-tuned DDG checkpoint paths."""
        input_data = BAddGInput(
            structure_path=sample_pdb,
            mutations=["EA63Q"],
            chains="A_B",
        )
        cfg = _written_config(runner, input_data, tmp_path)

        assert "mpnn_checkpoint_path" in cfg
        assert "ddg_checkpoint_path" in cfg
        assert cfg["mpnn_checkpoint_path"] != cfg["ddg_checkpoint_path"]

    def test_full_config_byte_compat(
        self, runner: BAddGRunner, tmp_path: Path, sample_pdb: Path
    ) -> None:
        """Full config.json equality — byte-compat contract, incl. checkpoint paths."""
        input_data = BAddGInput(
            structure_path=sample_pdb,
            mutations=["YH103H", "QD30V"],
            chains="ABC_DE",
            n_folds=1,
            seed=42,
            device="cpu",
        )
        cfg = _written_config(runner, input_data, tmp_path)

        assert cfg == {
            "pdb_path": "/workspace/inputs/complex.pdb",
            "mutations": "YH103H,QD30V",
            "chains": "ABC_DE",
            "mpnn_checkpoint_path": "/app/baddg/ckpt/soluble_model_weights/v_48_020.pt",
            "ddg_checkpoint_path": "/app/baddg/ckpt/ddg_model.ckpt",
            "output_dir": "/workspace/outputs/raw",
            "n_folds": 1,
            "seed": 42,
            "device": "cpu",
        }


# ---------------------------------------------------------------------------
# TestBAddGValidation
# ---------------------------------------------------------------------------


class TestBAddGValidation:
    """Tests for host-side input validation."""

    def test_missing_structure_raises(self, runner: BAddGRunner, tmp_path: Path) -> None:
        """Missing structure file raises AutobioError."""
        workspace = Workspace.create(tmp_path / "ws")
        fake_path = tmp_path / "nonexistent.pdb"
        input_data = BAddGInput(
            structure_path=fake_path,
            mutations=["EA63Q"],
            chains="A_B",
        )
        with pytest.raises(AutobioError, match="does not exist"):
            runner.prepare_workspace(input_data, workspace)

    def test_empty_mutations_raises(
        self, runner: BAddGRunner, tmp_path: Path, sample_pdb: Path
    ) -> None:
        """Empty mutations list raises AutobioError."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = BAddGInput(
            structure_path=sample_pdb,
            mutations=[],
            chains="A_B",
        )
        with pytest.raises(AutobioError, match="requires at least one mutation"):
            runner.prepare_workspace(input_data, workspace)

    def test_missing_chains_raises(
        self, runner: BAddGRunner, tmp_path: Path, sample_pdb: Path
    ) -> None:
        """Empty chains raises AutobioError."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = BAddGInput(
            structure_path=sample_pdb,
            mutations=["EA63Q"],
            chains="",
        )
        with pytest.raises(AutobioError, match="requires a non-empty 'chains'"):
            runner.prepare_workspace(input_data, workspace)

    def test_invalid_chains_no_underscore_raises(
        self, runner: BAddGRunner, tmp_path: Path, sample_pdb: Path
    ) -> None:
        """Chains without underscore raises AutobioError."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = BAddGInput(
            structure_path=sample_pdb,
            mutations=["EA63Q"],
            chains="ABC",
        )
        with pytest.raises(AutobioError, match="exactly one underscore"):
            runner.prepare_workspace(input_data, workspace)

    def test_invalid_chains_multiple_underscores_raises(
        self, runner: BAddGRunner, tmp_path: Path, sample_pdb: Path
    ) -> None:
        """Chains with multiple underscores raises AutobioError."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = BAddGInput(
            structure_path=sample_pdb,
            mutations=["EA63Q"],
            chains="A_B_C",
        )
        with pytest.raises(AutobioError, match="exactly one underscore"):
            runner.prepare_workspace(input_data, workspace)


# ---------------------------------------------------------------------------
# TestBAddGParseOutput
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

_MULTI_FOLD_RESULT = {
    "scores": [
        {
            "total_score": 1.35,
            "score_breakdown": {
                "chains": "A_B",
                "fold_values": {"fold_1": 1.23, "fold_2": 1.47, "fold_3": 1.35},
                "n_folds": 3,
            },
            "units": "kcal/mol",
            "per_residue_scores": None,
            "structure_path": None,
            "ddg": 1.35,
            "mutations": ["EA63Q"],
        }
    ]
}


class TestBAddGParseOutput:
    """Tests for BAddGRunner.parse_output."""

    def test_parse_ddg_output(self, runner: BAddGRunner, tmp_path: Path) -> None:
        """Standard result_data.json is deserialized correctly."""
        workspace = Workspace.create(tmp_path / "ws")
        (workspace.std_output_dir / "result_data.json").write_text(json.dumps(_DDG_RESULT))

        output = runner.parse_output(workspace)
        assert isinstance(output, ScoringOutput)
        assert len(output.scores) == 1

    def test_ddg_value(self, runner: BAddGRunner, tmp_path: Path) -> None:
        """DDG and total_score are correct."""
        workspace = Workspace.create(tmp_path / "ws")
        (workspace.std_output_dir / "result_data.json").write_text(json.dumps(_DDG_RESULT))

        output = runner.parse_output(workspace)
        s = output.scores[0]
        assert s.ddg == pytest.approx(1.23)
        assert s.total_score == pytest.approx(1.23)

    def test_units_kcal_mol(self, runner: BAddGRunner, tmp_path: Path) -> None:
        """Units are kcal/mol."""
        workspace = Workspace.create(tmp_path / "ws")
        (workspace.std_output_dir / "result_data.json").write_text(json.dumps(_DDG_RESULT))

        output = runner.parse_output(workspace)
        assert output.scores[0].units == "kcal/mol"

    def test_mutations_in_output(self, runner: BAddGRunner, tmp_path: Path) -> None:
        """Mutations list is preserved."""
        workspace = Workspace.create(tmp_path / "ws")
        (workspace.std_output_dir / "result_data.json").write_text(json.dumps(_DDG_RESULT))

        output = runner.parse_output(workspace)
        assert output.scores[0].mutations == ["EA63Q"]

    def test_no_structure_path(self, runner: BAddGRunner, tmp_path: Path) -> None:
        """BA-ddG does not produce output structures."""
        workspace = Workspace.create(tmp_path / "ws")
        (workspace.std_output_dir / "result_data.json").write_text(json.dumps(_DDG_RESULT))

        output = runner.parse_output(workspace)
        assert output.scores[0].structure_path is None

    def test_output_type(self, runner: BAddGRunner, tmp_path: Path) -> None:
        """Returns ScoringOutput."""
        workspace = Workspace.create(tmp_path / "ws")
        (workspace.std_output_dir / "result_data.json").write_text(json.dumps(_DDG_RESULT))
        output = runner.parse_output(workspace)
        assert isinstance(output, ScoringOutput)

    def test_raw_output_path(self, runner: BAddGRunner, tmp_path: Path) -> None:
        """raw_output_path points to the raw output directory."""
        workspace = Workspace.create(tmp_path / "ws")
        (workspace.std_output_dir / "result_data.json").write_text(json.dumps(_DDG_RESULT))
        output = runner.parse_output(workspace)
        assert output.raw_output_path == workspace.raw_output_dir

    def test_multi_fold_output(self, runner: BAddGRunner, tmp_path: Path) -> None:
        """Multi-fold results include fold breakdown."""
        workspace = Workspace.create(tmp_path / "ws")
        (workspace.std_output_dir / "result_data.json").write_text(json.dumps(_MULTI_FOLD_RESULT))

        output = runner.parse_output(workspace)
        s = output.scores[0]
        assert s.ddg == pytest.approx(1.35)
        assert s.score_breakdown is not None
        assert s.score_breakdown["n_folds"] == 3


# ---------------------------------------------------------------------------
# TestBAddGRegistration
# ---------------------------------------------------------------------------


class TestBAddGRegistration:
    """Tests for tool and runner registration in the catalog."""

    def test_registered_as_catalog_tool(self) -> None:
        import autobio.tools  # noqa: F401
        from autobio.core.catalog import CATALOG
        from autobio.core.registry import TOOL_REGISTRY

        assert "baddg" in CATALOG
        assert set(get_tool("baddg").modes) == {"predict"}
        assert get_tool("baddg").default_mode == "predict"
        assert "baddg" not in TOOL_REGISTRY

    def test_in_tool_runners(self) -> None:
        assert "baddg" in TOOL_RUNNERS
        assert TOOL_RUNNERS["baddg"] is BAddGRunner

    def test_scoring_category(self) -> None:
        assert get_tool("baddg").category == ToolCategory.SCORING

    def test_gpu_required(self) -> None:
        tool = get_tool("baddg")
        assert tool.requires_gpu is True
        assert tool.gpu_count == 1

    def test_schema_types(self) -> None:
        mode = get_tool("baddg").modes["predict"]
        assert mode.input_schema is BAddGInput
        assert mode.output_schema is ScoringOutput

    def test_image_tag(self) -> None:
        assert get_tool("baddg").image_tag == "baddg:1.0.0"

    def test_get_runner_returns_baddg_runner(self, config: AutobioConfig) -> None:
        with (
            patch("autobio.tools.base.ContainerManager"),
            patch("autobio.tools.base.GPUManager"),
        ):
            r = get_runner("baddg", config)
        assert isinstance(r, BAddGRunner)
        assert r.tool_name == "baddg"

    def test_tool_constant_registered(self) -> None:
        import autobio.tools  # noqa: F401
        from autobio.tools.baddg import BADDG_TOOL

        assert BADDG_TOOL.name == "baddg"
        assert get_tool("baddg") is BADDG_TOOL


def test_info_snapshot_baddg() -> None:
    import autobio.tools  # noqa: F401
    from autobio.cli.formatters import OutputFormat, format_tool_info_catalog

    parsed = json.loads(format_tool_info_catalog(get_tool("baddg"), OutputFormat.JSON))
    assert [m["name"] for m in parsed["modes"]] == ["predict"]
    predict = parsed["modes"][0]
    struct = predict["input_schema"]["properties"]["structure_path"]
    assert struct["x-autobio"]["widget"] == "file"
    assert "output_schema" in predict
