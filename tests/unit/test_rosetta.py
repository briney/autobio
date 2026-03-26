"""Tests for RosettaRunner — prepare_workspace, parse_output, and registration."""

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
from autobio.tools.rosetta import _VARIANT_CONFIG, RosettaRunner

if TYPE_CHECKING:
    from pathlib import Path


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def config() -> AutobioConfig:
    return AutobioConfig.resolve()


@pytest.fixture()
def runner(config: AutobioConfig) -> RosettaRunner:
    """Create a RosettaRunner for rosetta_score with mocked deps."""
    with (
        patch("autobio.tools.base.ContainerManager"),
        patch("autobio.tools.base.GPUManager"),
    ):
        return RosettaRunner("rosetta_score", config)


@pytest.fixture()
def relax_runner(config: AutobioConfig) -> RosettaRunner:
    """Create a RosettaRunner for rosetta_relax."""
    with (
        patch("autobio.tools.base.ContainerManager"),
        patch("autobio.tools.base.GPUManager"),
    ):
        return RosettaRunner("rosetta_relax", config)


@pytest.fixture()
def minimize_runner(config: AutobioConfig) -> RosettaRunner:
    """Create a RosettaRunner for rosetta_minimize."""
    with (
        patch("autobio.tools.base.ContainerManager"),
        patch("autobio.tools.base.GPUManager"),
    ):
        return RosettaRunner("rosetta_minimize", config)


@pytest.fixture()
def flexddg_runner(config: AutobioConfig) -> RosettaRunner:
    """Create a RosettaRunner for rosetta_flexddg."""
    with (
        patch("autobio.tools.base.ContainerManager"),
        patch("autobio.tools.base.GPUManager"),
    ):
        return RosettaRunner("rosetta_flexddg", config)


@pytest.fixture()
def sample_pdb(tmp_path: Path) -> Path:
    """Write a minimal PDB file for testing."""
    pdb_path = tmp_path / "test.pdb"
    pdb_path.write_text("ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00\nEND\n")
    return pdb_path


# ---------------------------------------------------------------------------
# TestRosettaPrepareWorkspace
# ---------------------------------------------------------------------------

_ALL_TOOL_NAMES = list(_VARIANT_CONFIG.keys())
_NON_DDG_TOOLS = [t for t in _ALL_TOOL_NAMES if not _VARIANT_CONFIG[t]["requires_mutations"]]
_DDG_TOOLS = [t for t in _ALL_TOOL_NAMES if _VARIANT_CONFIG[t]["requires_mutations"]]


class TestRosettaPrepareWorkspace:
    """Tests for RosettaRunner.prepare_workspace."""

    @pytest.mark.parametrize("tool_name", _NON_DDG_TOOLS)
    def test_basic_config(
        self,
        tool_name: str,
        config: AutobioConfig,
        tmp_path: Path,
        sample_pdb: Path,
    ) -> None:
        """Config contains correct binary, protocol, and database path."""
        with (
            patch("autobio.tools.base.ContainerManager"),
            patch("autobio.tools.base.GPUManager"),
        ):
            r = RosettaRunner(tool_name, config)
        workspace = Workspace.create(tmp_path / "ws")
        input_data = ScoringInput(structure_path=sample_pdb)
        r.prepare_workspace(input_data, workspace)

        cfg = json.loads(workspace.config_path.read_text())
        expected = _VARIANT_CONFIG[tool_name]
        assert cfg["binary"] == expected["binary"]
        assert cfg["protocol"] == expected["protocol"]
        assert cfg["database_path"] == "/usr/local/lib/python3.8/dist-packages/pyrosetta/database"
        assert cfg["score_function"] == "ref2015"

    def test_structure_file_copied(
        self, runner: RosettaRunner, tmp_path: Path, sample_pdb: Path
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

    def test_score_function_override(
        self, runner: RosettaRunner, tmp_path: Path, sample_pdb: Path
    ) -> None:
        """Custom score function from extra dict is used."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = ScoringInput(structure_path=sample_pdb, extra={"score_function": "beta_nov16"})
        runner.prepare_workspace(input_data, workspace)

        cfg = json.loads(workspace.config_path.read_text())
        assert cfg["score_function"] == "beta_nov16"

    def test_nstruct_from_extra(
        self, relax_runner: RosettaRunner, tmp_path: Path, sample_pdb: Path
    ) -> None:
        """nstruct from extra overrides variant default."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = ScoringInput(structure_path=sample_pdb, extra={"nstruct": 10})
        relax_runner.prepare_workspace(input_data, workspace)

        cfg = json.loads(workspace.config_path.read_text())
        assert cfg["nstruct"] == 10

    def test_nstruct_defaults(
        self,
        config: AutobioConfig,
        tmp_path: Path,
        sample_pdb: Path,
    ) -> None:
        """Each variant uses its own default nstruct."""
        for tool_name in _NON_DDG_TOOLS:
            with (
                patch("autobio.tools.base.ContainerManager"),
                patch("autobio.tools.base.GPUManager"),
            ):
                r = RosettaRunner(tool_name, config)
            workspace = Workspace.create(tmp_path / f"ws_{tool_name}")
            input_data = ScoringInput(structure_path=sample_pdb)
            r.prepare_workspace(input_data, workspace)

            cfg = json.loads(workspace.config_path.read_text())
            assert cfg["nstruct"] == _VARIANT_CONFIG[tool_name]["default_nstruct"]

    def test_xml_path_for_script_tools(
        self, relax_runner: RosettaRunner, tmp_path: Path, sample_pdb: Path
    ) -> None:
        """Tools using rosetta_scripts include xml_path in config."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = ScoringInput(structure_path=sample_pdb)
        relax_runner.prepare_workspace(input_data, workspace)

        cfg = json.loads(workspace.config_path.read_text())
        assert cfg["xml_path"] == "/opt/tool/xml/relax.xml"

    def test_no_xml_path_for_score(
        self, runner: RosettaRunner, tmp_path: Path, sample_pdb: Path
    ) -> None:
        """Score tool does not include xml_path in config."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = ScoringInput(structure_path=sample_pdb)
        runner.prepare_workspace(input_data, workspace)

        cfg = json.loads(workspace.config_path.read_text())
        assert "xml_path" not in cfg

    def test_extra_dict_merged(
        self, runner: RosettaRunner, tmp_path: Path, sample_pdb: Path
    ) -> None:
        """Extra dict keys (non-consumed) appear at top level of config.json."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = ScoringInput(
            structure_path=sample_pdb,
            extra={"ex1": True, "ex2": True, "custom_flag": "value"},
        )
        runner.prepare_workspace(input_data, workspace)

        cfg = json.loads(workspace.config_path.read_text())
        assert cfg["ex1"] is True
        assert cfg["ex2"] is True
        assert cfg["custom_flag"] == "value"

    def test_consumed_keys_not_merged(
        self, runner: RosettaRunner, tmp_path: Path, sample_pdb: Path
    ) -> None:
        """Consumed extra keys don't leak into config.json top level."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = ScoringInput(
            structure_path=sample_pdb,
            extra={"nstruct": 3, "score_function": "ref2015", "ex1": True},
        )
        runner.prepare_workspace(input_data, workspace)

        cfg = json.loads(workspace.config_path.read_text())
        # nstruct and score_function are consumed but placed explicitly
        assert cfg["nstruct"] == 3
        assert cfg["score_function"] == "ref2015"
        # ex1 is NOT consumed — it should be merged
        assert cfg["ex1"] is True


# ---------------------------------------------------------------------------
# TestRosettaDDGWorkspace
# ---------------------------------------------------------------------------


class TestRosettaDDGWorkspace:
    """Tests for DDG-specific prepare_workspace logic."""

    def test_flexddg_with_chains_to_move(
        self, flexddg_runner: RosettaRunner, tmp_path: Path, sample_pdb: Path
    ) -> None:
        """Flex-ddG accepts mutations and chains_to_move."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = ScoringInput(
            structure_path=sample_pdb,
            extra={"mutations": ["A42F"], "chains_to_move": "B"},
        )
        flexddg_runner.prepare_workspace(input_data, workspace)

        cfg = json.loads(workspace.config_path.read_text())
        assert cfg["mutations"] == ["A42F"]
        assert cfg["chains_to_move"] == "B"

    def test_resfile_passthrough(
        self, flexddg_runner: RosettaRunner, tmp_path: Path, sample_pdb: Path
    ) -> None:
        """Raw resfile content is written to inputs/ when provided."""
        workspace = Workspace.create(tmp_path / "ws")
        resfile_content = "NATAA\nstart\n42 A PIKAA F\n"
        input_data = ScoringInput(
            structure_path=sample_pdb,
            extra={
                "mutations": ["A42F"],
                "chains_to_move": "B",
                "resfile": resfile_content,
            },
        )
        flexddg_runner.prepare_workspace(input_data, workspace)

        resfile_path = workspace.inputs_dir / "mutations.resfile"
        assert resfile_path.exists()
        assert resfile_path.read_text() == resfile_content

        cfg = json.loads(workspace.config_path.read_text())
        assert cfg["resfile_path"] == "/workspace/inputs/mutations.resfile"


# ---------------------------------------------------------------------------
# TestRosettaValidation
# ---------------------------------------------------------------------------


class TestRosettaValidation:
    """Tests for host-side input validation."""

    def test_missing_structure_raises(self, runner: RosettaRunner, tmp_path: Path) -> None:
        """Missing structure file raises AutobioError."""
        workspace = Workspace.create(tmp_path / "ws")
        fake_path = tmp_path / "nonexistent.pdb"
        input_data = ScoringInput(structure_path=fake_path)
        with pytest.raises(AutobioError, match="does not exist"):
            runner.prepare_workspace(input_data, workspace)

    def test_flexddg_missing_mutations_raises(
        self, flexddg_runner: RosettaRunner, tmp_path: Path, sample_pdb: Path
    ) -> None:
        """Flex-ddG without mutations raises AutobioError."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = ScoringInput(structure_path=sample_pdb, extra={"chains_to_move": "B"})
        with pytest.raises(AutobioError, match="requires 'mutations'"):
            flexddg_runner.prepare_workspace(input_data, workspace)

    def test_flexddg_invalid_mutations_type_raises(
        self, flexddg_runner: RosettaRunner, tmp_path: Path, sample_pdb: Path
    ) -> None:
        """Non-list mutations raises AutobioError."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = ScoringInput(
            structure_path=sample_pdb,
            extra={"mutations": "A42F", "chains_to_move": "B"},
        )
        with pytest.raises(AutobioError, match="must be a list"):
            flexddg_runner.prepare_workspace(input_data, workspace)

    def test_flexddg_missing_chains_raises(
        self, flexddg_runner: RosettaRunner, tmp_path: Path, sample_pdb: Path
    ) -> None:
        """Flex-ddG without chains_to_move raises AutobioError."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = ScoringInput(structure_path=sample_pdb, extra={"mutations": ["A42F"]})
        with pytest.raises(AutobioError, match="chains_to_move"):
            flexddg_runner.prepare_workspace(input_data, workspace)


# ---------------------------------------------------------------------------
# TestRosettaParseOutput
# ---------------------------------------------------------------------------

_SCORE_RESULT = {
    "scores": [
        {
            "total_score": -198.432,
            "score_breakdown": {"fa_atr": -320.12, "fa_rep": 45.67, "fa_sol": 189.32},
            "units": "REU",
            "per_residue_scores": None,
            "structure_path": None,
            "ddg": None,
            "mutations": None,
        }
    ]
}

_RELAX_RESULT = {
    "scores": [
        {
            "total_score": -205.1,
            "score_breakdown": {"fa_atr": -325.0, "fa_rep": 40.0, "fa_sol": 180.0},
            "units": "REU",
            "per_residue_scores": None,
            "structure_path": "/workspace/outputs/standardized/relaxed_0001.pdb",
            "ddg": None,
            "mutations": None,
        }
    ]
}

_DDG_RESULT = {
    "scores": [
        {
            "total_score": 2.5,
            "score_breakdown": {"wt_score": -200.0, "mut_score": -197.5},
            "units": "REU",
            "per_residue_scores": None,
            "structure_path": None,
            "ddg": 2.5,
            "mutations": ["A42F"],
        }
    ]
}


class TestRosettaParseOutput:
    """Tests for RosettaRunner.parse_output."""

    def test_parse_score_output(self, runner: RosettaRunner, tmp_path: Path) -> None:
        workspace = Workspace.create(tmp_path / "ws")
        (workspace.std_output_dir / "result_data.json").write_text(json.dumps(_SCORE_RESULT))

        output = runner.parse_output(workspace)
        assert isinstance(output, ScoringOutput)
        assert len(output.scores) == 1

        s = output.scores[0]
        assert s.total_score == pytest.approx(-198.432)
        assert s.units == "REU"
        assert s.score_breakdown is not None
        assert s.score_breakdown["fa_atr"] == pytest.approx(-320.12)
        assert s.structure_path is None
        assert s.ddg is None

    def test_parse_relax_output_with_structure(
        self, relax_runner: RosettaRunner, tmp_path: Path
    ) -> None:
        workspace = Workspace.create(tmp_path / "ws")
        # Create the expected output PDB
        pdb_dest = workspace.std_output_dir / "relaxed_0001.pdb"
        pdb_dest.write_text("ATOM  relaxed\nEND\n")
        (workspace.std_output_dir / "result_data.json").write_text(json.dumps(_RELAX_RESULT))

        output = relax_runner.parse_output(workspace)
        s = output.scores[0]
        assert s.structure_path is not None
        assert s.structure_path.name == "relaxed_0001.pdb"

    def test_parse_ddg_output(self, flexddg_runner: RosettaRunner, tmp_path: Path) -> None:
        workspace = Workspace.create(tmp_path / "ws")
        (workspace.std_output_dir / "result_data.json").write_text(json.dumps(_DDG_RESULT))

        output = flexddg_runner.parse_output(workspace)
        s = output.scores[0]
        assert s.ddg == pytest.approx(2.5)
        assert s.mutations == ["A42F"]

    def test_output_type(self, runner: RosettaRunner, tmp_path: Path) -> None:
        workspace = Workspace.create(tmp_path / "ws")
        (workspace.std_output_dir / "result_data.json").write_text(json.dumps(_SCORE_RESULT))
        output = runner.parse_output(workspace)
        assert isinstance(output, ScoringOutput)

    def test_raw_output_path(self, runner: RosettaRunner, tmp_path: Path) -> None:
        workspace = Workspace.create(tmp_path / "ws")
        (workspace.std_output_dir / "result_data.json").write_text(json.dumps(_SCORE_RESULT))
        output = runner.parse_output(workspace)
        assert output.raw_output_path == workspace.raw_output_dir


# ---------------------------------------------------------------------------
# TestRosettaRegistration
# ---------------------------------------------------------------------------


class TestRosettaRegistration:
    """Tests for tool and runner registration."""

    @pytest.mark.parametrize("tool_name", _ALL_TOOL_NAMES)
    def test_in_tool_registry(self, tool_name: str) -> None:
        assert tool_name in TOOL_REGISTRY

    @pytest.mark.parametrize("tool_name", _ALL_TOOL_NAMES)
    def test_in_tool_runners(self, tool_name: str) -> None:
        assert tool_name in TOOL_RUNNERS
        assert TOOL_RUNNERS[tool_name] is RosettaRunner

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

    def test_each_tool_has_unique_image_tag(self) -> None:
        """Unlike Complexa, each Rosetta tool has its own image."""
        tags = {TOOL_REGISTRY[t].image_tag for t in _ALL_TOOL_NAMES}
        assert len(tags) == len(_ALL_TOOL_NAMES)

    def test_get_runner_returns_rosetta_runner(self, config: AutobioConfig) -> None:
        with (
            patch("autobio.tools.base.ContainerManager"),
            patch("autobio.tools.base.GPUManager"),
        ):
            r = get_runner("rosetta_score", config)
        assert isinstance(r, RosettaRunner)
        assert r.tool_name == "rosetta_score"
