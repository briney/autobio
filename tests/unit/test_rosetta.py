"""Tests for the migrated rosetta Tool (modes: score, relax, minimize, flexddg)."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

from autobio.core.catalog import get_tool
from autobio.core.config import AutobioConfig
from autobio.core.registry import ToolCategory
from autobio.core.result import AutobioError
from autobio.core.workspace import Workspace
from autobio.schemas.scoring import (
    RosettaBaseInput,
    RosettaFlexDdgInput,
    RosettaRelaxInput,
    ScoringOutput,
)
from autobio.tools import TOOL_RUNNERS, get_runner
from autobio.tools.rosetta import _MODE_CONFIG, _ROSETTA_DB, RosettaRunner

if TYPE_CHECKING:
    from pathlib import Path


_MODES = ("score", "relax", "minimize", "flexddg")
_NON_DDG_MODES = ("score", "relax", "minimize")

_OLD_FLAT_NAMES = ("rosetta_score", "rosetta_relax", "rosetta_minimize", "rosetta_flexddg")


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


@pytest.fixture()
def config() -> AutobioConfig:
    return AutobioConfig.resolve()


def _make_runner(mode_name: str, config: AutobioConfig) -> RosettaRunner:
    """Create a RosettaRunner with mocked deps, current_mode pinned to *mode_name*."""
    with (
        patch("autobio.tools.base.ContainerManager"),
        patch("autobio.tools.base.GPUManager"),
    ):
        runner = RosettaRunner("rosetta", config)
    runner.current_mode = get_tool("rosetta").modes[mode_name]
    return runner


def _input_for_mode(mode_name: str, structure_path: Path, **kwargs: object):
    """Build the correct typed input class for *mode_name*."""
    if mode_name == "relax":
        return RosettaRelaxInput(structure_path=structure_path, **kwargs)  # type: ignore[arg-type]
    if mode_name == "flexddg":
        kwargs.setdefault("mutations", ["A42F"])
        kwargs.setdefault("chains_to_move", "B")
        return RosettaFlexDdgInput(structure_path=structure_path, **kwargs)  # type: ignore[arg-type]
    return RosettaBaseInput(structure_path=structure_path, **kwargs)  # type: ignore[arg-type]


@pytest.fixture()
def runner(config: AutobioConfig) -> RosettaRunner:
    """Score-mode runner (the common case)."""
    return _make_runner("score", config)


@pytest.fixture()
def relax_runner(config: AutobioConfig) -> RosettaRunner:
    return _make_runner("relax", config)


@pytest.fixture()
def minimize_runner(config: AutobioConfig) -> RosettaRunner:
    return _make_runner("minimize", config)


@pytest.fixture()
def flexddg_runner(config: AutobioConfig) -> RosettaRunner:
    return _make_runner("flexddg", config)


@pytest.fixture()
def sample_pdb(tmp_path: Path) -> Path:
    """Write a minimal PDB file for testing."""
    pdb_path = tmp_path / "test.pdb"
    pdb_path.write_text("ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00\nEND\n")
    return pdb_path


# ---------------------------------------------------------------------------
# TestRosettaPrepareWorkspace
# ---------------------------------------------------------------------------


class TestRosettaPrepareWorkspace:
    """Tests for RosettaRunner.prepare_workspace."""

    @pytest.mark.parametrize("mode_name", _NON_DDG_MODES)
    def test_basic_config(
        self,
        mode_name: str,
        config: AutobioConfig,
        tmp_path: Path,
        sample_pdb: Path,
    ) -> None:
        """Config contains correct binary, protocol, and database path."""
        r = _make_runner(mode_name, config)
        workspace = Workspace.create(tmp_path / "ws")
        input_data = _input_for_mode(mode_name, sample_pdb)
        r.prepare_workspace(input_data, workspace)

        cfg = json.loads(workspace.config_path.read_text())
        expected = _MODE_CONFIG[mode_name]
        assert cfg["binary"] == expected["binary"]
        assert cfg["protocol"] == expected["protocol"]
        assert cfg["database_path"] == _ROSETTA_DB
        assert cfg["score_function"] == "ref2015"

    def test_structure_file_copied(
        self, runner: RosettaRunner, tmp_path: Path, sample_pdb: Path
    ) -> None:
        """Input structure is copied to inputs/ and config references container path."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = RosettaBaseInput(structure_path=sample_pdb)
        runner.prepare_workspace(input_data, workspace)

        copied = workspace.inputs_dir / "test.pdb"
        assert copied.exists()
        assert copied.read_text() == sample_pdb.read_text()

        cfg = json.loads(workspace.config_path.read_text())
        assert cfg["structure_path"] == "/workspace/inputs/test.pdb"

    def test_score_function_override(
        self, runner: RosettaRunner, tmp_path: Path, sample_pdb: Path
    ) -> None:
        """Custom score function (typed field) is used."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = RosettaBaseInput(structure_path=sample_pdb, score_function="beta_nov16")
        runner.prepare_workspace(input_data, workspace)

        cfg = json.loads(workspace.config_path.read_text())
        assert cfg["score_function"] == "beta_nov16"

    def test_nstruct_override(
        self, relax_runner: RosettaRunner, tmp_path: Path, sample_pdb: Path
    ) -> None:
        """nstruct (typed field) overrides the mode default."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = RosettaRelaxInput(structure_path=sample_pdb, nstruct=10)
        relax_runner.prepare_workspace(input_data, workspace)

        cfg = json.loads(workspace.config_path.read_text())
        assert cfg["nstruct"] == 10

    @pytest.mark.parametrize(
        ("mode_name", "expected_nstruct"),
        [("score", 1), ("relax", 5), ("minimize", 1), ("flexddg", 35)],
    )
    def test_nstruct_defaults(
        self,
        mode_name: str,
        expected_nstruct: int,
        config: AutobioConfig,
        tmp_path: Path,
        sample_pdb: Path,
    ) -> None:
        """Each mode uses its own default nstruct."""
        r = _make_runner(mode_name, config)
        workspace = Workspace.create(tmp_path / f"ws_{mode_name}")
        input_data = _input_for_mode(mode_name, sample_pdb)
        r.prepare_workspace(input_data, workspace)

        cfg = json.loads(workspace.config_path.read_text())
        assert cfg["nstruct"] == expected_nstruct

    def test_xml_path_for_script_modes(
        self, relax_runner: RosettaRunner, tmp_path: Path, sample_pdb: Path
    ) -> None:
        """Modes using rosetta_scripts include xml_path in config."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = RosettaRelaxInput(structure_path=sample_pdb)
        relax_runner.prepare_workspace(input_data, workspace)

        cfg = json.loads(workspace.config_path.read_text())
        assert cfg["xml_path"] == "/opt/tool/xml/relax.xml"

    def test_no_xml_path_for_score(
        self, runner: RosettaRunner, tmp_path: Path, sample_pdb: Path
    ) -> None:
        """Score mode does not include xml_path in config."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = RosettaBaseInput(structure_path=sample_pdb)
        runner.prepare_workspace(input_data, workspace)

        cfg = json.loads(workspace.config_path.read_text())
        assert "xml_path" not in cfg

    def test_extra_dict_merged(
        self, runner: RosettaRunner, tmp_path: Path, sample_pdb: Path
    ) -> None:
        """Extra dict keys (non-typed) appear at top level of config.json."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = RosettaBaseInput(
            structure_path=sample_pdb,
            extra={"ex1": True, "ex2": True, "custom_flag": "value"},
        )
        runner.prepare_workspace(input_data, workspace)

        cfg = json.loads(workspace.config_path.read_text())
        assert cfg["ex1"] is True
        assert cfg["ex2"] is True
        assert cfg["custom_flag"] == "value"

    def test_extra_shadowing_typed_field_rejected(
        self, runner: RosettaRunner, tmp_path: Path, sample_pdb: Path
    ) -> None:
        """extra containing a typed field name (score_function/nstruct) raises."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = RosettaBaseInput(
            structure_path=sample_pdb,
            extra={"score_function": "ref2015"},
        )
        with pytest.raises(AutobioError, match="collide with typed input fields"):
            runner.prepare_workspace(input_data, workspace)

    def test_extra_shadowing_config_key_rejected(
        self, runner: RosettaRunner, tmp_path: Path, sample_pdb: Path
    ) -> None:
        """extra containing a runner-derived config key (out_dir) raises."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = RosettaBaseInput(
            structure_path=sample_pdb,
            extra={"out_dir": "/somewhere/else"},
        )
        with pytest.raises(AutobioError, match="collide"):
            runner.prepare_workspace(input_data, workspace)


# ---------------------------------------------------------------------------
# TestRosettaDDGWorkspace
# ---------------------------------------------------------------------------


class TestRosettaDDGWorkspace:
    """Tests for flex-ddG-specific prepare_workspace logic."""

    def test_flexddg_with_chains_to_move(
        self, flexddg_runner: RosettaRunner, tmp_path: Path, sample_pdb: Path
    ) -> None:
        """Flex-ddG accepts mutations and chains_to_move (typed fields)."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = RosettaFlexDdgInput(
            structure_path=sample_pdb, mutations=["A42F"], chains_to_move="B"
        )
        flexddg_runner.prepare_workspace(input_data, workspace)

        cfg = json.loads(workspace.config_path.read_text())
        assert cfg["mutations"] == ["A42F"]
        assert cfg["chains_to_move"] == "B"
        assert cfg["mutation_list"] == ["A42F"]
        assert "resfile_path" not in cfg

    def test_resfile_passthrough(
        self, flexddg_runner: RosettaRunner, tmp_path: Path, sample_pdb: Path
    ) -> None:
        """Raw resfile content is written to inputs/ when provided."""
        workspace = Workspace.create(tmp_path / "ws")
        resfile_content = "NATAA\nstart\n42 A PIKAA F\n"
        input_data = RosettaFlexDdgInput(
            structure_path=sample_pdb,
            mutations=["A42F"],
            chains_to_move="B",
            resfile=resfile_content,
        )
        flexddg_runner.prepare_workspace(input_data, workspace)

        resfile_path = workspace.inputs_dir / "mutations.resfile"
        assert resfile_path.exists()
        assert resfile_path.read_text() == resfile_content

        cfg = json.loads(workspace.config_path.read_text())
        assert cfg["resfile_path"] == "/workspace/inputs/mutations.resfile"
        assert "mutation_list" not in cfg

    def test_flexddg_extra_passthrough(
        self, flexddg_runner: RosettaRunner, tmp_path: Path, sample_pdb: Path
    ) -> None:
        """backrub_trials/max_minimization_iter remain extra-only pass-through keys."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = RosettaFlexDdgInput(
            structure_path=sample_pdb,
            mutations=["A42F"],
            chains_to_move="B",
            extra={"backrub_trials": 5000, "max_minimization_iter": 2000},
        )
        flexddg_runner.prepare_workspace(input_data, workspace)

        cfg = json.loads(workspace.config_path.read_text())
        assert cfg["backrub_trials"] == 5000
        assert cfg["max_minimization_iter"] == 2000


# ---------------------------------------------------------------------------
# TestRosettaValidation
# ---------------------------------------------------------------------------


class TestRosettaValidation:
    """Tests for host-side input validation."""

    def test_missing_structure_raises(self, runner: RosettaRunner, tmp_path: Path) -> None:
        """Missing structure file raises AutobioError."""
        workspace = Workspace.create(tmp_path / "ws")
        fake_path = tmp_path / "nonexistent.pdb"
        input_data = RosettaBaseInput(structure_path=fake_path)
        with pytest.raises(AutobioError, match="does not exist"):
            runner.prepare_workspace(input_data, workspace)

    def test_flexddg_missing_mutations_raises(
        self, flexddg_runner: RosettaRunner, tmp_path: Path, sample_pdb: Path
    ) -> None:
        """Flex-ddG with an empty mutations list raises AutobioError."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = RosettaFlexDdgInput(
            structure_path=sample_pdb, mutations=[], chains_to_move="B"
        )
        with pytest.raises(AutobioError, match="requires at least one mutation"):
            flexddg_runner.prepare_workspace(input_data, workspace)

    def test_flexddg_missing_chains_raises(
        self, flexddg_runner: RosettaRunner, tmp_path: Path, sample_pdb: Path
    ) -> None:
        """Flex-ddG with an empty chains_to_move raises AutobioError."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = RosettaFlexDdgInput(
            structure_path=sample_pdb, mutations=["A42F"], chains_to_move=""
        )
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
# TestRosettaByteCompatConfig — full-dict config.json equality, per mode
# ---------------------------------------------------------------------------


class TestRosettaByteCompatConfig:
    """Full-dict ``config.json`` equality tests, pinning key order per mode."""

    def test_score_full_config(
        self, config: AutobioConfig, tmp_path: Path, sample_pdb: Path
    ) -> None:
        r = _make_runner("score", config)
        workspace = Workspace.create(tmp_path / "ws")
        r.prepare_workspace(RosettaBaseInput(structure_path=sample_pdb), workspace)

        cfg = json.loads(workspace.config_path.read_text())
        expected = {
            "binary": "score_jd2",
            "protocol": "score",
            "structure_path": f"/workspace/inputs/{sample_pdb.name}",
            "database_path": _ROSETTA_DB,
            "score_function": "ref2015",
            "out_dir": "/workspace/outputs/raw",
            "nstruct": 1,
        }
        assert cfg == expected
        assert list(cfg.keys()) == list(expected.keys())

    def test_relax_full_config(
        self, config: AutobioConfig, tmp_path: Path, sample_pdb: Path
    ) -> None:
        r = _make_runner("relax", config)
        workspace = Workspace.create(tmp_path / "ws")
        r.prepare_workspace(RosettaRelaxInput(structure_path=sample_pdb), workspace)

        cfg = json.loads(workspace.config_path.read_text())
        expected = {
            "binary": "rosetta_scripts",
            "protocol": "relax",
            "structure_path": f"/workspace/inputs/{sample_pdb.name}",
            "database_path": _ROSETTA_DB,
            "score_function": "ref2015",
            "out_dir": "/workspace/outputs/raw",
            "nstruct": 5,
            "xml_path": "/opt/tool/xml/relax.xml",
        }
        assert cfg == expected
        assert list(cfg.keys()) == list(expected.keys())

    def test_minimize_full_config(
        self, config: AutobioConfig, tmp_path: Path, sample_pdb: Path
    ) -> None:
        r = _make_runner("minimize", config)
        workspace = Workspace.create(tmp_path / "ws")
        r.prepare_workspace(RosettaBaseInput(structure_path=sample_pdb), workspace)

        cfg = json.loads(workspace.config_path.read_text())
        expected = {
            "binary": "rosetta_scripts",
            "protocol": "minimize",
            "structure_path": f"/workspace/inputs/{sample_pdb.name}",
            "database_path": _ROSETTA_DB,
            "score_function": "ref2015",
            "out_dir": "/workspace/outputs/raw",
            "nstruct": 1,
            "xml_path": "/opt/tool/xml/minimize.xml",
        }
        assert cfg == expected
        assert list(cfg.keys()) == list(expected.keys())

    def test_flexddg_full_config_mutation_list(
        self, config: AutobioConfig, tmp_path: Path, sample_pdb: Path
    ) -> None:
        """No resfile: config carries mutation_list (not resfile_path)."""
        r = _make_runner("flexddg", config)
        workspace = Workspace.create(tmp_path / "ws")
        r.prepare_workspace(
            RosettaFlexDdgInput(structure_path=sample_pdb, mutations=["A42F"], chains_to_move="B"),
            workspace,
        )

        cfg = json.loads(workspace.config_path.read_text())
        expected = {
            "binary": "rosetta_scripts",
            "protocol": "flexddg",
            "structure_path": f"/workspace/inputs/{sample_pdb.name}",
            "database_path": _ROSETTA_DB,
            "score_function": "ref2015",
            "out_dir": "/workspace/outputs/raw",
            "nstruct": 35,
            "mutations": ["A42F"],
            "chains_to_move": "B",
            "mutation_list": ["A42F"],
        }
        assert cfg == expected
        assert list(cfg.keys()) == list(expected.keys())

    def test_flexddg_full_config_resfile(
        self, config: AutobioConfig, tmp_path: Path, sample_pdb: Path
    ) -> None:
        """With resfile: config carries resfile_path (not mutation_list)."""
        r = _make_runner("flexddg", config)
        workspace = Workspace.create(tmp_path / "ws")
        resfile_content = "NATAA\nstart\n42 A PIKAA F\n"
        r.prepare_workspace(
            RosettaFlexDdgInput(
                structure_path=sample_pdb,
                mutations=["A42F"],
                chains_to_move="B",
                resfile=resfile_content,
            ),
            workspace,
        )

        cfg = json.loads(workspace.config_path.read_text())
        expected = {
            "binary": "rosetta_scripts",
            "protocol": "flexddg",
            "structure_path": f"/workspace/inputs/{sample_pdb.name}",
            "database_path": _ROSETTA_DB,
            "score_function": "ref2015",
            "out_dir": "/workspace/outputs/raw",
            "nstruct": 35,
            "mutations": ["A42F"],
            "chains_to_move": "B",
            "resfile_path": "/workspace/inputs/mutations.resfile",
        }
        assert cfg == expected
        assert list(cfg.keys()) == list(expected.keys())


# ---------------------------------------------------------------------------
# TestRosettaRegistration
# ---------------------------------------------------------------------------


class TestRosettaRegistration:
    """Tests for the catalog Tool + runner registration."""

    def test_rosetta_registered_as_single_tool(self) -> None:
        import autobio.tools  # noqa: F401 - populate registries

        tool = get_tool("rosetta")
        assert set(tool.modes) == {"score", "relax", "minimize", "flexddg"}
        assert tool.default_mode == "score"
        assert tool.category == ToolCategory.SCORING
        assert tool.requires_gpu is False
        assert tool.gpu_count == 0

    @pytest.mark.parametrize("flat_name", _OLD_FLAT_NAMES)
    def test_old_flat_names_absent_from_tool_runners(self, flat_name: str) -> None:
        import autobio.tools  # noqa: F401

        assert flat_name not in TOOL_RUNNERS

    def test_rosetta_in_tool_runners(self) -> None:
        import autobio.tools  # noqa: F401

        assert "rosetta" in TOOL_RUNNERS
        assert TOOL_RUNNERS["rosetta"] is RosettaRunner

    def test_modes_have_distinct_image_tags(self) -> None:
        """Each of the 4 modes resolves to its own Docker image."""
        tool = get_tool("rosetta")
        tags = {mode.image_tag for mode in tool.modes.values()}
        assert len(tags) == 4
        assert tags == {
            "rosetta-score:1.0.0",
            "rosetta-relax:1.0.0",
            "rosetta-minimize:1.0.0",
            "rosetta-flexddg:1.0.0",
        }

    @pytest.mark.parametrize(
        ("mode_name", "timeout"),
        [("score", 600), ("relax", 3600), ("minimize", 1800), ("flexddg", 14400)],
    )
    def test_modes_have_per_mode_timeout(self, mode_name: str, timeout: int) -> None:
        assert get_tool("rosetta").modes[mode_name].default_timeout == timeout

    def test_get_runner_returns_rosetta_runner(self, config: AutobioConfig) -> None:
        with (
            patch("autobio.tools.base.ContainerManager"),
            patch("autobio.tools.base.GPUManager"),
        ):
            r = get_runner("rosetta", config)
        assert isinstance(r, RosettaRunner)
        assert r.tool_name == "rosetta"

    def test_get_runner_removed_flat_name_raises(self, config: AutobioConfig) -> None:
        with pytest.raises(KeyError, match="rosetta_score"):
            get_runner("rosetta_score", config)


# ---------------------------------------------------------------------------
# TestRosettaInfoSnapshot
# ---------------------------------------------------------------------------


class TestRosettaInfoSnapshot:
    """``autobio info rosetta`` output — per-mode notes, hints, output_schema."""

    def test_info_snapshot(self) -> None:
        import autobio.tools  # noqa: F401
        from autobio.cli.formatters import OutputFormat, format_tool_info_catalog

        parsed = json.loads(format_tool_info_catalog(get_tool("rosetta"), OutputFormat.JSON))
        assert [m["name"] for m in parsed["modes"]] == ["score", "relax", "minimize", "flexddg"]

        score_mode = parsed["modes"][0]
        assert len(score_mode["notes"]) > 0
        assert "output_schema" in score_mode

        score_function_prop = score_mode["input_schema"]["properties"]["score_function"]
        assert score_function_prop["x-autobio"]["widget"] == "select"

        flexddg_mode = parsed["modes"][3]
        mutations_prop = flexddg_mode["input_schema"]["properties"]["mutations"]
        assert mutations_prop["x-autobio"]["widget"] == "text"
        assert "output_schema" in flexddg_mode


# ---------------------------------------------------------------------------
# TestRosettaRunMetadataMode — full run() lifecycle threads mode into metadata
# ---------------------------------------------------------------------------


_MIN_RESULT = {
    "scores": [
        {
            "total_score": -10.0,
            "score_breakdown": None,
            "units": "REU",
            "per_residue_scores": None,
            "structure_path": None,
            "ddg": None,
            "mutations": None,
        }
    ]
}


class TestRosettaRunMetadataMode:
    """``run(...).metadata.mode`` reflects the selected mode for each mode."""

    @pytest.mark.parametrize("mode_name", _MODES)
    def test_run_metadata_mode(
        self,
        mode_name: str,
        config: AutobioConfig,
        tmp_path: Path,
        sample_pdb: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import autobio.tools  # noqa: F401

        output_dir = tmp_path / "ws"
        std_dir = output_dir / "outputs" / "standardized"
        std_dir.mkdir(parents=True)
        (std_dir / "result_data.json").write_text(json.dumps(_MIN_RESULT))

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
            r = RosettaRunner("rosetta", config)

        input_data = _input_for_mode(mode_name, sample_pdb)
        out = r.run(input_data, gpu="none", output_dir=output_dir, mode=mode_name)
        assert out.metadata.mode == mode_name
        assert out.metadata.tool_name == "rosetta"
