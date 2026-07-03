"""Tests for the migrated esm_if1 Tool (modes: design, score)."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

from autobio.core.catalog import get_tool, tool_categories
from autobio.core.config import AutobioConfig
from autobio.core.registry import ToolCategory
from autobio.core.result import AutobioError
from autobio.core.workspace import Workspace
from autobio.schemas.inverse_folding import (
    InverseFoldingInput,
    InverseFoldingOutput,
)
from autobio.schemas.scoring import ScoringInput, ScoringOutput
from autobio.tools import TOOL_RUNNERS, get_runner
from autobio.tools.esm_if1 import ESMIF1Runner

if TYPE_CHECKING:
    from pathlib import Path


_OLD_FLAT_NAMES = ("esm_if1_score",)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


@pytest.fixture()
def config() -> AutobioConfig:
    return AutobioConfig.resolve()


def _make_runner(mode_name: str, config: AutobioConfig) -> ESMIF1Runner:
    """Create an ESMIF1Runner with mocked deps, current_mode pinned to *mode_name*."""
    with (
        patch("autobio.tools.base.ContainerManager"),
        patch("autobio.tools.base.GPUManager"),
    ):
        runner = ESMIF1Runner("esm_if1", config)
    runner.current_mode = get_tool("esm_if1").modes[mode_name]
    return runner


@pytest.fixture()
def runner(config: AutobioConfig) -> ESMIF1Runner:
    """Design-mode runner (the common case)."""
    return _make_runner("design", config)


@pytest.fixture()
def score_runner(config: AutobioConfig) -> ESMIF1Runner:
    return _make_runner("score", config)


@pytest.fixture()
def sample_pdb(tmp_path: Path) -> Path:
    """Write a minimal PDB file for testing."""
    pdb_path = tmp_path / "test.pdb"
    pdb_path.write_text("ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00\nEND\n")
    return pdb_path


# ---------------------------------------------------------------------------
# TestESMIF1PrepareWorkspace (design mode)
# ---------------------------------------------------------------------------


class TestESMIF1PrepareWorkspace:
    """Tests for ESMIF1Runner.prepare_workspace in design mode."""

    def test_structure_file_copied(
        self, runner: ESMIF1Runner, tmp_path: Path, sample_pdb: Path
    ) -> None:
        """Input structure is copied to inputs/ and config references container path."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = InverseFoldingInput(structure_path=sample_pdb)
        runner.prepare_workspace(input_data, workspace)

        copied = workspace.inputs_dir / "test.pdb"
        assert copied.exists()
        assert copied.read_text() == sample_pdb.read_text()

        cfg = json.loads(workspace.config_path.read_text())
        assert cfg["structure_path"] == "/workspace/inputs/test.pdb"

    def test_mode_is_design(self, runner: ESMIF1Runner, tmp_path: Path, sample_pdb: Path) -> None:
        """Config mode is set to 'design'."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = InverseFoldingInput(structure_path=sample_pdb)
        runner.prepare_workspace(input_data, workspace)

        cfg = json.loads(workspace.config_path.read_text())
        assert cfg["mode"] == "design"

    def test_defaults_applied(self, runner: ESMIF1Runner, tmp_path: Path, sample_pdb: Path) -> None:
        """Minimal input produces sensible defaults."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = InverseFoldingInput(structure_path=sample_pdb)
        runner.prepare_workspace(input_data, workspace)

        cfg = json.loads(workspace.config_path.read_text())
        assert cfg["temperature"] == 0.1
        assert cfg["num_sequences"] == 1

    def test_num_sequences_passthrough(
        self, runner: ESMIF1Runner, tmp_path: Path, sample_pdb: Path
    ) -> None:
        workspace = Workspace.create(tmp_path / "ws")
        input_data = InverseFoldingInput(structure_path=sample_pdb, num_sequences=5)
        runner.prepare_workspace(input_data, workspace)

        cfg = json.loads(workspace.config_path.read_text())
        assert cfg["num_sequences"] == 5

    def test_temperature_passthrough(
        self, runner: ESMIF1Runner, tmp_path: Path, sample_pdb: Path
    ) -> None:
        workspace = Workspace.create(tmp_path / "ws")
        input_data = InverseFoldingInput(structure_path=sample_pdb, temperature=0.5)
        runner.prepare_workspace(input_data, workspace)

        cfg = json.loads(workspace.config_path.read_text())
        assert cfg["temperature"] == 0.5

    def test_chains_to_design_passthrough(
        self, runner: ESMIF1Runner, tmp_path: Path, sample_pdb: Path
    ) -> None:
        workspace = Workspace.create(tmp_path / "ws")
        input_data = InverseFoldingInput(structure_path=sample_pdb, chains_to_design=["A", "B"])
        runner.prepare_workspace(input_data, workspace)

        cfg = json.loads(workspace.config_path.read_text())
        assert cfg["chains_to_design"] == ["A", "B"]

    def test_chains_to_design_absent_when_none(
        self, runner: ESMIF1Runner, tmp_path: Path, sample_pdb: Path
    ) -> None:
        workspace = Workspace.create(tmp_path / "ws")
        input_data = InverseFoldingInput(structure_path=sample_pdb)
        runner.prepare_workspace(input_data, workspace)

        cfg = json.loads(workspace.config_path.read_text())
        assert "chains_to_design" not in cfg

    def test_fixed_positions_passthrough(
        self, runner: ESMIF1Runner, tmp_path: Path, sample_pdb: Path
    ) -> None:
        workspace = Workspace.create(tmp_path / "ws")
        input_data = InverseFoldingInput(
            structure_path=sample_pdb, fixed_positions={"A": [1, 5, 10]}
        )
        runner.prepare_workspace(input_data, workspace)

        cfg = json.loads(workspace.config_path.read_text())
        assert cfg["fixed_positions"] == {"A": [1, 5, 10]}

    def test_extra_dict_merged(
        self, runner: ESMIF1Runner, tmp_path: Path, sample_pdb: Path
    ) -> None:
        """Extra dict keys appear at the top level of config.json."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = InverseFoldingInput(structure_path=sample_pdb, extra={"seed": 42})
        runner.prepare_workspace(input_data, workspace)

        cfg = json.loads(workspace.config_path.read_text())
        assert cfg["seed"] == 42

    def test_extra_shadowing_typed_field_rejected(
        self, runner: ESMIF1Runner, tmp_path: Path, sample_pdb: Path
    ) -> None:
        """extra containing a typed field name (temperature) raises."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = InverseFoldingInput(structure_path=sample_pdb, extra={"temperature": 0.5})
        with pytest.raises(AutobioError, match="collide with typed input fields"):
            runner.prepare_workspace(input_data, workspace)

    def test_extra_shadowing_config_key_rejected(
        self, runner: ESMIF1Runner, tmp_path: Path, sample_pdb: Path
    ) -> None:
        """extra containing a runner-derived config key (mode) raises."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = InverseFoldingInput(structure_path=sample_pdb, extra={"mode": "score"})
        with pytest.raises(AutobioError, match="collide"):
            runner.prepare_workspace(input_data, workspace)


# ---------------------------------------------------------------------------
# TestESMIF1ParseOutput (design mode)
# ---------------------------------------------------------------------------

_SINGLE_SEQ_RESULT = {
    "designed_sequences": [
        {
            "rank": 1,
            "sequence": {"A": "TVCCPSEEAKKKYEECRKPGTPDEECAKATGCIIIPGTKCPPDYPY"},
            "score": None,
            "recovery": 0.45,
        }
    ],
    "native_sequence": {"A": "TTCCPSIVARSNFNVCRLPGTPEAICATYTGCIIIPGATCPGDYAN"},
}

_MULTI_SEQ_RESULT = {
    "designed_sequences": [
        {
            "rank": 1,
            "sequence": {"A": "MKWVTFIS", "B": "GVSEKL"},
            "score": None,
            "recovery": 0.75,
        },
        {
            "rank": 2,
            "sequence": {"A": "MKWVTFLS", "B": "GVSERL"},
            "score": None,
            "recovery": 0.65,
        },
        {
            "rank": 3,
            "sequence": {"A": "MKWVTFAS", "B": "GVSEKR"},
            "score": None,
            "recovery": 0.55,
        },
    ],
    "native_sequence": {"A": "MKWVTFIS", "B": "GVSEKL"},
}


class TestESMIF1ParseOutput:
    """Tests for ESMIF1Runner.parse_output in design mode."""

    def test_parse_single_sequence(self, runner: ESMIF1Runner, tmp_path: Path) -> None:
        workspace = Workspace.create(tmp_path / "ws")
        (workspace.std_output_dir / "result_data.json").write_text(json.dumps(_SINGLE_SEQ_RESULT))

        output = runner.parse_output(workspace)
        assert isinstance(output, InverseFoldingOutput)
        assert len(output.designed_sequences) == 1
        seq = output.designed_sequences[0]
        assert seq.rank == 1
        assert seq.sequence["A"].startswith("TVCCPS")
        assert seq.recovery == pytest.approx(0.45)
        assert seq.score is None

    def test_parse_multiple_sequences(self, runner: ESMIF1Runner, tmp_path: Path) -> None:
        workspace = Workspace.create(tmp_path / "ws")
        (workspace.std_output_dir / "result_data.json").write_text(json.dumps(_MULTI_SEQ_RESULT))

        output = runner.parse_output(workspace)
        assert isinstance(output, InverseFoldingOutput)
        assert len(output.designed_sequences) == 3
        assert output.designed_sequences[0].rank == 1
        assert output.designed_sequences[2].rank == 3

    def test_parse_multi_chain(self, runner: ESMIF1Runner, tmp_path: Path) -> None:
        workspace = Workspace.create(tmp_path / "ws")
        (workspace.std_output_dir / "result_data.json").write_text(json.dumps(_MULTI_SEQ_RESULT))

        output = runner.parse_output(workspace)
        assert isinstance(output, InverseFoldingOutput)
        seq = output.designed_sequences[0]
        assert "A" in seq.sequence
        assert "B" in seq.sequence

    def test_parse_with_native_sequence(self, runner: ESMIF1Runner, tmp_path: Path) -> None:
        workspace = Workspace.create(tmp_path / "ws")
        (workspace.std_output_dir / "result_data.json").write_text(json.dumps(_SINGLE_SEQ_RESULT))

        output = runner.parse_output(workspace)
        assert isinstance(output, InverseFoldingOutput)
        assert output.native_sequence is not None
        assert "A" in output.native_sequence

    def test_parse_without_native_sequence(self, runner: ESMIF1Runner, tmp_path: Path) -> None:
        data = {**_SINGLE_SEQ_RESULT, "native_sequence": None}
        workspace = Workspace.create(tmp_path / "ws")
        (workspace.std_output_dir / "result_data.json").write_text(json.dumps(data))

        output = runner.parse_output(workspace)
        assert isinstance(output, InverseFoldingOutput)
        assert output.native_sequence is None

    def test_output_type(self, runner: ESMIF1Runner, tmp_path: Path) -> None:
        workspace = Workspace.create(tmp_path / "ws")
        (workspace.std_output_dir / "result_data.json").write_text(json.dumps(_SINGLE_SEQ_RESULT))

        output = runner.parse_output(workspace)
        assert isinstance(output, InverseFoldingOutput)

    def test_raw_output_path(self, runner: ESMIF1Runner, tmp_path: Path) -> None:
        workspace = Workspace.create(tmp_path / "ws")
        (workspace.std_output_dir / "result_data.json").write_text(json.dumps(_SINGLE_SEQ_RESULT))

        output = runner.parse_output(workspace)
        assert output.raw_output_path == workspace.raw_output_dir


# ---------------------------------------------------------------------------
# TestESMIF1ScorePrepareWorkspace (score mode)
# ---------------------------------------------------------------------------


class TestESMIF1ScorePrepareWorkspace:
    """Tests for ESMIF1Runner.prepare_workspace in score mode."""

    def test_mode_is_score(
        self, score_runner: ESMIF1Runner, tmp_path: Path, sample_pdb: Path
    ) -> None:
        workspace = Workspace.create(tmp_path / "ws")
        input_data = ScoringInput(structure_path=sample_pdb, sequences={"A": "MKWVTFIS"})
        score_runner.prepare_workspace(input_data, workspace)

        cfg = json.loads(workspace.config_path.read_text())
        assert cfg["mode"] == "score"

    def test_structure_file_copied(
        self, score_runner: ESMIF1Runner, tmp_path: Path, sample_pdb: Path
    ) -> None:
        workspace = Workspace.create(tmp_path / "ws")
        input_data = ScoringInput(structure_path=sample_pdb, sequences={"A": "MKWVTFIS"})
        score_runner.prepare_workspace(input_data, workspace)

        copied = workspace.inputs_dir / "test.pdb"
        assert copied.exists()

        cfg = json.loads(workspace.config_path.read_text())
        assert cfg["structure_path"] == "/workspace/inputs/test.pdb"

    def test_sequences_passthrough(
        self, score_runner: ESMIF1Runner, tmp_path: Path, sample_pdb: Path
    ) -> None:
        workspace = Workspace.create(tmp_path / "ws")
        sequences = {"A": "MKWVTFIS", "B": "GVSEKL"}
        input_data = ScoringInput(structure_path=sample_pdb, sequences=sequences)
        score_runner.prepare_workspace(input_data, workspace)

        cfg = json.loads(workspace.config_path.read_text())
        assert cfg["sequences"] == sequences

    def test_extra_dict_merged(
        self, score_runner: ESMIF1Runner, tmp_path: Path, sample_pdb: Path
    ) -> None:
        workspace = Workspace.create(tmp_path / "ws")
        input_data = ScoringInput(
            structure_path=sample_pdb,
            sequences={"A": "MKWVTFIS"},
            extra={"custom_param": "value"},
        )
        score_runner.prepare_workspace(input_data, workspace)

        cfg = json.loads(workspace.config_path.read_text())
        assert cfg["custom_param"] == "value"

    def test_extra_shadowing_typed_field_rejected(
        self, score_runner: ESMIF1Runner, tmp_path: Path, sample_pdb: Path
    ) -> None:
        """extra containing a typed field name (sequences) raises."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = ScoringInput(
            structure_path=sample_pdb,
            sequences={"A": "MKWVTFIS"},
            extra={"sequences": {"A": "GVSEKL"}},
        )
        with pytest.raises(AutobioError, match="collide with typed input fields"):
            score_runner.prepare_workspace(input_data, workspace)

    def test_extra_shadowing_config_key_rejected(
        self, score_runner: ESMIF1Runner, tmp_path: Path, sample_pdb: Path
    ) -> None:
        """extra containing a runner-derived config key (mode) raises."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = ScoringInput(
            structure_path=sample_pdb,
            sequences={"A": "MKWVTFIS"},
            extra={"mode": "design"},
        )
        with pytest.raises(AutobioError, match="collide"):
            score_runner.prepare_workspace(input_data, workspace)


# ---------------------------------------------------------------------------
# TestESMIF1ScoreParseOutput (score mode)
# ---------------------------------------------------------------------------

_SCORE_RESULT = {
    "scores": [
        {
            "total_score": -1.234,
            "per_residue_scores": None,
            "score_breakdown": {
                "A_ll_fullseq": -1.234,
                "A_ll_withcoord": -0.987,
                "ll_fullseq": -1.234,
                "ll_withcoord": -0.987,
            },
            "units": "avg_nll",
            "structure_path": None,
            "ddg": None,
            "mutations": None,
        }
    ]
}


class TestESMIF1ScoreParseOutput:
    """Tests for ESMIF1Runner.parse_output in score mode."""

    def test_parse_score(self, score_runner: ESMIF1Runner, tmp_path: Path) -> None:
        workspace = Workspace.create(tmp_path / "ws")
        (workspace.std_output_dir / "result_data.json").write_text(json.dumps(_SCORE_RESULT))

        output = score_runner.parse_output(workspace)
        assert isinstance(output, ScoringOutput)
        assert len(output.scores) == 1
        score = output.scores[0]
        assert score.total_score == pytest.approx(-1.234)
        assert score.units == "avg_nll"
        assert score.score_breakdown is not None
        assert "ll_fullseq" in score.score_breakdown

    def test_output_type(self, score_runner: ESMIF1Runner, tmp_path: Path) -> None:
        workspace = Workspace.create(tmp_path / "ws")
        (workspace.std_output_dir / "result_data.json").write_text(json.dumps(_SCORE_RESULT))

        output = score_runner.parse_output(workspace)
        assert isinstance(output, ScoringOutput)

    def test_raw_output_path(self, score_runner: ESMIF1Runner, tmp_path: Path) -> None:
        workspace = Workspace.create(tmp_path / "ws")
        (workspace.std_output_dir / "result_data.json").write_text(json.dumps(_SCORE_RESULT))

        output = score_runner.parse_output(workspace)
        assert output.raw_output_path == workspace.raw_output_dir


# ---------------------------------------------------------------------------
# TestESMIF1ByteCompatConfig — full-dict config.json equality, per mode
# ---------------------------------------------------------------------------


class TestESMIF1ByteCompatConfig:
    """Full-dict ``config.json`` equality tests, pinning key order per mode."""

    def test_design_full_config_minimal(
        self, config: AutobioConfig, tmp_path: Path, sample_pdb: Path
    ) -> None:
        """Minimal design input: no chains_to_design/fixed_positions in config."""
        r = _make_runner("design", config)
        workspace = Workspace.create(tmp_path / "ws")
        r.prepare_workspace(InverseFoldingInput(structure_path=sample_pdb), workspace)

        cfg = json.loads(workspace.config_path.read_text())
        expected = {
            "mode": "design",
            "structure_path": f"/workspace/inputs/{sample_pdb.name}",
            "num_sequences": 1,
            "temperature": 0.1,
        }
        assert cfg == expected
        assert list(cfg.keys()) == list(expected.keys())

    def test_design_full_config_with_chains_and_fixed_positions(
        self, config: AutobioConfig, tmp_path: Path, sample_pdb: Path
    ) -> None:
        """Design input with chains_to_design + fixed_positions set — full key order."""
        r = _make_runner("design", config)
        workspace = Workspace.create(tmp_path / "ws")
        r.prepare_workspace(
            InverseFoldingInput(
                structure_path=sample_pdb,
                num_sequences=3,
                temperature=0.5,
                chains_to_design=["A", "B"],
                fixed_positions={"A": [1, 5, 10]},
                extra={"seed": 42},
            ),
            workspace,
        )

        cfg = json.loads(workspace.config_path.read_text())
        expected = {
            "mode": "design",
            "structure_path": f"/workspace/inputs/{sample_pdb.name}",
            "num_sequences": 3,
            "temperature": 0.5,
            "chains_to_design": ["A", "B"],
            "fixed_positions": {"A": [1, 5, 10]},
            "seed": 42,
        }
        assert cfg == expected
        assert list(cfg.keys()) == list(expected.keys())

    def test_score_full_config(
        self, config: AutobioConfig, tmp_path: Path, sample_pdb: Path
    ) -> None:
        r = _make_runner("score", config)
        workspace = Workspace.create(tmp_path / "ws")
        r.prepare_workspace(
            ScoringInput(
                structure_path=sample_pdb,
                sequences={"A": "MKWVTFIS"},
                extra={"custom_param": "value"},
            ),
            workspace,
        )

        cfg = json.loads(workspace.config_path.read_text())
        expected = {
            "mode": "score",
            "structure_path": f"/workspace/inputs/{sample_pdb.name}",
            "sequences": {"A": "MKWVTFIS"},
            "custom_param": "value",
        }
        assert cfg == expected
        assert list(cfg.keys()) == list(expected.keys())


# ---------------------------------------------------------------------------
# TestESMIF1Registration
# ---------------------------------------------------------------------------


class TestESMIF1Registration:
    """Tests for the catalog Tool + runner registration."""

    def test_esm_if1_registered_as_single_tool(self) -> None:
        tool = get_tool("esm_if1")
        assert set(tool.modes) == {"design", "score"}
        assert tool.default_mode == "design"
        assert tool.category == ToolCategory.INVERSE_FOLDING
        assert tool.requires_gpu is True
        assert tool.gpu_count == 1
        assert tool.image_tag == "esm-if1:1.0.0"

    @pytest.mark.parametrize("flat_name", _OLD_FLAT_NAMES)
    def test_old_flat_names_absent_from_tool_runners(self, flat_name: str) -> None:
        assert flat_name not in TOOL_RUNNERS

    def test_esm_if1_in_tool_runners(self) -> None:
        assert "esm_if1" in TOOL_RUNNERS
        assert TOOL_RUNNERS["esm_if1"] is ESMIF1Runner

    def test_esm_if1_score_runner_class_removed(self) -> None:
        import autobio.tools.esm_if1 as esm_if1_module

        assert not hasattr(esm_if1_module, "ESMIF1ScoreRunner")

    @pytest.mark.parametrize(
        ("mode_name", "timeout"),
        [("design", 600), ("score", 300)],
    )
    def test_modes_have_per_mode_timeout(self, mode_name: str, timeout: int) -> None:
        assert get_tool("esm_if1").modes[mode_name].default_timeout == timeout

    def test_modes_share_tool_image_tag(self) -> None:
        """Neither mode overrides image_tag — both fall back to the Tool's."""
        tool = get_tool("esm_if1")
        assert tool.modes["design"].image_tag is None
        assert tool.modes["score"].image_tag is None

    def test_mode_schemas(self) -> None:
        tool = get_tool("esm_if1")
        assert tool.modes["design"].input_schema is InverseFoldingInput
        assert tool.modes["design"].output_schema is InverseFoldingOutput
        assert tool.modes["score"].input_schema is ScoringInput
        assert tool.modes["score"].output_schema is ScoringOutput

    def test_get_runner_returns_esm_if1_runner(self, config: AutobioConfig) -> None:
        with (
            patch("autobio.tools.base.ContainerManager"),
            patch("autobio.tools.base.GPUManager"),
        ):
            r = get_runner("esm_if1", config)
        assert isinstance(r, ESMIF1Runner)
        assert r.tool_name == "esm_if1"

    def test_get_runner_removed_flat_name_raises(self, config: AutobioConfig) -> None:
        with pytest.raises(KeyError, match="esm_if1_score"):
            get_runner("esm_if1_score", config)


# ---------------------------------------------------------------------------
# TestESMIF1CrossCategory — first cross-category catalog Tool
# ---------------------------------------------------------------------------


class TestESMIF1CrossCategory:
    """esm_if1 is the first Tool whose modes span two categories."""

    def test_tool_categories_union(self) -> None:
        tool = get_tool("esm_if1")
        assert tool_categories(tool) == (ToolCategory.INVERSE_FOLDING, ToolCategory.SCORING)

    def test_listed_under_inverse_folding(self) -> None:
        from autobio.core.catalog import list_tools

        assert "esm_if1" in list_tools(category=ToolCategory.INVERSE_FOLDING)

    def test_listed_under_scoring(self) -> None:
        from autobio.core.catalog import list_tools

        assert "esm_if1" in list_tools(category=ToolCategory.SCORING)


# ---------------------------------------------------------------------------
# TestESMIF1InfoSnapshot
# ---------------------------------------------------------------------------


class TestESMIF1InfoSnapshot:
    """``autobio info esm_if1`` output — per-mode notes, output_schema, category."""

    def test_info_snapshot(self) -> None:
        from autobio.cli.formatters import OutputFormat, format_tool_info_catalog

        parsed = json.loads(format_tool_info_catalog(get_tool("esm_if1"), OutputFormat.JSON))
        assert [m["name"] for m in parsed["modes"]] == ["design", "score"]

        design_mode = parsed["modes"][0]
        assert len(design_mode["notes"]) > 0
        assert "output_schema" in design_mode

        score_mode = parsed["modes"][1]
        assert len(score_mode["notes"]) > 0
        assert "output_schema" in score_mode
        assert score_mode["category"] == "scoring"


# ---------------------------------------------------------------------------
# TestESMIF1RunMetadataMode — full run() lifecycle threads mode into metadata
# ---------------------------------------------------------------------------

_MIN_DESIGN_RESULT = {
    "designed_sequences": [
        {"rank": 1, "sequence": {"A": "MKWVTFIS"}, "score": None, "recovery": 0.5}
    ],
    "native_sequence": {"A": "MKWVTFIS"},
}

_MIN_SCORE_RESULT = {
    "scores": [
        {
            "total_score": -1.0,
            "per_residue_scores": None,
            "score_breakdown": None,
            "units": "avg_nll",
            "structure_path": None,
            "ddg": None,
            "mutations": None,
        }
    ]
}


class TestESMIF1RunMetadataMode:
    """``run(...).metadata.mode`` reflects the selected mode for each mode."""

    @pytest.mark.parametrize(
        ("mode_name", "result_data"),
        [("design", _MIN_DESIGN_RESULT), ("score", _MIN_SCORE_RESULT)],
    )
    def test_run_metadata_mode(
        self,
        mode_name: str,
        result_data: dict,
        config: AutobioConfig,
        tmp_path: Path,
        sample_pdb: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        output_dir = tmp_path / "ws"
        std_dir = output_dir / "outputs" / "standardized"
        std_dir.mkdir(parents=True)
        (std_dir / "result_data.json").write_text(json.dumps(result_data))

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
            r = ESMIF1Runner("esm_if1", config)

        if mode_name == "design":
            input_data: InverseFoldingInput | ScoringInput = InverseFoldingInput(
                structure_path=sample_pdb
            )
        else:
            input_data = ScoringInput(structure_path=sample_pdb, sequences={"A": "MKWVTFIS"})

        out = r.run(input_data, gpu="none", output_dir=output_dir, mode=mode_name)
        assert out.metadata.mode == mode_name
        assert out.metadata.tool_name == "esm_if1"
