"""Tests for ESMIF1Runner, ESMIF1ScoreRunner — prepare_workspace, parse_output, and registration."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

from autobio.core.config import AutobioConfig
from autobio.core.registry import TOOL_REGISTRY, ToolCategory
from autobio.core.workspace import Workspace
from autobio.schemas.inverse_folding import (
    InverseFoldingInput,
    InverseFoldingOutput,
)
from autobio.schemas.scoring import ScoringInput, ScoringOutput
from autobio.tools import TOOL_RUNNERS, get_runner
from autobio.tools.esm_if1 import ESMIF1Runner, ESMIF1ScoreRunner

if TYPE_CHECKING:
    from pathlib import Path


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def config() -> AutobioConfig:
    return AutobioConfig.resolve()


@pytest.fixture()
def runner(config: AutobioConfig) -> ESMIF1Runner:
    """Create an ESMIF1Runner with mocked ContainerManager and GPUManager."""
    with (
        patch("autobio.tools.base.ContainerManager"),
        patch("autobio.tools.base.GPUManager"),
    ):
        return ESMIF1Runner("esm_if1", config)


@pytest.fixture()
def score_runner(config: AutobioConfig) -> ESMIF1ScoreRunner:
    """Create an ESMIF1ScoreRunner with mocked ContainerManager and GPUManager."""
    with (
        patch("autobio.tools.base.ContainerManager"),
        patch("autobio.tools.base.GPUManager"),
    ):
        return ESMIF1ScoreRunner("esm_if1_score", config)


@pytest.fixture()
def sample_pdb(tmp_path: Path) -> Path:
    """Write a minimal PDB file for testing."""
    pdb_path = tmp_path / "test.pdb"
    pdb_path.write_text("ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00\nEND\n")
    return pdb_path


# ---------------------------------------------------------------------------
# TestESMIF1PrepareWorkspace
# ---------------------------------------------------------------------------


class TestESMIF1PrepareWorkspace:
    """Tests for ESMIF1Runner.prepare_workspace."""

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


# ---------------------------------------------------------------------------
# TestESMIF1ParseOutput
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
    """Tests for ESMIF1Runner.parse_output."""

    def test_parse_single_sequence(self, runner: ESMIF1Runner, tmp_path: Path) -> None:
        workspace = Workspace.create(tmp_path / "ws")
        (workspace.std_output_dir / "result_data.json").write_text(json.dumps(_SINGLE_SEQ_RESULT))

        output = runner.parse_output(workspace)
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
        assert len(output.designed_sequences) == 3
        assert output.designed_sequences[0].rank == 1
        assert output.designed_sequences[2].rank == 3

    def test_parse_multi_chain(self, runner: ESMIF1Runner, tmp_path: Path) -> None:
        workspace = Workspace.create(tmp_path / "ws")
        (workspace.std_output_dir / "result_data.json").write_text(json.dumps(_MULTI_SEQ_RESULT))

        output = runner.parse_output(workspace)
        seq = output.designed_sequences[0]
        assert "A" in seq.sequence
        assert "B" in seq.sequence

    def test_parse_with_native_sequence(self, runner: ESMIF1Runner, tmp_path: Path) -> None:
        workspace = Workspace.create(tmp_path / "ws")
        (workspace.std_output_dir / "result_data.json").write_text(json.dumps(_SINGLE_SEQ_RESULT))

        output = runner.parse_output(workspace)
        assert output.native_sequence is not None
        assert "A" in output.native_sequence

    def test_parse_without_native_sequence(self, runner: ESMIF1Runner, tmp_path: Path) -> None:
        data = {**_SINGLE_SEQ_RESULT, "native_sequence": None}
        workspace = Workspace.create(tmp_path / "ws")
        (workspace.std_output_dir / "result_data.json").write_text(json.dumps(data))

        output = runner.parse_output(workspace)
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
# TestESMIF1ScorePrepareWorkspace
# ---------------------------------------------------------------------------


class TestESMIF1ScorePrepareWorkspace:
    """Tests for ESMIF1ScoreRunner.prepare_workspace."""

    def test_mode_is_score(
        self, score_runner: ESMIF1ScoreRunner, tmp_path: Path, sample_pdb: Path
    ) -> None:
        workspace = Workspace.create(tmp_path / "ws")
        input_data = ScoringInput(structure_path=sample_pdb, sequences={"A": "MKWVTFIS"})
        score_runner.prepare_workspace(input_data, workspace)

        cfg = json.loads(workspace.config_path.read_text())
        assert cfg["mode"] == "score"

    def test_structure_file_copied(
        self, score_runner: ESMIF1ScoreRunner, tmp_path: Path, sample_pdb: Path
    ) -> None:
        workspace = Workspace.create(tmp_path / "ws")
        input_data = ScoringInput(structure_path=sample_pdb, sequences={"A": "MKWVTFIS"})
        score_runner.prepare_workspace(input_data, workspace)

        copied = workspace.inputs_dir / "test.pdb"
        assert copied.exists()

        cfg = json.loads(workspace.config_path.read_text())
        assert cfg["structure_path"] == "/workspace/inputs/test.pdb"

    def test_sequences_passthrough(
        self, score_runner: ESMIF1ScoreRunner, tmp_path: Path, sample_pdb: Path
    ) -> None:
        workspace = Workspace.create(tmp_path / "ws")
        sequences = {"A": "MKWVTFIS", "B": "GVSEKL"}
        input_data = ScoringInput(structure_path=sample_pdb, sequences=sequences)
        score_runner.prepare_workspace(input_data, workspace)

        cfg = json.loads(workspace.config_path.read_text())
        assert cfg["sequences"] == sequences

    def test_extra_dict_merged(
        self, score_runner: ESMIF1ScoreRunner, tmp_path: Path, sample_pdb: Path
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


# ---------------------------------------------------------------------------
# TestESMIF1ScoreParseOutput
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
    """Tests for ESMIF1ScoreRunner.parse_output."""

    def test_parse_score(self, score_runner: ESMIF1ScoreRunner, tmp_path: Path) -> None:
        workspace = Workspace.create(tmp_path / "ws")
        (workspace.std_output_dir / "result_data.json").write_text(json.dumps(_SCORE_RESULT))

        output = score_runner.parse_output(workspace)
        assert len(output.scores) == 1
        score = output.scores[0]
        assert score.total_score == pytest.approx(-1.234)
        assert score.units == "avg_nll"
        assert score.score_breakdown is not None
        assert "ll_fullseq" in score.score_breakdown

    def test_output_type(self, score_runner: ESMIF1ScoreRunner, tmp_path: Path) -> None:
        workspace = Workspace.create(tmp_path / "ws")
        (workspace.std_output_dir / "result_data.json").write_text(json.dumps(_SCORE_RESULT))

        output = score_runner.parse_output(workspace)
        assert isinstance(output, ScoringOutput)

    def test_raw_output_path(self, score_runner: ESMIF1ScoreRunner, tmp_path: Path) -> None:
        workspace = Workspace.create(tmp_path / "ws")
        (workspace.std_output_dir / "result_data.json").write_text(json.dumps(_SCORE_RESULT))

        output = score_runner.parse_output(workspace)
        assert output.raw_output_path == workspace.raw_output_dir


# ---------------------------------------------------------------------------
# TestESMIF1Registration
# ---------------------------------------------------------------------------


class TestESMIF1Registration:
    """Tests for tool and runner registration."""

    def test_esm_if1_in_registry(self) -> None:
        assert "esm_if1" in TOOL_REGISTRY
        entry = TOOL_REGISTRY["esm_if1"]
        assert entry.category == ToolCategory.INVERSE_FOLDING
        assert entry.input_schema is InverseFoldingInput
        assert entry.output_schema is InverseFoldingOutput
        assert entry.requires_gpu is True
        assert entry.gpu_count == 1

    def test_esm_if1_score_in_registry(self) -> None:
        assert "esm_if1_score" in TOOL_REGISTRY
        entry = TOOL_REGISTRY["esm_if1_score"]
        assert entry.category == ToolCategory.SCORING
        assert entry.input_schema is ScoringInput
        assert entry.output_schema is ScoringOutput
        assert entry.requires_gpu is True

    def test_both_share_image_tag(self) -> None:
        assert (
            TOOL_REGISTRY["esm_if1"].image_tag
            == TOOL_REGISTRY["esm_if1_score"].image_tag
            == "esm-if1:1.0.0"
        )

    def test_tool_runners_registered(self) -> None:
        assert "esm_if1" in TOOL_RUNNERS
        assert "esm_if1_score" in TOOL_RUNNERS
        assert TOOL_RUNNERS["esm_if1"] is ESMIF1Runner
        assert TOOL_RUNNERS["esm_if1_score"] is ESMIF1ScoreRunner

    def test_get_runner_returns_correct_class(self, config: AutobioConfig) -> None:
        with (
            patch("autobio.tools.base.ContainerManager"),
            patch("autobio.tools.base.GPUManager"),
        ):
            r = get_runner("esm_if1", config)
        assert isinstance(r, ESMIF1Runner)
        assert r.tool_name == "esm_if1"

    def test_get_score_runner_returns_correct_class(self, config: AutobioConfig) -> None:
        with (
            patch("autobio.tools.base.ContainerManager"),
            patch("autobio.tools.base.GPUManager"),
        ):
            r = get_runner("esm_if1_score", config)
        assert isinstance(r, ESMIF1ScoreRunner)
        assert r.tool_name == "esm_if1_score"
