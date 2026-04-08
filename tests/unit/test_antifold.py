"""Tests for AntiFoldRunner, AntiFoldScoreRunner — prepare_workspace,
parse_output, and registration."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

from autobio.core.config import AutobioConfig
from autobio.core.registry import TOOL_REGISTRY, ToolCategory
from autobio.core.result import AutobioError
from autobio.core.workspace import Workspace
from autobio.schemas.inverse_folding import (
    InverseFoldingInput,
    InverseFoldingOutput,
)
from autobio.schemas.scoring import ScoringInput, ScoringOutput
from autobio.tools import TOOL_RUNNERS, get_runner
from autobio.tools.antifold import AntiFoldRunner, AntiFoldScoreRunner

if TYPE_CHECKING:
    from pathlib import Path


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def config() -> AutobioConfig:
    return AutobioConfig.resolve()


@pytest.fixture()
def runner(config: AutobioConfig) -> AntiFoldRunner:
    """Create an AntiFoldRunner with mocked ContainerManager and GPUManager."""
    with (
        patch("autobio.tools.base.ContainerManager"),
        patch("autobio.tools.base.GPUManager"),
    ):
        return AntiFoldRunner("antifold", config)


@pytest.fixture()
def score_runner(config: AutobioConfig) -> AntiFoldScoreRunner:
    """Create an AntiFoldScoreRunner with mocked ContainerManager and GPUManager."""
    with (
        patch("autobio.tools.base.ContainerManager"),
        patch("autobio.tools.base.GPUManager"),
    ):
        return AntiFoldScoreRunner("antifold_score", config)


@pytest.fixture()
def sample_pdb(tmp_path: Path) -> Path:
    """Write a minimal PDB file for testing."""
    pdb_path = tmp_path / "test.pdb"
    pdb_path.write_text("ATOM      1  N   ALA H   1       0.000   0.000   0.000  1.00  0.00\nEND\n")
    return pdb_path


def _make_design_input(pdb: Path, **extra: object) -> InverseFoldingInput:
    """Helper to create InverseFoldingInput with default chain IDs."""
    default_extra = {"heavy_chain": "H", "light_chain": "L"}
    default_extra.update(extra)
    return InverseFoldingInput(structure_path=pdb, extra=default_extra)


def _make_score_input(
    pdb: Path, sequences: dict[str, str] | None = None, **extra: object
) -> ScoringInput:
    """Helper to create ScoringInput with default chain IDs."""
    default_extra = {"heavy_chain": "H", "light_chain": "L"}
    default_extra.update(extra)
    return ScoringInput(structure_path=pdb, sequences=sequences, extra=default_extra)


# ---------------------------------------------------------------------------
# TestAntiFoldPrepareWorkspace
# ---------------------------------------------------------------------------


class TestAntiFoldPrepareWorkspace:
    """Tests for AntiFoldRunner.prepare_workspace."""

    def test_structure_file_copied(
        self, runner: AntiFoldRunner, tmp_path: Path, sample_pdb: Path
    ) -> None:
        """Input structure is copied to inputs/ and config references container path."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = _make_design_input(sample_pdb)
        runner.prepare_workspace(input_data, workspace)

        copied = workspace.inputs_dir / "test.pdb"
        assert copied.exists()
        assert copied.read_text() == sample_pdb.read_text()

        cfg = json.loads(workspace.config_path.read_text())
        assert cfg["structure_path"] == "/workspace/inputs/test.pdb"

    def test_mode_is_design(self, runner: AntiFoldRunner, tmp_path: Path, sample_pdb: Path) -> None:
        """Config mode is set to 'design'."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = _make_design_input(sample_pdb)
        runner.prepare_workspace(input_data, workspace)

        cfg = json.loads(workspace.config_path.read_text())
        assert cfg["mode"] == "design"

    def test_defaults_applied(
        self, runner: AntiFoldRunner, tmp_path: Path, sample_pdb: Path
    ) -> None:
        """Minimal input produces sensible defaults."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = _make_design_input(sample_pdb)
        runner.prepare_workspace(input_data, workspace)

        cfg = json.loads(workspace.config_path.read_text())
        assert cfg["temperature"] == 0.1
        assert cfg["num_sequences"] == 1

    def test_num_sequences_passthrough(
        self, runner: AntiFoldRunner, tmp_path: Path, sample_pdb: Path
    ) -> None:
        workspace = Workspace.create(tmp_path / "ws")
        input_data = InverseFoldingInput(
            structure_path=sample_pdb,
            num_sequences=5,
            extra={"heavy_chain": "H", "light_chain": "L"},
        )
        runner.prepare_workspace(input_data, workspace)

        cfg = json.loads(workspace.config_path.read_text())
        assert cfg["num_sequences"] == 5

    def test_temperature_passthrough(
        self, runner: AntiFoldRunner, tmp_path: Path, sample_pdb: Path
    ) -> None:
        workspace = Workspace.create(tmp_path / "ws")
        input_data = InverseFoldingInput(
            structure_path=sample_pdb,
            temperature=0.25,
            extra={"heavy_chain": "H", "light_chain": "L"},
        )
        runner.prepare_workspace(input_data, workspace)

        cfg = json.loads(workspace.config_path.read_text())
        assert cfg["temperature"] == 0.25

    def test_heavy_chain_passthrough(
        self, runner: AntiFoldRunner, tmp_path: Path, sample_pdb: Path
    ) -> None:
        workspace = Workspace.create(tmp_path / "ws")
        input_data = _make_design_input(sample_pdb, heavy_chain="A")
        runner.prepare_workspace(input_data, workspace)

        cfg = json.loads(workspace.config_path.read_text())
        assert cfg["heavy_chain"] == "A"

    def test_light_chain_passthrough(
        self, runner: AntiFoldRunner, tmp_path: Path, sample_pdb: Path
    ) -> None:
        workspace = Workspace.create(tmp_path / "ws")
        input_data = _make_design_input(sample_pdb, light_chain="B")
        runner.prepare_workspace(input_data, workspace)

        cfg = json.loads(workspace.config_path.read_text())
        assert cfg["light_chain"] == "B"

    def test_antigen_chain_passthrough(
        self, runner: AntiFoldRunner, tmp_path: Path, sample_pdb: Path
    ) -> None:
        workspace = Workspace.create(tmp_path / "ws")
        input_data = _make_design_input(sample_pdb, antigen_chain="C")
        runner.prepare_workspace(input_data, workspace)

        cfg = json.loads(workspace.config_path.read_text())
        assert cfg["antigen_chain"] == "C"

    def test_regions_passthrough(
        self, runner: AntiFoldRunner, tmp_path: Path, sample_pdb: Path
    ) -> None:
        workspace = Workspace.create(tmp_path / "ws")
        input_data = _make_design_input(sample_pdb, regions=["CDRH1", "CDRH2", "CDRH3"])
        runner.prepare_workspace(input_data, workspace)

        cfg = json.loads(workspace.config_path.read_text())
        assert cfg["regions"] == ["CDRH1", "CDRH2", "CDRH3"]

    def test_extra_dict_merged(
        self, runner: AntiFoldRunner, tmp_path: Path, sample_pdb: Path
    ) -> None:
        """Extra dict keys appear at the top level of config.json."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = _make_design_input(sample_pdb, seed=42)
        runner.prepare_workspace(input_data, workspace)

        cfg = json.loads(workspace.config_path.read_text())
        assert cfg["seed"] == 42

    def test_chains_to_design_not_mapped(
        self, runner: AntiFoldRunner, tmp_path: Path, sample_pdb: Path
    ) -> None:
        """chains_to_design from InverseFoldingInput is NOT passed to config."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = InverseFoldingInput(
            structure_path=sample_pdb,
            chains_to_design=["A", "B"],
            extra={"heavy_chain": "H", "light_chain": "L"},
        )
        runner.prepare_workspace(input_data, workspace)

        cfg = json.loads(workspace.config_path.read_text())
        assert "chains_to_design" not in cfg

    def test_fixed_positions_not_mapped(
        self, runner: AntiFoldRunner, tmp_path: Path, sample_pdb: Path
    ) -> None:
        """fixed_positions from InverseFoldingInput is NOT passed to config."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = InverseFoldingInput(
            structure_path=sample_pdb,
            fixed_positions={"A": [1, 5, 10]},
            extra={"heavy_chain": "H", "light_chain": "L"},
        )
        runner.prepare_workspace(input_data, workspace)

        cfg = json.loads(workspace.config_path.read_text())
        assert "fixed_positions" not in cfg

    def test_validation_requires_chain_id(
        self, runner: AntiFoldRunner, tmp_path: Path, sample_pdb: Path
    ) -> None:
        """Raises AutobioError when neither heavy_chain nor light_chain provided."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = InverseFoldingInput(structure_path=sample_pdb, extra={})
        with pytest.raises(AutobioError, match="heavy_chain.*light_chain"):
            runner.prepare_workspace(input_data, workspace)

    def test_validation_accepts_heavy_only(
        self, runner: AntiFoldRunner, tmp_path: Path, sample_pdb: Path
    ) -> None:
        """Does NOT raise when only heavy_chain is provided."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = InverseFoldingInput(structure_path=sample_pdb, extra={"heavy_chain": "H"})
        runner.prepare_workspace(input_data, workspace)

        cfg = json.loads(workspace.config_path.read_text())
        assert cfg["heavy_chain"] == "H"

    def test_validation_accepts_light_only(
        self, runner: AntiFoldRunner, tmp_path: Path, sample_pdb: Path
    ) -> None:
        """Does NOT raise when only light_chain is provided."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = InverseFoldingInput(structure_path=sample_pdb, extra={"light_chain": "L"})
        runner.prepare_workspace(input_data, workspace)

        cfg = json.loads(workspace.config_path.read_text())
        assert cfg["light_chain"] == "L"


# ---------------------------------------------------------------------------
# TestAntiFoldParseOutput
# ---------------------------------------------------------------------------

_SINGLE_SEQ_RESULT = {
    "designed_sequences": [
        {
            "rank": 1,
            "sequence": {"H": "EVQLVESGGGLVQPGGSLRLSCAASGFNIKD"},
            "score": -1.234,
            "recovery": 0.85,
        }
    ],
    "native_sequence": {"H": "EVQLVESGGGLVQPGGSLRLSCAASGFTFSD"},
}

_MULTI_SEQ_RESULT = {
    "designed_sequences": [
        {
            "rank": 1,
            "sequence": {"H": "EVQLVESGG", "L": "DIQMTQSPS"},
            "score": -0.95,
            "recovery": 0.88,
        },
        {
            "rank": 2,
            "sequence": {"H": "EVQLVEAGG", "L": "DIQMTQAPS"},
            "score": -1.12,
            "recovery": 0.75,
        },
        {
            "rank": 3,
            "sequence": {"H": "EVQLVESGK", "L": "DIQMTQSPK"},
            "score": -1.45,
            "recovery": 0.62,
        },
    ],
    "native_sequence": {"H": "EVQLVESGG", "L": "DIQMTQSPS"},
}


class TestAntiFoldParseOutput:
    """Tests for AntiFoldRunner.parse_output."""

    def test_parse_single_sequence(self, runner: AntiFoldRunner, tmp_path: Path) -> None:
        workspace = Workspace.create(tmp_path / "ws")
        (workspace.std_output_dir / "result_data.json").write_text(json.dumps(_SINGLE_SEQ_RESULT))

        output = runner.parse_output(workspace)
        assert len(output.designed_sequences) == 1
        seq = output.designed_sequences[0]
        assert seq.rank == 1
        assert seq.sequence["H"].startswith("EVQLVES")
        assert seq.recovery == pytest.approx(0.85)

    def test_parse_multiple_sequences(self, runner: AntiFoldRunner, tmp_path: Path) -> None:
        workspace = Workspace.create(tmp_path / "ws")
        (workspace.std_output_dir / "result_data.json").write_text(json.dumps(_MULTI_SEQ_RESULT))

        output = runner.parse_output(workspace)
        assert len(output.designed_sequences) == 3
        assert output.designed_sequences[0].rank == 1
        assert output.designed_sequences[2].rank == 3

    def test_parse_multi_chain(self, runner: AntiFoldRunner, tmp_path: Path) -> None:
        workspace = Workspace.create(tmp_path / "ws")
        (workspace.std_output_dir / "result_data.json").write_text(json.dumps(_MULTI_SEQ_RESULT))

        output = runner.parse_output(workspace)
        seq = output.designed_sequences[0]
        assert "H" in seq.sequence
        assert "L" in seq.sequence

    def test_parse_with_score(self, runner: AntiFoldRunner, tmp_path: Path) -> None:
        """AntiFold populates score (unlike ESM-IF1 where it is None)."""
        workspace = Workspace.create(tmp_path / "ws")
        (workspace.std_output_dir / "result_data.json").write_text(json.dumps(_SINGLE_SEQ_RESULT))

        output = runner.parse_output(workspace)
        seq = output.designed_sequences[0]
        assert seq.score is not None
        assert seq.score == pytest.approx(-1.234)

    def test_parse_with_native_sequence(self, runner: AntiFoldRunner, tmp_path: Path) -> None:
        workspace = Workspace.create(tmp_path / "ws")
        (workspace.std_output_dir / "result_data.json").write_text(json.dumps(_SINGLE_SEQ_RESULT))

        output = runner.parse_output(workspace)
        assert output.native_sequence is not None
        assert "H" in output.native_sequence

    def test_output_type(self, runner: AntiFoldRunner, tmp_path: Path) -> None:
        workspace = Workspace.create(tmp_path / "ws")
        (workspace.std_output_dir / "result_data.json").write_text(json.dumps(_SINGLE_SEQ_RESULT))

        output = runner.parse_output(workspace)
        assert isinstance(output, InverseFoldingOutput)

    def test_raw_output_path(self, runner: AntiFoldRunner, tmp_path: Path) -> None:
        workspace = Workspace.create(tmp_path / "ws")
        (workspace.std_output_dir / "result_data.json").write_text(json.dumps(_SINGLE_SEQ_RESULT))

        output = runner.parse_output(workspace)
        assert output.raw_output_path == workspace.raw_output_dir


# ---------------------------------------------------------------------------
# TestAntiFoldScorePrepareWorkspace
# ---------------------------------------------------------------------------


class TestAntiFoldScorePrepareWorkspace:
    """Tests for AntiFoldScoreRunner.prepare_workspace."""

    def test_mode_is_score(
        self, score_runner: AntiFoldScoreRunner, tmp_path: Path, sample_pdb: Path
    ) -> None:
        workspace = Workspace.create(tmp_path / "ws")
        input_data = _make_score_input(sample_pdb, sequences={"H": "EVQLVES"})
        score_runner.prepare_workspace(input_data, workspace)

        cfg = json.loads(workspace.config_path.read_text())
        assert cfg["mode"] == "score"

    def test_structure_file_copied(
        self, score_runner: AntiFoldScoreRunner, tmp_path: Path, sample_pdb: Path
    ) -> None:
        workspace = Workspace.create(tmp_path / "ws")
        input_data = _make_score_input(sample_pdb, sequences={"H": "EVQLVES"})
        score_runner.prepare_workspace(input_data, workspace)

        copied = workspace.inputs_dir / "test.pdb"
        assert copied.exists()

        cfg = json.loads(workspace.config_path.read_text())
        assert cfg["structure_path"] == "/workspace/inputs/test.pdb"

    def test_sequences_none_passthrough(
        self, score_runner: AntiFoldScoreRunner, tmp_path: Path, sample_pdb: Path
    ) -> None:
        """sequences=None is passed through (score native)."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = _make_score_input(sample_pdb, sequences=None)
        score_runner.prepare_workspace(input_data, workspace)

        cfg = json.loads(workspace.config_path.read_text())
        assert cfg["sequences"] is None

    def test_sequences_dict_passthrough(
        self, score_runner: AntiFoldScoreRunner, tmp_path: Path, sample_pdb: Path
    ) -> None:
        workspace = Workspace.create(tmp_path / "ws")
        sequences = {"H": "EVQLVES", "L": "DIQMTQS"}
        input_data = _make_score_input(sample_pdb, sequences=sequences)
        score_runner.prepare_workspace(input_data, workspace)

        cfg = json.loads(workspace.config_path.read_text())
        assert cfg["sequences"] == sequences

    def test_extra_dict_merged(
        self, score_runner: AntiFoldScoreRunner, tmp_path: Path, sample_pdb: Path
    ) -> None:
        workspace = Workspace.create(tmp_path / "ws")
        input_data = _make_score_input(sample_pdb, sequences={"H": "EVQLVES"}, regions=["CDRH3"])
        score_runner.prepare_workspace(input_data, workspace)

        cfg = json.loads(workspace.config_path.read_text())
        assert cfg["regions"] == ["CDRH3"]

    def test_validation_requires_chain_id(
        self, score_runner: AntiFoldScoreRunner, tmp_path: Path, sample_pdb: Path
    ) -> None:
        workspace = Workspace.create(tmp_path / "ws")
        input_data = ScoringInput(structure_path=sample_pdb, sequences={"H": "EVQLVES"}, extra={})
        with pytest.raises(AutobioError, match="heavy_chain.*light_chain"):
            score_runner.prepare_workspace(input_data, workspace)


# ---------------------------------------------------------------------------
# TestAntiFoldScoreParseOutput
# ---------------------------------------------------------------------------

_SCORE_RESULT = {
    "scores": [
        {
            "total_score": -0.85,
            "per_residue_scores": [-0.5, -1.2, -0.8, -0.9],
            "score_breakdown": {
                "H_mean_ll": -0.90,
                "L_mean_ll": -0.80,
                "H_perplexity": 2.46,
                "L_perplexity": 2.23,
                "perplexity": 2.34,
            },
            "units": "avg_nll",
            "structure_path": None,
            "ddg": None,
            "mutations": None,
        }
    ]
}


class TestAntiFoldScoreParseOutput:
    """Tests for AntiFoldScoreRunner.parse_output."""

    def test_parse_score(self, score_runner: AntiFoldScoreRunner, tmp_path: Path) -> None:
        workspace = Workspace.create(tmp_path / "ws")
        (workspace.std_output_dir / "result_data.json").write_text(json.dumps(_SCORE_RESULT))

        output = score_runner.parse_output(workspace)
        assert len(output.scores) == 1
        score = output.scores[0]
        assert score.total_score == pytest.approx(-0.85)
        assert score.units == "avg_nll"

    def test_parse_with_per_residue_scores(
        self, score_runner: AntiFoldScoreRunner, tmp_path: Path
    ) -> None:
        workspace = Workspace.create(tmp_path / "ws")
        (workspace.std_output_dir / "result_data.json").write_text(json.dumps(_SCORE_RESULT))

        output = score_runner.parse_output(workspace)
        score = output.scores[0]
        assert score.per_residue_scores is not None
        assert len(score.per_residue_scores) == 4

    def test_parse_with_breakdown(self, score_runner: AntiFoldScoreRunner, tmp_path: Path) -> None:
        workspace = Workspace.create(tmp_path / "ws")
        (workspace.std_output_dir / "result_data.json").write_text(json.dumps(_SCORE_RESULT))

        output = score_runner.parse_output(workspace)
        score = output.scores[0]
        assert score.score_breakdown is not None
        assert "perplexity" in score.score_breakdown
        assert "H_mean_ll" in score.score_breakdown

    def test_output_type(self, score_runner: AntiFoldScoreRunner, tmp_path: Path) -> None:
        workspace = Workspace.create(tmp_path / "ws")
        (workspace.std_output_dir / "result_data.json").write_text(json.dumps(_SCORE_RESULT))

        output = score_runner.parse_output(workspace)
        assert isinstance(output, ScoringOutput)


# ---------------------------------------------------------------------------
# TestAntiFoldRegistration
# ---------------------------------------------------------------------------


class TestAntiFoldRegistration:
    """Tests for tool and runner registration."""

    def test_antifold_in_registry(self) -> None:
        assert "antifold" in TOOL_REGISTRY
        entry = TOOL_REGISTRY["antifold"]
        assert entry.category == ToolCategory.INVERSE_FOLDING
        assert entry.input_schema is InverseFoldingInput
        assert entry.output_schema is InverseFoldingOutput
        assert entry.requires_gpu is True
        assert entry.gpu_count == 1

    def test_antifold_score_in_registry(self) -> None:
        assert "antifold_score" in TOOL_REGISTRY
        entry = TOOL_REGISTRY["antifold_score"]
        assert entry.category == ToolCategory.SCORING
        assert entry.input_schema is ScoringInput
        assert entry.output_schema is ScoringOutput
        assert entry.requires_gpu is True

    def test_both_share_image_tag(self) -> None:
        assert (
            TOOL_REGISTRY["antifold"].image_tag
            == TOOL_REGISTRY["antifold_score"].image_tag
            == "antifold:1.0.0"
        )

    def test_tool_runners_registered(self) -> None:
        assert "antifold" in TOOL_RUNNERS
        assert "antifold_score" in TOOL_RUNNERS
        assert TOOL_RUNNERS["antifold"] is AntiFoldRunner
        assert TOOL_RUNNERS["antifold_score"] is AntiFoldScoreRunner

    def test_get_runner_returns_correct_class(self, config: AutobioConfig) -> None:
        with (
            patch("autobio.tools.base.ContainerManager"),
            patch("autobio.tools.base.GPUManager"),
        ):
            r = get_runner("antifold", config)
        assert isinstance(r, AntiFoldRunner)
        assert r.tool_name == "antifold"

    def test_get_score_runner_returns_correct_class(self, config: AutobioConfig) -> None:
        with (
            patch("autobio.tools.base.ContainerManager"),
            patch("autobio.tools.base.GPUManager"),
        ):
            r = get_runner("antifold_score", config)
        assert isinstance(r, AntiFoldScoreRunner)
        assert r.tool_name == "antifold_score"
