"""Tests for MPNNScoreRunner — prepare_workspace, parse_output, and registration."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

from autobio.core.config import AutobioConfig
from autobio.core.registry import TOOL_REGISTRY, ToolCategory
from autobio.core.workspace import Workspace
from autobio.schemas.scoring import ScoringInput, ScoringOutput
from autobio.tools import TOOL_RUNNERS, get_runner
from autobio.tools.mpnn import _SCORE_CHECKPOINT_DIR, MPNNScoreRunner

if TYPE_CHECKING:
    from pathlib import Path


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def config() -> AutobioConfig:
    return AutobioConfig.resolve()


@pytest.fixture()
def runner(config: AutobioConfig) -> MPNNScoreRunner:
    """Create an MPNNScoreRunner for proteinmpnn_score."""
    with (
        patch("autobio.tools.base.ContainerManager"),
        patch("autobio.tools.base.GPUManager"),
    ):
        return MPNNScoreRunner("proteinmpnn_score", config)


@pytest.fixture()
def ligand_runner(config: AutobioConfig) -> MPNNScoreRunner:
    """Create an MPNNScoreRunner for ligandmpnn_score."""
    with (
        patch("autobio.tools.base.ContainerManager"),
        patch("autobio.tools.base.GPUManager"),
    ):
        return MPNNScoreRunner("ligandmpnn_score", config)


@pytest.fixture()
def sample_pdb(tmp_path: Path) -> Path:
    """Write a minimal PDB file for testing."""
    pdb_path = tmp_path / "test.pdb"
    pdb_path.write_text("ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00\nEND\n")
    return pdb_path


# ---------------------------------------------------------------------------
# TestMPNNScorePrepareWorkspace
# ---------------------------------------------------------------------------


class TestMPNNScorePrepareWorkspace:
    """Tests for MPNNScoreRunner.prepare_workspace."""

    @pytest.mark.parametrize(
        ("tool_name", "expected_model_type", "expected_checkpoint"),
        [
            ("proteinmpnn_score", "protein_mpnn", "proteinmpnn_v_48_020.pt"),
            ("ligandmpnn_score", "ligand_mpnn", "ligandmpnn_v_32_010_25.pt"),
        ],
    )
    def test_model_config_per_tool(
        self,
        tool_name: str,
        expected_model_type: str,
        expected_checkpoint: str,
        config: AutobioConfig,
        tmp_path: Path,
        sample_pdb: Path,
    ) -> None:
        """Config contains correct model_type and checkpoint_path per tool."""
        with (
            patch("autobio.tools.base.ContainerManager"),
            patch("autobio.tools.base.GPUManager"),
        ):
            r = MPNNScoreRunner(tool_name, config)
        workspace = Workspace.create(tmp_path / "ws")
        input_data = ScoringInput(structure_path=sample_pdb)
        r.prepare_workspace(input_data, workspace)

        cfg = json.loads(workspace.config_path.read_text())
        assert cfg["model_type"] == expected_model_type
        assert cfg["checkpoint_path"] == f"{_SCORE_CHECKPOINT_DIR}/{expected_checkpoint}"

    def test_mode_is_score(self, runner: MPNNScoreRunner, tmp_path: Path, sample_pdb: Path) -> None:
        workspace = Workspace.create(tmp_path / "ws")
        input_data = ScoringInput(structure_path=sample_pdb)
        runner.prepare_workspace(input_data, workspace)

        cfg = json.loads(workspace.config_path.read_text())
        assert cfg["mode"] == "score"

    def test_structure_file_copied(
        self, runner: MPNNScoreRunner, tmp_path: Path, sample_pdb: Path
    ) -> None:
        """Input structure is copied to inputs/ and config references container path."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = ScoringInput(structure_path=sample_pdb)
        runner.prepare_workspace(input_data, workspace)

        # File exists in workspace inputs
        copied = workspace.inputs_dir / "test.pdb"
        assert copied.exists()
        assert copied.read_text() == sample_pdb.read_text()

        # Config references container path
        cfg = json.loads(workspace.config_path.read_text())
        assert cfg["structure_path"] == "/workspace/inputs/test.pdb"

    def test_sequences_passthrough(
        self, runner: MPNNScoreRunner, tmp_path: Path, sample_pdb: Path
    ) -> None:
        """Explicit sequences dict is passed through to config."""
        workspace = Workspace.create(tmp_path / "ws")
        seqs = {"A": "MKWVTFIS", "B": "GVSEKL"}
        input_data = ScoringInput(structure_path=sample_pdb, sequences=seqs)
        runner.prepare_workspace(input_data, workspace)

        cfg = json.loads(workspace.config_path.read_text())
        assert cfg["sequences"] == seqs

    def test_sequences_none_passthrough(
        self, runner: MPNNScoreRunner, tmp_path: Path, sample_pdb: Path
    ) -> None:
        """When sequences is None, config has null (score native)."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = ScoringInput(structure_path=sample_pdb, sequences=None)
        runner.prepare_workspace(input_data, workspace)

        cfg = json.loads(workspace.config_path.read_text())
        assert cfg["sequences"] is None

    def test_extra_dict_merged(
        self, runner: MPNNScoreRunner, tmp_path: Path, sample_pdb: Path
    ) -> None:
        """Extra dict keys appear at the top level of config.json."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = ScoringInput(
            structure_path=sample_pdb,
            extra={"seed": 42, "batch_size": 4},
        )
        runner.prepare_workspace(input_data, workspace)

        cfg = json.loads(workspace.config_path.read_text())
        assert cfg["seed"] == 42
        assert cfg["batch_size"] == 4

    def test_checkpoint_uses_score_container_paths(
        self, runner: MPNNScoreRunner, tmp_path: Path, sample_pdb: Path
    ) -> None:
        """Checkpoint path uses /app/LigandMPNN/model_params/ (not foundry)."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = ScoringInput(structure_path=sample_pdb)
        runner.prepare_workspace(input_data, workspace)

        cfg = json.loads(workspace.config_path.read_text())
        assert cfg["checkpoint_path"].startswith("/app/LigandMPNN/model_params/")


# ---------------------------------------------------------------------------
# TestMPNNScoreParseOutput
# ---------------------------------------------------------------------------

_SINGLE_CHAIN_SCORE = {
    "scores": [
        {
            "total_score": -1.85,
            "per_residue_scores": [-1.2, -2.1, -1.5, -2.6],
            "score_breakdown": {
                "A_mean_nll": -1.85,
                "A_perplexity": 6.36,
                "perplexity": 6.36,
            },
            "units": "avg_nll",
            "structure_path": None,
            "ddg": None,
            "mutations": None,
        }
    ]
}

_MULTI_CHAIN_SCORE = {
    "scores": [
        {
            "total_score": -2.01,
            "per_residue_scores": [-1.2, -2.1, -1.5, -2.6, -2.5, -1.8],
            "score_breakdown": {
                "A_mean_nll": -1.70,
                "A_perplexity": 5.47,
                "B_mean_nll": -2.32,
                "B_perplexity": 10.18,
                "perplexity": 7.46,
            },
            "units": "avg_nll",
            "structure_path": None,
            "ddg": None,
            "mutations": None,
        }
    ]
}


class TestMPNNScoreParseOutput:
    """Tests for MPNNScoreRunner.parse_output."""

    def test_parse_single_chain_score(self, runner: MPNNScoreRunner, tmp_path: Path) -> None:
        workspace = Workspace.create(tmp_path / "ws")
        (workspace.std_output_dir / "result_data.json").write_text(json.dumps(_SINGLE_CHAIN_SCORE))

        output = runner.parse_output(workspace)
        assert len(output.scores) == 1
        assert output.scores[0].total_score == pytest.approx(-1.85)
        assert output.scores[0].units == "avg_nll"

    def test_parse_per_residue_scores(self, runner: MPNNScoreRunner, tmp_path: Path) -> None:
        workspace = Workspace.create(tmp_path / "ws")
        (workspace.std_output_dir / "result_data.json").write_text(json.dumps(_SINGLE_CHAIN_SCORE))

        output = runner.parse_output(workspace)
        assert output.scores[0].per_residue_scores is not None
        assert len(output.scores[0].per_residue_scores) == 4

    def test_parse_multi_chain_breakdown(self, runner: MPNNScoreRunner, tmp_path: Path) -> None:
        workspace = Workspace.create(tmp_path / "ws")
        (workspace.std_output_dir / "result_data.json").write_text(json.dumps(_MULTI_CHAIN_SCORE))

        output = runner.parse_output(workspace)
        breakdown = output.scores[0].score_breakdown
        assert breakdown is not None
        assert "A_mean_nll" in breakdown
        assert "B_mean_nll" in breakdown
        assert "perplexity" in breakdown

    def test_output_type(self, runner: MPNNScoreRunner, tmp_path: Path) -> None:
        workspace = Workspace.create(tmp_path / "ws")
        (workspace.std_output_dir / "result_data.json").write_text(json.dumps(_SINGLE_CHAIN_SCORE))

        output = runner.parse_output(workspace)
        assert isinstance(output, ScoringOutput)

    def test_raw_output_path(self, runner: MPNNScoreRunner, tmp_path: Path) -> None:
        workspace = Workspace.create(tmp_path / "ws")
        (workspace.std_output_dir / "result_data.json").write_text(json.dumps(_SINGLE_CHAIN_SCORE))

        output = runner.parse_output(workspace)
        assert output.raw_output_path == workspace.raw_output_dir


# ---------------------------------------------------------------------------
# TestMPNNScoreRegistration
# ---------------------------------------------------------------------------


class TestMPNNScoreRegistration:
    """Tests for tool and runner registration."""

    def test_proteinmpnn_score_in_registry(self) -> None:
        assert "proteinmpnn_score" in TOOL_REGISTRY
        entry = TOOL_REGISTRY["proteinmpnn_score"]
        assert entry.category == ToolCategory.SCORING
        assert entry.input_schema is ScoringInput
        assert entry.output_schema is ScoringOutput
        assert entry.requires_gpu is True

    def test_ligandmpnn_score_in_registry(self) -> None:
        assert "ligandmpnn_score" in TOOL_REGISTRY
        entry = TOOL_REGISTRY["ligandmpnn_score"]
        assert entry.category == ToolCategory.SCORING
        assert "ligand" in entry.description.lower()

    def test_both_share_image_tag(self) -> None:
        assert (
            TOOL_REGISTRY["proteinmpnn_score"].image_tag
            == TOOL_REGISTRY["ligandmpnn_score"].image_tag
        )
        assert TOOL_REGISTRY["proteinmpnn_score"].image_tag == "mpnn-score:1.0.0"

    def test_tool_runners_registered(self) -> None:
        assert "proteinmpnn_score" in TOOL_RUNNERS
        assert "ligandmpnn_score" in TOOL_RUNNERS
        assert TOOL_RUNNERS["proteinmpnn_score"] is MPNNScoreRunner
        assert TOOL_RUNNERS["ligandmpnn_score"] is MPNNScoreRunner

    def test_get_runner_returns_score_runner(self, config: AutobioConfig) -> None:
        with (
            patch("autobio.tools.base.ContainerManager"),
            patch("autobio.tools.base.GPUManager"),
        ):
            r = get_runner("proteinmpnn_score", config)
        assert isinstance(r, MPNNScoreRunner)
        assert r.tool_name == "proteinmpnn_score"
