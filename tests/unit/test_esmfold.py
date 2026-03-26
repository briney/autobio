"""Tests for ESMFoldRunner — prepare_workspace, parse_output, host validation, and registration."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

from autobio.core.config import AutobioConfig
from autobio.core.registry import TOOL_REGISTRY, ToolCategory
from autobio.core.result import AutobioError
from autobio.core.workspace import Workspace
from autobio.schemas.structure_prediction import (
    StructurePredictionInput,
    StructurePredictionOutput,
)
from autobio.tools import TOOL_RUNNERS, get_runner
from autobio.tools.esmfold import ESMFoldRunner

if TYPE_CHECKING:
    from pathlib import Path


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def config() -> AutobioConfig:
    return AutobioConfig.resolve()


@pytest.fixture()
def runner(config: AutobioConfig) -> ESMFoldRunner:
    """Create an ESMFoldRunner with mocked ContainerManager and GPUManager."""
    with (
        patch("autobio.tools.base.ContainerManager"),
        patch("autobio.tools.base.GPUManager"),
    ):
        return ESMFoldRunner("esmfold", config)


# ---------------------------------------------------------------------------
# TestESMFoldPrepareWorkspace
# ---------------------------------------------------------------------------


class TestESMFoldPrepareWorkspace:
    """Tests for ESMFoldRunner.prepare_workspace."""

    def test_fasta_written(self, runner: ESMFoldRunner, tmp_path: Path) -> None:
        """Input sequence is written as a FASTA file."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = StructurePredictionInput(sequences={"A": "MVLSPADKTNVKAAWGKVGA"})
        runner.prepare_workspace(input_data, workspace)

        fasta_path = workspace.inputs_dir / "sequences.fasta"
        assert fasta_path.exists()
        content = fasta_path.read_text()
        assert ">A" in content
        assert "MVLSPADKTNVKAAWGKVGA" in content

    def test_config_defaults(self, runner: ESMFoldRunner, tmp_path: Path) -> None:
        """Minimal input produces correct config.json."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = StructurePredictionInput(sequences={"A": "MVLSPADKTNVKAAWGKVGA"})
        runner.prepare_workspace(input_data, workspace)

        cfg = json.loads(workspace.config_path.read_text())
        assert cfg["model_name"] == "facebook/esmfold_v1"
        assert cfg["input_fasta"] == "/workspace/inputs/sequences.fasta"
        assert cfg["output_dir"] == "/workspace/outputs/raw"
        assert cfg["hf_cache"] == "/app/esmfold/hf_cache"

    def test_extra_merged(self, runner: ESMFoldRunner, tmp_path: Path) -> None:
        """Extra dict keys appear in config.json."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = StructurePredictionInput(
            sequences={"A": "MVLSPADKTNVKAAWGKVGA"},
            extra={"chunk_size": 128},
        )
        runner.prepare_workspace(input_data, workspace)

        cfg = json.loads(workspace.config_path.read_text())
        assert cfg["chunk_size"] == 128


# ---------------------------------------------------------------------------
# TestESMFoldHostValidation
# ---------------------------------------------------------------------------


class TestESMFoldHostValidation:
    """Tests for host-side input validation."""

    def test_empty_sequences_raises(self, runner: ESMFoldRunner, tmp_path: Path) -> None:
        workspace = Workspace.create(tmp_path / "ws")
        input_data = StructurePredictionInput(sequences={})
        with pytest.raises(AutobioError, match="sequences must be non-empty"):
            runner.prepare_workspace(input_data, workspace)

    def test_multi_chain_raises(self, runner: ESMFoldRunner, tmp_path: Path) -> None:
        """ESMFold rejects multi-chain input with a clear message."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = StructurePredictionInput(
            sequences={"A": "MVLSPADKTNVKAAWGKVGA", "B": "GVSEKL"}
        )
        with pytest.raises(AutobioError, match="single-chain only"):
            runner.prepare_workspace(input_data, workspace)

    def test_templates_raises(self, runner: ESMFoldRunner, tmp_path: Path) -> None:
        """ESMFold rejects templates."""
        tmpl = tmp_path / "template.pdb"
        tmpl.write_text("ATOM      1  CA  ALA A   1       0.0  0.0  0.0\n")
        workspace = Workspace.create(tmp_path / "ws")
        input_data = StructurePredictionInput(
            sequences={"A": "MVLSPADKTNVKAAWGKVGA"},
            templates=[tmpl],
        )
        with pytest.raises(AutobioError, match="does not use templates"):
            runner.prepare_workspace(input_data, workspace)

    def test_num_models_gt1_raises(self, runner: ESMFoldRunner, tmp_path: Path) -> None:
        """ESMFold rejects num_models > 1."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = StructurePredictionInput(
            sequences={"A": "MVLSPADKTNVKAAWGKVGA"},
            num_models=5,
        )
        with pytest.raises(AutobioError, match="deterministic"):
            runner.prepare_workspace(input_data, workspace)

    def test_invalid_sequence_raises(self, runner: ESMFoldRunner, tmp_path: Path) -> None:
        workspace = Workspace.create(tmp_path / "ws")
        input_data = StructurePredictionInput(sequences={"A": "MVLS123INVALID"})
        with pytest.raises(AutobioError, match="Invalid protein sequence"):
            runner.prepare_workspace(input_data, workspace)

    def test_single_chain_accepted(self, runner: ESMFoldRunner, tmp_path: Path) -> None:
        """Single chain with num_models=1 is accepted."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = StructurePredictionInput(
            sequences={"A": "MVLSPADKTNVKAAWGKVGA"},
            num_models=1,
        )
        runner.prepare_workspace(input_data, workspace)
        assert (workspace.inputs_dir / "sequences.fasta").exists()


# ---------------------------------------------------------------------------
# TestESMFoldParseOutput
# ---------------------------------------------------------------------------

_SINGLE_STRUCTURE_RESULT = {
    "structures": [
        {
            "model_rank": 1,
            "structure_path": "/workspace/outputs/standardized/prediction.pdb",
            "plddt_per_residue": [85.2, 90.1, 88.4, 92.0],
            "plddt_mean": 88.925,
            "ptm": 0.82,
            "iptm": None,
            "chain_mapping": None,
        }
    ],
    "confidence": {
        "best_plddt_mean": 88.925,
        "best_ptm": 0.82,
        "best_iptm": None,
    },
}


class TestESMFoldParseOutput:
    """Tests for ESMFoldRunner.parse_output."""

    def test_parse_single_structure(self, runner: ESMFoldRunner, tmp_path: Path) -> None:
        workspace = Workspace.create(tmp_path / "ws")
        (workspace.std_output_dir / "result_data.json").write_text(
            json.dumps(_SINGLE_STRUCTURE_RESULT)
        )

        output = runner.parse_output(workspace)
        assert len(output.structures) == 1
        s = output.structures[0]
        assert s.model_rank == 1
        assert s.plddt_mean == pytest.approx(88.925)
        assert s.ptm == pytest.approx(0.82)

    def test_iptm_is_none(self, runner: ESMFoldRunner, tmp_path: Path) -> None:
        """Single-chain prediction has no ipTM."""
        workspace = Workspace.create(tmp_path / "ws")
        (workspace.std_output_dir / "result_data.json").write_text(
            json.dumps(_SINGLE_STRUCTURE_RESULT)
        )

        output = runner.parse_output(workspace)
        assert output.structures[0].iptm is None

    def test_confidence_metrics(self, runner: ESMFoldRunner, tmp_path: Path) -> None:
        workspace = Workspace.create(tmp_path / "ws")
        (workspace.std_output_dir / "result_data.json").write_text(
            json.dumps(_SINGLE_STRUCTURE_RESULT)
        )

        output = runner.parse_output(workspace)
        assert output.confidence.best_plddt_mean == pytest.approx(88.925)
        assert output.confidence.best_ptm == pytest.approx(0.82)
        assert output.confidence.best_iptm is None

    def test_container_paths_resolved(self, runner: ESMFoldRunner, tmp_path: Path) -> None:
        """Container-internal paths are remapped to host workspace."""
        workspace = Workspace.create(tmp_path / "ws")
        (workspace.std_output_dir / "result_data.json").write_text(
            json.dumps(_SINGLE_STRUCTURE_RESULT)
        )

        output = runner.parse_output(workspace)
        expected = workspace.root / "outputs" / "standardized" / "prediction.pdb"
        assert output.structures[0].structure_path == expected

    def test_output_type(self, runner: ESMFoldRunner, tmp_path: Path) -> None:
        workspace = Workspace.create(tmp_path / "ws")
        (workspace.std_output_dir / "result_data.json").write_text(
            json.dumps(_SINGLE_STRUCTURE_RESULT)
        )

        output = runner.parse_output(workspace)
        assert isinstance(output, StructurePredictionOutput)

    def test_plddt_per_residue(self, runner: ESMFoldRunner, tmp_path: Path) -> None:
        workspace = Workspace.create(tmp_path / "ws")
        (workspace.std_output_dir / "result_data.json").write_text(
            json.dumps(_SINGLE_STRUCTURE_RESULT)
        )

        output = runner.parse_output(workspace)
        assert output.structures[0].plddt_per_residue is not None
        assert len(output.structures[0].plddt_per_residue) == 4


# ---------------------------------------------------------------------------
# TestESMFoldRegistration
# ---------------------------------------------------------------------------


class TestESMFoldRegistration:
    """Tests for tool and runner registration."""

    def test_esmfold_in_registry(self) -> None:
        assert "esmfold" in TOOL_REGISTRY
        entry = TOOL_REGISTRY["esmfold"]
        assert entry.category == ToolCategory.STRUCTURE_PREDICTION
        assert entry.input_schema is StructurePredictionInput
        assert entry.output_schema is StructurePredictionOutput
        assert entry.requires_gpu is True

    def test_supports_batch_false(self) -> None:
        assert TOOL_REGISTRY["esmfold"].supports_batch is False

    def test_tool_runner_registered(self) -> None:
        assert "esmfold" in TOOL_RUNNERS
        assert TOOL_RUNNERS["esmfold"] is ESMFoldRunner

    def test_get_runner_returns_esmfold_runner(self, config: AutobioConfig) -> None:
        with (
            patch("autobio.tools.base.ContainerManager"),
            patch("autobio.tools.base.GPUManager"),
        ):
            r = get_runner("esmfold", config)
        assert isinstance(r, ESMFoldRunner)
        assert r.tool_name == "esmfold"

    def test_notes_populated(self) -> None:
        """Notes contain key ESMFold limitations."""
        notes = " ".join(TOOL_REGISTRY["esmfold"].notes)
        assert "single-chain" in notes.lower()
        assert "deterministic" in notes.lower()
        assert "templates" in notes.lower()

    def test_input_format_populated(self) -> None:
        fmt = " ".join(TOOL_REGISTRY["esmfold"].input_format)
        assert "single" in fmt.lower()
        assert "amino acid" in fmt.lower()
