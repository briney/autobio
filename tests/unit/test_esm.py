"""Tests for ESMRunner — prepare_workspace, parse_output, host validation, and registration."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

from autobio.core.config import AutobioConfig
from autobio.core.registry import TOOL_REGISTRY, ToolCategory
from autobio.core.result import AutobioError
from autobio.core.workspace import Workspace
from autobio.schemas.embedding import (
    EmbeddingInput,
    EmbeddingOutput,
)
from autobio.tools import TOOL_RUNNERS, get_runner
from autobio.tools.esm import ESMRunner

if TYPE_CHECKING:
    from pathlib import Path


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def config() -> AutobioConfig:
    return AutobioConfig.resolve()


@pytest.fixture()
def esm2_runner(config: AutobioConfig) -> ESMRunner:
    """Create an ESMRunner (esm2) with mocked ContainerManager and GPUManager."""
    with (
        patch("autobio.tools.base.ContainerManager"),
        patch("autobio.tools.base.GPUManager"),
    ):
        return ESMRunner("esm2", config)


@pytest.fixture()
def esm1b_runner(config: AutobioConfig) -> ESMRunner:
    """Create an ESMRunner for esm1b."""
    with (
        patch("autobio.tools.base.ContainerManager"),
        patch("autobio.tools.base.GPUManager"),
    ):
        return ESMRunner("esm1b", config)


# ---------------------------------------------------------------------------
# TestESMPrepareWorkspace
# ---------------------------------------------------------------------------


class TestESMPrepareWorkspace:
    """Tests for ESMRunner.prepare_workspace."""

    @pytest.mark.parametrize(
        ("tool_name", "expected_model"),
        [
            ("esm1b", "facebook/esm1b_t33_650M_UR50S"),
            ("esm2", "facebook/esm2_t33_650M_UR50D"),
        ],
    )
    def test_model_config_per_tool(
        self,
        tool_name: str,
        expected_model: str,
        config: AutobioConfig,
        tmp_path: Path,
    ) -> None:
        """Config contains correct model_name per tool name."""
        with (
            patch("autobio.tools.base.ContainerManager"),
            patch("autobio.tools.base.GPUManager"),
        ):
            r = ESMRunner(tool_name, config)
        workspace = Workspace.create(tmp_path / "ws")
        input_data = EmbeddingInput(sequences={"seq1": "MVLSPADKTNVKAAWGKVGA"})
        r.prepare_workspace(input_data, workspace)

        cfg = json.loads(workspace.config_path.read_text())
        assert cfg["model_name"] == expected_model

    def test_fasta_written(self, esm2_runner: ESMRunner, tmp_path: Path) -> None:
        """Input sequences are written as a FASTA file in workspace/inputs/."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = EmbeddingInput(sequences={"seq1": "MKWVTFIS", "seq2": "GVSEKL"})
        esm2_runner.prepare_workspace(input_data, workspace)

        fasta_path = workspace.inputs_dir / "sequences.fasta"
        assert fasta_path.exists()
        content = fasta_path.read_text()
        assert ">seq1" in content
        assert ">seq2" in content
        assert "MKWVTFIS" in content
        assert "GVSEKL" in content

    def test_layer_in_config(self, esm2_runner: ESMRunner, tmp_path: Path) -> None:
        """Explicit layer value appears in config.json."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = EmbeddingInput(sequences={"seq1": "MVLSPADKTNVKAAWGKVGA"}, layer=20)
        esm2_runner.prepare_workspace(input_data, workspace)

        cfg = json.loads(workspace.config_path.read_text())
        assert cfg["layer"] == 20

    def test_layer_default_none(self, esm2_runner: ESMRunner, tmp_path: Path) -> None:
        """Layer defaults to None (final layer) in config.json."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = EmbeddingInput(sequences={"seq1": "MVLSPADKTNVKAAWGKVGA"})
        esm2_runner.prepare_workspace(input_data, workspace)

        cfg = json.loads(workspace.config_path.read_text())
        assert cfg["layer"] is None

    def test_pooling_default_mean(self, esm2_runner: ESMRunner, tmp_path: Path) -> None:
        """Pooling defaults to 'mean' when not specified."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = EmbeddingInput(sequences={"seq1": "MVLSPADKTNVKAAWGKVGA"})
        esm2_runner.prepare_workspace(input_data, workspace)

        cfg = json.loads(workspace.config_path.read_text())
        assert cfg["pooling"] == "mean"

    def test_pooling_in_config(self, esm2_runner: ESMRunner, tmp_path: Path) -> None:
        """Custom pooling value appears in config.json."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = EmbeddingInput(
            sequences={"seq1": "MVLSPADKTNVKAAWGKVGA"}, pooling="per_residue"
        )
        esm2_runner.prepare_workspace(input_data, workspace)

        cfg = json.loads(workspace.config_path.read_text())
        assert cfg["pooling"] == "per_residue"

    def test_esm2_checkpoint_selection(self, esm2_runner: ESMRunner, tmp_path: Path) -> None:
        """extra['checkpoint'] selects the correct ESM-2 model."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = EmbeddingInput(
            sequences={"seq1": "MVLSPADKTNVKAAWGKVGA"},
            extra={"checkpoint": "150M"},
        )
        esm2_runner.prepare_workspace(input_data, workspace)

        cfg = json.loads(workspace.config_path.read_text())
        assert cfg["model_name"] == "facebook/esm2_t30_150M_UR50D"

    def test_esm2_checkpoint_default(self, esm2_runner: ESMRunner, tmp_path: Path) -> None:
        """Without extra['checkpoint'], ESM-2 defaults to 650M."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = EmbeddingInput(sequences={"seq1": "MVLSPADKTNVKAAWGKVGA"})
        esm2_runner.prepare_workspace(input_data, workspace)

        cfg = json.loads(workspace.config_path.read_text())
        assert cfg["model_name"] == "facebook/esm2_t33_650M_UR50D"

    @pytest.mark.parametrize("checkpoint", ["8M", "35M", "150M", "650M", "3B", "15B"])
    def test_esm2_all_checkpoints_valid(
        self, config: AutobioConfig, checkpoint: str, tmp_path: Path
    ) -> None:
        """All defined checkpoint codes resolve without error."""
        with (
            patch("autobio.tools.base.ContainerManager"),
            patch("autobio.tools.base.GPUManager"),
        ):
            r = ESMRunner("esm2", config)
        workspace = Workspace.create(tmp_path / "ws")
        input_data = EmbeddingInput(
            sequences={"seq1": "MVLSPADKTNVKAAWGKVGA"},
            extra={"checkpoint": checkpoint},
        )
        r.prepare_workspace(input_data, workspace)

        cfg = json.loads(workspace.config_path.read_text())
        assert "facebook/esm2" in cfg["model_name"]

    def test_esm1b_ignores_checkpoint(self, esm1b_runner: ESMRunner, tmp_path: Path) -> None:
        """ESM-1b always uses its fixed model regardless of extra['checkpoint']."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = EmbeddingInput(
            sequences={"seq1": "MVLSPADKTNVKAAWGKVGA"},
            extra={"checkpoint": "8M"},
        )
        esm1b_runner.prepare_workspace(input_data, workspace)

        cfg = json.loads(workspace.config_path.read_text())
        assert cfg["model_name"] == "facebook/esm1b_t33_650M_UR50S"

    def test_extra_dict_merged(self, esm2_runner: ESMRunner, tmp_path: Path) -> None:
        """Non-consumed extra keys appear in config.json."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = EmbeddingInput(
            sequences={"seq1": "MVLSPADKTNVKAAWGKVGA"},
            extra={"batch_size": 16, "seed": 42},
        )
        esm2_runner.prepare_workspace(input_data, workspace)

        cfg = json.loads(workspace.config_path.read_text())
        assert cfg["batch_size"] == 16
        assert cfg["seed"] == 42

    def test_consumed_keys_excluded(self, esm2_runner: ESMRunner, tmp_path: Path) -> None:
        """Consumed extra key 'checkpoint' does not appear in config.json."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = EmbeddingInput(
            sequences={"seq1": "MVLSPADKTNVKAAWGKVGA"},
            extra={"checkpoint": "150M", "seed": 42},
        )
        esm2_runner.prepare_workspace(input_data, workspace)

        cfg = json.loads(workspace.config_path.read_text())
        assert "checkpoint" not in cfg
        assert cfg["seed"] == 42

    def test_defaults_applied(self, esm2_runner: ESMRunner, tmp_path: Path) -> None:
        """Minimal input produces sensible defaults."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = EmbeddingInput(sequences={"seq1": "MVLSPADKTNVKAAWGKVGA"})
        esm2_runner.prepare_workspace(input_data, workspace)

        cfg = json.loads(workspace.config_path.read_text())
        assert cfg["model_name"] == "facebook/esm2_t33_650M_UR50D"
        assert cfg["input_fasta"] == "/workspace/inputs/sequences.fasta"
        assert cfg["output_dir"] == "/workspace/outputs/raw"
        assert cfg["hf_cache"] == "/app/esm/hf_cache"
        assert cfg["pooling"] == "mean"
        assert cfg["layer"] is None


# ---------------------------------------------------------------------------
# TestESMHostValidation
# ---------------------------------------------------------------------------


class TestESMHostValidation:
    """Tests for host-side input validation."""

    def test_empty_sequences_raises(self, esm2_runner: ESMRunner, tmp_path: Path) -> None:
        workspace = Workspace.create(tmp_path / "ws")
        input_data = EmbeddingInput(sequences={})
        with pytest.raises(AutobioError, match="sequences must be non-empty"):
            esm2_runner.prepare_workspace(input_data, workspace)

    def test_invalid_sequence_raises(self, esm2_runner: ESMRunner, tmp_path: Path) -> None:
        workspace = Workspace.create(tmp_path / "ws")
        input_data = EmbeddingInput(sequences={"seq1": "MVLS123INVALID"})
        with pytest.raises(AutobioError, match="Invalid protein sequence"):
            esm2_runner.prepare_workspace(input_data, workspace)

    def test_invalid_layer_too_high_raises(self, esm2_runner: ESMRunner, tmp_path: Path) -> None:
        workspace = Workspace.create(tmp_path / "ws")
        input_data = EmbeddingInput(sequences={"seq1": "MVLSPADKTNVKAAWGKVGA"}, layer=50)
        with pytest.raises(AutobioError, match="layer must be between 0 and 33"):
            esm2_runner.prepare_workspace(input_data, workspace)

    def test_invalid_layer_negative_raises(self, esm2_runner: ESMRunner, tmp_path: Path) -> None:
        workspace = Workspace.create(tmp_path / "ws")
        input_data = EmbeddingInput(sequences={"seq1": "MVLSPADKTNVKAAWGKVGA"}, layer=-1)
        with pytest.raises(AutobioError, match="layer must be between 0 and 33"):
            esm2_runner.prepare_workspace(input_data, workspace)

    def test_layer_validation_checkpoint_aware(self, config: AutobioConfig, tmp_path: Path) -> None:
        """Layer validation uses the correct num_layers for the selected checkpoint."""
        with (
            patch("autobio.tools.base.ContainerManager"),
            patch("autobio.tools.base.GPUManager"),
        ):
            r = ESMRunner("esm2", config)
        workspace = Workspace.create(tmp_path / "ws")
        # 8M model has only 6 layers — layer=10 should fail
        input_data = EmbeddingInput(
            sequences={"seq1": "MVLSPADKTNVKAAWGKVGA"},
            layer=10,
            extra={"checkpoint": "8M"},
        )
        with pytest.raises(AutobioError, match="layer must be between 0 and 6"):
            r.prepare_workspace(input_data, workspace)

    def test_invalid_pooling_raises(self, esm2_runner: ESMRunner, tmp_path: Path) -> None:
        workspace = Workspace.create(tmp_path / "ws")
        input_data = EmbeddingInput(sequences={"seq1": "MVLSPADKTNVKAAWGKVGA"}, pooling="max")
        with pytest.raises(AutobioError, match="pooling must be one of"):
            esm2_runner.prepare_workspace(input_data, workspace)

    def test_invalid_checkpoint_raises(self, esm2_runner: ESMRunner, tmp_path: Path) -> None:
        workspace = Workspace.create(tmp_path / "ws")
        input_data = EmbeddingInput(
            sequences={"seq1": "MVLSPADKTNVKAAWGKVGA"},
            extra={"checkpoint": "999B"},
        )
        with pytest.raises(AutobioError, match="Unknown ESM-2 checkpoint"):
            esm2_runner.prepare_workspace(input_data, workspace)

    def test_valid_layer_zero(self, esm2_runner: ESMRunner, tmp_path: Path) -> None:
        """Layer 0 (input embedding) is valid."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = EmbeddingInput(sequences={"seq1": "MVLSPADKTNVKAAWGKVGA"}, layer=0)
        esm2_runner.prepare_workspace(input_data, workspace)

        cfg = json.loads(workspace.config_path.read_text())
        assert cfg["layer"] == 0

    def test_valid_pooling_cls(self, esm2_runner: ESMRunner, tmp_path: Path) -> None:
        """CLS pooling is accepted."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = EmbeddingInput(sequences={"seq1": "MVLSPADKTNVKAAWGKVGA"}, pooling="cls")
        esm2_runner.prepare_workspace(input_data, workspace)

        cfg = json.loads(workspace.config_path.read_text())
        assert cfg["pooling"] == "cls"


# ---------------------------------------------------------------------------
# TestESMParseOutput
# ---------------------------------------------------------------------------

_SINGLE_EMBEDDING_RESULT = {
    "embeddings": [
        {
            "sequence_id": "seq1",
            "embedding_path": "/workspace/outputs/standardized/seq1.npy",
            "dimension": 1280,
            "layer": 33,
            "pooling": "mean",
        }
    ],
    "model_name": "esm2_t33_650M_UR50D",
    "embedding_dimension": 1280,
}

_MULTI_EMBEDDING_RESULT = {
    "embeddings": [
        {
            "sequence_id": "seq1",
            "embedding_path": "/workspace/outputs/standardized/seq1.npy",
            "dimension": 1280,
            "layer": 33,
            "pooling": "mean",
        },
        {
            "sequence_id": "seq2",
            "embedding_path": "/workspace/outputs/standardized/seq2.npy",
            "dimension": 1280,
            "layer": 33,
            "pooling": "mean",
        },
    ],
    "model_name": "esm2_t33_650M_UR50D",
    "embedding_dimension": 1280,
}


class TestESMParseOutput:
    """Tests for ESMRunner.parse_output."""

    def test_parse_single_embedding(self, esm2_runner: ESMRunner, tmp_path: Path) -> None:
        workspace = Workspace.create(tmp_path / "ws")
        (workspace.std_output_dir / "result_data.json").write_text(
            json.dumps(_SINGLE_EMBEDDING_RESULT)
        )

        output = esm2_runner.parse_output(workspace)
        assert len(output.embeddings) == 1
        e = output.embeddings[0]
        assert e.sequence_id == "seq1"
        assert e.dimension == 1280
        assert e.layer == 33
        assert e.pooling == "mean"

    def test_parse_multiple_embeddings(self, esm2_runner: ESMRunner, tmp_path: Path) -> None:
        workspace = Workspace.create(tmp_path / "ws")
        (workspace.std_output_dir / "result_data.json").write_text(
            json.dumps(_MULTI_EMBEDDING_RESULT)
        )

        output = esm2_runner.parse_output(workspace)
        assert len(output.embeddings) == 2
        ids = {e.sequence_id for e in output.embeddings}
        assert ids == {"seq1", "seq2"}

    def test_model_name_populated(self, esm2_runner: ESMRunner, tmp_path: Path) -> None:
        workspace = Workspace.create(tmp_path / "ws")
        (workspace.std_output_dir / "result_data.json").write_text(
            json.dumps(_SINGLE_EMBEDDING_RESULT)
        )

        output = esm2_runner.parse_output(workspace)
        assert output.model_name == "esm2_t33_650M_UR50D"

    def test_embedding_dimension(self, esm2_runner: ESMRunner, tmp_path: Path) -> None:
        workspace = Workspace.create(tmp_path / "ws")
        (workspace.std_output_dir / "result_data.json").write_text(
            json.dumps(_SINGLE_EMBEDDING_RESULT)
        )

        output = esm2_runner.parse_output(workspace)
        assert output.embedding_dimension == 1280

    def test_container_paths_resolved(self, esm2_runner: ESMRunner, tmp_path: Path) -> None:
        """Container-internal /workspace/... paths are remapped to host workspace."""
        workspace = Workspace.create(tmp_path / "ws")
        (workspace.std_output_dir / "result_data.json").write_text(
            json.dumps(_SINGLE_EMBEDDING_RESULT)
        )

        output = esm2_runner.parse_output(workspace)
        expected = workspace.root / "outputs" / "standardized" / "seq1.npy"
        assert output.embeddings[0].embedding_path == expected

    def test_output_type(self, esm2_runner: ESMRunner, tmp_path: Path) -> None:
        workspace = Workspace.create(tmp_path / "ws")
        (workspace.std_output_dir / "result_data.json").write_text(
            json.dumps(_SINGLE_EMBEDDING_RESULT)
        )

        output = esm2_runner.parse_output(workspace)
        assert isinstance(output, EmbeddingOutput)

    def test_raw_output_path(self, esm2_runner: ESMRunner, tmp_path: Path) -> None:
        workspace = Workspace.create(tmp_path / "ws")
        (workspace.std_output_dir / "result_data.json").write_text(
            json.dumps(_SINGLE_EMBEDDING_RESULT)
        )

        output = esm2_runner.parse_output(workspace)
        assert output.raw_output_path == workspace.raw_output_dir


# ---------------------------------------------------------------------------
# TestESMRegistration
# ---------------------------------------------------------------------------


class TestESMRegistration:
    """Tests for tool and runner registration."""

    def test_esm1b_in_registry(self) -> None:
        assert "esm1b" in TOOL_REGISTRY
        entry = TOOL_REGISTRY["esm1b"]
        assert entry.category == ToolCategory.EMBEDDING
        assert entry.input_schema is EmbeddingInput
        assert entry.output_schema is EmbeddingOutput
        assert entry.requires_gpu is True

    def test_esm2_in_registry(self) -> None:
        assert "esm2" in TOOL_REGISTRY
        entry = TOOL_REGISTRY["esm2"]
        assert entry.category == ToolCategory.EMBEDDING
        assert "checkpoint" in entry.description.lower()

    def test_both_share_image_tag(self) -> None:
        assert TOOL_REGISTRY["esm1b"].image_tag == TOOL_REGISTRY["esm2"].image_tag

    def test_supports_batch(self) -> None:
        assert TOOL_REGISTRY["esm1b"].supports_batch is True
        assert TOOL_REGISTRY["esm2"].supports_batch is True

    def test_tool_runners_registered(self) -> None:
        assert "esm1b" in TOOL_RUNNERS
        assert "esm2" in TOOL_RUNNERS
        assert TOOL_RUNNERS["esm1b"] is ESMRunner
        assert TOOL_RUNNERS["esm2"] is ESMRunner

    def test_get_runner_returns_esm_runner(self, config: AutobioConfig) -> None:
        with (
            patch("autobio.tools.base.ContainerManager"),
            patch("autobio.tools.base.GPUManager"),
        ):
            r = get_runner("esm2", config)
        assert isinstance(r, ESMRunner)
        assert r.tool_name == "esm2"

    def test_notes_populated(self) -> None:
        """Notes contain key operational guidance."""
        esm2_notes = " ".join(TOOL_REGISTRY["esm2"].notes)
        assert "checkpoint" in esm2_notes.lower()
        assert "1022" in esm2_notes

    def test_input_format_populated(self) -> None:
        """Input format describes sequence input."""
        fmt = " ".join(TOOL_REGISTRY["esm2"].input_format)
        assert "amino acid" in fmt.lower()
