"""Unit tests for the migrated esm1b / esm2 Tools (mode: embed)."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

from autobio.core.catalog import get_tool
from autobio.core.config import AutobioConfig
from autobio.core.result import AutobioError
from autobio.tools.esm import ESMRunner

if TYPE_CHECKING:
    from pathlib import Path


def _make_runner(tool_name: str) -> ESMRunner:
    with patch("autobio.tools.base.ContainerManager"), patch("autobio.tools.base.GPUManager"):
        runner = ESMRunner(tool_name, AutobioConfig.resolve())
    runner.current_mode = get_tool(tool_name).modes["embed"]
    return runner


def _written_config(runner: ESMRunner, input_data, tmp_path: Path) -> dict:
    from autobio.core.workspace import Workspace

    ws = Workspace.create(tmp_path / "ws")
    try:
        runner.prepare_workspace(input_data, ws)
        return json.loads((ws.root / "config.json").read_text())
    finally:
        ws.cleanup()


def test_esm_registered_as_single_mode_tools() -> None:
    import autobio.tools  # noqa: F401
    from autobio.core.catalog import CATALOG
    from autobio.core.registry import TOOL_REGISTRY

    assert "esm1b" in CATALOG and "esm2" in CATALOG
    assert set(get_tool("esm1b").modes) == {"embed"}
    assert "esm1b" not in TOOL_REGISTRY and "esm2" not in TOOL_REGISTRY


def test_esm1b_config_model_name(tmp_path: Path) -> None:
    from autobio.schemas.embedding import ESMEmbedInput

    runner = _make_runner("esm1b")
    cfg = _written_config(runner, ESMEmbedInput(sequences={"s1": "MKT"}), tmp_path)
    assert cfg["model_name"] == "facebook/esm1b_t33_650M_UR50S"
    assert cfg["pooling"] == "mean"
    assert cfg["input_fasta"] == "/workspace/inputs/sequences.fasta"


def test_esm2_checkpoint_resolves_model(tmp_path: Path) -> None:
    from autobio.schemas.embedding import ESM2Input

    runner = _make_runner("esm2")
    cfg = _written_config(runner, ESM2Input(sequences={"s1": "MKT"}, checkpoint="150M"), tmp_path)
    assert cfg["model_name"] == "facebook/esm2_t30_150M_UR50D"


def test_esm_accepts_fasta_text(tmp_path: Path) -> None:
    from autobio.schemas.embedding import ESMEmbedInput

    runner = _make_runner("esm1b")
    inp = ESMEmbedInput(sequences=">s1\nMKT\n>s2\nGGG\n")
    assert inp.sequences == {"s1": "MKT", "s2": "GGG"}  # GenericSequenceSet normalized it
    cfg = _written_config(runner, inp, tmp_path)
    assert cfg["model_name"] == "facebook/esm1b_t33_650M_UR50S"


def test_esm_invalid_protein_sequence_rejected(tmp_path: Path) -> None:
    from autobio.schemas.embedding import ESMEmbedInput

    runner = _make_runner("esm1b")
    with pytest.raises(AutobioError, match="Invalid protein sequence"):
        _written_config(runner, ESMEmbedInput(sequences={"s1": "XZ123"}), tmp_path)


def test_esm2_layer_out_of_range_checkpoint_aware_rejected(tmp_path: Path) -> None:
    """esm2 8M has num_layers=6; layer=10 must be rejected against that bound, not esm1b's 33."""
    from autobio.schemas.embedding import ESM2Input

    runner = _make_runner("esm2")
    with pytest.raises(AutobioError, match="between 0 and 6"):
        _written_config(
            runner,
            ESM2Input(sequences={"s1": "MKT"}, checkpoint="8M", layer=10),
            tmp_path,
        )


def test_esm2_negative_layer_rejected(tmp_path: Path) -> None:
    from autobio.schemas.embedding import ESM2Input

    runner = _make_runner("esm2")
    with pytest.raises(AutobioError, match="between 0 and 6"):
        _written_config(
            runner,
            ESM2Input(sequences={"s1": "MKT"}, checkpoint="8M", layer=-1),
            tmp_path,
        )


def test_esm_empty_sequences_rejected(tmp_path: Path) -> None:
    from autobio.schemas.embedding import ESMEmbedInput

    runner = _make_runner("esm1b")
    with pytest.raises(AutobioError, match="sequences must be non-empty"):
        _written_config(runner, ESMEmbedInput(sequences={}), tmp_path)


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


def test_parse_output_single_embedding(tmp_path: Path) -> None:
    from autobio.core.workspace import Workspace
    from autobio.schemas.embedding import EmbeddingOutput

    runner = _make_runner("esm2")
    ws = Workspace.create(tmp_path / "ws")
    try:
        (ws.std_output_dir / "result_data.json").write_text(json.dumps(_SINGLE_EMBEDDING_RESULT))
        output = runner.parse_output(ws)
        assert isinstance(output, EmbeddingOutput)
        assert output.model_name == "esm2_t33_650M_UR50D"
        assert output.embedding_dimension == 1280
        assert len(output.embeddings) == 1
        e = output.embeddings[0]
        assert e.sequence_id == "seq1"
        assert e.dimension == 1280
        assert e.layer == 33
        assert e.pooling == "mean"
        # container-path remapping: /workspace/... -> host workspace root
        assert e.embedding_path == ws.root / "outputs" / "standardized" / "seq1.npy"
    finally:
        ws.cleanup()


def test_parse_output_multiple_embeddings(tmp_path: Path) -> None:
    from autobio.core.workspace import Workspace

    runner = _make_runner("esm2")
    ws = Workspace.create(tmp_path / "ws")
    try:
        (ws.std_output_dir / "result_data.json").write_text(json.dumps(_MULTI_EMBEDDING_RESULT))
        output = runner.parse_output(ws)
        assert len(output.embeddings) == 2
        assert {e.sequence_id for e in output.embeddings} == {"seq1", "seq2"}
    finally:
        ws.cleanup()


def test_extra_shadowing_typed_field_rejected(tmp_path: Path) -> None:
    from autobio.schemas.embedding import ESMEmbedInput

    runner = _make_runner("esm1b")
    with pytest.raises(AutobioError, match="shadow typed input fields"):
        _written_config(
            runner, ESMEmbedInput(sequences={"s1": "MKT"}, extra={"layer": 5}), tmp_path
        )


def test_info_snapshot_esm2() -> None:
    import autobio.tools  # noqa: F401
    from autobio.cli.formatters import OutputFormat, format_tool_info_catalog

    parsed = json.loads(format_tool_info_catalog(get_tool("esm2"), OutputFormat.JSON))
    assert parsed["modes"][0]["name"] == "embed"
    props = parsed["modes"][0]["input_schema"]["properties"]
    assert props["sequences"]["x-autobio"]["widget"] == "sequence"
    assert props["sequences"]["x-autobio"]["flavor"] == "generic"
    assert props["layer"]["x-autobio"]["widget"] == "number"
    assert props["layer"]["x-autobio"]["tier"] == "advanced"
    assert props["pooling"]["x-autobio"]["widget"] == "select"
    assert props["pooling"]["x-autobio"]["tier"] == "primary"
    assert props["checkpoint"]["x-autobio"]["widget"] == "select"
    assert props["checkpoint"]["default"] == "650M"
    assert "output_schema" in parsed["modes"][0]


def test_esm_tool_constants_registered() -> None:
    import autobio.tools  # noqa: F401
    from autobio.tools.esm import ESM1B_TOOL, ESM2_TOOL

    assert ESM1B_TOOL.name == "esm1b"
    assert ESM2_TOOL.name == "esm2"
    assert get_tool("esm1b") is ESM1B_TOOL
    assert get_tool("esm2") is ESM2_TOOL
