"""Unit tests for the migrated esm1b / esm2 Tools (mode: embed)."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING
from unittest.mock import patch

from autobio.core.catalog import get_tool
from autobio.core.config import AutobioConfig
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


def test_info_snapshot_esm2() -> None:
    import autobio.tools  # noqa: F401
    from autobio.cli.formatters import OutputFormat, format_tool_info_catalog

    parsed = json.loads(format_tool_info_catalog(get_tool("esm2"), OutputFormat.JSON))
    props = parsed["modes"][0]["input_schema"]["properties"]
    assert props["sequences"]["x-autobio"]["widget"] == "sequence"
    assert props["sequences"]["x-autobio"]["flavor"] == "generic"
    assert props["checkpoint"]["default"] == "650M"
    assert "output_schema" in parsed["modes"][0]
