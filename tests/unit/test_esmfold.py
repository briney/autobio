"""Tests for ESMFoldRunner — prepare_workspace, parse_output, host validation, and registration."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

from autobio.core.catalog import CATALOG, get_tool
from autobio.core.config import AutobioConfig
from autobio.core.registry import ToolCategory
from autobio.core.result import AutobioError
from autobio.core.workspace import Workspace
from autobio.schemas.structure_prediction import ESMFoldInput, StructurePredictionOutput
from autobio.tools import TOOL_RUNNERS, get_runner
from autobio.tools.esmfold import ESMFOLD_TOOL, ESMFoldRunner

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
    """Create an ESMFoldRunner with mocked ContainerManager and GPUManager.

    ``current_mode`` is set directly (rather than via ``run()``) so that
    ``prepare_workspace`` — which calls ``_apply_extra`` — can be exercised
    in isolation.
    """
    with (
        patch("autobio.tools.base.ContainerManager"),
        patch("autobio.tools.base.GPUManager"),
    ):
        runner = ESMFoldRunner("esmfold", config)
    runner.current_mode = get_tool("esmfold").modes["predict"]
    return runner


# ---------------------------------------------------------------------------
# TestESMFoldPrepareWorkspace
# ---------------------------------------------------------------------------


class TestESMFoldPrepareWorkspace:
    """Tests for ESMFoldRunner.prepare_workspace."""

    def test_fasta_written(self, runner: ESMFoldRunner, tmp_path: Path) -> None:
        """Input sequence is written as a FASTA file."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = ESMFoldInput(sequences={"A": "MVLSPADKTNVKAAWGKVGA"})
        runner.prepare_workspace(input_data, workspace)

        fasta_path = workspace.inputs_dir / "sequences.fasta"
        assert fasta_path.exists()
        content = fasta_path.read_text()
        assert ">A" in content
        assert "MVLSPADKTNVKAAWGKVGA" in content

    def test_fasta_text_input_normalizes(self, runner: ESMFoldRunner, tmp_path: Path) -> None:
        """FASTA text is normalized to a dict via GenericSequenceSet."""
        input_data = ESMFoldInput(sequences=">A\nMKT\n")
        assert input_data.sequences == {"A": "MKT"}

        workspace = Workspace.create(tmp_path / "ws")
        runner.prepare_workspace(input_data, workspace)

        content = (workspace.inputs_dir / "sequences.fasta").read_text()
        assert ">A" in content
        assert "MKT" in content

    def test_config_defaults(self, runner: ESMFoldRunner, tmp_path: Path) -> None:
        """Minimal input produces correct config.json."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = ESMFoldInput(sequences={"A": "MVLSPADKTNVKAAWGKVGA"})
        runner.prepare_workspace(input_data, workspace)

        cfg = json.loads(workspace.config_path.read_text())
        assert cfg["model_name"] == "facebook/esmfold_v1"
        assert cfg["input_fasta"] == "/workspace/inputs/sequences.fasta"
        assert cfg["output_dir"] == "/workspace/outputs/raw"
        assert cfg["hf_cache"] == "/app/esmfold/hf_cache"

    def test_config_full_dict_equality(self, runner: ESMFoldRunner, tmp_path: Path) -> None:
        """The full config.json dict is byte-compat with the pre-migration output."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = ESMFoldInput(sequences={"A": "MVLSPADKTNVKAAWGKVGA"})
        runner.prepare_workspace(input_data, workspace)

        cfg = json.loads(workspace.config_path.read_text())
        assert cfg == {
            "model_name": "facebook/esmfold_v1",
            "input_fasta": "/workspace/inputs/sequences.fasta",
            "output_dir": "/workspace/outputs/raw",
            "hf_cache": "/app/esmfold/hf_cache",
        }

    def test_extra_merged(self, runner: ESMFoldRunner, tmp_path: Path) -> None:
        """Extra dict keys appear in config.json."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = ESMFoldInput(
            sequences={"A": "MVLSPADKTNVKAAWGKVGA"},
            extra={"chunk_size": 128},
        )
        runner.prepare_workspace(input_data, workspace)

        cfg = json.loads(workspace.config_path.read_text())
        assert cfg["chunk_size"] == 128

    def test_extra_shadowing_typed_field_rejected(
        self, runner: ESMFoldRunner, tmp_path: Path
    ) -> None:
        """Extra keys that collide with typed input fields are rejected."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = ESMFoldInput(
            sequences={"A": "MVLSPADKTNVKAAWGKVGA"},
            extra={"num_models": 1},
        )
        with pytest.raises(AutobioError, match="collide with typed input fields"):
            runner.prepare_workspace(input_data, workspace)


# ---------------------------------------------------------------------------
# TestESMFoldHostValidation
# ---------------------------------------------------------------------------


class TestESMFoldHostValidation:
    """Tests for host-side input validation."""

    def test_empty_sequences_raises(self, runner: ESMFoldRunner, tmp_path: Path) -> None:
        workspace = Workspace.create(tmp_path / "ws")
        input_data = ESMFoldInput(sequences={})
        with pytest.raises(AutobioError, match="sequences must be non-empty"):
            runner.prepare_workspace(input_data, workspace)

    def test_multi_chain_raises(self, runner: ESMFoldRunner, tmp_path: Path) -> None:
        """ESMFold rejects multi-chain input with a clear message."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = ESMFoldInput(sequences={"A": "MVLSPADKTNVKAAWGKVGA", "B": "GVSEKL"})
        with pytest.raises(AutobioError, match="single-chain only"):
            runner.prepare_workspace(input_data, workspace)

    def test_templates_raises(self, runner: ESMFoldRunner, tmp_path: Path) -> None:
        """ESMFold rejects templates."""
        tmpl = tmp_path / "template.pdb"
        tmpl.write_text("ATOM      1  CA  ALA A   1       0.0  0.0  0.0\n")
        workspace = Workspace.create(tmp_path / "ws")
        input_data = ESMFoldInput(
            sequences={"A": "MVLSPADKTNVKAAWGKVGA"},
            templates=[tmpl],
        )
        with pytest.raises(AutobioError, match="does not use templates"):
            runner.prepare_workspace(input_data, workspace)

    def test_num_models_gt1_raises(self, runner: ESMFoldRunner, tmp_path: Path) -> None:
        """ESMFold rejects num_models > 1."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = ESMFoldInput(
            sequences={"A": "MVLSPADKTNVKAAWGKVGA"},
            num_models=5,
        )
        with pytest.raises(AutobioError, match="deterministic"):
            runner.prepare_workspace(input_data, workspace)

    def test_invalid_sequence_raises(self, runner: ESMFoldRunner, tmp_path: Path) -> None:
        workspace = Workspace.create(tmp_path / "ws")
        input_data = ESMFoldInput(sequences={"A": "MVLS123INVALID"})
        with pytest.raises(AutobioError, match="Invalid protein sequence"):
            runner.prepare_workspace(input_data, workspace)

    def test_single_chain_accepted(self, runner: ESMFoldRunner, tmp_path: Path) -> None:
        """Single chain with num_models=1 is accepted."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = ESMFoldInput(
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
    """Tests for catalog Tool and runner registration."""

    def test_esmfold_registered_as_catalog_tool(self) -> None:
        assert "esmfold" in CATALOG
        assert set(get_tool("esmfold").modes) == {"predict"}
        assert get_tool("esmfold").default_mode == "predict"
        assert get_tool("esmfold").category == ToolCategory.STRUCTURE_PREDICTION

    def test_supports_batch_false(self) -> None:
        assert get_tool("esmfold").modes["predict"].supports_batch is False

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
        notes = " ".join(get_tool("esmfold").modes["predict"].notes)
        assert "single-chain" in notes.lower()
        assert "deterministic" in notes.lower()
        assert "templates" in notes.lower()

    def test_esmfold_tool_constant_registered(self) -> None:
        assert ESMFOLD_TOOL.name == "esmfold"
        assert get_tool("esmfold") is ESMFOLD_TOOL


def test_info_snapshot_esmfold() -> None:
    from autobio.cli.formatters import OutputFormat, format_tool_info_catalog

    parsed = json.loads(format_tool_info_catalog(get_tool("esmfold"), OutputFormat.JSON))
    assert parsed["modes"][0]["name"] == "predict"
    props = parsed["modes"][0]["input_schema"]["properties"]
    assert props["sequences"]["x-autobio"]["widget"] == "sequence"
    assert props["sequences"]["x-autobio"]["flavor"] == "generic"
    assert "output_schema" in parsed["modes"][0]
