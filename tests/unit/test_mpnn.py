"""Tests for MPNNRunner — prepare_workspace, parse_output, and registration."""

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
from autobio.tools import TOOL_RUNNERS, get_runner
from autobio.tools.mpnn import _CHECKPOINT_DIR, MPNNRunner

if TYPE_CHECKING:
    from pathlib import Path


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def config() -> AutobioConfig:
    return AutobioConfig.resolve()


@pytest.fixture()
def runner(config: AutobioConfig) -> MPNNRunner:
    """Create an MPNNRunner with mocked ContainerManager and GPUManager."""
    with (
        patch("autobio.tools.base.ContainerManager"),
        patch("autobio.tools.base.GPUManager"),
    ):
        return MPNNRunner("proteinmpnn", config)


@pytest.fixture()
def ligand_runner(config: AutobioConfig) -> MPNNRunner:
    """Create an MPNNRunner for ligandmpnn."""
    with (
        patch("autobio.tools.base.ContainerManager"),
        patch("autobio.tools.base.GPUManager"),
    ):
        return MPNNRunner("ligandmpnn", config)


@pytest.fixture()
def sample_pdb(tmp_path: Path) -> Path:
    """Write a minimal PDB file for testing."""
    pdb_path = tmp_path / "test.pdb"
    pdb_path.write_text("ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00\nEND\n")
    return pdb_path


# ---------------------------------------------------------------------------
# TestMPNNPrepareWorkspace
# ---------------------------------------------------------------------------


class TestMPNNPrepareWorkspace:
    """Tests for MPNNRunner.prepare_workspace."""

    @pytest.mark.parametrize(
        ("tool_name", "expected_model_type", "expected_checkpoint"),
        [
            ("proteinmpnn", "protein_mpnn", "proteinmpnn_v_48_020.pt"),
            ("ligandmpnn", "ligand_mpnn", "ligandmpnn_v_32_010_25.pt"),
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
            r = MPNNRunner(tool_name, config)
        workspace = Workspace.create(tmp_path / "ws")
        input_data = InverseFoldingInput(structure_path=sample_pdb)
        r.prepare_workspace(input_data, workspace)

        cfg = json.loads(workspace.config_path.read_text())
        assert cfg["model_type"] == expected_model_type
        assert cfg["checkpoint_path"] == f"{_CHECKPOINT_DIR}/{expected_checkpoint}"
        assert cfg["is_legacy_weights"] is True

    def test_structure_file_copied(
        self, runner: MPNNRunner, tmp_path: Path, sample_pdb: Path
    ) -> None:
        """Input structure is copied to inputs/ and config references container path."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = InverseFoldingInput(structure_path=sample_pdb)
        runner.prepare_workspace(input_data, workspace)

        # File exists in workspace inputs
        copied = workspace.inputs_dir / "test.pdb"
        assert copied.exists()
        assert copied.read_text() == sample_pdb.read_text()

        # Config references container path
        cfg = json.loads(workspace.config_path.read_text())
        assert cfg["structure_path"] == "/workspace/inputs/test.pdb"

    def test_num_sequences_maps_to_number_of_batches(
        self, runner: MPNNRunner, tmp_path: Path, sample_pdb: Path
    ) -> None:
        workspace = Workspace.create(tmp_path / "ws")
        input_data = InverseFoldingInput(structure_path=sample_pdb, num_sequences=5)
        runner.prepare_workspace(input_data, workspace)

        cfg = json.loads(workspace.config_path.read_text())
        assert cfg["number_of_batches"] == 5

    def test_temperature_passthrough(
        self, runner: MPNNRunner, tmp_path: Path, sample_pdb: Path
    ) -> None:
        workspace = Workspace.create(tmp_path / "ws")
        input_data = InverseFoldingInput(structure_path=sample_pdb, temperature=0.5)
        runner.prepare_workspace(input_data, workspace)

        cfg = json.loads(workspace.config_path.read_text())
        assert cfg["temperature"] == 0.5

    def test_defaults_applied(self, runner: MPNNRunner, tmp_path: Path, sample_pdb: Path) -> None:
        """Minimal input produces sensible defaults."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = InverseFoldingInput(structure_path=sample_pdb)
        runner.prepare_workspace(input_data, workspace)

        cfg = json.loads(workspace.config_path.read_text())
        assert cfg["temperature"] == 0.1
        assert cfg["number_of_batches"] == 1

    def test_chains_to_design(self, runner: MPNNRunner, tmp_path: Path, sample_pdb: Path) -> None:
        workspace = Workspace.create(tmp_path / "ws")
        input_data = InverseFoldingInput(structure_path=sample_pdb, chains_to_design=["A", "B"])
        runner.prepare_workspace(input_data, workspace)

        cfg = json.loads(workspace.config_path.read_text())
        assert cfg["designed_chains"] == "A,B"

    def test_fixed_positions_mapping(
        self, runner: MPNNRunner, tmp_path: Path, sample_pdb: Path
    ) -> None:
        """fixed_positions dict is transformed to comma-separated residue IDs."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = InverseFoldingInput(
            structure_path=sample_pdb, fixed_positions={"A": [1, 5, 10], "B": [3]}
        )
        runner.prepare_workspace(input_data, workspace)

        cfg = json.loads(workspace.config_path.read_text())
        ids = cfg["fixed_residues"].split(",")
        assert set(ids) == {"A1", "A5", "A10", "B3"}

    def test_fixed_positions_overrides_designed_chains(
        self, runner: MPNNRunner, tmp_path: Path, sample_pdb: Path
    ) -> None:
        """When both are given, fixed_positions takes precedence (they're mutually exclusive)."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = InverseFoldingInput(
            structure_path=sample_pdb,
            chains_to_design=["A"],
            fixed_positions={"B": [1, 2]},
        )
        runner.prepare_workspace(input_data, workspace)

        cfg = json.loads(workspace.config_path.read_text())
        assert "fixed_residues" in cfg
        assert "designed_chains" not in cfg

    def test_extra_dict_merged(self, runner: MPNNRunner, tmp_path: Path, sample_pdb: Path) -> None:
        """Extra dict keys appear at the top level of config.json."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = InverseFoldingInput(
            structure_path=sample_pdb,
            extra={"omit": '["CYS","TRP"]', "seed": 42, "batch_size": 4},
        )
        runner.prepare_workspace(input_data, workspace)

        cfg = json.loads(workspace.config_path.read_text())
        assert cfg["omit"] == '["CYS","TRP"]'
        assert cfg["seed"] == 42
        assert cfg["batch_size"] == 4


# ---------------------------------------------------------------------------
# TestMPNNParseOutput
# ---------------------------------------------------------------------------

_SINGLE_SEQ_RESULT = {
    "designed_sequences": [
        {
            "rank": 1,
            "sequence": {"A": "TVCCPSEEAKKKYEECRKPGTPDEECAKATGCIIIPGTKCPPDYPY"},
            "score": None,
            "recovery": 0.5652,
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


class TestMPNNParseOutput:
    """Tests for MPNNRunner.parse_output."""

    def test_parse_single_sequence(self, runner: MPNNRunner, tmp_path: Path) -> None:
        workspace = Workspace.create(tmp_path / "ws")
        (workspace.std_output_dir / "result_data.json").write_text(json.dumps(_SINGLE_SEQ_RESULT))

        output = runner.parse_output(workspace)
        assert len(output.designed_sequences) == 1
        seq = output.designed_sequences[0]
        assert seq.rank == 1
        assert seq.sequence["A"].startswith("TVCCPS")
        assert seq.recovery == pytest.approx(0.5652)
        assert seq.score is None

    def test_parse_multiple_sequences(self, runner: MPNNRunner, tmp_path: Path) -> None:
        workspace = Workspace.create(tmp_path / "ws")
        (workspace.std_output_dir / "result_data.json").write_text(json.dumps(_MULTI_SEQ_RESULT))

        output = runner.parse_output(workspace)
        assert len(output.designed_sequences) == 3
        assert output.designed_sequences[0].rank == 1
        assert output.designed_sequences[2].rank == 3

    def test_parse_multi_chain(self, runner: MPNNRunner, tmp_path: Path) -> None:
        workspace = Workspace.create(tmp_path / "ws")
        (workspace.std_output_dir / "result_data.json").write_text(json.dumps(_MULTI_SEQ_RESULT))

        output = runner.parse_output(workspace)
        seq = output.designed_sequences[0]
        assert "A" in seq.sequence
        assert "B" in seq.sequence

    def test_parse_with_native_sequence(self, runner: MPNNRunner, tmp_path: Path) -> None:
        workspace = Workspace.create(tmp_path / "ws")
        (workspace.std_output_dir / "result_data.json").write_text(json.dumps(_SINGLE_SEQ_RESULT))

        output = runner.parse_output(workspace)
        assert output.native_sequence is not None
        assert "A" in output.native_sequence

    def test_parse_without_native_sequence(self, runner: MPNNRunner, tmp_path: Path) -> None:
        data = {**_SINGLE_SEQ_RESULT, "native_sequence": None}
        workspace = Workspace.create(tmp_path / "ws")
        (workspace.std_output_dir / "result_data.json").write_text(json.dumps(data))

        output = runner.parse_output(workspace)
        assert output.native_sequence is None

    def test_output_type(self, runner: MPNNRunner, tmp_path: Path) -> None:
        workspace = Workspace.create(tmp_path / "ws")
        (workspace.std_output_dir / "result_data.json").write_text(json.dumps(_SINGLE_SEQ_RESULT))

        output = runner.parse_output(workspace)
        assert isinstance(output, InverseFoldingOutput)

    def test_raw_output_path(self, runner: MPNNRunner, tmp_path: Path) -> None:
        workspace = Workspace.create(tmp_path / "ws")
        (workspace.std_output_dir / "result_data.json").write_text(json.dumps(_SINGLE_SEQ_RESULT))

        output = runner.parse_output(workspace)
        assert output.raw_output_path == workspace.raw_output_dir


# ---------------------------------------------------------------------------
# TestMPNNRegistration
# ---------------------------------------------------------------------------


class TestMPNNRegistration:
    """Tests for tool and runner registration."""

    def test_proteinmpnn_in_registry(self) -> None:
        assert "proteinmpnn" in TOOL_REGISTRY
        entry = TOOL_REGISTRY["proteinmpnn"]
        assert entry.category == ToolCategory.INVERSE_FOLDING
        assert entry.input_schema is InverseFoldingInput
        assert entry.output_schema is InverseFoldingOutput
        assert entry.requires_gpu is True

    def test_ligandmpnn_in_registry(self) -> None:
        assert "ligandmpnn" in TOOL_REGISTRY
        entry = TOOL_REGISTRY["ligandmpnn"]
        assert entry.category == ToolCategory.INVERSE_FOLDING
        assert entry.description.lower().count("ligand") >= 1

    def test_both_share_image_tag(self) -> None:
        assert TOOL_REGISTRY["proteinmpnn"].image_tag == TOOL_REGISTRY["ligandmpnn"].image_tag

    def test_tool_runners_registered(self) -> None:
        assert "proteinmpnn" in TOOL_RUNNERS
        assert "ligandmpnn" in TOOL_RUNNERS
        assert TOOL_RUNNERS["proteinmpnn"] is MPNNRunner
        assert TOOL_RUNNERS["ligandmpnn"] is MPNNRunner

    def test_get_runner_returns_mpnn_runner(self, config: AutobioConfig) -> None:
        with (
            patch("autobio.tools.base.ContainerManager"),
            patch("autobio.tools.base.GPUManager"),
        ):
            r = get_runner("proteinmpnn", config)
        assert isinstance(r, MPNNRunner)
        assert r.tool_name == "proteinmpnn"
