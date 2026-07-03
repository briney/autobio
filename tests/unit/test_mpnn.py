"""Tests for MPNNRunner — prepare_workspace, parse_output, and registration."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

from autobio.core.catalog import get_tool
from autobio.core.config import AutobioConfig
from autobio.core.registry import ToolCategory
from autobio.core.result import AutobioError
from autobio.core.workspace import Workspace
from autobio.schemas.inverse_folding import InverseFoldingOutput, MPNNInput
from autobio.tools import TOOL_RUNNERS, get_runner
from autobio.tools.mpnn import _CHECKPOINT_DIR, LIGANDMPNN_TOOL, PROTEINMPNN_TOOL, MPNNRunner

if TYPE_CHECKING:
    from pathlib import Path


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_runner(tool_name: str, config: AutobioConfig) -> MPNNRunner:
    """Create an MPNNRunner with mocked ContainerManager/GPUManager and current_mode set."""
    with (
        patch("autobio.tools.base.ContainerManager"),
        patch("autobio.tools.base.GPUManager"),
    ):
        runner = MPNNRunner(tool_name, config)
    runner.current_mode = get_tool(tool_name).modes["design"]
    return runner


@pytest.fixture()
def config() -> AutobioConfig:
    return AutobioConfig.resolve()


@pytest.fixture()
def runner(config: AutobioConfig) -> MPNNRunner:
    """Create an MPNNRunner for proteinmpnn."""
    return _make_runner("proteinmpnn", config)


@pytest.fixture()
def ligand_runner(config: AutobioConfig) -> MPNNRunner:
    """Create an MPNNRunner for ligandmpnn."""
    return _make_runner("ligandmpnn", config)


@pytest.fixture()
def sample_pdb(tmp_path: Path) -> Path:
    """Write a minimal PDB file for testing."""
    pdb_path = tmp_path / "test.pdb"
    pdb_path.write_text("ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00\nEND\n")
    return pdb_path


def _written_config(runner: MPNNRunner, input_data: MPNNInput, tmp_path: Path) -> dict:
    workspace = Workspace.create(tmp_path / "ws")
    try:
        runner.prepare_workspace(input_data, workspace)
        return json.loads(workspace.config_path.read_text())
    finally:
        workspace.cleanup()


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
        r = _make_runner(tool_name, config)
        cfg = _written_config(r, MPNNInput(structure_path=sample_pdb), tmp_path)

        assert cfg["model_type"] == expected_model_type
        assert cfg["checkpoint_path"] == f"{_CHECKPOINT_DIR}/{expected_checkpoint}"
        assert cfg["is_legacy_weights"] is True

    def test_structure_file_copied(
        self, runner: MPNNRunner, tmp_path: Path, sample_pdb: Path
    ) -> None:
        """Input structure is copied to inputs/ and config references container path."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = MPNNInput(structure_path=sample_pdb)
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
        cfg = _written_config(
            runner, MPNNInput(structure_path=sample_pdb, num_sequences=5), tmp_path
        )
        assert cfg["number_of_batches"] == 5

    def test_temperature_passthrough(
        self, runner: MPNNRunner, tmp_path: Path, sample_pdb: Path
    ) -> None:
        cfg = _written_config(
            runner, MPNNInput(structure_path=sample_pdb, temperature=0.5), tmp_path
        )
        assert cfg["temperature"] == 0.5

    def test_defaults_applied(self, runner: MPNNRunner, tmp_path: Path, sample_pdb: Path) -> None:
        """Minimal input produces sensible defaults."""
        cfg = _written_config(runner, MPNNInput(structure_path=sample_pdb), tmp_path)
        assert cfg["temperature"] == 0.1
        assert cfg["number_of_batches"] == 1

    def test_chains_to_design(self, runner: MPNNRunner, tmp_path: Path, sample_pdb: Path) -> None:
        cfg = _written_config(
            runner, MPNNInput(structure_path=sample_pdb, chains_to_design=["A", "B"]), tmp_path
        )
        assert cfg["designed_chains"] == "A,B"

    def test_fixed_positions_mapping(
        self, runner: MPNNRunner, tmp_path: Path, sample_pdb: Path
    ) -> None:
        """fixed_positions dict is transformed to comma-separated residue IDs."""
        cfg = _written_config(
            runner,
            MPNNInput(structure_path=sample_pdb, fixed_positions={"A": [1, 5, 10], "B": [3]}),
            tmp_path,
        )
        ids = cfg["fixed_residues"].split(",")
        assert set(ids) == {"A1", "A5", "A10", "B3"}

    def test_fixed_positions_overrides_designed_chains(
        self, runner: MPNNRunner, tmp_path: Path, sample_pdb: Path
    ) -> None:
        """When both are given, fixed_positions takes precedence (they're mutually exclusive)."""
        cfg = _written_config(
            runner,
            MPNNInput(
                structure_path=sample_pdb,
                chains_to_design=["A"],
                fixed_positions={"B": [1, 2]},
            ),
            tmp_path,
        )
        assert "fixed_residues" in cfg
        assert "designed_chains" not in cfg

    def test_extra_dict_merged(self, runner: MPNNRunner, tmp_path: Path, sample_pdb: Path) -> None:
        """LigandMPNN/CLI knobs not promoted to typed fields flat-merge from extra."""
        cfg = _written_config(
            runner,
            MPNNInput(
                structure_path=sample_pdb,
                extra={
                    "omit": '["CYS","TRP"]',
                    "seed": 42,
                    "batch_size": 4,
                    "bias": {"A": 0.1},
                    "temperature_per_residue": [0.1, 0.2],
                    "atomize_side_chains": True,
                },
            ),
            tmp_path,
        )
        assert cfg["omit"] == '["CYS","TRP"]'
        assert cfg["seed"] == 42
        assert cfg["batch_size"] == 4
        assert cfg["bias"] == {"A": 0.1}
        assert cfg["temperature_per_residue"] == [0.1, 0.2]
        assert cfg["atomize_side_chains"] is True


# ---------------------------------------------------------------------------
# TestMPNNFullConfigEquality — byte-compat full-dict config.json contract
# ---------------------------------------------------------------------------


class TestMPNNFullConfigEquality:
    """Full-dict equality tests pinning the exact config.json contract."""

    def test_proteinmpnn_full_config(
        self, runner: MPNNRunner, tmp_path: Path, sample_pdb: Path
    ) -> None:
        cfg = _written_config(
            runner,
            MPNNInput(structure_path=sample_pdb, num_sequences=3, temperature=0.25),
            tmp_path,
        )
        assert cfg == {
            "model_type": "protein_mpnn",
            "checkpoint_path": f"{_CHECKPOINT_DIR}/proteinmpnn_v_48_020.pt",
            "is_legacy_weights": True,
            "structure_path": "/workspace/inputs/test.pdb",
            "number_of_batches": 3,
            "temperature": 0.25,
        }

    def test_ligandmpnn_full_config_with_extra(
        self, ligand_runner: MPNNRunner, tmp_path: Path, sample_pdb: Path
    ) -> None:
        cfg = _written_config(
            ligand_runner,
            MPNNInput(
                structure_path=sample_pdb,
                chains_to_design=["A", "B"],
                extra={"omit": '["CYS"]', "seed": 7},
            ),
            tmp_path,
        )
        assert cfg == {
            "model_type": "ligand_mpnn",
            "checkpoint_path": f"{_CHECKPOINT_DIR}/ligandmpnn_v_32_010_25.pt",
            "is_legacy_weights": True,
            "structure_path": "/workspace/inputs/test.pdb",
            "number_of_batches": 1,
            "temperature": 0.1,
            "designed_chains": "A,B",
            "omit": '["CYS"]',
            "seed": 7,
        }

    def test_fixed_positions_mutual_exclusivity_full_config(
        self, runner: MPNNRunner, tmp_path: Path, sample_pdb: Path
    ) -> None:
        """chains_to_design + fixed_positions: fixed_residues wins, no designed_chains."""
        cfg = _written_config(
            runner,
            MPNNInput(
                structure_path=sample_pdb,
                chains_to_design=["A"],
                fixed_positions={"B": [1, 2]},
                num_sequences=2,
            ),
            tmp_path,
        )
        assert cfg == {
            "model_type": "protein_mpnn",
            "checkpoint_path": f"{_CHECKPOINT_DIR}/proteinmpnn_v_48_020.pt",
            "is_legacy_weights": True,
            "structure_path": "/workspace/inputs/test.pdb",
            "number_of_batches": 2,
            "temperature": 0.1,
            "fixed_residues": "B1,B2",
        }


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
# TestMPNNExtraShadowRejection
# ---------------------------------------------------------------------------


class TestMPNNExtraShadowRejection:
    """`extra` keys that collide with typed fields or derived config keys raise."""

    @pytest.mark.parametrize("extra_key", ["temperature", "num_sequences"])
    def test_typed_field_collision_rejected(
        self, runner: MPNNRunner, tmp_path: Path, sample_pdb: Path, extra_key: str
    ) -> None:
        workspace = Workspace.create(tmp_path / "ws")
        input_data = MPNNInput(structure_path=sample_pdb, extra={extra_key: 999})
        with pytest.raises(AutobioError, match="collide with typed input fields"):
            runner.prepare_workspace(input_data, workspace)

    def test_derived_config_key_collision_rejected(
        self, runner: MPNNRunner, tmp_path: Path, sample_pdb: Path
    ) -> None:
        """``designed_chains`` is a runner-derived config key, not a typed field."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = MPNNInput(
            structure_path=sample_pdb,
            chains_to_design=["A"],
            extra={"designed_chains": "B"},
        )
        with pytest.raises(AutobioError, match="runner-derived config keys"):
            runner.prepare_workspace(input_data, workspace)


# ---------------------------------------------------------------------------
# TestMPNNRegistration
# ---------------------------------------------------------------------------


class TestMPNNRegistration:
    """Tests for catalog Tool and runner registration."""

    def test_proteinmpnn_registered_as_catalog_tool(self) -> None:
        from autobio.core.catalog import CATALOG

        assert "proteinmpnn" in CATALOG
        tool = get_tool("proteinmpnn")
        assert tool.category == ToolCategory.INVERSE_FOLDING
        assert tool.requires_gpu is True
        assert set(tool.modes) == {"design"}
        assert tool.modes["design"].input_schema is MPNNInput
        assert tool.modes["design"].output_schema is InverseFoldingOutput

    def test_ligandmpnn_registered_as_catalog_tool(self) -> None:
        from autobio.core.catalog import CATALOG

        assert "ligandmpnn" in CATALOG
        tool = get_tool("ligandmpnn")
        assert tool.category == ToolCategory.INVERSE_FOLDING
        assert set(tool.modes) == {"design", "build_mutant"}
        assert tool.description.lower().count("ligand") >= 1

    def test_both_share_image_tag(self) -> None:
        assert get_tool("proteinmpnn").image_tag == get_tool("ligandmpnn").image_tag

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

    def test_tool_constants_registered(self) -> None:
        assert PROTEINMPNN_TOOL.name == "proteinmpnn"
        assert LIGANDMPNN_TOOL.name == "ligandmpnn"
        assert get_tool("proteinmpnn") is PROTEINMPNN_TOOL
        assert get_tool("ligandmpnn") is LIGANDMPNN_TOOL

    def test_ligandmpnn_build_mutant_mode(self) -> None:
        from autobio.core.catalog import tool_categories
        from autobio.schemas.scoring import LigandMPNNPackerInput, ScoringOutput

        tool = get_tool("ligandmpnn")
        bm = tool.modes["build_mutant"]
        assert bm.input_schema is LigandMPNNPackerInput
        assert bm.output_schema is ScoringOutput
        assert bm.image_tag == "ligandmpnn-packer:1.0.0"
        assert bm.category == ToolCategory.SCORING
        assert tool_categories(tool) == (
            ToolCategory.INVERSE_FOLDING,
            ToolCategory.SCORING,
        )

    def test_ligandmpnn_build_mutant_not_a_tool_name(self) -> None:
        from autobio.core.catalog import CATALOG

        assert "ligandmpnn_build_mutant" not in CATALOG
        assert "ligandmpnn_build_mutant" not in TOOL_RUNNERS


# ---------------------------------------------------------------------------
# TestMPNNInfoSnapshot
# ---------------------------------------------------------------------------


class TestMPNNInfoSnapshot:
    """Snapshot the `autobio info` catalog rendering for both mpnn Tools."""

    def test_info_snapshot_proteinmpnn(self) -> None:
        from autobio.cli.formatters import OutputFormat, format_tool_info_catalog

        parsed = json.loads(format_tool_info_catalog(get_tool("proteinmpnn"), OutputFormat.JSON))
        assert parsed["modes"][0]["name"] == "design"
        props = parsed["modes"][0]["input_schema"]["properties"]
        assert props["structure_path"]["x-autobio"]["widget"] == "file"
        assert props["structure_path"]["x-autobio"]["tier"] == "primary"
        assert props["temperature"]["x-autobio"]["tier"] == "advanced"
        assert "output_schema" in parsed["modes"][0]
        assert parsed["modes"][0]["notes"]

    def test_info_snapshot_ligandmpnn_includes_ligand_note(self) -> None:
        from autobio.cli.formatters import OutputFormat, format_tool_info_catalog

        parsed = json.loads(format_tool_info_catalog(get_tool("ligandmpnn"), OutputFormat.JSON))
        notes = parsed["modes"][0]["notes"]
        assert any("non-polymer" in n for n in notes)
        assert "output_schema" in parsed["modes"][0]
