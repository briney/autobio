"""End-to-end tests for ligandmpnn_build_mutant.

Each test exercises the full pipeline:
    input construction -> validation -> prepare_workspace ->
    (simulated packing output) -> standardize.py -> parse_output -> verify

The only thing not tested is the actual LigandMPNN sidechain packing.
The standardize script is imported and run directly against realistic
output data.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from autobio.core.catalog import get_tool
from autobio.core.config import AutobioConfig
from autobio.core.result import AutobioError
from autobio.core.workspace import Workspace
from autobio.schemas.scoring import LigandMPNNPackerInput, ScoringOutput
from autobio.tools import TOOL_RUNNERS
from autobio.tools.ligandmpnn_packer import LIGANDMPNN_PACKER_TOOL, LigandMPNNPackerRunner

# ---------------------------------------------------------------------------
# Realistic simulated output data
# ---------------------------------------------------------------------------

# Minimal but valid two-chain PDB content for testing
_MINIMAL_COMPLEX_PDB = (
    "HEADER    TEST COMPLEX\n"
    "ATOM      1  N   ALA A   1       1.000   2.000   3.000  1.00 10.00           N\n"
    "ATOM      2  CA  ALA A   1       2.000   3.000   4.000  1.00 10.00           C\n"
    "ATOM      3  C   ALA A   1       3.000   4.000   5.000  1.00 10.00           C\n"
    "ATOM      4  O   ALA A   1       3.500   4.500   5.500  1.00 10.00           O\n"
    "ATOM      5  CB  ALA A   1       2.500   3.500   3.000  1.00 10.00           C\n"
    "ATOM      6  N   GLU E  63       4.000   5.000   6.000  1.00 12.00           N\n"
    "ATOM      7  CA  GLU E  63       5.000   6.000   7.000  1.00 12.00           C\n"
    "ATOM      8  C   GLU E  63       6.000   7.000   8.000  1.00 12.00           C\n"
    "ATOM      9  O   GLU E  63       6.500   7.500   8.500  1.00 12.00           O\n"
    "ATOM     10  CB  GLU E  63       5.500   6.500   6.000  1.00 12.00           C\n"
    "ATOM     11  CG  GLU E  63       5.800   6.800   5.000  1.00 12.00           C\n"
    "ATOM     12  CD  GLU E  63       6.100   7.100   4.000  1.00 12.00           C\n"
    "ATOM     13  OE1 GLU E  63       6.400   7.400   3.500  1.00 12.00           O\n"
    "ATOM     14  OE2 GLU E  63       6.000   7.000   3.000  1.00 12.00           O\n"
    "ATOM     15  N   LYS K  42       7.000   8.000   9.000  1.00 11.00           N\n"
    "ATOM     16  CA  LYS K  42       8.000   9.000  10.000  1.00 11.00           C\n"
    "ATOM     17  C   LYS K  42       9.000  10.000  11.000  1.00 11.00           C\n"
    "ATOM     18  O   LYS K  42       9.500  10.500  11.500  1.00 11.00           O\n"
    "ATOM     19  CB  LYS K  42       8.500   9.500   9.000  1.00 11.00           C\n"
    "ATOM     20  CG  LYS K  42       8.800   9.800   8.000  1.00 11.00           C\n"
    "ATOM     21  CD  LYS K  42       9.100  10.100   7.000  1.00 11.00           C\n"
    "ATOM     22  CE  LYS K  42       9.400  10.400   6.000  1.00 11.00           C\n"
    "ATOM     23  NZ  LYS K  42       9.700  10.700   5.000  1.00 11.00           N\n"
    "END\n"
)

# Simulated packed PDB (minimal — just backbone + sidechain atoms)
_PACKED_PDB_CONTENT = (
    "ATOM      1  N   ALA A   1       1.000   2.000   3.000  1.00  0.95           N\n"
    "ATOM      2  CA  ALA A   1       2.000   3.000   4.000  1.00  0.95           C\n"
    "ATOM      3  C   ALA A   1       3.000   4.000   5.000  1.00  0.95           C\n"
    "ATOM      4  O   ALA A   1       3.500   4.500   5.500  1.00  0.95           O\n"
    "ATOM      5  CB  ALA A   1       2.500   3.500   3.000  1.00  0.95           C\n"
    "ATOM      6  N   GLN E  63       4.000   5.000   6.000  1.00  0.88           N\n"
    "ATOM      7  CA  GLN E  63       5.000   6.000   7.000  1.00  0.88           C\n"
    "ATOM      8  C   GLN E  63       6.000   7.000   8.000  1.00  0.88           C\n"
    "ATOM      9  O   GLN E  63       6.500   7.500   8.500  1.00  0.88           O\n"
    "ATOM     10  CB  GLN E  63       5.200   6.200   6.200  1.00  0.88           C\n"
    "ATOM     11  CG  GLN E  63       5.500   6.500   5.200  1.00  0.88           C\n"
    "ATOM     12  CD  GLN E  63       5.800   6.800   4.200  1.00  0.88           C\n"
    "ATOM     13  OE1 GLN E  63       6.100   7.100   3.700  1.00  0.88           O\n"
    "ATOM     14  NE2 GLN E  63       5.700   6.700   3.200  1.00  0.88           N\n"
    "END\n"
)

# Simulated packing_scores.json
_PACKING_SCORES = [
    {
        "pack_id": 0,
        "total_score": -1.25,
        "per_residue_scores": [-1.10, -1.40],
        "structure_file": "packed_0000.pdb",
    },
]

_PACKING_SCORES_MULTI = [
    {
        "pack_id": 0,
        "total_score": -1.25,
        "per_residue_scores": [-1.10, -1.40],
        "structure_file": "packed_0000.pdb",
    },
    {
        "pack_id": 1,
        "total_score": -1.30,
        "per_residue_scores": [-1.15, -1.45],
        "structure_file": "packed_0001.pdb",
    },
    {
        "pack_id": 2,
        "total_score": -1.18,
        "per_residue_scores": [-1.05, -1.31],
        "structure_file": "packed_0002.pdb",
    },
]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def config() -> AutobioConfig:
    return AutobioConfig.resolve()


@pytest.fixture()
def complex_pdb(tmp_path: Path) -> Path:
    """Write a minimal multi-chain PDB."""
    pdb_path = tmp_path / "complex.pdb"
    pdb_path.write_text(_MINIMAL_COMPLEX_PDB)
    return pdb_path


def _make_runner(config: AutobioConfig) -> LigandMPNNPackerRunner:
    """Create a runner with mocked container/GPU and current_mode set."""
    with (
        patch("autobio.tools.base.ContainerManager"),
        patch("autobio.tools.base.GPUManager"),
    ):
        runner = LigandMPNNPackerRunner("ligandmpnn_build_mutant", config)
    runner.current_mode = get_tool("ligandmpnn_build_mutant").modes["build_mutant"]
    return runner


def _import_standardize():
    """Import the container's standardize module by file path.

    Uses spec_from_file_location to avoid module name collisions with other
    containers' standardize.py when the full test suite runs together.
    """
    script_path = (
        Path(__file__).resolve().parent.parent.parent
        / "containers"
        / "ligandmpnn-packer"
        / "standardize.py"
    )
    spec = importlib.util.spec_from_file_location("ligandmpnn_packer_standardize", script_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _written_config(
    runner: LigandMPNNPackerRunner, input_data: LigandMPNNPackerInput, tmp_path: Path
) -> dict:
    workspace = Workspace.create(tmp_path / "ws")
    try:
        runner.prepare_workspace(input_data, workspace)
        return json.loads(workspace.config_path.read_text())
    finally:
        workspace.cleanup()


def _run_e2e(
    config: AutobioConfig,
    input_data: LigandMPNNPackerInput,
    raw_files: dict[str, str],
    tmp_path: Path,
) -> ScoringOutput:
    """Full end-to-end pipeline without Docker.

    1. prepare_workspace
    2. Write simulated raw output files
    3. Run the container's standardize.py
    4. parse_output
    """
    runner = _make_runner(config)
    workspace = Workspace.create(tmp_path / "ws")

    # Step 1: prepare workspace
    runner.prepare_workspace(input_data, workspace)

    # Step 2: write simulated raw output files
    for filename, content in raw_files.items():
        (workspace.raw_output_dir / filename).write_text(content)

    # Step 3: run the actual standardize.py script
    std_mod = _import_standardize()
    std_mod.standardize(workspace.root)

    # Verify result_data.json was produced
    result_data_path = workspace.std_output_dir / "result_data.json"
    assert result_data_path.exists(), "standardize.py did not produce result_data.json"

    # Step 4: parse output
    output = runner.parse_output(workspace)
    assert isinstance(output, ScoringOutput)
    return output


# ---------------------------------------------------------------------------
# TestPrepareWorkspace
# ---------------------------------------------------------------------------


class TestPrepareWorkspace:
    """Tests for prepare_workspace config generation."""

    def test_config_has_required_fields(
        self, config: AutobioConfig, complex_pdb: Path, tmp_path: Path
    ) -> None:
        """Config.json has all required fields."""
        runner = _make_runner(config)
        workspace = Workspace.create(tmp_path / "ws")
        input_data = LigandMPNNPackerInput(structure_path=complex_pdb, mutations=["EE63Q"])
        runner.prepare_workspace(input_data, workspace)

        cfg = json.loads(workspace.config_path.read_text())
        assert cfg["structure_path"].startswith("/workspace/inputs/")
        assert cfg["mutations"] == ["EE63Q"]
        assert cfg["checkpoint_sc"] == "/app/LigandMPNN/model_params/ligandmpnn_sc_v_32_002_16.pt"
        assert cfg["checkpoint_bb"] == "/app/LigandMPNN/model_params/ligandmpnn_v_32_010_25.pt"

    def test_config_defaults(
        self, config: AutobioConfig, complex_pdb: Path, tmp_path: Path
    ) -> None:
        """Config.json uses correct defaults for optional parameters."""
        runner = _make_runner(config)
        workspace = Workspace.create(tmp_path / "ws")
        input_data = LigandMPNNPackerInput(structure_path=complex_pdb, mutations=["EE63Q"])
        runner.prepare_workspace(input_data, workspace)

        cfg = json.loads(workspace.config_path.read_text())
        assert cfg["num_packs"] == 4
        assert cfg["num_denoising_steps"] == 3
        assert cfg["num_samples"] == 16
        assert cfg["repack_everything"] is True
        assert cfg["pack_with_ligand_context"] is True

    def test_config_custom_parameters(
        self, config: AutobioConfig, complex_pdb: Path, tmp_path: Path
    ) -> None:
        """Custom packing parameters are written to config; seed flat-merges from extra."""
        runner = _make_runner(config)
        workspace = Workspace.create(tmp_path / "ws")
        input_data = LigandMPNNPackerInput(
            structure_path=complex_pdb,
            mutations=["EE63Q"],
            num_packs=8,
            num_denoising_steps=5,
            num_samples=32,
            repack_everything=False,
            extra={"seed": 42},
        )
        runner.prepare_workspace(input_data, workspace)

        cfg = json.loads(workspace.config_path.read_text())
        assert cfg["num_packs"] == 8
        assert cfg["num_denoising_steps"] == 5
        assert cfg["num_samples"] == 32
        assert cfg["repack_everything"] is False
        assert cfg["seed"] == 42

    def test_structure_copied(
        self, config: AutobioConfig, complex_pdb: Path, tmp_path: Path
    ) -> None:
        """Input PDB is copied to workspace inputs dir."""
        runner = _make_runner(config)
        workspace = Workspace.create(tmp_path / "ws")
        input_data = LigandMPNNPackerInput(structure_path=complex_pdb, mutations=["EE63Q"])
        runner.prepare_workspace(input_data, workspace)

        copied = workspace.inputs_dir / complex_pdb.name
        assert copied.exists()
        assert copied.read_text() == complex_pdb.read_text()

    def test_multiple_mutations(
        self, config: AutobioConfig, complex_pdb: Path, tmp_path: Path
    ) -> None:
        """Multiple mutations are stored in config."""
        runner = _make_runner(config)
        workspace = Workspace.create(tmp_path / "ws")
        input_data = LigandMPNNPackerInput(structure_path=complex_pdb, mutations=["EE63Q", "KK42A"])
        runner.prepare_workspace(input_data, workspace)

        cfg = json.loads(workspace.config_path.read_text())
        assert cfg["mutations"] == ["EE63Q", "KK42A"]

    def test_pack_with_ligand_context_passthrough(
        self, config: AutobioConfig, complex_pdb: Path, tmp_path: Path
    ) -> None:
        """pack_with_ligand_context is written to config."""
        runner = _make_runner(config)
        cfg = _written_config(
            runner,
            LigandMPNNPackerInput(
                structure_path=complex_pdb,
                mutations=["EE63Q"],
                pack_with_ligand_context=False,
            ),
            tmp_path,
        )
        assert cfg["pack_with_ligand_context"] is False


# ---------------------------------------------------------------------------
# TestFullConfigEquality — byte-compat full-dict config.json contract
# ---------------------------------------------------------------------------


class TestFullConfigEquality:
    """Full-dict equality test pinning the exact config.json contract."""

    def test_full_config_byte_compat(
        self, config: AutobioConfig, complex_pdb: Path, tmp_path: Path
    ) -> None:
        runner = _make_runner(config)
        cfg = _written_config(
            runner,
            LigandMPNNPackerInput(
                structure_path=complex_pdb,
                mutations=["EE63Q", "KK42A"],
                num_packs=8,
                num_denoising_steps=5,
                num_samples=32,
                repack_everything=False,
                pack_with_ligand_context=False,
                extra={"seed": 42},
            ),
            tmp_path,
        )
        assert cfg == {
            "structure_path": "/workspace/inputs/complex.pdb",
            "mutations": ["EE63Q", "KK42A"],
            "checkpoint_sc": "/app/LigandMPNN/model_params/ligandmpnn_sc_v_32_002_16.pt",
            "checkpoint_bb": "/app/LigandMPNN/model_params/ligandmpnn_v_32_010_25.pt",
            "num_packs": 8,
            "num_denoising_steps": 5,
            "num_samples": 32,
            "repack_everything": False,
            "pack_with_ligand_context": False,
            "seed": 42,
        }

    def test_minimal_full_config_byte_compat(
        self, config: AutobioConfig, complex_pdb: Path, tmp_path: Path
    ) -> None:
        """Minimal input (defaults only, no extra) produces the exact default config."""
        runner = _make_runner(config)
        cfg = _written_config(
            runner,
            LigandMPNNPackerInput(structure_path=complex_pdb, mutations=["EE63Q"]),
            tmp_path,
        )
        assert cfg == {
            "structure_path": "/workspace/inputs/complex.pdb",
            "mutations": ["EE63Q"],
            "checkpoint_sc": "/app/LigandMPNN/model_params/ligandmpnn_sc_v_32_002_16.pt",
            "checkpoint_bb": "/app/LigandMPNN/model_params/ligandmpnn_v_32_010_25.pt",
            "num_packs": 4,
            "num_denoising_steps": 3,
            "num_samples": 16,
            "repack_everything": True,
            "pack_with_ligand_context": True,
        }


# ---------------------------------------------------------------------------
# TestParseOutput
# ---------------------------------------------------------------------------


class TestParseOutput:
    """Tests for the full prepare -> standardize -> parse pipeline."""

    def test_single_pack_pipeline(
        self, config: AutobioConfig, complex_pdb: Path, tmp_path: Path
    ) -> None:
        """Single packed structure produces correct ScoringOutput."""
        input_data = LigandMPNNPackerInput(structure_path=complex_pdb, mutations=["EE63Q"])
        output = _run_e2e(
            config,
            input_data,
            raw_files={
                "packed_0000.pdb": _PACKED_PDB_CONTENT,
                "packing_scores.json": json.dumps(_PACKING_SCORES),
            },
            tmp_path=tmp_path,
        )

        assert len(output.scores) == 1
        s = output.scores[0]
        assert s.total_score == pytest.approx(-1.25)
        assert s.units == "LigandMPNN_SC_logprob"
        assert s.structure_path is not None
        assert s.structure_path.name == "packed_0000.pdb"
        assert s.mutations == ["EE63Q"]
        assert s.ddg is None

    def test_multiple_packs_pipeline(
        self, config: AutobioConfig, complex_pdb: Path, tmp_path: Path
    ) -> None:
        """Multiple packed structures produce one ScoredStructure each."""
        input_data = LigandMPNNPackerInput(structure_path=complex_pdb, mutations=["EE63Q"])
        output = _run_e2e(
            config,
            input_data,
            raw_files={
                "packed_0000.pdb": _PACKED_PDB_CONTENT,
                "packed_0001.pdb": _PACKED_PDB_CONTENT,
                "packed_0002.pdb": _PACKED_PDB_CONTENT,
                "packing_scores.json": json.dumps(_PACKING_SCORES_MULTI),
            },
            tmp_path=tmp_path,
        )

        assert len(output.scores) == 3
        for s in output.scores:
            assert s.mutations == ["EE63Q"]
            assert s.structure_path is not None
            assert s.units == "LigandMPNN_SC_logprob"

        # Verify different scores per pack
        assert output.scores[0].total_score == pytest.approx(-1.25)
        assert output.scores[1].total_score == pytest.approx(-1.30)
        assert output.scores[2].total_score == pytest.approx(-1.18)

    def test_per_residue_scores(
        self, config: AutobioConfig, complex_pdb: Path, tmp_path: Path
    ) -> None:
        """Per-residue scores are passed through."""
        input_data = LigandMPNNPackerInput(structure_path=complex_pdb, mutations=["EE63Q"])
        output = _run_e2e(
            config,
            input_data,
            raw_files={
                "packed_0000.pdb": _PACKED_PDB_CONTENT,
                "packing_scores.json": json.dumps(_PACKING_SCORES),
            },
            tmp_path=tmp_path,
        )

        s = output.scores[0]
        assert s.per_residue_scores is not None
        assert len(s.per_residue_scores) == 2
        assert s.per_residue_scores[0] == pytest.approx(-1.10)
        assert s.per_residue_scores[1] == pytest.approx(-1.40)

    def test_multiple_mutations_in_output(
        self, config: AutobioConfig, complex_pdb: Path, tmp_path: Path
    ) -> None:
        """Multiple mutations are echoed in output."""
        input_data = LigandMPNNPackerInput(structure_path=complex_pdb, mutations=["EE63Q", "KK42A"])
        output = _run_e2e(
            config,
            input_data,
            raw_files={
                "packed_0000.pdb": _PACKED_PDB_CONTENT,
                "packing_scores.json": json.dumps(_PACKING_SCORES),
            },
            tmp_path=tmp_path,
        )

        assert output.scores[0].mutations == ["EE63Q", "KK42A"]


# ---------------------------------------------------------------------------
# TestValidation
# ---------------------------------------------------------------------------


class TestValidation:
    """Tests for host-side input validation."""

    def test_nonexistent_structure_fails(self, config: AutobioConfig, tmp_path: Path) -> None:
        """Nonexistent input structure raises validation error."""
        runner = _make_runner(config)
        workspace = Workspace.create(tmp_path / "ws")
        input_data = LigandMPNNPackerInput(
            structure_path=tmp_path / "nonexistent.pdb", mutations=["EA63Q"]
        )
        with pytest.raises(AutobioError, match="does not exist"):
            runner.prepare_workspace(input_data, workspace)

    def test_non_pdb_format_fails(self, config: AutobioConfig, tmp_path: Path) -> None:
        """Non-PDB format raises validation error."""
        cif_path = tmp_path / "structure.cif"
        cif_path.write_text("data_test\n")
        runner = _make_runner(config)
        workspace = Workspace.create(tmp_path / "ws")
        input_data = LigandMPNNPackerInput(structure_path=cif_path, mutations=["EA63Q"])
        with pytest.raises(AutobioError, match="PDB format"):
            runner.prepare_workspace(input_data, workspace)

    def test_empty_mutations_fails(
        self, config: AutobioConfig, complex_pdb: Path, tmp_path: Path
    ) -> None:
        """Empty mutations list raises validation error with the accurate message."""
        runner = _make_runner(config)
        workspace = Workspace.create(tmp_path / "ws")
        input_data = LigandMPNNPackerInput(structure_path=complex_pdb, mutations=[])
        with pytest.raises(AutobioError, match="requires at least one mutation"):
            runner.prepare_workspace(input_data, workspace)

    def test_invalid_mutation_format_fails(
        self, config: AutobioConfig, complex_pdb: Path, tmp_path: Path
    ) -> None:
        """Invalid mutation format raises validation error."""
        runner = _make_runner(config)
        workspace = Workspace.create(tmp_path / "ws")
        input_data = LigandMPNNPackerInput(structure_path=complex_pdb, mutations=["invalid"])
        with pytest.raises(AutobioError, match="Invalid mutation format"):
            runner.prepare_workspace(input_data, workspace)


# ---------------------------------------------------------------------------
# TestExtraShadowRejection
# ---------------------------------------------------------------------------


class TestExtraShadowRejection:
    """`extra` keys that collide with typed fields or derived config keys raise."""

    @pytest.mark.parametrize("extra_key", ["mutations", "num_packs"])
    def test_typed_field_collision_rejected(
        self, config: AutobioConfig, complex_pdb: Path, tmp_path: Path, extra_key: str
    ) -> None:
        runner = _make_runner(config)
        workspace = Workspace.create(tmp_path / "ws")
        input_data = LigandMPNNPackerInput(
            structure_path=complex_pdb,
            mutations=["EE63Q"],
            extra={extra_key: "bogus"},
        )
        with pytest.raises(AutobioError, match="collide with typed input fields"):
            runner.prepare_workspace(input_data, workspace)

    def test_derived_config_key_collision_rejected(
        self, config: AutobioConfig, complex_pdb: Path, tmp_path: Path
    ) -> None:
        """``checkpoint_sc`` is a runner-derived config key, not a typed field."""
        runner = _make_runner(config)
        workspace = Workspace.create(tmp_path / "ws")
        input_data = LigandMPNNPackerInput(
            structure_path=complex_pdb,
            mutations=["EE63Q"],
            extra={"checkpoint_sc": "/other/path.pt"},
        )
        with pytest.raises(AutobioError, match="runner-derived config keys"):
            runner.prepare_workspace(input_data, workspace)


# ---------------------------------------------------------------------------
# TestRegistration
# ---------------------------------------------------------------------------


class TestRegistration:
    """Tests for catalog Tool and runner registration."""

    def test_registered_as_catalog_tool(self) -> None:
        from autobio.core.catalog import CATALOG

        assert "ligandmpnn_build_mutant" in CATALOG
        tool = get_tool("ligandmpnn_build_mutant")
        assert tool.category.value == "scoring"
        assert tool.requires_gpu is True
        assert tool.gpu_count == 1
        assert set(tool.modes) == {"build_mutant"}
        assert tool.default_mode == "build_mutant"

    def test_in_tool_runners(self) -> None:
        assert "ligandmpnn_build_mutant" in TOOL_RUNNERS
        assert TOOL_RUNNERS["ligandmpnn_build_mutant"] is LigandMPNNPackerRunner

    def test_schema_types(self) -> None:
        from autobio.schemas.scoring import LigandMPNNPackerInput, ScoringOutput

        mode = get_tool("ligandmpnn_build_mutant").modes["build_mutant"]
        assert mode.input_schema is LigandMPNNPackerInput
        assert mode.output_schema is ScoringOutput

    def test_image_tag(self) -> None:
        assert get_tool("ligandmpnn_build_mutant").image_tag == "ligandmpnn-packer:1.0.0"

    def test_timeout(self) -> None:
        assert get_tool("ligandmpnn_build_mutant").modes["build_mutant"].default_timeout == 600

    def test_has_notes(self) -> None:
        notes = get_tool("ligandmpnn_build_mutant").modes["build_mutant"].notes
        assert len(notes) > 0
        assert any("chi" in note.lower() or "sidechain" in note.lower() for note in notes)

    def test_tool_constant_registered(self) -> None:
        assert LIGANDMPNN_PACKER_TOOL.name == "ligandmpnn_build_mutant"
        assert get_tool("ligandmpnn_build_mutant") is LIGANDMPNN_PACKER_TOOL


# ---------------------------------------------------------------------------
# TestInfoSnapshot
# ---------------------------------------------------------------------------


class TestInfoSnapshot:
    """Snapshot the `autobio info` catalog rendering for the packer Tool."""

    def test_info_snapshot(self) -> None:
        from autobio.cli.formatters import OutputFormat, format_tool_info_catalog

        parsed = json.loads(
            format_tool_info_catalog(get_tool("ligandmpnn_build_mutant"), OutputFormat.JSON)
        )
        assert [m["name"] for m in parsed["modes"]] == ["build_mutant"]
        mode = parsed["modes"][0]
        props = mode["input_schema"]["properties"]
        assert props["structure_path"]["x-autobio"]["widget"] == "file"
        assert props["structure_path"]["x-autobio"]["tier"] == "primary"
        assert props["num_packs"]["x-autobio"]["tier"] == "advanced"
        assert "output_schema" in mode
        assert mode["notes"]
