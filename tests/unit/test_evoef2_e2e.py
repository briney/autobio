"""End-to-end tests for EvoEF2.

Each test exercises the full pipeline:
    input construction -> validation -> prepare_workspace ->
    (simulated EvoEF2 output) -> standardize.py -> parse_output -> verify

The only thing not tested is the actual EvoEF2 binary execution.
The standardize script is imported and run directly against realistic
output data.
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from autobio.core.config import AutobioConfig
from autobio.core.result import AutobioError
from autobio.core.workspace import Workspace
from autobio.schemas.scoring import ScoringInput, ScoringOutput
from autobio.tools.evoef2 import EvoEF2Runner

# ---------------------------------------------------------------------------
# Realistic EvoEF2 output data
# ---------------------------------------------------------------------------

# Minimal but valid two-chain PDB content for testing
_MINIMAL_COMPLEX_PDB = (
    "HEADER    TEST COMPLEX\n"
    "ATOM      1  N   ALA A   1       1.000   2.000   3.000  1.00 10.00           N\n"
    "ATOM      2  CA  ALA A   1       2.000   3.000   4.000  1.00 10.00           C\n"
    "ATOM      3  C   ALA A   1       3.000   4.000   5.000  1.00 10.00           C\n"
    "ATOM      4  O   ALA A   1       3.500   4.500   5.500  1.00 10.00           O\n"
    "ATOM      5  N   GLY B   1       4.000   5.000   6.000  1.00 12.00           N\n"
    "ATOM      6  CA  GLY B   1       5.000   6.000   7.000  1.00 12.00           C\n"
    "ATOM      7  C   GLY B   1       6.000   7.000   8.000  1.00 12.00           C\n"
    "ATOM      8  O   GLY B   1       6.500   7.500   8.500  1.00 12.00           O\n"
    "END\n"
)

# Simulated EvoEF2 energy output for RepairStructure (stdout format)
_REPAIR_ENERGY_OUTPUT = (
    "EvoEF2 RepairStructure\n"
    "reference_ALA         =     -1.25\n"
    "reference_CYS         =      0.00\n"
    "intraR_vdwatt         =    -45.30\n"
    "intraR_vdwrep         =     12.50\n"
    "intraR_electr         =     -8.20\n"
    "interS_vdwatt         =    -30.10\n"
    "interS_vdwrep         =      5.40\n"
    "interS_electr         =     -6.80\n"
    "----------------------------------------------------\n"
    "Total                 =    -73.75\n\n"
)

# Simulated EvoEF2 energy output for ComputeBinding (stdout format)
_BINDING_ENERGY_OUTPUT = (
    "Binding energy details between chain(s) A and chain(s) B:\n"
    "interD_vdwatt         =    -15.30\n"
    "interD_vdwrep         =      3.20\n"
    "interD_electr         =     -4.50\n"
    "interD_deslvP         =      2.10\n"
    "interD_deslvH         =     -1.80\n"
    "interD_hbbbbb_dis     =     -2.30\n"
    "----------------------------------------------------\n"
    "Total                 =    -18.60\n\n"
)

# Simulated EvoEF2 energy output for BuildMutant (stdout format)
_BUILD_MUTANT_ENERGY_OUTPUT = (
    "EvoEF2 BuildMutant\n"
    "reference_ALA         =     -1.10\n"
    "intraR_vdwatt         =    -40.00\n"
    "----------------------------------------------------\n"
    "Total                 =    -41.10\n\n"
)

# Minimal repaired PDB content
_REPAIRED_PDB_CONTENT = (
    "HEADER    REPAIRED STRUCTURE\n"
    "ATOM      1  N   ALA A   1       1.000   2.000   3.000  1.00 10.00           N\n"
    "END\n"
)

# Minimal mutant model PDB content
_MODEL_PDB_CONTENT = (
    "HEADER    MUTANT MODEL\n"
    "ATOM      1  N   GLN A  63       1.000   2.000   3.000  1.00 10.00           N\n"
    "END\n"
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def config() -> AutobioConfig:
    return AutobioConfig.resolve()


@pytest.fixture()
def complex_pdb(tmp_path: Path) -> Path:
    """Write a minimal two-chain PDB."""
    pdb_path = tmp_path / "complex.pdb"
    pdb_path.write_text(_MINIMAL_COMPLEX_PDB)
    return pdb_path


def _make_runner(tool_name: str, config: AutobioConfig) -> EvoEF2Runner:
    """Create a runner with mocked container/GPU (we simulate container output)."""
    with (
        patch("autobio.tools.base.ContainerManager"),
        patch("autobio.tools.base.GPUManager"),
    ):
        return EvoEF2Runner(tool_name, config)


def _import_standardize():
    """Import the container's standardize module."""
    container_dir = str(Path(__file__).resolve().parent.parent.parent / "containers" / "evoef2")
    if container_dir not in sys.path:
        sys.path.insert(0, container_dir)
    mod = importlib.import_module("standardize")
    importlib.reload(mod)  # ensure fresh import
    return mod


def _run_e2e(
    tool_name: str,
    config: AutobioConfig,
    input_data: ScoringInput,
    raw_files: dict[str, str],
    tmp_path: Path,
    log_files: dict[str, str] | None = None,
) -> ScoringOutput:
    """Full end-to-end pipeline without Docker.

    1. prepare_workspace
    2. Write simulated raw output files
    3. Write simulated log files
    4. Run the container's standardize.py
    5. parse_output
    """
    runner = _make_runner(tool_name, config)
    workspace = Workspace.create(tmp_path / "ws")

    # Step 1: prepare workspace
    runner.prepare_workspace(input_data, workspace)

    # Step 2: write simulated raw output files
    for filename, content in raw_files.items():
        (workspace.raw_output_dir / filename).write_text(content)

    # Step 3: write simulated log files
    if log_files:
        for filename, content in log_files.items():
            (workspace.logs_dir / filename).write_text(content)

    # Step 4: run the actual standardize.py script
    std_mod = _import_standardize()
    std_mod.standardize(workspace.root)

    # Verify result_data.json was produced
    result_data_path = workspace.std_output_dir / "result_data.json"
    assert result_data_path.exists(), "standardize.py did not produce result_data.json"

    # Step 5: parse output
    output = runner.parse_output(workspace)
    assert isinstance(output, ScoringOutput)
    return output


# ---------------------------------------------------------------------------
# TestEvoEF2RepairE2E
# ---------------------------------------------------------------------------


class TestEvoEF2RepairE2E:
    """End-to-end tests for evoef2_repair."""

    def test_repair_full_pipeline(
        self, config: AutobioConfig, complex_pdb: Path, tmp_path: Path
    ) -> None:
        """Repair pipeline produces structure and energy score."""
        input_data = ScoringInput(structure_path=complex_pdb, extra={})
        output = _run_e2e(
            "evoef2_repair",
            config,
            input_data,
            raw_files={"complex_Repair.pdb": _REPAIRED_PDB_CONTENT},
            log_files={"tool.log": _REPAIR_ENERGY_OUTPUT},
            tmp_path=tmp_path,
        )

        assert len(output.scores) == 1
        s = output.scores[0]
        assert s.total_score == pytest.approx(-73.75)
        assert s.units == "EvoEF2"
        assert s.structure_path is not None
        assert s.structure_path.name == "complex_Repair.pdb"
        assert s.ddg is None
        assert s.mutations is None

    def test_repair_score_breakdown(
        self, config: AutobioConfig, complex_pdb: Path, tmp_path: Path
    ) -> None:
        """Repair output includes energy term breakdown."""
        input_data = ScoringInput(structure_path=complex_pdb, extra={})
        output = _run_e2e(
            "evoef2_repair",
            config,
            input_data,
            raw_files={"complex_Repair.pdb": _REPAIRED_PDB_CONTENT},
            log_files={"tool.log": _REPAIR_ENERGY_OUTPUT},
            tmp_path=tmp_path,
        )

        breakdown = output.scores[0].score_breakdown
        assert breakdown is not None
        assert breakdown["intraR_vdwatt"] == pytest.approx(-45.30)
        assert breakdown["interS_electr"] == pytest.approx(-6.80)
        # "Total" should NOT be in breakdown — it's in total_score
        assert "Total" not in breakdown

    def test_repair_config_params(
        self, config: AutobioConfig, complex_pdb: Path, tmp_path: Path
    ) -> None:
        """Config.json has correct command and default parameters."""
        runner = _make_runner("evoef2_repair", config)
        workspace = Workspace.create(tmp_path / "ws")
        input_data = ScoringInput(structure_path=complex_pdb, extra={})
        runner.prepare_workspace(input_data, workspace)

        cfg = json.loads(workspace.config_path.read_text())
        assert cfg["command"] == "RepairStructure"
        assert cfg["evoef2_bin"] == "/app/evoef2/EvoEF2"
        assert cfg["structure_path"].startswith("/workspace/inputs/")


# ---------------------------------------------------------------------------
# TestEvoEF2MinimizeE2E
# ---------------------------------------------------------------------------


class TestEvoEF2MinimizeE2E:
    """End-to-end tests for evoef2_minimize."""

    def test_minimize_full_pipeline(
        self, config: AutobioConfig, complex_pdb: Path, tmp_path: Path
    ) -> None:
        """Minimize pipeline produces structure and energy score."""
        input_data = ScoringInput(structure_path=complex_pdb, extra={})
        output = _run_e2e(
            "evoef2_minimize",
            config,
            input_data,
            raw_files={"complex_Repair.pdb": _REPAIRED_PDB_CONTENT},
            log_files={"tool.log": _REPAIR_ENERGY_OUTPUT},
            tmp_path=tmp_path,
        )

        assert len(output.scores) == 1
        s = output.scores[0]
        assert s.total_score == pytest.approx(-73.75)
        assert s.units == "EvoEF2"
        assert s.structure_path is not None
        assert s.structure_path.name == "complex_Repair.pdb"
        assert s.ddg is None
        assert s.mutations is None

    def test_minimize_score_breakdown(
        self, config: AutobioConfig, complex_pdb: Path, tmp_path: Path
    ) -> None:
        """Minimize output includes energy term breakdown."""
        input_data = ScoringInput(structure_path=complex_pdb, extra={})
        output = _run_e2e(
            "evoef2_minimize",
            config,
            input_data,
            raw_files={"complex_Repair.pdb": _REPAIRED_PDB_CONTENT},
            log_files={"tool.log": _REPAIR_ENERGY_OUTPUT},
            tmp_path=tmp_path,
        )

        breakdown = output.scores[0].score_breakdown
        assert breakdown is not None
        assert breakdown["intraR_vdwatt"] == pytest.approx(-45.30)
        assert breakdown["interS_electr"] == pytest.approx(-6.80)
        # "Total" should NOT be in breakdown — it's in total_score
        assert "Total" not in breakdown

    def test_minimize_config_params(
        self, config: AutobioConfig, complex_pdb: Path, tmp_path: Path
    ) -> None:
        """Config.json has correct command and default parameters."""
        runner = _make_runner("evoef2_minimize", config)
        workspace = Workspace.create(tmp_path / "ws")
        input_data = ScoringInput(structure_path=complex_pdb, extra={})
        runner.prepare_workspace(input_data, workspace)

        cfg = json.loads(workspace.config_path.read_text())
        assert cfg["command"] == "RepairStructure"
        assert cfg["evoef2_bin"] == "/app/evoef2/EvoEF2"
        assert cfg["structure_path"].startswith("/workspace/inputs/")


# ---------------------------------------------------------------------------
# TestEvoEF2BindingE2E
# ---------------------------------------------------------------------------


class TestEvoEF2BindingE2E:
    """End-to-end tests for evoef2_binding."""

    def test_binding_full_pipeline(
        self, config: AutobioConfig, complex_pdb: Path, tmp_path: Path
    ) -> None:
        """Binding pipeline produces energy score with breakdown."""
        input_data = ScoringInput(structure_path=complex_pdb, extra={"repair": False})
        output = _run_e2e(
            "evoef2_binding",
            config,
            input_data,
            raw_files={"binding_output.txt": _BINDING_ENERGY_OUTPUT},
            tmp_path=tmp_path,
        )

        assert len(output.scores) == 1
        s = output.scores[0]
        assert s.total_score == pytest.approx(-18.60)
        assert s.units == "EvoEF2"
        assert s.ddg is None

    def test_binding_with_repair(
        self, config: AutobioConfig, complex_pdb: Path, tmp_path: Path
    ) -> None:
        """Binding with auto-repair includes repaired structure path."""
        input_data = ScoringInput(structure_path=complex_pdb, extra={})
        output = _run_e2e(
            "evoef2_binding",
            config,
            input_data,
            raw_files={
                "binding_output.txt": _BINDING_ENERGY_OUTPUT,
                "complex_Repair.pdb": _REPAIRED_PDB_CONTENT,
            },
            tmp_path=tmp_path,
        )

        s = output.scores[0]
        assert s.structure_path is not None
        assert s.structure_path.name == "complex_Repair.pdb"

    def test_binding_without_repair(
        self, config: AutobioConfig, complex_pdb: Path, tmp_path: Path
    ) -> None:
        """Binding without repair has no structure path."""
        input_data = ScoringInput(structure_path=complex_pdb, extra={"repair": False})
        output = _run_e2e(
            "evoef2_binding",
            config,
            input_data,
            raw_files={"binding_output.txt": _BINDING_ENERGY_OUTPUT},
            tmp_path=tmp_path,
        )

        assert output.scores[0].structure_path is None

    def test_binding_config_repair_default(
        self, config: AutobioConfig, complex_pdb: Path, tmp_path: Path
    ) -> None:
        """Repair defaults to true in config."""
        runner = _make_runner("evoef2_binding", config)
        workspace = Workspace.create(tmp_path / "ws")
        input_data = ScoringInput(structure_path=complex_pdb, extra={})
        runner.prepare_workspace(input_data, workspace)

        cfg = json.loads(workspace.config_path.read_text())
        assert cfg["repair"] is True

    def test_binding_config_with_split(
        self, config: AutobioConfig, complex_pdb: Path, tmp_path: Path
    ) -> None:
        """split_chains parameter is written to config."""
        runner = _make_runner("evoef2_binding", config)
        workspace = Workspace.create(tmp_path / "ws")
        input_data = ScoringInput(structure_path=complex_pdb, extra={"split_chains": "A,B"})
        runner.prepare_workspace(input_data, workspace)

        cfg = json.loads(workspace.config_path.read_text())
        assert cfg["split_chains"] == "A,B"

    def test_binding_score_breakdown(
        self, config: AutobioConfig, complex_pdb: Path, tmp_path: Path
    ) -> None:
        """Binding output includes per-term energy breakdown."""
        input_data = ScoringInput(structure_path=complex_pdb, extra={"repair": False})
        output = _run_e2e(
            "evoef2_binding",
            config,
            input_data,
            raw_files={"binding_output.txt": _BINDING_ENERGY_OUTPUT},
            tmp_path=tmp_path,
        )

        breakdown = output.scores[0].score_breakdown
        assert breakdown is not None
        assert breakdown["interD_vdwatt"] == pytest.approx(-15.30)
        assert breakdown["interD_electr"] == pytest.approx(-4.50)
        assert breakdown["interD_deslvP"] == pytest.approx(2.10)
        assert "Total" not in breakdown


# ---------------------------------------------------------------------------
# TestEvoEF2BuildMutantE2E
# ---------------------------------------------------------------------------


class TestEvoEF2BuildMutantE2E:
    """End-to-end tests for evoef2_build_mutant."""

    def test_build_mutant_pipeline(
        self, config: AutobioConfig, complex_pdb: Path, tmp_path: Path
    ) -> None:
        """Build mutant pipeline produces model structure."""
        input_data = ScoringInput(
            structure_path=complex_pdb,
            extra={"mutations": ["EA63Q"]},
        )
        output = _run_e2e(
            "evoef2_build_mutant",
            config,
            input_data,
            raw_files={"complex_Model_0001.pdb": _MODEL_PDB_CONTENT},
            log_files={"tool.log": _BUILD_MUTANT_ENERGY_OUTPUT},
            tmp_path=tmp_path,
        )

        assert len(output.scores) == 1
        s = output.scores[0]
        assert s.total_score == pytest.approx(-41.10)
        assert s.structure_path is not None
        assert s.structure_path.name == "complex_Model_0001.pdb"
        assert s.mutations == ["EA63Q"]

    def test_build_mutant_multiple_models(
        self, config: AutobioConfig, complex_pdb: Path, tmp_path: Path
    ) -> None:
        """Multiple model PDBs produce multiple ScoredStructure entries."""
        input_data = ScoringInput(
            structure_path=complex_pdb,
            extra={"mutations": ["EA63Q"]},
        )
        output = _run_e2e(
            "evoef2_build_mutant",
            config,
            input_data,
            raw_files={
                "complex_Model_0001.pdb": _MODEL_PDB_CONTENT,
                "complex_Model_0002.pdb": _MODEL_PDB_CONTENT,
                "complex_Model_0003.pdb": _MODEL_PDB_CONTENT,
            },
            log_files={"tool.log": _BUILD_MUTANT_ENERGY_OUTPUT},
            tmp_path=tmp_path,
        )

        assert len(output.scores) == 3
        for s in output.scores:
            assert s.mutations == ["EA63Q"]
            assert s.structure_path is not None

    def test_build_mutant_mutation_file_written(
        self, config: AutobioConfig, complex_pdb: Path, tmp_path: Path
    ) -> None:
        """Mutation file is written in EvoEF2 format."""
        runner = _make_runner("evoef2_build_mutant", config)
        workspace = Workspace.create(tmp_path / "ws")
        input_data = ScoringInput(
            structure_path=complex_pdb,
            extra={"mutations": ["EA63Q"]},
        )
        runner.prepare_workspace(input_data, workspace)

        mut_file = workspace.inputs_dir / "individual_list.txt"
        assert mut_file.exists()
        assert mut_file.read_text().strip() == "EA63Q;"

    def test_build_mutant_multiple_mutations_file(
        self, config: AutobioConfig, complex_pdb: Path, tmp_path: Path
    ) -> None:
        """Multiple mutations are comma-separated in mutation file."""
        runner = _make_runner("evoef2_build_mutant", config)
        workspace = Workspace.create(tmp_path / "ws")
        input_data = ScoringInput(
            structure_path=complex_pdb,
            extra={"mutations": ["EA63Q", "KB42A"]},
        )
        runner.prepare_workspace(input_data, workspace)

        mut_file = workspace.inputs_dir / "individual_list.txt"
        assert mut_file.read_text().strip() == "EA63Q,KB42A;"

    def test_build_mutant_config(
        self, config: AutobioConfig, complex_pdb: Path, tmp_path: Path
    ) -> None:
        """Config.json has correct command and parameters."""
        runner = _make_runner("evoef2_build_mutant", config)
        workspace = Workspace.create(tmp_path / "ws")
        input_data = ScoringInput(
            structure_path=complex_pdb,
            extra={"mutations": ["EA63Q"]},
        )
        runner.prepare_workspace(input_data, workspace)

        cfg = json.loads(workspace.config_path.read_text())
        assert cfg["command"] == "BuildMutant"
        assert cfg["mutant_file"] == "/workspace/inputs/individual_list.txt"
        assert cfg["mutations"] == ["EA63Q"]


# ---------------------------------------------------------------------------
# TestEvoEF2ValidationE2E
# ---------------------------------------------------------------------------


class TestEvoEF2ValidationE2E:
    """Tests for host-side input validation."""

    def test_nonexistent_structure_fails(self, config: AutobioConfig, tmp_path: Path) -> None:
        """Nonexistent input structure raises validation error."""
        runner = _make_runner("evoef2_repair", config)
        workspace = Workspace.create(tmp_path / "ws")
        input_data = ScoringInput(
            structure_path=tmp_path / "nonexistent.pdb",
            extra={},
        )
        with pytest.raises(AutobioError, match="does not exist"):
            runner.prepare_workspace(input_data, workspace)

    def test_non_pdb_format_fails(self, config: AutobioConfig, tmp_path: Path) -> None:
        """Non-PDB format raises validation error."""
        cif_path = tmp_path / "structure.cif"
        cif_path.write_text("data_test\n")
        runner = _make_runner("evoef2_binding", config)
        workspace = Workspace.create(tmp_path / "ws")
        input_data = ScoringInput(structure_path=cif_path, extra={})
        with pytest.raises(AutobioError, match="PDB format"):
            runner.prepare_workspace(input_data, workspace)

    def test_build_mutant_missing_mutations_fails(
        self, config: AutobioConfig, complex_pdb: Path, tmp_path: Path
    ) -> None:
        """Missing mutations for build_mutant raises validation error."""
        runner = _make_runner("evoef2_build_mutant", config)
        workspace = Workspace.create(tmp_path / "ws")
        input_data = ScoringInput(structure_path=complex_pdb, extra={})
        with pytest.raises(AutobioError, match="requires 'mutations'"):
            runner.prepare_workspace(input_data, workspace)

    def test_build_mutant_invalid_mutation_type_fails(
        self, config: AutobioConfig, complex_pdb: Path, tmp_path: Path
    ) -> None:
        """Non-list mutations raises validation error."""
        runner = _make_runner("evoef2_build_mutant", config)
        workspace = Workspace.create(tmp_path / "ws")
        input_data = ScoringInput(
            structure_path=complex_pdb,
            extra={"mutations": "EA63Q"},  # string, not list
        )
        with pytest.raises(AutobioError, match="list of strings"):
            runner.prepare_workspace(input_data, workspace)

    def test_build_mutant_invalid_mutation_format_fails(
        self, config: AutobioConfig, complex_pdb: Path, tmp_path: Path
    ) -> None:
        """Invalid mutation format raises validation error."""
        runner = _make_runner("evoef2_build_mutant", config)
        workspace = Workspace.create(tmp_path / "ws")
        input_data = ScoringInput(
            structure_path=complex_pdb,
            extra={"mutations": ["invalid"]},
        )
        with pytest.raises(AutobioError, match="Invalid mutation format"):
            runner.prepare_workspace(input_data, workspace)

    def test_binding_invalid_split_chains_fails(
        self, config: AutobioConfig, complex_pdb: Path, tmp_path: Path
    ) -> None:
        """Invalid split_chains format raises validation error."""
        runner = _make_runner("evoef2_binding", config)
        workspace = Workspace.create(tmp_path / "ws")
        input_data = ScoringInput(
            structure_path=complex_pdb,
            extra={"split_chains": "ABC"},  # missing comma
        )
        with pytest.raises(AutobioError, match="exactly one comma"):
            runner.prepare_workspace(input_data, workspace)
