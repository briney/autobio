"""Tests for the migrated evoef2 Tool (modes: repair, binding, build_mutant).

Each E2E test exercises the full pipeline:
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

from autobio.core.catalog import get_tool
from autobio.core.config import AutobioConfig
from autobio.core.registry import TOOL_REGISTRY, ToolCategory
from autobio.core.result import AutobioError
from autobio.core.workspace import Workspace
from autobio.schemas.scoring import (
    EvoEF2BindingInput,
    EvoEF2BuildMutantInput,
    EvoEF2RepairInput,
    ScoringOutput,
)
from autobio.tools import TOOL_RUNNERS, get_runner
from autobio.tools.evoef2 import EvoEF2Runner

_OLD_FLAT_NAMES = ("evoef2_repair", "evoef2_binding", "evoef2_build_mutant")

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
# Fixtures / helpers
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


def _make_runner(mode_name: str, config: AutobioConfig) -> EvoEF2Runner:
    """Create a runner with mocked container/GPU, current_mode pinned to *mode_name*."""
    with (
        patch("autobio.tools.base.ContainerManager"),
        patch("autobio.tools.base.GPUManager"),
    ):
        runner = EvoEF2Runner("evoef2", config)
    runner.current_mode = get_tool("evoef2").modes[mode_name]
    return runner


def _import_standardize():
    """Import the container's standardize module."""
    container_dir = str(Path(__file__).resolve().parent.parent.parent / "containers" / "evoef2")
    if container_dir not in sys.path:
        sys.path.insert(0, container_dir)
    mod = importlib.import_module("standardize")
    importlib.reload(mod)  # ensure fresh import
    return mod


def _run_e2e(
    mode_name: str,
    config: AutobioConfig,
    input_data,
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
    runner = _make_runner(mode_name, config)
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
    """End-to-end tests for evoef2 repair mode."""

    def test_repair_full_pipeline(
        self, config: AutobioConfig, complex_pdb: Path, tmp_path: Path
    ) -> None:
        """Repair pipeline produces structure and energy score."""
        input_data = EvoEF2RepairInput(structure_path=complex_pdb)
        output = _run_e2e(
            "repair",
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
        input_data = EvoEF2RepairInput(structure_path=complex_pdb)
        output = _run_e2e(
            "repair",
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
        runner = _make_runner("repair", config)
        workspace = Workspace.create(tmp_path / "ws")
        input_data = EvoEF2RepairInput(structure_path=complex_pdb)
        runner.prepare_workspace(input_data, workspace)

        cfg = json.loads(workspace.config_path.read_text())
        assert cfg["command"] == "RepairStructure"
        assert cfg["evoef2_bin"] == "/app/evoef2/EvoEF2"
        assert cfg["structure_path"].startswith("/workspace/inputs/")


# ---------------------------------------------------------------------------
# TestEvoEF2BindingE2E
# ---------------------------------------------------------------------------


class TestEvoEF2BindingE2E:
    """End-to-end tests for evoef2 binding mode."""

    def test_binding_full_pipeline(
        self, config: AutobioConfig, complex_pdb: Path, tmp_path: Path
    ) -> None:
        """Binding pipeline produces energy score with breakdown."""
        input_data = EvoEF2BindingInput(structure_path=complex_pdb, repair=False)
        output = _run_e2e(
            "binding",
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
        input_data = EvoEF2BindingInput(structure_path=complex_pdb)
        output = _run_e2e(
            "binding",
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
        input_data = EvoEF2BindingInput(structure_path=complex_pdb, repair=False)
        output = _run_e2e(
            "binding",
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
        runner = _make_runner("binding", config)
        workspace = Workspace.create(tmp_path / "ws")
        input_data = EvoEF2BindingInput(structure_path=complex_pdb)
        runner.prepare_workspace(input_data, workspace)

        cfg = json.loads(workspace.config_path.read_text())
        assert cfg["repair"] is True

    def test_binding_config_with_split(
        self, config: AutobioConfig, complex_pdb: Path, tmp_path: Path
    ) -> None:
        """split_chains parameter is written to config."""
        runner = _make_runner("binding", config)
        workspace = Workspace.create(tmp_path / "ws")
        input_data = EvoEF2BindingInput(structure_path=complex_pdb, split_chains="A,B")
        runner.prepare_workspace(input_data, workspace)

        cfg = json.loads(workspace.config_path.read_text())
        assert cfg["split_chains"] == "A,B"

    def test_binding_score_breakdown(
        self, config: AutobioConfig, complex_pdb: Path, tmp_path: Path
    ) -> None:
        """Binding output includes per-term energy breakdown."""
        input_data = EvoEF2BindingInput(structure_path=complex_pdb, repair=False)
        output = _run_e2e(
            "binding",
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
    """End-to-end tests for evoef2 build_mutant mode."""

    def test_build_mutant_pipeline(
        self, config: AutobioConfig, complex_pdb: Path, tmp_path: Path
    ) -> None:
        """Build mutant pipeline produces model structure."""
        input_data = EvoEF2BuildMutantInput(structure_path=complex_pdb, mutations=["EA63Q"])
        output = _run_e2e(
            "build_mutant",
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
        input_data = EvoEF2BuildMutantInput(structure_path=complex_pdb, mutations=["EA63Q"])
        output = _run_e2e(
            "build_mutant",
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
        runner = _make_runner("build_mutant", config)
        workspace = Workspace.create(tmp_path / "ws")
        input_data = EvoEF2BuildMutantInput(structure_path=complex_pdb, mutations=["EA63Q"])
        runner.prepare_workspace(input_data, workspace)

        mut_file = workspace.inputs_dir / "individual_list.txt"
        assert mut_file.exists()
        assert mut_file.read_text().strip() == "EA63Q;"

    def test_build_mutant_multiple_mutations_file(
        self, config: AutobioConfig, complex_pdb: Path, tmp_path: Path
    ) -> None:
        """Multiple mutations are comma-separated in mutation file."""
        runner = _make_runner("build_mutant", config)
        workspace = Workspace.create(tmp_path / "ws")
        input_data = EvoEF2BuildMutantInput(
            structure_path=complex_pdb, mutations=["EA63Q", "KB42A"]
        )
        runner.prepare_workspace(input_data, workspace)

        mut_file = workspace.inputs_dir / "individual_list.txt"
        assert mut_file.read_text().strip() == "EA63Q,KB42A;"

    def test_build_mutant_config(
        self, config: AutobioConfig, complex_pdb: Path, tmp_path: Path
    ) -> None:
        """Config.json has correct command and parameters."""
        runner = _make_runner("build_mutant", config)
        workspace = Workspace.create(tmp_path / "ws")
        input_data = EvoEF2BuildMutantInput(structure_path=complex_pdb, mutations=["EA63Q"])
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
        runner = _make_runner("repair", config)
        workspace = Workspace.create(tmp_path / "ws")
        input_data = EvoEF2RepairInput(structure_path=tmp_path / "nonexistent.pdb")
        with pytest.raises(AutobioError, match="does not exist"):
            runner.prepare_workspace(input_data, workspace)

    def test_non_pdb_format_fails(self, config: AutobioConfig, tmp_path: Path) -> None:
        """Non-PDB format raises validation error."""
        cif_path = tmp_path / "structure.cif"
        cif_path.write_text("data_test\n")
        runner = _make_runner("binding", config)
        workspace = Workspace.create(tmp_path / "ws")
        input_data = EvoEF2BindingInput(structure_path=cif_path)
        with pytest.raises(AutobioError, match="PDB format"):
            runner.prepare_workspace(input_data, workspace)

    def test_build_mutant_empty_mutations_fails(
        self, config: AutobioConfig, complex_pdb: Path, tmp_path: Path
    ) -> None:
        """Empty mutations list for build_mutant raises the reworded validation error."""
        runner = _make_runner("build_mutant", config)
        workspace = Workspace.create(tmp_path / "ws")
        input_data = EvoEF2BuildMutantInput(structure_path=complex_pdb, mutations=[])
        with pytest.raises(
            AutobioError, match="EvoEF2 build_mutant requires at least one mutation"
        ):
            runner.prepare_workspace(input_data, workspace)

    def test_build_mutant_invalid_mutation_format_fails(
        self, config: AutobioConfig, complex_pdb: Path, tmp_path: Path
    ) -> None:
        """Invalid mutation format raises validation error."""
        runner = _make_runner("build_mutant", config)
        workspace = Workspace.create(tmp_path / "ws")
        input_data = EvoEF2BuildMutantInput(structure_path=complex_pdb, mutations=["invalid"])
        with pytest.raises(AutobioError, match="Invalid mutation format"):
            runner.prepare_workspace(input_data, workspace)

    def test_binding_invalid_split_chains_fails(
        self, config: AutobioConfig, complex_pdb: Path, tmp_path: Path
    ) -> None:
        """Invalid split_chains format raises validation error."""
        runner = _make_runner("binding", config)
        workspace = Workspace.create(tmp_path / "ws")
        input_data = EvoEF2BindingInput(structure_path=complex_pdb, split_chains="ABC")
        with pytest.raises(AutobioError, match="exactly one comma"):
            runner.prepare_workspace(input_data, workspace)


# ---------------------------------------------------------------------------
# TestEvoEF2ByteCompatConfig — full-dict config.json equality, per mode
# ---------------------------------------------------------------------------


class TestEvoEF2ByteCompatConfig:
    """Full-dict ``config.json`` equality tests, pinning key order per mode."""

    def test_repair_full_config(
        self, config: AutobioConfig, complex_pdb: Path, tmp_path: Path
    ) -> None:
        r = _make_runner("repair", config)
        workspace = Workspace.create(tmp_path / "ws")
        r.prepare_workspace(EvoEF2RepairInput(structure_path=complex_pdb), workspace)

        cfg = json.loads(workspace.config_path.read_text())
        expected = {
            "command": "RepairStructure",
            "structure_path": f"/workspace/inputs/{complex_pdb.name}",
            "evoef2_bin": "/app/evoef2/EvoEF2",
            "out_dir": "/workspace/outputs/raw",
        }
        assert cfg == expected
        assert list(cfg.keys()) == list(expected.keys())

    def test_binding_full_config_without_split_chains(
        self, config: AutobioConfig, complex_pdb: Path, tmp_path: Path
    ) -> None:
        """Binding config with default repair, no split_chains key present."""
        r = _make_runner("binding", config)
        workspace = Workspace.create(tmp_path / "ws")
        r.prepare_workspace(EvoEF2BindingInput(structure_path=complex_pdb), workspace)

        cfg = json.loads(workspace.config_path.read_text())
        expected = {
            "command": "ComputeBinding",
            "structure_path": f"/workspace/inputs/{complex_pdb.name}",
            "evoef2_bin": "/app/evoef2/EvoEF2",
            "out_dir": "/workspace/outputs/raw",
            "repair": True,
        }
        assert cfg == expected
        assert list(cfg.keys()) == list(expected.keys())
        assert "split_chains" not in cfg

    def test_binding_full_config_with_split_chains(
        self, config: AutobioConfig, complex_pdb: Path, tmp_path: Path
    ) -> None:
        """Binding config with repair disabled and split_chains present, in order."""
        r = _make_runner("binding", config)
        workspace = Workspace.create(tmp_path / "ws")
        r.prepare_workspace(
            EvoEF2BindingInput(structure_path=complex_pdb, repair=False, split_chains="A,B"),
            workspace,
        )

        cfg = json.loads(workspace.config_path.read_text())
        expected = {
            "command": "ComputeBinding",
            "structure_path": f"/workspace/inputs/{complex_pdb.name}",
            "evoef2_bin": "/app/evoef2/EvoEF2",
            "out_dir": "/workspace/outputs/raw",
            "repair": False,
            "split_chains": "A,B",
        }
        assert cfg == expected
        assert list(cfg.keys()) == list(expected.keys())

    def test_build_mutant_full_config(
        self, config: AutobioConfig, complex_pdb: Path, tmp_path: Path
    ) -> None:
        r = _make_runner("build_mutant", config)
        workspace = Workspace.create(tmp_path / "ws")
        r.prepare_workspace(
            EvoEF2BuildMutantInput(structure_path=complex_pdb, mutations=["EA63Q", "KB42A"]),
            workspace,
        )

        cfg = json.loads(workspace.config_path.read_text())
        expected = {
            "command": "BuildMutant",
            "structure_path": f"/workspace/inputs/{complex_pdb.name}",
            "evoef2_bin": "/app/evoef2/EvoEF2",
            "out_dir": "/workspace/outputs/raw",
            "mutations": ["EA63Q", "KB42A"],
            "mutant_file": "/workspace/inputs/individual_list.txt",
        }
        assert cfg == expected
        assert list(cfg.keys()) == list(expected.keys())

    def test_extra_shadowing_typed_field_rejected(
        self, config: AutobioConfig, complex_pdb: Path, tmp_path: Path
    ) -> None:
        """extra containing a typed field name (repair) raises."""
        r = _make_runner("binding", config)
        workspace = Workspace.create(tmp_path / "ws")
        input_data = EvoEF2BindingInput(structure_path=complex_pdb, extra={"repair": True})
        with pytest.raises(AutobioError, match="collide with typed input fields"):
            r.prepare_workspace(input_data, workspace)

    def test_extra_unknown_key_passed_through(
        self, config: AutobioConfig, complex_pdb: Path, tmp_path: Path
    ) -> None:
        r = _make_runner("repair", config)
        workspace = Workspace.create(tmp_path / "ws")
        input_data = EvoEF2RepairInput(structure_path=complex_pdb, extra={"custom_flag": True})
        r.prepare_workspace(input_data, workspace)

        cfg = json.loads(workspace.config_path.read_text())
        assert cfg["custom_flag"] is True


# ---------------------------------------------------------------------------
# TestEvoEF2Registration
# ---------------------------------------------------------------------------


class TestEvoEF2Registration:
    """Tests for the catalog Tool + runner registration."""

    def test_evoef2_registered_as_single_tool(self) -> None:
        import autobio.tools  # noqa: F401 - populate registries

        tool = get_tool("evoef2")
        assert set(tool.modes) == {"repair", "binding", "build_mutant"}
        assert tool.default_mode == "repair"
        assert tool.category == ToolCategory.SCORING
        assert tool.requires_gpu is False
        assert tool.gpu_count == 0

    @pytest.mark.parametrize("flat_name", _OLD_FLAT_NAMES)
    def test_old_flat_names_absent_from_tool_registry(self, flat_name: str) -> None:
        import autobio.tools  # noqa: F401

        assert flat_name not in TOOL_REGISTRY

    @pytest.mark.parametrize("flat_name", _OLD_FLAT_NAMES)
    def test_old_flat_names_absent_from_tool_runners(self, flat_name: str) -> None:
        import autobio.tools  # noqa: F401

        assert flat_name not in TOOL_RUNNERS

    def test_evoef2_in_tool_runners(self) -> None:
        import autobio.tools  # noqa: F401

        assert "evoef2" in TOOL_RUNNERS
        assert TOOL_RUNNERS["evoef2"] is EvoEF2Runner

    def test_get_runner_evoef2_resolves_catalog_tool(self, config: AutobioConfig) -> None:
        with (
            patch("autobio.tools.base.ContainerManager"),
            patch("autobio.tools.base.GPUManager"),
        ):
            r = get_runner("evoef2", config)
        assert isinstance(r, EvoEF2Runner)
        assert r.tool_name == "evoef2"
        assert r.tool is not None and r.tool.name == "evoef2"

    @pytest.mark.parametrize("flat_name", _OLD_FLAT_NAMES)
    def test_get_runner_removed_flat_name_raises(
        self, flat_name: str, config: AutobioConfig
    ) -> None:
        with pytest.raises(KeyError, match=flat_name):
            get_runner(flat_name, config)

    @pytest.mark.parametrize(
        ("mode_name", "timeout"),
        [("repair", 600), ("binding", 600), ("build_mutant", 600)],
    )
    def test_modes_have_uniform_timeout(self, mode_name: str, timeout: int) -> None:
        import autobio.tools  # noqa: F401

        assert get_tool("evoef2").modes[mode_name].default_timeout == timeout

    def test_modes_share_single_image(self) -> None:
        import autobio.tools  # noqa: F401

        tool = get_tool("evoef2")
        assert tool.image_tag == "evoef2:1.0.0"
        for mode in tool.modes.values():
            assert mode.image_tag is None  # falls back to Tool.image_tag


# ---------------------------------------------------------------------------
# TestEvoEF2InfoSnapshot
# ---------------------------------------------------------------------------


class TestEvoEF2InfoSnapshot:
    """``autobio info evoef2`` output — per-mode notes, hints, output_schema."""

    def test_info_snapshot(self) -> None:
        import autobio.tools  # noqa: F401
        from autobio.cli.formatters import OutputFormat, format_tool_info_catalog

        parsed = json.loads(format_tool_info_catalog(get_tool("evoef2"), OutputFormat.JSON))
        assert [m["name"] for m in parsed["modes"]] == ["repair", "binding", "build_mutant"]

        repair_mode = parsed["modes"][0]
        assert len(repair_mode["notes"]) > 0
        assert "output_schema" in repair_mode

        binding_mode = parsed["modes"][1]
        assert len(binding_mode["notes"]) > 0
        assert "output_schema" in binding_mode
        repair_prop = binding_mode["input_schema"]["properties"]["repair"]
        assert repair_prop["x-autobio"]["widget"] == "toggle"
        # Reworded notes no longer mention the extra dict.
        assert not any("extra[" in note for note in binding_mode["notes"])

        build_mutant_mode = parsed["modes"][2]
        assert "output_schema" in build_mutant_mode
        mutations_prop = build_mutant_mode["input_schema"]["properties"]["mutations"]
        assert mutations_prop["x-autobio"]["widget"] == "text"
        assert not any("extra[" in note for note in build_mutant_mode["notes"])
        assert not any("extra[" in note for note in repair_mode["notes"])
