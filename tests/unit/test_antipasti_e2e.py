"""End-to-end tests for ANTIPASTI.

Each test exercises the full pipeline:
    input construction -> validation -> prepare_workspace ->
    (simulated JSON output) -> standardize.py -> parse_output -> verify

The only thing not tested is the actual ANTIPASTI model execution and NMA
preprocessing. The standardize script is imported and run directly against
realistic JSON output data.
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
from autobio.core.result import AutobioError
from autobio.core.workspace import Workspace
from autobio.schemas.binding_affinity import AntipastiInput, BindingAffinityOutput
from autobio.tools.antipasti import AntipastiRunner

# ---------------------------------------------------------------------------
# Realistic ANTIPASTI output data
# ---------------------------------------------------------------------------

# Minimal but valid three-chain PDB content for testing
_MINIMAL_COMPLEX_PDB = (
    "HEADER    TEST ANTIBODY-ANTIGEN COMPLEX\n"
    "ATOM      1  N   ALA H   1       1.000   2.000   3.000  1.00 10.00           N\n"
    "ATOM      2  CA  ALA H   1       2.000   3.000   4.000  1.00 10.00           C\n"
    "ATOM      3  C   ALA H   1       3.000   4.000   5.000  1.00 10.00           C\n"
    "ATOM      4  O   ALA H   1       3.500   4.500   5.500  1.00 10.00           O\n"
    "ATOM      5  N   GLY L   1       4.000   5.000   6.000  1.00 12.00           N\n"
    "ATOM      6  CA  GLY L   1       5.000   6.000   7.000  1.00 12.00           C\n"
    "ATOM      7  C   GLY L   1       6.000   7.000   8.000  1.00 12.00           C\n"
    "ATOM      8  O   GLY L   1       6.500   7.500   8.500  1.00 12.00           O\n"
    "ATOM      9  N   LEU A   1       7.000   8.000   9.000  1.00 14.00           N\n"
    "ATOM     10  CA  LEU A   1       8.000   9.000  10.000  1.00 14.00           C\n"
    "ATOM     11  C   LEU A   1       9.000  10.000  11.000  1.00 14.00           C\n"
    "ATOM     12  O   LEU A   1       9.500  10.500  11.500  1.00 14.00           O\n"
    "END\n"
)

# Single structure output JSON (what inference.py produces)
_SINGLE_OUTPUT_JSON = json.dumps(
    {
        "pdb_id": "complex",
        "log10_kd": -8.542,
        "kd_molar": 2.87e-9,
        "heavy_chain": "H",
        "light_chain": "L",
        "antigen_chains": ["A"],
        "modes": "all",
        "checkpoint": "model_epochs_1044_modes_all_pool_1_filters_4_size_4",
    }
)

# Multi-chain antigen output
_MULTI_ANTIGEN_OUTPUT_JSON = json.dumps(
    {
        "pdb_id": "multi_ag",
        "log10_kd": -6.123,
        "kd_molar": 7.53e-7,
        "heavy_chain": "A",
        "light_chain": "B",
        "antigen_chains": ["C", "D"],
        "modes": "all",
        "checkpoint": "model_epochs_1044_modes_all_pool_1_filters_4_size_4",
    }
)

# Custom-modes output (integer normal-mode count)
_CUSTOM_MODES_OUTPUT_JSON = json.dumps(
    {
        "pdb_id": "complex",
        "log10_kd": -7.310,
        "kd_molar": 4.9e-8,
        "heavy_chain": "H",
        "light_chain": "L",
        "antigen_chains": ["A"],
        "modes": 100,
        "checkpoint": "model_epochs_1044_modes_all_pool_1_filters_4_size_4",
    }
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def config() -> AutobioConfig:
    return AutobioConfig.resolve()


@pytest.fixture()
def complex_pdb(tmp_path: Path) -> Path:
    """Write a minimal three-chain PDB."""
    pdb_path = tmp_path / "complex.pdb"
    pdb_path.write_text(_MINIMAL_COMPLEX_PDB)
    return pdb_path


def _make_runner(config: AutobioConfig) -> AntipastiRunner:
    """Create a runner with mocked container/GPU (we simulate container output)."""
    with (
        patch("autobio.tools.base.ContainerManager"),
        patch("autobio.tools.base.GPUManager"),
    ):
        runner = AntipastiRunner("antipasti", config)
    runner.current_mode = get_tool("antipasti").modes["predict"]
    return runner


def _import_standardize():
    """Import the container's standardize module."""
    container_dir = str(Path(__file__).resolve().parent.parent.parent / "containers" / "antipasti")
    if container_dir not in sys.path:
        sys.path.insert(0, container_dir)
    mod = importlib.import_module("standardize")
    importlib.reload(mod)  # ensure fresh import
    return mod


def _run_e2e(
    config: AutobioConfig,
    input_data: AntipastiInput,
    raw_json_content: str,
    tmp_path: Path,
) -> BindingAffinityOutput:
    """Full end-to-end pipeline without Docker.

    1. prepare_workspace
    2. Write simulated raw JSON output
    3. Run the container's standardize.py
    4. parse_output
    """
    runner = _make_runner(config)
    workspace = Workspace.create(tmp_path / "ws")

    # Step 1: prepare workspace
    runner.prepare_workspace(input_data, workspace)

    # Verify config.json was written
    cfg = json.loads(workspace.config_path.read_text())
    assert cfg["pdb_path"] is not None
    assert "checkpoints/full_ags_all_modes" in cfg["checkpoint_path"]

    # Step 2: write simulated raw JSON (what the container would produce)
    (workspace.raw_output_dir / "output.json").write_text(raw_json_content)

    # Step 3: run the actual standardize.py script
    std_mod = _import_standardize()
    std_mod.standardize(workspace.root)

    # Verify result_data.json was produced
    result_data_path = workspace.std_output_dir / "result_data.json"
    assert result_data_path.exists(), "standardize.py did not produce result_data.json"

    # Step 4: parse output
    output = runner.parse_output(workspace)
    assert isinstance(output, BindingAffinityOutput)
    return output


# ---------------------------------------------------------------------------
# TestAntipastiBasicE2E
# ---------------------------------------------------------------------------


class TestAntipastiBasicE2E:
    """End-to-end test for basic ANTIPASTI prediction."""

    def test_single_structure_pipeline(
        self, config: AutobioConfig, complex_pdb: Path, tmp_path: Path
    ) -> None:
        """Single structure, single antigen chain — basic pipeline."""
        input_data = AntipastiInput(
            structure_path=complex_pdb,
            heavy_chain="H",
            light_chain="L",
            antigen_chains=["A"],
        )
        output = _run_e2e(config, input_data, _SINGLE_OUTPUT_JSON, tmp_path)

        assert len(output.predictions) == 1
        p = output.predictions[0]
        assert p.log10_kd == pytest.approx(-8.542)
        assert p.kd_molar == pytest.approx(2.87e-9, rel=1e-2)
        assert p.units == "log10(Kd) [M]"

    def test_chain_info_preserved(
        self, config: AutobioConfig, complex_pdb: Path, tmp_path: Path
    ) -> None:
        """Chain information is preserved in score breakdown."""
        input_data = AntipastiInput(
            structure_path=complex_pdb,
            heavy_chain="H",
            light_chain="L",
            antigen_chains=["A"],
        )
        output = _run_e2e(config, input_data, _SINGLE_OUTPUT_JSON, tmp_path)

        breakdown = output.predictions[0].score_breakdown
        assert breakdown is not None
        assert breakdown["heavy_chain"] == "H"
        assert breakdown["light_chain"] == "L"
        assert breakdown["antigen_chains"] == ["A"]

    def test_modes_in_breakdown(
        self, config: AutobioConfig, complex_pdb: Path, tmp_path: Path
    ) -> None:
        """Normal modes setting is captured in score breakdown."""
        input_data = AntipastiInput(
            structure_path=complex_pdb,
            heavy_chain="H",
            light_chain="L",
            antigen_chains=["A"],
        )
        output = _run_e2e(config, input_data, _SINGLE_OUTPUT_JSON, tmp_path)

        breakdown = output.predictions[0].score_breakdown
        assert breakdown is not None
        assert breakdown["modes"] == "all"

    def test_checkpoint_in_breakdown(
        self, config: AutobioConfig, complex_pdb: Path, tmp_path: Path
    ) -> None:
        """Checkpoint name is captured in score breakdown."""
        input_data = AntipastiInput(
            structure_path=complex_pdb,
            heavy_chain="H",
            light_chain="L",
            antigen_chains=["A"],
        )
        output = _run_e2e(config, input_data, _SINGLE_OUTPUT_JSON, tmp_path)

        breakdown = output.predictions[0].score_breakdown
        assert breakdown is not None
        assert "model_epochs" in breakdown["checkpoint"]


# ---------------------------------------------------------------------------
# TestAntipastiMultiAntigenE2E
# ---------------------------------------------------------------------------


class TestAntipastiMultiAntigenE2E:
    """End-to-end test for multi-chain antigen predictions."""

    def test_multi_antigen_chains(
        self, config: AutobioConfig, complex_pdb: Path, tmp_path: Path
    ) -> None:
        """Multi-chain antigen with different chain IDs."""
        input_data = AntipastiInput(
            structure_path=complex_pdb,
            heavy_chain="A",
            light_chain="B",
            antigen_chains=["C", "D"],
        )
        output = _run_e2e(config, input_data, _MULTI_ANTIGEN_OUTPUT_JSON, tmp_path)

        assert len(output.predictions) == 1
        p = output.predictions[0]
        assert p.log10_kd == pytest.approx(-6.123)
        assert p.kd_molar == pytest.approx(7.53e-7, rel=1e-2)

    def test_multi_antigen_chains_in_breakdown(
        self, config: AutobioConfig, complex_pdb: Path, tmp_path: Path
    ) -> None:
        """Multi-chain antigen chain IDs are preserved."""
        input_data = AntipastiInput(
            structure_path=complex_pdb,
            heavy_chain="A",
            light_chain="B",
            antigen_chains=["C", "D"],
        )
        output = _run_e2e(config, input_data, _MULTI_ANTIGEN_OUTPUT_JSON, tmp_path)

        breakdown = output.predictions[0].score_breakdown
        assert breakdown is not None
        assert breakdown["antigen_chains"] == ["C", "D"]


# ---------------------------------------------------------------------------
# TestAntipastiCustomModesE2E
# ---------------------------------------------------------------------------


class TestAntipastiCustomModesE2E:
    """End-to-end test for an integer 'modes' override."""

    def test_integer_modes_pipeline(
        self, config: AutobioConfig, complex_pdb: Path, tmp_path: Path
    ) -> None:
        """modes=100 (typed field) flows through config.json and breakdown."""
        input_data = AntipastiInput(
            structure_path=complex_pdb,
            heavy_chain="H",
            light_chain="L",
            antigen_chains=["A"],
            modes=100,
        )
        runner = _make_runner(config)
        workspace = Workspace.create(tmp_path / "ws")
        runner.prepare_workspace(input_data, workspace)
        cfg = json.loads(workspace.config_path.read_text())
        assert cfg["modes"] == 100

        output = _run_e2e(config, input_data, _CUSTOM_MODES_OUTPUT_JSON, tmp_path)
        breakdown = output.predictions[0].score_breakdown
        assert breakdown is not None
        assert breakdown["modes"] == 100


# ---------------------------------------------------------------------------
# TestAntipastiConfigE2E
# ---------------------------------------------------------------------------


class TestAntipastiConfigE2E:
    """Tests for config.json generation and validation."""

    def test_config_params_written(
        self, config: AutobioConfig, complex_pdb: Path, tmp_path: Path
    ) -> None:
        """All parameters are correctly written to config.json."""
        runner = _make_runner(config)
        workspace = Workspace.create(tmp_path / "ws")
        input_data = AntipastiInput(
            structure_path=complex_pdb,
            heavy_chain="H",
            light_chain="L",
            antigen_chains=["A"],
            modes=100,
        )
        runner.prepare_workspace(input_data, workspace)

        cfg = json.loads(workspace.config_path.read_text())
        assert cfg["modes"] == 100
        assert cfg["heavy_chain"] == "H"
        assert cfg["light_chain"] == "L"
        assert cfg["antigen_chains"] == ["A"]
        assert cfg["antipasti_dir"] == "/app/antipasti"

    def test_nonexistent_structure_fails(self, config: AutobioConfig, tmp_path: Path) -> None:
        """Nonexistent input structure raises host-side validation error."""
        runner = _make_runner(config)
        workspace = Workspace.create(tmp_path / "ws")
        input_data = AntipastiInput(
            structure_path=tmp_path / "nonexistent.pdb",
            heavy_chain="H",
            light_chain="L",
            antigen_chains=["A"],
        )
        with pytest.raises(AutobioError, match="does not exist"):
            runner.prepare_workspace(input_data, workspace)

    def test_duplicate_chains_fails(
        self, config: AutobioConfig, complex_pdb: Path, tmp_path: Path
    ) -> None:
        """Duplicate chain IDs raises host-side validation error."""
        runner = _make_runner(config)
        workspace = Workspace.create(tmp_path / "ws")
        input_data = AntipastiInput(
            structure_path=complex_pdb,
            heavy_chain="A",
            light_chain="B",
            antigen_chains=["A"],
        )
        with pytest.raises(AutobioError, match="Duplicate"):
            runner.prepare_workspace(input_data, workspace)

    def test_empty_antigen_chains_fails(
        self, config: AutobioConfig, complex_pdb: Path, tmp_path: Path
    ) -> None:
        """Empty antigen chains list raises host-side validation error."""
        runner = _make_runner(config)
        workspace = Workspace.create(tmp_path / "ws")
        input_data = AntipastiInput(
            structure_path=complex_pdb,
            heavy_chain="H",
            light_chain="L",
            antigen_chains=[],
        )
        with pytest.raises(AutobioError, match="antigen_chains"):
            runner.prepare_workspace(input_data, workspace)
