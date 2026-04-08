"""End-to-end tests for PRODIGY.

Each test exercises the full pipeline:
    input construction -> validation -> prepare_workspace ->
    (simulated JSON output) -> standardize.py -> parse_output -> verify

The only thing not tested is the actual PRODIGY contact calculation.
The standardize script is imported and run directly against realistic
JSON output data.
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
from autobio.schemas.protein_binding_affinity import (
    ProteinBindingAffinityInput,
    ProteinBindingAffinityOutput,
)
from autobio.tools.prodigy import ProdigyRunner

# ---------------------------------------------------------------------------
# Realistic PRODIGY output data
# ---------------------------------------------------------------------------

_MINIMAL_COMPLEX_PDB = (
    "HEADER    TEST PROTEIN COMPLEX\n"
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

# What run_prodigy.py would produce (raw output)
_SINGLE_OUTPUT_JSON = json.dumps(
    {
        "delta_g": -10.2,
        "kd": 3.37e-08,
        "intermolecular_contacts": 42,
        "charged_charged_contacts": 3.0,
        "charged_polar_contacts": 5.0,
        "charged_apolar_contacts": 12.0,
        "polar_polar_contacts": 2.0,
        "polar_apolar_contacts": 8.0,
        "apolar_apolar_contacts": 12.0,
        "hydrophilic_hydrophilic_contacts": 6.0,
        "hydrophobic_hydrophilic_contacts": 14.0,
        "hydrophobic_hydrophobic_contacts": 22.0,
        "pct_apolar_nis": 42.31,
        "pct_charged_nis": 18.46,
        "pct_polar_nis": 39.23,
        "selection": ["A", "B"],
        "temperature": 25.0,
        "distance_cutoff": 5.5,
        "n_chains": 2,
        "n_residues": 350,
        "structure": "complex",
    }
)

# Custom temperature output
_CUSTOM_TEMP_OUTPUT_JSON = json.dumps(
    {
        "delta_g": -10.2,
        "kd": 4.52e-08,
        "intermolecular_contacts": 42,
        "charged_charged_contacts": 3.0,
        "charged_polar_contacts": 5.0,
        "charged_apolar_contacts": 12.0,
        "polar_polar_contacts": 2.0,
        "polar_apolar_contacts": 8.0,
        "apolar_apolar_contacts": 12.0,
        "hydrophilic_hydrophilic_contacts": 6.0,
        "hydrophobic_hydrophilic_contacts": 14.0,
        "hydrophobic_hydrophobic_contacts": 22.0,
        "pct_apolar_nis": 42.31,
        "pct_charged_nis": 18.46,
        "pct_polar_nis": 39.23,
        "selection": ["A", "B"],
        "temperature": 37.0,
        "distance_cutoff": 5.5,
        "n_chains": 2,
        "n_residues": 350,
        "structure": "complex",
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
    """Write a minimal two-chain PDB."""
    pdb_path = tmp_path / "complex.pdb"
    pdb_path.write_text(_MINIMAL_COMPLEX_PDB)
    return pdb_path


def _make_runner(config: AutobioConfig) -> ProdigyRunner:
    """Create a runner with mocked container/GPU (we simulate container output)."""
    with (
        patch("autobio.tools.base.ContainerManager"),
        patch("autobio.tools.base.GPUManager"),
    ):
        return ProdigyRunner("prodigy", config)


def _import_standardize():
    """Import the container's standardize module."""
    container_dir = str(Path(__file__).resolve().parent.parent.parent / "containers" / "prodigy")
    if container_dir not in sys.path:
        sys.path.insert(0, container_dir)
    mod = importlib.import_module("standardize")
    importlib.reload(mod)  # ensure fresh import
    return mod


def _run_e2e(
    config: AutobioConfig,
    input_data: ProteinBindingAffinityInput,
    raw_json_content: str,
    tmp_path: Path,
) -> ProteinBindingAffinityOutput:
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
    assert cfg["structure_path"] is not None

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
    assert isinstance(output, ProteinBindingAffinityOutput)
    return output


# ---------------------------------------------------------------------------
# TestProdigyBasicE2E
# ---------------------------------------------------------------------------


class TestProdigyBasicE2E:
    """End-to-end test for basic PRODIGY prediction."""

    def test_single_structure_pipeline(
        self, config: AutobioConfig, complex_pdb: Path, tmp_path: Path
    ) -> None:
        """Single structure — basic pipeline."""
        input_data = ProteinBindingAffinityInput(
            structure_path=complex_pdb,
            chain_selection="A B",
        )
        output = _run_e2e(config, input_data, _SINGLE_OUTPUT_JSON, tmp_path)

        assert len(output.predictions) == 1
        p = output.predictions[0]
        assert p.delta_g_kcal_mol == pytest.approx(-10.2)
        assert p.kd_molar == pytest.approx(3.37e-08, rel=1e-2)
        assert p.units == "kcal/mol"

    def test_contact_counts_preserved(
        self, config: AutobioConfig, complex_pdb: Path, tmp_path: Path
    ) -> None:
        """Contact counts are preserved in score breakdown."""
        input_data = ProteinBindingAffinityInput(
            structure_path=complex_pdb,
            chain_selection="A B",
        )
        output = _run_e2e(config, input_data, _SINGLE_OUTPUT_JSON, tmp_path)

        breakdown = output.predictions[0].score_breakdown
        assert breakdown is not None
        assert breakdown["intermolecular_contacts"] == 42
        assert breakdown["charged_charged_contacts"] == 3.0
        assert breakdown["apolar_apolar_contacts"] == 12.0

    def test_nis_percentages_preserved(
        self, config: AutobioConfig, complex_pdb: Path, tmp_path: Path
    ) -> None:
        """NIS percentages are preserved in score breakdown."""
        input_data = ProteinBindingAffinityInput(
            structure_path=complex_pdb,
            chain_selection="A B",
        )
        output = _run_e2e(config, input_data, _SINGLE_OUTPUT_JSON, tmp_path)

        breakdown = output.predictions[0].score_breakdown
        assert breakdown is not None
        assert breakdown["pct_apolar_nis"] == pytest.approx(42.31)
        assert breakdown["pct_charged_nis"] == pytest.approx(18.46)
        assert breakdown["pct_polar_nis"] == pytest.approx(39.23)

    def test_chain_selection_in_breakdown(
        self, config: AutobioConfig, complex_pdb: Path, tmp_path: Path
    ) -> None:
        """Chain selection is captured in score breakdown."""
        input_data = ProteinBindingAffinityInput(
            structure_path=complex_pdb,
            chain_selection="A B",
        )
        output = _run_e2e(config, input_data, _SINGLE_OUTPUT_JSON, tmp_path)

        breakdown = output.predictions[0].score_breakdown
        assert breakdown is not None
        assert breakdown["chain_selection"] == "A B"


# ---------------------------------------------------------------------------
# TestProdigyCustomParamsE2E
# ---------------------------------------------------------------------------


class TestProdigyCustomParamsE2E:
    """End-to-end tests with custom parameters."""

    def test_custom_temperature(
        self, config: AutobioConfig, complex_pdb: Path, tmp_path: Path
    ) -> None:
        """Custom temperature affects Kd but not delta-G."""
        input_data = ProteinBindingAffinityInput(
            structure_path=complex_pdb,
            chain_selection="A B",
            temperature=37.0,
        )
        output = _run_e2e(config, input_data, _CUSTOM_TEMP_OUTPUT_JSON, tmp_path)

        p = output.predictions[0]
        assert p.delta_g_kcal_mol == pytest.approx(-10.2)
        # Kd changes with temperature
        assert p.kd_molar == pytest.approx(4.52e-08, rel=1e-2)

        breakdown = p.score_breakdown
        assert breakdown is not None
        assert breakdown["temperature_celsius"] == pytest.approx(37.0)

    def test_no_chain_selection(
        self, config: AutobioConfig, complex_pdb: Path, tmp_path: Path
    ) -> None:
        """None chain_selection (all inter-chain contacts)."""
        input_data = ProteinBindingAffinityInput(
            structure_path=complex_pdb,
        )
        output = _run_e2e(config, input_data, _SINGLE_OUTPUT_JSON, tmp_path)

        assert len(output.predictions) == 1
        p = output.predictions[0]
        assert p.delta_g_kcal_mol == pytest.approx(-10.2)

        breakdown = p.score_breakdown
        assert breakdown is not None
        assert breakdown["chain_selection"] is None


# ---------------------------------------------------------------------------
# TestProdigyConfigE2E
# ---------------------------------------------------------------------------


class TestProdigyConfigE2E:
    """Tests for config.json generation and validation."""

    def test_config_params_written(
        self, config: AutobioConfig, complex_pdb: Path, tmp_path: Path
    ) -> None:
        """All parameters are correctly written to config.json."""
        runner = _make_runner(config)
        workspace = Workspace.create(tmp_path / "ws")
        input_data = ProteinBindingAffinityInput(
            structure_path=complex_pdb,
            chain_selection="A B",
            temperature=37.0,
            extra={"distance_cutoff": 4.0, "contact_list": True},
        )
        runner.prepare_workspace(input_data, workspace)

        cfg = json.loads(workspace.config_path.read_text())
        assert cfg["selection"] == "A B"
        assert cfg["temperature"] == 37.0
        assert cfg["distance_cutoff"] == 4.0
        assert cfg["contact_list"] is True

    def test_nonexistent_structure_fails(self, config: AutobioConfig, tmp_path: Path) -> None:
        """Nonexistent input structure raises host-side validation error."""
        runner = _make_runner(config)
        workspace = Workspace.create(tmp_path / "ws")
        input_data = ProteinBindingAffinityInput(
            structure_path=tmp_path / "nonexistent.pdb",
        )
        with pytest.raises(AutobioError, match="does not exist"):
            runner.prepare_workspace(input_data, workspace)

    def test_invalid_temperature_fails(
        self, config: AutobioConfig, complex_pdb: Path, tmp_path: Path
    ) -> None:
        """Temperature below absolute zero raises host-side validation error."""
        runner = _make_runner(config)
        workspace = Workspace.create(tmp_path / "ws")
        input_data = ProteinBindingAffinityInput(
            structure_path=complex_pdb,
            temperature=-300.0,
        )
        with pytest.raises(AutobioError, match="absolute zero"):
            runner.prepare_workspace(input_data, workspace)
