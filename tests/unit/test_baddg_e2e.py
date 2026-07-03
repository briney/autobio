"""End-to-end tests for BA-ddG.

Each test exercises the full pipeline:
    input construction -> validation -> prepare_workspace ->
    (simulated CSV output) -> standardize.py -> parse_output -> verify

The only thing not tested is the actual BA-ddG model execution.
The standardize script is imported and run directly against realistic
CSV output data.
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
from autobio.schemas.scoring import BAddGInput, ScoringOutput
from autobio.tools.baddg import BAddGRunner

# ---------------------------------------------------------------------------
# Realistic BA-ddG output data
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

# Single-fold output CSV (n_folds=1)
_SINGLE_FOLD_CSV = "mutation,ddg,fold_1\nEA63Q,-0.542,-0.542\n"

# Multi-fold output CSV (n_folds=3)
_MULTI_FOLD_CSV = "mutation,ddg,fold_1,fold_2,fold_3\nEA63Q,-0.544,-0.542,-0.601,-0.489\n"

# Multiple mutations combined
_MULTI_MUTATION_CSV = 'mutation,ddg,fold_1\n"YH103H,QD30V",1.230,1.230\n'

# Multiple rows (batch-like output)
_MULTI_ROW_CSV = "mutation,ddg,fold_1\nEA63Q,-0.542,-0.542\nKA66A,0.123,0.123\n"


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


def _make_runner(config: AutobioConfig) -> BAddGRunner:
    """Create a runner with mocked container/GPU (we simulate container output)."""
    with (
        patch("autobio.tools.base.ContainerManager"),
        patch("autobio.tools.base.GPUManager"),
    ):
        runner = BAddGRunner("baddg", config)
    runner.current_mode = get_tool("baddg").modes["predict"]
    return runner


def _import_standardize():
    """Import the container's standardize module."""
    container_dir = str(Path(__file__).resolve().parent.parent.parent / "containers" / "baddg")
    if container_dir not in sys.path:
        sys.path.insert(0, container_dir)
    mod = importlib.import_module("standardize")
    importlib.reload(mod)  # ensure fresh import
    return mod


def _run_e2e(
    config: AutobioConfig,
    input_data: BAddGInput,
    raw_csv_content: str,
    tmp_path: Path,
) -> ScoringOutput:
    """Full end-to-end pipeline without Docker.

    1. prepare_workspace
    2. Write simulated raw CSV output
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
    assert cfg["mpnn_checkpoint_path"] == "/app/baddg/ckpt/soluble_model_weights/v_48_020.pt"
    assert cfg["ddg_checkpoint_path"] == "/app/baddg/ckpt/ddg_model.ckpt"

    # Step 2: write simulated raw CSV (what the container would produce)
    (workspace.raw_output_dir / "output.csv").write_text(raw_csv_content)

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
# TestBAddGSingleFoldE2E
# ---------------------------------------------------------------------------


class TestBAddGSingleFoldE2E:
    """End-to-end test for single-fold BA-ddG prediction."""

    def test_single_mutation_pipeline(
        self, config: AutobioConfig, complex_pdb: Path, tmp_path: Path
    ) -> None:
        """Single mutation, single fold — basic pipeline."""
        input_data = BAddGInput(
            structure_path=complex_pdb,
            mutations=["EA63Q"],
            chains="A_B",
            n_folds=1,
        )
        output = _run_e2e(config, input_data, _SINGLE_FOLD_CSV, tmp_path)

        assert len(output.scores) == 1
        s = output.scores[0]
        assert s.ddg == pytest.approx(-0.542)
        assert s.total_score == pytest.approx(-0.542)
        assert s.units == "kcal/mol"
        assert s.structure_path is None

    def test_mutations_in_output(
        self, config: AutobioConfig, complex_pdb: Path, tmp_path: Path
    ) -> None:
        """Mutations list is preserved in output."""
        input_data = BAddGInput(
            structure_path=complex_pdb,
            mutations=["EA63Q"],
            chains="A_B",
            n_folds=1,
        )
        output = _run_e2e(config, input_data, _SINGLE_FOLD_CSV, tmp_path)
        assert output.scores[0].mutations == ["EA63Q"]

    def test_chains_in_breakdown(
        self, config: AutobioConfig, complex_pdb: Path, tmp_path: Path
    ) -> None:
        """Chain specification is included in score breakdown."""
        input_data = BAddGInput(
            structure_path=complex_pdb,
            mutations=["EA63Q"],
            chains="A_B",
            n_folds=1,
        )
        output = _run_e2e(config, input_data, _SINGLE_FOLD_CSV, tmp_path)
        assert output.scores[0].score_breakdown is not None
        assert output.scores[0].score_breakdown["chains"] == "A_B"


# ---------------------------------------------------------------------------
# TestBAddGMultiFoldE2E
# ---------------------------------------------------------------------------


class TestBAddGMultiFoldE2E:
    """End-to-end test for multi-fold BA-ddG prediction."""

    def test_multi_fold_averages(
        self, config: AutobioConfig, complex_pdb: Path, tmp_path: Path
    ) -> None:
        """Multi-fold output averages ddG across folds."""
        input_data = BAddGInput(
            structure_path=complex_pdb,
            mutations=["EA63Q"],
            chains="A_B",
        )
        output = _run_e2e(config, input_data, _MULTI_FOLD_CSV, tmp_path)

        s = output.scores[0]
        # Mean of -0.542, -0.601, -0.489
        expected_mean = (-0.542 + -0.601 + -0.489) / 3
        assert s.ddg == pytest.approx(expected_mean)
        assert s.total_score == pytest.approx(expected_mean)

    def test_multi_fold_breakdown(
        self, config: AutobioConfig, complex_pdb: Path, tmp_path: Path
    ) -> None:
        """Multi-fold output includes fold values in breakdown."""
        input_data = BAddGInput(
            structure_path=complex_pdb,
            mutations=["EA63Q"],
            chains="A_B",
        )
        output = _run_e2e(config, input_data, _MULTI_FOLD_CSV, tmp_path)

        breakdown = output.scores[0].score_breakdown
        assert breakdown is not None
        assert breakdown["n_folds"] == 3
        assert "fold_values" in breakdown
        assert breakdown["fold_values"]["fold_1"] == pytest.approx(-0.542)


# ---------------------------------------------------------------------------
# TestBAddGMultiMutationE2E
# ---------------------------------------------------------------------------


class TestBAddGMultiMutationE2E:
    """End-to-end test for combined multi-mutation predictions."""

    def test_combined_mutations(
        self, config: AutobioConfig, complex_pdb: Path, tmp_path: Path
    ) -> None:
        """Multiple mutations predict combined ddG."""
        input_data = BAddGInput(
            structure_path=complex_pdb,
            mutations=["YH103H", "QD30V"],
            chains="ABC_DE",
        )
        output = _run_e2e(config, input_data, _MULTI_MUTATION_CSV, tmp_path)

        assert len(output.scores) == 1
        s = output.scores[0]
        assert s.ddg == pytest.approx(1.230)
        assert s.mutations == ["YH103H", "QD30V"]


# ---------------------------------------------------------------------------
# TestBAddGMultiRowE2E
# ---------------------------------------------------------------------------


class TestBAddGMultiRowE2E:
    """End-to-end test for outputs with multiple rows."""

    def test_multiple_rows_produce_multiple_scores(
        self, config: AutobioConfig, complex_pdb: Path, tmp_path: Path
    ) -> None:
        """Multiple CSV rows produce multiple ScoredStructure entries."""
        input_data = BAddGInput(
            structure_path=complex_pdb,
            mutations=["EA63Q", "KA66A"],
            chains="A_B",
        )
        output = _run_e2e(config, input_data, _MULTI_ROW_CSV, tmp_path)

        assert len(output.scores) == 2
        assert output.scores[0].ddg == pytest.approx(-0.542)
        assert output.scores[1].ddg == pytest.approx(0.123)


# ---------------------------------------------------------------------------
# TestBAddGConfigE2E
# ---------------------------------------------------------------------------


class TestBAddGConfigE2E:
    """Tests for config.json generation and validation."""

    def test_config_params_written(
        self, config: AutobioConfig, complex_pdb: Path, tmp_path: Path
    ) -> None:
        """All parameters are correctly written to config.json."""
        runner = _make_runner(config)
        workspace = Workspace.create(tmp_path / "ws")
        input_data = BAddGInput(
            structure_path=complex_pdb,
            mutations=["EA63Q"],
            chains="A_B",
            n_folds=2,
            seed=42,
            device="cpu",
        )
        runner.prepare_workspace(input_data, workspace)

        cfg = json.loads(workspace.config_path.read_text())
        assert cfg["n_folds"] == 2
        assert cfg["seed"] == 42
        assert cfg["device"] == "cpu"

    def test_nonexistent_structure_fails(self, config: AutobioConfig, tmp_path: Path) -> None:
        """Nonexistent input structure raises host-side validation error."""
        runner = _make_runner(config)
        workspace = Workspace.create(tmp_path / "ws")
        input_data = BAddGInput(
            structure_path=tmp_path / "nonexistent.pdb",
            mutations=["EA63Q"],
            chains="A_B",
        )
        with pytest.raises(AutobioError, match="does not exist"):
            runner.prepare_workspace(input_data, workspace)

    def test_missing_mutations_fails(
        self, config: AutobioConfig, complex_pdb: Path, tmp_path: Path
    ) -> None:
        """Empty mutations raises host-side validation error."""
        runner = _make_runner(config)
        workspace = Workspace.create(tmp_path / "ws")
        input_data = BAddGInput(
            structure_path=complex_pdb,
            mutations=[],
            chains="A_B",
        )
        with pytest.raises(AutobioError, match="requires at least one mutation"):
            runner.prepare_workspace(input_data, workspace)

    def test_invalid_chains_format_fails(
        self, config: AutobioConfig, complex_pdb: Path, tmp_path: Path
    ) -> None:
        """Invalid chains format raises host-side validation error."""
        runner = _make_runner(config)
        workspace = Workspace.create(tmp_path / "ws")
        input_data = BAddGInput(
            structure_path=complex_pdb,
            mutations=["EA63Q"],
            chains="AB",
        )
        with pytest.raises(AutobioError, match="exactly one underscore"):
            runner.prepare_workspace(input_data, workspace)
