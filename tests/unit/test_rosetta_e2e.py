"""End-to-end tests for the rosetta Tool (modes: score, relax, minimize, flexddg).

Each test exercises the full pipeline:
    input construction → validation → prepare_workspace →
    (simulated raw output) → standardize.py → parse_output → verify

The only thing not tested is the actual Rosetta binary execution.
The standardize scripts are imported and run directly against realistic
Rosetta output data.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

from autobio.core.catalog import get_tool
from autobio.core.config import AutobioConfig
from autobio.core.result import AutobioError
from autobio.core.workspace import Workspace
from autobio.schemas.scoring import (
    RosettaBaseInput,
    RosettaFlexDdgInput,
    RosettaRelaxInput,
    ScoringOutput,
)
from autobio.tools.rosetta import RosettaRunner

if TYPE_CHECKING:
    from autobio.schemas.base import BaseInput

# Import the shared score file parser
_ROSETTA_BASE_DIR = str(
    Path(__file__).resolve().parent.parent.parent / "containers" / "rosetta-base"
)
if _ROSETTA_BASE_DIR not in sys.path:
    sys.path.insert(0, _ROSETTA_BASE_DIR)

from parse_scorefile import parse_score_file  # noqa: E402

# ---------------------------------------------------------------------------
# Realistic Rosetta output data
# ---------------------------------------------------------------------------

# Minimal but valid PDB content for testing
_MINIMAL_PDB = (
    "HEADER    TEST STRUCTURE\n"
    "ATOM      1  N   ALA A   1       1.000   2.000   3.000  1.00 10.00           N\n"
    "ATOM      2  CA  ALA A   1       2.000   3.000   4.000  1.00 10.00           C\n"
    "ATOM      3  C   ALA A   1       3.000   4.000   5.000  1.00 10.00           C\n"
    "ATOM      4  O   ALA A   1       3.500   4.500   5.500  1.00 10.00           O\n"
    "ATOM      5  N   GLY A   2       4.000   5.000   6.000  1.00 12.00           N\n"
    "ATOM      6  CA  GLY A   2       5.000   6.000   7.000  1.00 12.00           C\n"
    "ATOM      7  C   GLY A   2       6.000   7.000   8.000  1.00 12.00           C\n"
    "ATOM      8  O   GLY A   2       6.500   7.500   8.500  1.00 12.00           O\n"
    "END\n"
)

# Realistic Rosetta score file — matches actual ref2015 energy terms
_SCORE_SC = (
    "SEQUENCE: \n"
    "SCORE:     total_score     fa_atr     fa_rep     fa_sol"
    "     fa_intra_rep     fa_elec     pro_close"
    "     hbond_sr_bb     hbond_lr_bb     hbond_bb_sc     hbond_sc"
    "     dslf_fa13     omega     fa_dun     p_aa_pp"
    "     yhh_planarity     ref     rama_prepro  description\n"
    "SCORE:       -42.310    -68.500     12.300     38.100"
    "     0.250     -11.200     0.100"
    "     -2.500     -1.800     -1.100     -0.600"
    "     0.000     0.400     30.200     -4.500"
    "     0.010     -8.300     -1.200  test_0001\n"
)

# Multi-structure score file (relax output)
_RELAX_SCORE_SC = (
    "SEQUENCE: \n"
    "SCORE:     total_score     fa_atr     fa_rep     fa_sol  description\n"
    "SCORE:       -48.200    -72.100     10.500     36.200  relaxed_0001\n"
    "SCORE:       -50.100    -73.500      9.800     35.100  relaxed_0002\n"
    "SCORE:       -47.600    -71.800     11.200     37.000  relaxed_0003\n"
)

# DDG monomer output
_DDG_PREDICTIONS_OUT = "ddG: mut_A42F 2.340 -42.310 -39.970 -42.310 -39.970\n"

# Flex-ddG score files — separate WT and mutant (produced by multi-step workflow)
_FLEXDDG_WT_SCORE_SC = (
    "SEQUENCE: \n"
    "SCORE:     total_score     fa_atr     fa_rep  description\n"
    "SCORE:       -42.310    -68.500     12.300  wt_backrub_input_0001\n"
    "SCORE:       -43.100    -69.200     11.800  wt_backrub_input_0002\n"
    "SCORE:       -41.900    -68.100     12.600  wt_backrub_input_0003\n"
)

_FLEXDDG_MUT_SCORE_SC = (
    "SEQUENCE: \n"
    "SCORE:     total_score     fa_atr     fa_rep  description\n"
    "SCORE:       -39.970    -66.200     13.100  mut_backrub_input_0001\n"
    "SCORE:       -40.800    -67.100     12.500  mut_backrub_input_0002\n"
    "SCORE:       -40.100    -66.800     13.000  mut_backrub_input_0003\n"
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def config() -> AutobioConfig:
    return AutobioConfig.resolve()


@pytest.fixture()
def sample_pdb(tmp_path: Path) -> Path:
    """Write a minimal but valid PDB file."""
    pdb_path = tmp_path / "structure.pdb"
    pdb_path.write_text(_MINIMAL_PDB)
    return pdb_path


@pytest.fixture()
def complex_pdb(tmp_path: Path) -> Path:
    """Write a two-chain PDB for DDG tests."""
    chain_b = _MINIMAL_PDB.replace(" A ", " B ")
    content = _MINIMAL_PDB.replace("END\n", "") + chain_b
    pdb_path = tmp_path / "complex.pdb"
    pdb_path.write_text(content)
    return pdb_path


def _make_runner(mode_name: str, config: AutobioConfig) -> RosettaRunner:
    """Create a runner with mocked container/GPU (we simulate container output)."""
    with (
        patch("autobio.tools.base.ContainerManager"),
        patch("autobio.tools.base.GPUManager"),
    ):
        runner = RosettaRunner("rosetta", config)
    runner.current_mode = get_tool("rosetta").modes[mode_name]
    return runner


def _import_standardize(tool_dir_name: str):
    """Import a container's standardize module."""
    container_dir = str(
        Path(__file__).resolve().parent.parent.parent / "containers" / tool_dir_name
    )
    if container_dir not in sys.path:
        sys.path.insert(0, container_dir)
    import importlib

    mod = importlib.import_module("standardize")
    importlib.reload(mod)  # ensure fresh import if multiple tools share name
    return mod


def _run_e2e(
    mode_name: str,
    config: AutobioConfig,
    input_data: BaseInput,
    raw_output_files: dict[str, str],
    tmp_path: Path,
) -> ScoringOutput:
    """Full end-to-end pipeline without Docker.

    1. prepare_workspace
    2. Write simulated raw output files
    3. Run the container's standardize.py
    4. parse_output
    """
    runner = _make_runner(mode_name, config)
    workspace = Workspace.create(tmp_path / "ws")

    # Step 1: prepare workspace (host-side validation + config writing)
    runner.prepare_workspace(input_data, workspace)

    # Verify config.json was written
    cfg = json.loads(workspace.config_path.read_text())
    assert cfg["binary"] is not None
    assert cfg["database_path"] == "/usr/local/lib/python3.8/dist-packages/pyrosetta/database"

    # Step 2: write simulated raw output (what the container would produce)
    for filename, content in raw_output_files.items():
        (workspace.raw_output_dir / filename).write_text(content)

    # Step 3: run the actual standardize.py script
    std_mod = _import_standardize(f"rosetta-{mode_name}")
    std_mod.standardize(workspace.root)

    # Verify result_data.json was produced
    result_data_path = workspace.std_output_dir / "result_data.json"
    assert result_data_path.exists(), "standardize.py did not produce result_data.json"

    # Step 4: parse output
    output = runner.parse_output(workspace)
    assert isinstance(output, ScoringOutput)
    return output


# ---------------------------------------------------------------------------
# TestRosettaScoreE2E
# ---------------------------------------------------------------------------


class TestRosettaScoreE2E:
    """End-to-end test for rosetta mode=score: full lifecycle."""

    def test_score_full_pipeline(
        self, config: AutobioConfig, sample_pdb: Path, tmp_path: Path
    ) -> None:
        """Score a structure and verify score breakdown."""
        input_data = RosettaBaseInput(structure_path=sample_pdb)
        output = _run_e2e(
            mode_name="score",
            config=config,
            input_data=input_data,
            raw_output_files={"score.sc": _SCORE_SC},
            tmp_path=tmp_path,
        )

        assert len(output.scores) == 1
        s = output.scores[0]
        assert s.total_score == pytest.approx(-42.310)
        assert s.units == "REU"
        assert s.structure_path is None
        assert s.ddg is None
        assert s.mutations is None

        # Verify score breakdown has expected Rosetta energy terms
        assert s.score_breakdown is not None
        assert "fa_atr" in s.score_breakdown
        assert "fa_rep" in s.score_breakdown
        assert "fa_sol" in s.score_breakdown
        assert "hbond_sr_bb" in s.score_breakdown
        assert s.score_breakdown["fa_atr"] == pytest.approx(-68.500)

    def test_score_custom_scorefunction(
        self, config: AutobioConfig, sample_pdb: Path, tmp_path: Path
    ) -> None:
        """Custom score function (typed field) is written to config.json."""
        runner = _make_runner("score", config)
        workspace = Workspace.create(tmp_path / "ws")
        input_data = RosettaBaseInput(structure_path=sample_pdb, score_function="beta_nov16")
        runner.prepare_workspace(input_data, workspace)

        cfg = json.loads(workspace.config_path.read_text())
        assert cfg["score_function"] == "beta_nov16"

    def test_score_with_extra_rotamers(
        self, config: AutobioConfig, sample_pdb: Path, tmp_path: Path
    ) -> None:
        """Extra rotamer flags pass through to config.json."""
        runner = _make_runner("score", config)
        workspace = Workspace.create(tmp_path / "ws")
        input_data = RosettaBaseInput(
            structure_path=sample_pdb,
            extra={"ex1": True, "ex2": True},
        )
        runner.prepare_workspace(input_data, workspace)

        cfg = json.loads(workspace.config_path.read_text())
        assert cfg["ex1"] is True
        assert cfg["ex2"] is True


# ---------------------------------------------------------------------------
# TestRosettaRelaxE2E
# ---------------------------------------------------------------------------


class TestRosettaRelaxE2E:
    """End-to-end test for rosetta mode=relax: refinement + scoring."""

    def test_relax_full_pipeline(
        self, config: AutobioConfig, sample_pdb: Path, tmp_path: Path
    ) -> None:
        """Relax a structure and verify refined PDBs + scores."""
        input_data = RosettaRelaxInput(structure_path=sample_pdb, nstruct=3)

        # Simulated output: 3 relaxed PDBs + score file
        raw_files = {
            "score.sc": _RELAX_SCORE_SC,
            "relaxed_0001.pdb": _MINIMAL_PDB,
            "relaxed_0002.pdb": _MINIMAL_PDB,
            "relaxed_0003.pdb": _MINIMAL_PDB,
        }

        output = _run_e2e(
            mode_name="relax",
            config=config,
            input_data=input_data,
            raw_output_files=raw_files,
            tmp_path=tmp_path,
        )

        assert len(output.scores) == 3

        # Best structure (lowest score) should be relaxed_0002
        scores_sorted = sorted(output.scores, key=lambda s: s.total_score)
        assert scores_sorted[0].total_score == pytest.approx(-50.100)

        # All structures should have output PDB paths
        for s in output.scores:
            assert s.structure_path is not None
            assert s.structure_path.exists()
            assert s.structure_path.suffix == ".pdb"
            content = s.structure_path.read_text()
            assert "ATOM" in content

        # Verify units and breakdown
        assert output.scores[0].units == "REU"
        assert output.scores[0].score_breakdown is not None

    def test_relax_nstruct_default(
        self, config: AutobioConfig, sample_pdb: Path, tmp_path: Path
    ) -> None:
        """Default nstruct for relax is 5."""
        runner = _make_runner("relax", config)
        workspace = Workspace.create(tmp_path / "ws")
        input_data = RosettaRelaxInput(structure_path=sample_pdb)
        runner.prepare_workspace(input_data, workspace)

        cfg = json.loads(workspace.config_path.read_text())
        assert cfg["nstruct"] == 5

    def test_relax_xml_path_in_config(
        self, config: AutobioConfig, sample_pdb: Path, tmp_path: Path
    ) -> None:
        """Relax config includes the XML protocol path."""
        runner = _make_runner("relax", config)
        workspace = Workspace.create(tmp_path / "ws")
        input_data = RosettaRelaxInput(structure_path=sample_pdb)
        runner.prepare_workspace(input_data, workspace)

        cfg = json.loads(workspace.config_path.read_text())
        assert cfg["xml_path"] == "/opt/tool/xml/relax.xml"
        assert cfg["binary"] == "rosetta_scripts"


# ---------------------------------------------------------------------------
# TestRosettaMinimizeE2E
# ---------------------------------------------------------------------------


class TestRosettaMinimizeE2E:
    """End-to-end test for rosetta mode=minimize: energy minimization."""

    def test_minimize_full_pipeline(
        self, config: AutobioConfig, sample_pdb: Path, tmp_path: Path
    ) -> None:
        """Minimize a structure and verify output."""
        input_data = RosettaBaseInput(structure_path=sample_pdb)

        # Single minimized structure
        minimize_sc = (
            "SEQUENCE: \n"
            "SCORE:     total_score     fa_atr     fa_rep     fa_sol  description\n"
            "SCORE:       -40.500    -67.200     13.100     38.500  minimized_0001\n"
        )
        raw_files = {
            "score.sc": minimize_sc,
            "minimized_0001.pdb": _MINIMAL_PDB,
        }

        output = _run_e2e(
            mode_name="minimize",
            config=config,
            input_data=input_data,
            raw_output_files=raw_files,
            tmp_path=tmp_path,
        )

        assert len(output.scores) == 1
        s = output.scores[0]
        assert s.total_score == pytest.approx(-40.500)
        assert s.structure_path is not None
        assert s.structure_path.exists()

    def test_minimize_nstruct_default(
        self, config: AutobioConfig, sample_pdb: Path, tmp_path: Path
    ) -> None:
        """Default nstruct for minimize is 1."""
        runner = _make_runner("minimize", config)
        workspace = Workspace.create(tmp_path / "ws")
        input_data = RosettaBaseInput(structure_path=sample_pdb)
        runner.prepare_workspace(input_data, workspace)

        cfg = json.loads(workspace.config_path.read_text())
        assert cfg["nstruct"] == 1


# ---------------------------------------------------------------------------
# TestRosettaFlexddGE2E
# ---------------------------------------------------------------------------


class TestRosettaFlexddGE2E:
    """End-to-end test for rosetta mode=flexddg: ensemble DDG prediction."""

    def test_flexddg_full_pipeline(
        self, config: AutobioConfig, complex_pdb: Path, tmp_path: Path
    ) -> None:
        """Run flex-ddG ensemble and verify DDG statistics."""
        input_data = RosettaFlexDdgInput(
            structure_path=complex_pdb,
            mutations=["A42F"],
            chains_to_move="B",
            nstruct=3,
            extra={"backrub_trials": 1000},
        )

        output = _run_e2e(
            mode_name="flexddg",
            config=config,
            input_data=input_data,
            raw_output_files={
                "wt_score.sc": _FLEXDDG_WT_SCORE_SC,
                "mut_score.sc": _FLEXDDG_MUT_SCORE_SC,
            },
            tmp_path=tmp_path,
        )

        # First entry should be the ensemble summary
        assert len(output.scores) >= 1
        summary = output.scores[0]
        assert summary.ddg is not None
        assert summary.mutations == ["A42F"]

        # DDG should be positive (destabilizing mutation in our test data)
        # WT scores: -42.31, -43.10, -41.90 → mean -42.43
        # Mut scores: -39.97, -40.80, -40.10 → mean -40.29
        # DDG per sample: 2.34, 2.30, 1.80 → mean ~2.15
        assert summary.ddg > 0

        # Ensemble stats should be in breakdown
        assert summary.score_breakdown is not None
        assert "ddg_mean" in summary.score_breakdown
        assert "ddg_std" in summary.score_breakdown
        assert "n_samples" in summary.score_breakdown
        assert summary.score_breakdown["n_samples"] == 3
        assert summary.score_breakdown["ddg_std"] >= 0

        # Per-sample DDG values should be present
        assert "ddg_values" in summary.score_breakdown
        assert len(summary.score_breakdown["ddg_values"]) == 3

        # Individual sample entries should follow the summary
        assert len(output.scores) == 4  # 1 summary + 3 samples
        for sample in output.scores[1:]:
            assert sample.ddg is not None
            assert sample.mutations == ["A42F"]

    def test_flexddg_config_params(
        self, config: AutobioConfig, complex_pdb: Path, tmp_path: Path
    ) -> None:
        """Flex-ddG parameters are correctly written to config.json."""
        runner = _make_runner("flexddg", config)
        workspace = Workspace.create(tmp_path / "ws")
        input_data = RosettaFlexDdgInput(
            structure_path=complex_pdb,
            mutations=["A42F"],
            chains_to_move="B",
            nstruct=10,
            extra={"backrub_trials": 5000, "max_minimization_iter": 2000},
        )
        runner.prepare_workspace(input_data, workspace)

        cfg = json.loads(workspace.config_path.read_text())
        assert cfg["chains_to_move"] == "B"
        assert cfg["mutations"] == ["A42F"]
        assert cfg["nstruct"] == 10
        assert cfg["backrub_trials"] == 5000
        assert cfg["max_minimization_iter"] == 2000
        assert "xml_path" not in cfg

    def test_flexddg_missing_chains_fails(
        self, config: AutobioConfig, complex_pdb: Path, tmp_path: Path
    ) -> None:
        """Empty chains_to_move raises host-side validation error."""
        runner = _make_runner("flexddg", config)
        workspace = Workspace.create(tmp_path / "ws")
        input_data = RosettaFlexDdgInput(
            structure_path=complex_pdb, mutations=["A42F"], chains_to_move=""
        )
        with pytest.raises(AutobioError, match="chains_to_move"):
            runner.prepare_workspace(input_data, workspace)

    def test_flexddg_resfile_passthrough(
        self, config: AutobioConfig, complex_pdb: Path, tmp_path: Path
    ) -> None:
        """Custom resfile content is written to inputs/."""
        runner = _make_runner("flexddg", config)
        workspace = Workspace.create(tmp_path / "ws")
        resfile = "NATAA\nstart\n42 A PIKAA F\n"
        input_data = RosettaFlexDdgInput(
            structure_path=complex_pdb,
            mutations=["A42F"],
            chains_to_move="B",
            resfile=resfile,
        )
        runner.prepare_workspace(input_data, workspace)

        resfile_path = workspace.inputs_dir / "mutations.resfile"
        assert resfile_path.exists()
        assert resfile_path.read_text() == resfile

    def test_flexddg_nstruct_default(
        self, config: AutobioConfig, complex_pdb: Path, tmp_path: Path
    ) -> None:
        """Default nstruct for flexddg is 35."""
        runner = _make_runner("flexddg", config)
        workspace = Workspace.create(tmp_path / "ws")
        input_data = RosettaFlexDdgInput(
            structure_path=complex_pdb, mutations=["A42F"], chains_to_move="B"
        )
        runner.prepare_workspace(input_data, workspace)

        cfg = json.loads(workspace.config_path.read_text())
        assert cfg["nstruct"] == 35


# ---------------------------------------------------------------------------
# TestInputValidation — cross-cutting validation tests
# ---------------------------------------------------------------------------


class TestInputValidation:
    """Validation tests that apply across all modes."""

    def test_nonexistent_structure_fails(self, config: AutobioConfig, tmp_path: Path) -> None:
        """All modes reject nonexistent input structures."""
        for mode_name in ("score", "relax", "minimize"):
            runner = _make_runner(mode_name, config)
            workspace = Workspace.create(tmp_path / f"ws_{mode_name}")
            input_data = RosettaBaseInput(structure_path=tmp_path / "nonexistent.pdb")
            with pytest.raises(AutobioError, match="does not exist"):
                runner.prepare_workspace(input_data, workspace)

    def test_ddg_empty_mutations_fails(
        self, config: AutobioConfig, sample_pdb: Path, tmp_path: Path
    ) -> None:
        """Flex-ddG rejects an empty mutations list."""
        runner = _make_runner("flexddg", config)
        workspace = Workspace.create(tmp_path / "ws_flexddg")
        input_data = RosettaFlexDdgInput(
            structure_path=sample_pdb, mutations=[], chains_to_move="B"
        )
        with pytest.raises(AutobioError, match="requires at least one mutation"):
            runner.prepare_workspace(input_data, workspace)

    def test_structure_copied_correctly(
        self, config: AutobioConfig, sample_pdb: Path, tmp_path: Path
    ) -> None:
        """Input structure is copied to workspace inputs/ for all modes."""
        for mode_name in ("score", "relax", "minimize"):
            runner = _make_runner(mode_name, config)
            workspace = Workspace.create(tmp_path / f"ws_{mode_name}")
            input_data = RosettaBaseInput(structure_path=sample_pdb)
            runner.prepare_workspace(input_data, workspace)

            copied = workspace.inputs_dir / sample_pdb.name
            assert copied.exists()
            assert copied.read_text() == sample_pdb.read_text()


# ---------------------------------------------------------------------------
# TestScoreFileParser — verify parser handles edge cases
# ---------------------------------------------------------------------------


class TestScoreFileParserEdgeCases:
    """Edge cases for the shared Rosetta score file parser."""

    def test_empty_file(self, tmp_path: Path) -> None:
        sc = tmp_path / "empty.sc"
        sc.write_text("")
        assert parse_score_file(sc) == []

    def test_header_only(self, tmp_path: Path) -> None:
        sc = tmp_path / "header.sc"
        sc.write_text("SEQUENCE: \nSCORE:     total_score     fa_atr  description\n")
        assert parse_score_file(sc) == []

    def test_non_numeric_values(self, tmp_path: Path) -> None:
        """Non-numeric score values are stored as strings."""
        sc = tmp_path / "mixed.sc"
        sc.write_text(
            "SEQUENCE: \n"
            "SCORE:     total_score     status  description\n"
            "SCORE:       -42.310     OK       test_0001\n"
        )
        results = parse_score_file(sc)
        assert len(results) == 1
        assert results[0]["total_score"] == pytest.approx(-42.310)
        assert results[0]["status"] == "OK"

    def test_many_structures(self, tmp_path: Path) -> None:
        """Parser handles large score files correctly."""
        lines = [
            "SEQUENCE: ",
            "SCORE:     total_score     fa_atr  description",
        ]
        for i in range(100):
            score = -40.0 - i * 0.1
            lines.append(f"SCORE:       {score:.3f}    -65.000  struct_{i:04d}")
        sc = tmp_path / "many.sc"
        sc.write_text("\n".join(lines) + "\n")

        results = parse_score_file(sc)
        assert len(results) == 100
        assert results[0]["total_score"] == pytest.approx(-40.0)
        assert results[99]["total_score"] == pytest.approx(-49.9)
