"""Tests for FreeSASARunner — prepare_workspace, parse_output, and registration."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

from autobio.core.config import AutobioConfig
from autobio.core.registry import TOOL_REGISTRY, ToolCategory
from autobio.core.result import AutobioError
from autobio.core.workspace import Workspace
from autobio.schemas.scoring import ScoringInput, ScoringOutput
from autobio.tools import TOOL_RUNNERS, get_runner
from autobio.tools.freesasa import FreeSASARunner

if TYPE_CHECKING:
    from pathlib import Path


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def config() -> AutobioConfig:
    return AutobioConfig.resolve()


@pytest.fixture()
def bsa_runner(config: AutobioConfig) -> FreeSASARunner:
    """Create a FreeSASARunner for BSA with mocked deps."""
    with (
        patch("autobio.tools.base.ContainerManager"),
        patch("autobio.tools.base.GPUManager"),
    ):
        return FreeSASARunner("freesasa_bsa", config)


@pytest.fixture()
def sasa_runner(config: AutobioConfig) -> FreeSASARunner:
    """Create a FreeSASARunner for SASA with mocked deps."""
    with (
        patch("autobio.tools.base.ContainerManager"),
        patch("autobio.tools.base.GPUManager"),
    ):
        return FreeSASARunner("freesasa_sasa", config)


@pytest.fixture()
def sample_pdb(tmp_path: Path) -> Path:
    """Write a minimal two-chain PDB file for testing."""
    content = (
        "ATOM      1  N   ALA A   1       1.000   2.000   3.000  1.00 10.00           N\n"
        "ATOM      2  CA  ALA A   1       2.000   3.000   4.000  1.00 10.00           C\n"
        "ATOM      3  N   GLY B   1       4.000   5.000   6.000  1.00 12.00           N\n"
        "ATOM      4  CA  GLY B   1       5.000   6.000   7.000  1.00 12.00           C\n"
        "END\n"
    )
    pdb_path = tmp_path / "complex.pdb"
    pdb_path.write_text(content)
    return pdb_path


# ---------------------------------------------------------------------------
# TestFreeSASABSAPrepareWorkspace
# ---------------------------------------------------------------------------


class TestFreeSASABSAPrepareWorkspace:
    """Tests for FreeSASARunner.prepare_workspace in BSA mode."""

    def test_basic_config(
        self, bsa_runner: FreeSASARunner, tmp_path: Path, sample_pdb: Path
    ) -> None:
        """Config contains correct fields for BSA."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = ScoringInput(
            structure_path=sample_pdb,
            extra={"partner1": "A", "partner2": "B"},
        )
        bsa_runner.prepare_workspace(input_data, workspace)

        cfg = json.loads(workspace.config_path.read_text())
        assert cfg["mode"] == "bsa"
        assert cfg["structure_path"] == "/workspace/inputs/complex.pdb"
        assert cfg["partner1"] == "A"
        assert cfg["partner2"] == "B"
        assert cfg["output_dir"] == "/workspace/outputs/raw"

    def test_structure_file_copied(
        self, bsa_runner: FreeSASARunner, tmp_path: Path, sample_pdb: Path
    ) -> None:
        """Input structure is copied to inputs/ directory."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = ScoringInput(
            structure_path=sample_pdb,
            extra={"partner1": "A", "partner2": "B"},
        )
        bsa_runner.prepare_workspace(input_data, workspace)

        copied = workspace.inputs_dir / "complex.pdb"
        assert copied.exists()
        assert copied.read_text() == sample_pdb.read_text()

    def test_default_algorithm(
        self, bsa_runner: FreeSASARunner, tmp_path: Path, sample_pdb: Path
    ) -> None:
        """Default algorithm is LeeRichards."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = ScoringInput(
            structure_path=sample_pdb,
            extra={"partner1": "A", "partner2": "B"},
        )
        bsa_runner.prepare_workspace(input_data, workspace)

        cfg = json.loads(workspace.config_path.read_text())
        assert cfg["algorithm"] == "LeeRichards"

    def test_custom_algorithm(
        self, bsa_runner: FreeSASARunner, tmp_path: Path, sample_pdb: Path
    ) -> None:
        """Custom algorithm from extra dict."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = ScoringInput(
            structure_path=sample_pdb,
            extra={"partner1": "A", "partner2": "B", "algorithm": "ShrakeRupley"},
        )
        bsa_runner.prepare_workspace(input_data, workspace)

        cfg = json.loads(workspace.config_path.read_text())
        assert cfg["algorithm"] == "ShrakeRupley"

    def test_default_probe_radius(
        self, bsa_runner: FreeSASARunner, tmp_path: Path, sample_pdb: Path
    ) -> None:
        """Default probe_radius is 1.4."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = ScoringInput(
            structure_path=sample_pdb,
            extra={"partner1": "A", "partner2": "B"},
        )
        bsa_runner.prepare_workspace(input_data, workspace)

        cfg = json.loads(workspace.config_path.read_text())
        assert cfg["probe_radius"] == pytest.approx(1.4)

    def test_custom_probe_radius(
        self, bsa_runner: FreeSASARunner, tmp_path: Path, sample_pdb: Path
    ) -> None:
        """Custom probe_radius from extra dict."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = ScoringInput(
            structure_path=sample_pdb,
            extra={"partner1": "A", "partner2": "B", "probe_radius": 1.8},
        )
        bsa_runner.prepare_workspace(input_data, workspace)

        cfg = json.loads(workspace.config_path.read_text())
        assert cfg["probe_radius"] == pytest.approx(1.8)

    def test_default_per_residue_false(
        self, bsa_runner: FreeSASARunner, tmp_path: Path, sample_pdb: Path
    ) -> None:
        """Default per_residue is False."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = ScoringInput(
            structure_path=sample_pdb,
            extra={"partner1": "A", "partner2": "B"},
        )
        bsa_runner.prepare_workspace(input_data, workspace)

        cfg = json.loads(workspace.config_path.read_text())
        assert cfg["per_residue"] is False

    def test_per_residue_true(
        self, bsa_runner: FreeSASARunner, tmp_path: Path, sample_pdb: Path
    ) -> None:
        """per_residue=True from extra dict."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = ScoringInput(
            structure_path=sample_pdb,
            extra={"partner1": "A", "partner2": "B", "per_residue": True},
        )
        bsa_runner.prepare_workspace(input_data, workspace)

        cfg = json.loads(workspace.config_path.read_text())
        assert cfg["per_residue"] is True

    def test_multi_chain_partners(
        self, bsa_runner: FreeSASARunner, tmp_path: Path, sample_pdb: Path
    ) -> None:
        """Multi-chain partner specs are written correctly."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = ScoringInput(
            structure_path=sample_pdb,
            extra={"partner1": "A,B", "partner2": "C,D"},
        )
        bsa_runner.prepare_workspace(input_data, workspace)

        cfg = json.loads(workspace.config_path.read_text())
        assert cfg["partner1"] == "A,B"
        assert cfg["partner2"] == "C,D"

    def test_extra_dict_merged(
        self, bsa_runner: FreeSASARunner, tmp_path: Path, sample_pdb: Path
    ) -> None:
        """Non-consumed extra dict keys appear at top level of config.json."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = ScoringInput(
            structure_path=sample_pdb,
            extra={"partner1": "A", "partner2": "B", "custom_flag": "value"},
        )
        bsa_runner.prepare_workspace(input_data, workspace)

        cfg = json.loads(workspace.config_path.read_text())
        assert cfg["custom_flag"] == "value"

    def test_consumed_keys_not_leaked(
        self, bsa_runner: FreeSASARunner, tmp_path: Path, sample_pdb: Path
    ) -> None:
        """Consumed keys are placed explicitly, not duplicated at top level."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = ScoringInput(
            structure_path=sample_pdb,
            extra={
                "partner1": "A",
                "partner2": "B",
                "algorithm": "ShrakeRupley",
                "probe_radius": 1.8,
                "per_residue": True,
                "custom_flag": "value",
            },
        )
        bsa_runner.prepare_workspace(input_data, workspace)

        cfg = json.loads(workspace.config_path.read_text())
        assert cfg["partner1"] == "A"
        assert cfg["partner2"] == "B"
        assert cfg["algorithm"] == "ShrakeRupley"
        assert cfg["probe_radius"] == pytest.approx(1.8)
        assert cfg["per_residue"] is True
        assert cfg["custom_flag"] == "value"


# ---------------------------------------------------------------------------
# TestFreeSASASASAPrepareWorkspace
# ---------------------------------------------------------------------------


class TestFreeSASASASAPrepareWorkspace:
    """Tests for FreeSASARunner.prepare_workspace in SASA mode."""

    def test_basic_config(
        self, sasa_runner: FreeSASARunner, tmp_path: Path, sample_pdb: Path
    ) -> None:
        """Config contains correct fields for SASA."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = ScoringInput(structure_path=sample_pdb)
        sasa_runner.prepare_workspace(input_data, workspace)

        cfg = json.loads(workspace.config_path.read_text())
        assert cfg["mode"] == "sasa"
        assert cfg["structure_path"] == "/workspace/inputs/complex.pdb"
        assert cfg["output_dir"] == "/workspace/outputs/raw"
        assert "partner1" not in cfg
        assert "partner2" not in cfg

    def test_default_params(
        self, sasa_runner: FreeSASARunner, tmp_path: Path, sample_pdb: Path
    ) -> None:
        """Default algorithm, probe_radius, per_residue for SASA."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = ScoringInput(structure_path=sample_pdb)
        sasa_runner.prepare_workspace(input_data, workspace)

        cfg = json.loads(workspace.config_path.read_text())
        assert cfg["algorithm"] == "LeeRichards"
        assert cfg["probe_radius"] == pytest.approx(1.4)
        assert cfg["per_residue"] is False

    def test_custom_params(
        self, sasa_runner: FreeSASARunner, tmp_path: Path, sample_pdb: Path
    ) -> None:
        """Custom algorithm and probe_radius from extra dict."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = ScoringInput(
            structure_path=sample_pdb,
            extra={"algorithm": "ShrakeRupley", "probe_radius": 2.0},
        )
        sasa_runner.prepare_workspace(input_data, workspace)

        cfg = json.loads(workspace.config_path.read_text())
        assert cfg["algorithm"] == "ShrakeRupley"
        assert cfg["probe_radius"] == pytest.approx(2.0)

    def test_structure_file_copied(
        self, sasa_runner: FreeSASARunner, tmp_path: Path, sample_pdb: Path
    ) -> None:
        """Input structure is copied to inputs/ directory."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = ScoringInput(structure_path=sample_pdb)
        sasa_runner.prepare_workspace(input_data, workspace)

        copied = workspace.inputs_dir / "complex.pdb"
        assert copied.exists()


# ---------------------------------------------------------------------------
# TestFreeSASAValidation
# ---------------------------------------------------------------------------


class TestFreeSASAValidation:
    """Tests for host-side input validation."""

    def test_missing_structure_raises(self, bsa_runner: FreeSASARunner, tmp_path: Path) -> None:
        """Missing structure file raises AutobioError."""
        workspace = Workspace.create(tmp_path / "ws")
        fake_path = tmp_path / "nonexistent.pdb"
        input_data = ScoringInput(
            structure_path=fake_path,
            extra={"partner1": "A", "partner2": "B"},
        )
        with pytest.raises(AutobioError, match="does not exist"):
            bsa_runner.prepare_workspace(input_data, workspace)

    def test_non_pdb_suffix_raises(self, bsa_runner: FreeSASARunner, tmp_path: Path) -> None:
        """Non-PDB suffix raises AutobioError."""
        cif_path = tmp_path / "structure.cif"
        cif_path.write_text("dummy")
        workspace = Workspace.create(tmp_path / "ws")
        input_data = ScoringInput(
            structure_path=cif_path,
            extra={"partner1": "A", "partner2": "B"},
        )
        with pytest.raises(AutobioError, match="PDB format"):
            bsa_runner.prepare_workspace(input_data, workspace)

    def test_bsa_missing_partner1_raises(
        self, bsa_runner: FreeSASARunner, tmp_path: Path, sample_pdb: Path
    ) -> None:
        """BSA without partner1 raises AutobioError."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = ScoringInput(
            structure_path=sample_pdb,
            extra={"partner2": "B"},
        )
        with pytest.raises(AutobioError, match="partner1"):
            bsa_runner.prepare_workspace(input_data, workspace)

    def test_bsa_missing_partner2_raises(
        self, bsa_runner: FreeSASARunner, tmp_path: Path, sample_pdb: Path
    ) -> None:
        """BSA without partner2 raises AutobioError."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = ScoringInput(
            structure_path=sample_pdb,
            extra={"partner1": "A"},
        )
        with pytest.raises(AutobioError, match="partner2"):
            bsa_runner.prepare_workspace(input_data, workspace)

    def test_overlapping_chains_raises(
        self, bsa_runner: FreeSASARunner, tmp_path: Path, sample_pdb: Path
    ) -> None:
        """Overlapping partner chains raise AutobioError."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = ScoringInput(
            structure_path=sample_pdb,
            extra={"partner1": "A,B", "partner2": "B,C"},
        )
        with pytest.raises(AutobioError, match="overlap"):
            bsa_runner.prepare_workspace(input_data, workspace)

    def test_empty_partner_string_raises(
        self, bsa_runner: FreeSASARunner, tmp_path: Path, sample_pdb: Path
    ) -> None:
        """Empty partner string raises AutobioError."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = ScoringInput(
            structure_path=sample_pdb,
            extra={"partner1": "", "partner2": "B"},
        )
        with pytest.raises(AutobioError, match="partner1"):
            bsa_runner.prepare_workspace(input_data, workspace)

    def test_invalid_algorithm_raises(
        self, bsa_runner: FreeSASARunner, tmp_path: Path, sample_pdb: Path
    ) -> None:
        """Invalid algorithm raises AutobioError."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = ScoringInput(
            structure_path=sample_pdb,
            extra={"partner1": "A", "partner2": "B", "algorithm": "FooBar"},
        )
        with pytest.raises(AutobioError, match="Invalid algorithm"):
            bsa_runner.prepare_workspace(input_data, workspace)

    def test_non_positive_probe_radius_raises(
        self, bsa_runner: FreeSASARunner, tmp_path: Path, sample_pdb: Path
    ) -> None:
        """Non-positive probe_radius raises AutobioError."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = ScoringInput(
            structure_path=sample_pdb,
            extra={"partner1": "A", "partner2": "B", "probe_radius": -1.0},
        )
        with pytest.raises(AutobioError, match="probe_radius"):
            bsa_runner.prepare_workspace(input_data, workspace)

    def test_zero_probe_radius_raises(
        self, bsa_runner: FreeSASARunner, tmp_path: Path, sample_pdb: Path
    ) -> None:
        """Zero probe_radius raises AutobioError."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = ScoringInput(
            structure_path=sample_pdb,
            extra={"partner1": "A", "partner2": "B", "probe_radius": 0},
        )
        with pytest.raises(AutobioError, match="probe_radius"):
            bsa_runner.prepare_workspace(input_data, workspace)

    def test_sasa_missing_structure_raises(
        self, sasa_runner: FreeSASARunner, tmp_path: Path
    ) -> None:
        """SASA mode also validates structure exists."""
        workspace = Workspace.create(tmp_path / "ws")
        fake_path = tmp_path / "nonexistent.pdb"
        input_data = ScoringInput(structure_path=fake_path)
        with pytest.raises(AutobioError, match="does not exist"):
            sasa_runner.prepare_workspace(input_data, workspace)

    def test_sasa_invalid_algorithm_raises(
        self, sasa_runner: FreeSASARunner, tmp_path: Path, sample_pdb: Path
    ) -> None:
        """SASA mode also validates algorithm."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = ScoringInput(
            structure_path=sample_pdb,
            extra={"algorithm": "BadAlgo"},
        )
        with pytest.raises(AutobioError, match="Invalid algorithm"):
            sasa_runner.prepare_workspace(input_data, workspace)


# ---------------------------------------------------------------------------
# TestFreeSASAParseOutput
# ---------------------------------------------------------------------------

_BSA_RESULT = {
    "scores": [
        {
            "total_score": 1850.3,
            "units": "angstrom^2",
            "score_breakdown": {
                "polar_bsa": 620.1,
                "apolar_bsa": 1230.2,
                "complex_sasa": 15200.5,
                "partner1_sasa": 9800.2,
                "partner2_sasa": 7250.6,
                "partner1_chains": "A,B",
                "partner2_chains": "C",
                "algorithm": "LeeRichards",
                "probe_radius": 1.4,
                "per_chain_sasa": {"A": 5200.1, "B": 4600.1, "C": 7250.6},
            },
            "per_residue_scores": None,
        }
    ]
}

_SASA_RESULT = {
    "scores": [
        {
            "total_score": 15200.5,
            "units": "angstrom^2",
            "score_breakdown": {
                "polar_sasa": 6100.2,
                "apolar_sasa": 9100.3,
                "algorithm": "LeeRichards",
                "probe_radius": 1.4,
                "per_chain_sasa": {"A": 5200.1, "B": 4600.1, "C": 5400.3},
            },
            "per_residue_scores": None,
        }
    ]
}

_BSA_RESULT_WITH_RESIDUES = {
    "scores": [
        {
            "total_score": 1850.3,
            "units": "angstrom^2",
            "score_breakdown": {
                "polar_bsa": 620.1,
                "apolar_bsa": 1230.2,
                "complex_sasa": 15200.5,
                "partner1_sasa": 9800.2,
                "partner2_sasa": 7250.6,
                "partner1_chains": "A",
                "partner2_chains": "B",
                "algorithm": "LeeRichards",
                "probe_radius": 1.4,
                "per_chain_sasa": {"A": 9800.2, "B": 7250.6},
            },
            "per_residue_scores": [15.2, 22.1, 0.0, 8.5],
        }
    ]
}


class TestFreeSASAParseOutput:
    """Tests for FreeSASARunner.parse_output."""

    def test_parse_bsa_output(self, bsa_runner: FreeSASARunner, tmp_path: Path) -> None:
        """Standard BSA result_data.json is deserialized correctly."""
        workspace = Workspace.create(tmp_path / "ws")
        (workspace.std_output_dir / "result_data.json").write_text(json.dumps(_BSA_RESULT))

        output = bsa_runner.parse_output(workspace)
        assert isinstance(output, ScoringOutput)
        assert len(output.scores) == 1

    def test_bsa_total_score(self, bsa_runner: FreeSASARunner, tmp_path: Path) -> None:
        """BSA total_score is correct."""
        workspace = Workspace.create(tmp_path / "ws")
        (workspace.std_output_dir / "result_data.json").write_text(json.dumps(_BSA_RESULT))

        output = bsa_runner.parse_output(workspace)
        assert output.scores[0].total_score == pytest.approx(1850.3)

    def test_bsa_units(self, bsa_runner: FreeSASARunner, tmp_path: Path) -> None:
        """BSA units are angstrom^2."""
        workspace = Workspace.create(tmp_path / "ws")
        (workspace.std_output_dir / "result_data.json").write_text(json.dumps(_BSA_RESULT))

        output = bsa_runner.parse_output(workspace)
        assert output.scores[0].units == "angstrom^2"

    def test_bsa_score_breakdown(self, bsa_runner: FreeSASARunner, tmp_path: Path) -> None:
        """BSA score_breakdown contains expected keys."""
        workspace = Workspace.create(tmp_path / "ws")
        (workspace.std_output_dir / "result_data.json").write_text(json.dumps(_BSA_RESULT))

        output = bsa_runner.parse_output(workspace)
        breakdown = output.scores[0].score_breakdown
        assert breakdown is not None
        assert breakdown["polar_bsa"] == pytest.approx(620.1)
        assert breakdown["apolar_bsa"] == pytest.approx(1230.2)
        assert breakdown["complex_sasa"] == pytest.approx(15200.5)
        assert breakdown["partner1_sasa"] == pytest.approx(9800.2)
        assert breakdown["partner2_sasa"] == pytest.approx(7250.6)
        assert breakdown["partner1_chains"] == "A,B"
        assert breakdown["partner2_chains"] == "C"

    def test_bsa_no_per_residue_by_default(
        self, bsa_runner: FreeSASARunner, tmp_path: Path
    ) -> None:
        """per_residue_scores is None when not requested."""
        workspace = Workspace.create(tmp_path / "ws")
        (workspace.std_output_dir / "result_data.json").write_text(json.dumps(_BSA_RESULT))

        output = bsa_runner.parse_output(workspace)
        assert output.scores[0].per_residue_scores is None

    def test_bsa_with_per_residue(self, bsa_runner: FreeSASARunner, tmp_path: Path) -> None:
        """per_residue_scores present when populated."""
        workspace = Workspace.create(tmp_path / "ws")
        (workspace.std_output_dir / "result_data.json").write_text(
            json.dumps(_BSA_RESULT_WITH_RESIDUES)
        )

        output = bsa_runner.parse_output(workspace)
        assert output.scores[0].per_residue_scores is not None
        assert len(output.scores[0].per_residue_scores) == 4
        assert output.scores[0].per_residue_scores[0] == pytest.approx(15.2)

    def test_parse_sasa_output(self, sasa_runner: FreeSASARunner, tmp_path: Path) -> None:
        """Standard SASA result_data.json is deserialized correctly."""
        workspace = Workspace.create(tmp_path / "ws")
        (workspace.std_output_dir / "result_data.json").write_text(json.dumps(_SASA_RESULT))

        output = sasa_runner.parse_output(workspace)
        assert isinstance(output, ScoringOutput)
        assert len(output.scores) == 1

    def test_sasa_total_score(self, sasa_runner: FreeSASARunner, tmp_path: Path) -> None:
        """SASA total_score is correct."""
        workspace = Workspace.create(tmp_path / "ws")
        (workspace.std_output_dir / "result_data.json").write_text(json.dumps(_SASA_RESULT))

        output = sasa_runner.parse_output(workspace)
        assert output.scores[0].total_score == pytest.approx(15200.5)

    def test_sasa_score_breakdown(self, sasa_runner: FreeSASARunner, tmp_path: Path) -> None:
        """SASA score_breakdown contains expected keys."""
        workspace = Workspace.create(tmp_path / "ws")
        (workspace.std_output_dir / "result_data.json").write_text(json.dumps(_SASA_RESULT))

        output = sasa_runner.parse_output(workspace)
        breakdown = output.scores[0].score_breakdown
        assert breakdown is not None
        assert breakdown["polar_sasa"] == pytest.approx(6100.2)
        assert breakdown["apolar_sasa"] == pytest.approx(9100.3)
        assert "per_chain_sasa" in breakdown

    def test_output_type(self, bsa_runner: FreeSASARunner, tmp_path: Path) -> None:
        """Returns ScoringOutput."""
        workspace = Workspace.create(tmp_path / "ws")
        (workspace.std_output_dir / "result_data.json").write_text(json.dumps(_BSA_RESULT))
        output = bsa_runner.parse_output(workspace)
        assert isinstance(output, ScoringOutput)

    def test_raw_output_path(self, bsa_runner: FreeSASARunner, tmp_path: Path) -> None:
        """raw_output_path points to the raw output directory."""
        workspace = Workspace.create(tmp_path / "ws")
        (workspace.std_output_dir / "result_data.json").write_text(json.dumps(_BSA_RESULT))
        output = bsa_runner.parse_output(workspace)
        assert output.raw_output_path == workspace.raw_output_dir


# ---------------------------------------------------------------------------
# TestFreeSASARegistration
# ---------------------------------------------------------------------------


class TestFreeSASARegistration:
    """Tests for tool and runner registration."""

    def test_bsa_in_tool_registry(self) -> None:
        assert "freesasa_bsa" in TOOL_REGISTRY

    def test_sasa_in_tool_registry(self) -> None:
        assert "freesasa_sasa" in TOOL_REGISTRY

    def test_bsa_in_tool_runners(self) -> None:
        assert "freesasa_bsa" in TOOL_RUNNERS
        assert TOOL_RUNNERS["freesasa_bsa"] is FreeSASARunner

    def test_sasa_in_tool_runners(self) -> None:
        assert "freesasa_sasa" in TOOL_RUNNERS
        assert TOOL_RUNNERS["freesasa_sasa"] is FreeSASARunner

    def test_bsa_scoring_category(self) -> None:
        assert TOOL_REGISTRY["freesasa_bsa"].category == ToolCategory.SCORING

    def test_sasa_scoring_category(self) -> None:
        assert TOOL_REGISTRY["freesasa_sasa"].category == ToolCategory.SCORING

    def test_bsa_no_gpu_required(self) -> None:
        entry = TOOL_REGISTRY["freesasa_bsa"]
        assert entry.requires_gpu is False
        assert entry.gpu_count == 0

    def test_sasa_no_gpu_required(self) -> None:
        entry = TOOL_REGISTRY["freesasa_sasa"]
        assert entry.requires_gpu is False
        assert entry.gpu_count == 0

    def test_bsa_schema_types(self) -> None:
        entry = TOOL_REGISTRY["freesasa_bsa"]
        assert entry.input_schema is ScoringInput
        assert entry.output_schema is ScoringOutput

    def test_sasa_schema_types(self) -> None:
        entry = TOOL_REGISTRY["freesasa_sasa"]
        assert entry.input_schema is ScoringInput
        assert entry.output_schema is ScoringOutput

    def test_image_tag(self) -> None:
        assert TOOL_REGISTRY["freesasa_bsa"].image_tag == "freesasa:2.2.1"
        assert TOOL_REGISTRY["freesasa_sasa"].image_tag == "freesasa:2.2.1"

    def test_timeout(self) -> None:
        assert TOOL_REGISTRY["freesasa_bsa"].default_timeout == 300
        assert TOOL_REGISTRY["freesasa_sasa"].default_timeout == 300

    def test_get_runner_returns_freesasa_runner_bsa(self, config: AutobioConfig) -> None:
        with (
            patch("autobio.tools.base.ContainerManager"),
            patch("autobio.tools.base.GPUManager"),
        ):
            r = get_runner("freesasa_bsa", config)
        assert isinstance(r, FreeSASARunner)
        assert r.tool_name == "freesasa_bsa"

    def test_get_runner_returns_freesasa_runner_sasa(self, config: AutobioConfig) -> None:
        with (
            patch("autobio.tools.base.ContainerManager"),
            patch("autobio.tools.base.GPUManager"),
        ):
            r = get_runner("freesasa_sasa", config)
        assert isinstance(r, FreeSASARunner)
        assert r.tool_name == "freesasa_sasa"
