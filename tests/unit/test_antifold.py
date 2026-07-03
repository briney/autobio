"""Tests for the migrated antifold Tool (modes: design, score)."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

from autobio.core.catalog import get_tool, tool_categories
from autobio.core.config import AutobioConfig
from autobio.core.registry import TOOL_REGISTRY, ToolCategory
from autobio.core.result import AutobioError
from autobio.core.workspace import Workspace
from autobio.schemas.inverse_folding import (
    InverseFoldingInput,
    InverseFoldingOutput,
)
from autobio.schemas.scoring import ScoringInput, ScoringOutput
from autobio.tools import TOOL_RUNNERS, get_runner
from autobio.tools.antifold import AntiFoldRunner

if TYPE_CHECKING:
    from pathlib import Path


_OLD_FLAT_NAMES = ("antifold_score",)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


@pytest.fixture()
def config() -> AutobioConfig:
    return AutobioConfig.resolve()


def _make_runner(mode_name: str, config: AutobioConfig) -> AntiFoldRunner:
    """Create an AntiFoldRunner with mocked deps, current_mode pinned to *mode_name*."""
    with (
        patch("autobio.tools.base.ContainerManager"),
        patch("autobio.tools.base.GPUManager"),
    ):
        runner = AntiFoldRunner("antifold", config)
    runner.current_mode = get_tool("antifold").modes[mode_name]
    return runner


@pytest.fixture()
def runner(config: AutobioConfig) -> AntiFoldRunner:
    """Design-mode runner (the common case)."""
    return _make_runner("design", config)


@pytest.fixture()
def score_runner(config: AutobioConfig) -> AntiFoldRunner:
    return _make_runner("score", config)


@pytest.fixture()
def sample_pdb(tmp_path: Path) -> Path:
    """Write a minimal PDB file for testing."""
    pdb_path = tmp_path / "test.pdb"
    pdb_path.write_text("ATOM      1  N   ALA H   1       0.000   0.000   0.000  1.00  0.00\nEND\n")
    return pdb_path


def _make_design_input(pdb: Path, **extra: object) -> InverseFoldingInput:
    """Helper to create InverseFoldingInput with default chain IDs."""
    default_extra = {"heavy_chain": "H", "light_chain": "L"}
    default_extra.update(extra)
    return InverseFoldingInput(structure_path=pdb, extra=default_extra)


def _make_score_input(
    pdb: Path, sequences: dict[str, str] | None = None, **extra: object
) -> ScoringInput:
    """Helper to create ScoringInput with default chain IDs."""
    default_extra = {"heavy_chain": "H", "light_chain": "L"}
    default_extra.update(extra)
    return ScoringInput(structure_path=pdb, sequences=sequences, extra=default_extra)


# ---------------------------------------------------------------------------
# TestAntiFoldPrepareWorkspace (design mode)
# ---------------------------------------------------------------------------


class TestAntiFoldPrepareWorkspace:
    """Tests for AntiFoldRunner.prepare_workspace in design mode."""

    def test_structure_file_copied(
        self, runner: AntiFoldRunner, tmp_path: Path, sample_pdb: Path
    ) -> None:
        """Input structure is copied to inputs/ and config references container path."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = _make_design_input(sample_pdb)
        runner.prepare_workspace(input_data, workspace)

        copied = workspace.inputs_dir / "test.pdb"
        assert copied.exists()
        assert copied.read_text() == sample_pdb.read_text()

        cfg = json.loads(workspace.config_path.read_text())
        assert cfg["structure_path"] == "/workspace/inputs/test.pdb"

    def test_mode_is_design(self, runner: AntiFoldRunner, tmp_path: Path, sample_pdb: Path) -> None:
        """Config mode is set to 'design'."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = _make_design_input(sample_pdb)
        runner.prepare_workspace(input_data, workspace)

        cfg = json.loads(workspace.config_path.read_text())
        assert cfg["mode"] == "design"

    def test_defaults_applied(
        self, runner: AntiFoldRunner, tmp_path: Path, sample_pdb: Path
    ) -> None:
        """Minimal input produces sensible defaults."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = _make_design_input(sample_pdb)
        runner.prepare_workspace(input_data, workspace)

        cfg = json.loads(workspace.config_path.read_text())
        assert cfg["temperature"] == 0.1
        assert cfg["num_sequences"] == 1

    def test_num_sequences_passthrough(
        self, runner: AntiFoldRunner, tmp_path: Path, sample_pdb: Path
    ) -> None:
        workspace = Workspace.create(tmp_path / "ws")
        input_data = InverseFoldingInput(
            structure_path=sample_pdb,
            num_sequences=5,
            extra={"heavy_chain": "H", "light_chain": "L"},
        )
        runner.prepare_workspace(input_data, workspace)

        cfg = json.loads(workspace.config_path.read_text())
        assert cfg["num_sequences"] == 5

    def test_temperature_passthrough(
        self, runner: AntiFoldRunner, tmp_path: Path, sample_pdb: Path
    ) -> None:
        workspace = Workspace.create(tmp_path / "ws")
        input_data = InverseFoldingInput(
            structure_path=sample_pdb,
            temperature=0.25,
            extra={"heavy_chain": "H", "light_chain": "L"},
        )
        runner.prepare_workspace(input_data, workspace)

        cfg = json.loads(workspace.config_path.read_text())
        assert cfg["temperature"] == 0.25

    def test_heavy_chain_passthrough(
        self, runner: AntiFoldRunner, tmp_path: Path, sample_pdb: Path
    ) -> None:
        workspace = Workspace.create(tmp_path / "ws")
        input_data = _make_design_input(sample_pdb, heavy_chain="A")
        runner.prepare_workspace(input_data, workspace)

        cfg = json.loads(workspace.config_path.read_text())
        assert cfg["heavy_chain"] == "A"

    def test_light_chain_passthrough(
        self, runner: AntiFoldRunner, tmp_path: Path, sample_pdb: Path
    ) -> None:
        workspace = Workspace.create(tmp_path / "ws")
        input_data = _make_design_input(sample_pdb, light_chain="B")
        runner.prepare_workspace(input_data, workspace)

        cfg = json.loads(workspace.config_path.read_text())
        assert cfg["light_chain"] == "B"

    def test_antigen_chain_passthrough(
        self, runner: AntiFoldRunner, tmp_path: Path, sample_pdb: Path
    ) -> None:
        workspace = Workspace.create(tmp_path / "ws")
        input_data = _make_design_input(sample_pdb, antigen_chain="C")
        runner.prepare_workspace(input_data, workspace)

        cfg = json.loads(workspace.config_path.read_text())
        assert cfg["antigen_chain"] == "C"

    def test_regions_passthrough(
        self, runner: AntiFoldRunner, tmp_path: Path, sample_pdb: Path
    ) -> None:
        workspace = Workspace.create(tmp_path / "ws")
        input_data = _make_design_input(sample_pdb, regions=["CDRH1", "CDRH2", "CDRH3"])
        runner.prepare_workspace(input_data, workspace)

        cfg = json.loads(workspace.config_path.read_text())
        assert cfg["regions"] == ["CDRH1", "CDRH2", "CDRH3"]

    def test_extra_dict_merged(
        self, runner: AntiFoldRunner, tmp_path: Path, sample_pdb: Path
    ) -> None:
        """Extra dict keys appear at the top level of config.json."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = _make_design_input(sample_pdb, seed=42)
        runner.prepare_workspace(input_data, workspace)

        cfg = json.loads(workspace.config_path.read_text())
        assert cfg["seed"] == 42

    def test_chains_to_design_not_mapped(
        self, runner: AntiFoldRunner, tmp_path: Path, sample_pdb: Path
    ) -> None:
        """chains_to_design from InverseFoldingInput is NOT passed to config."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = InverseFoldingInput(
            structure_path=sample_pdb,
            chains_to_design=["A", "B"],
            extra={"heavy_chain": "H", "light_chain": "L"},
        )
        runner.prepare_workspace(input_data, workspace)

        cfg = json.loads(workspace.config_path.read_text())
        assert "chains_to_design" not in cfg

    def test_fixed_positions_not_mapped(
        self, runner: AntiFoldRunner, tmp_path: Path, sample_pdb: Path
    ) -> None:
        """fixed_positions from InverseFoldingInput is NOT passed to config."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = InverseFoldingInput(
            structure_path=sample_pdb,
            fixed_positions={"A": [1, 5, 10]},
            extra={"heavy_chain": "H", "light_chain": "L"},
        )
        runner.prepare_workspace(input_data, workspace)

        cfg = json.loads(workspace.config_path.read_text())
        assert "fixed_positions" not in cfg

    def test_validation_requires_chain_id(
        self, runner: AntiFoldRunner, tmp_path: Path, sample_pdb: Path
    ) -> None:
        """Raises AutobioError when neither heavy_chain nor light_chain provided."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = InverseFoldingInput(structure_path=sample_pdb, extra={})
        with pytest.raises(AutobioError, match="heavy_chain.*light_chain"):
            runner.prepare_workspace(input_data, workspace)

    def test_validation_accepts_heavy_only(
        self, runner: AntiFoldRunner, tmp_path: Path, sample_pdb: Path
    ) -> None:
        """Does NOT raise when only heavy_chain is provided."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = InverseFoldingInput(structure_path=sample_pdb, extra={"heavy_chain": "H"})
        runner.prepare_workspace(input_data, workspace)

        cfg = json.loads(workspace.config_path.read_text())
        assert cfg["heavy_chain"] == "H"

    def test_validation_accepts_light_only(
        self, runner: AntiFoldRunner, tmp_path: Path, sample_pdb: Path
    ) -> None:
        """Does NOT raise when only light_chain is provided."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = InverseFoldingInput(structure_path=sample_pdb, extra={"light_chain": "L"})
        runner.prepare_workspace(input_data, workspace)

        cfg = json.loads(workspace.config_path.read_text())
        assert cfg["light_chain"] == "L"

    def test_extra_shadowing_typed_field_rejected(
        self, runner: AntiFoldRunner, tmp_path: Path, sample_pdb: Path
    ) -> None:
        """extra containing a typed field name (temperature) raises."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = _make_design_input(sample_pdb, temperature=0.5)
        with pytest.raises(AutobioError, match="collide with typed input fields"):
            runner.prepare_workspace(input_data, workspace)

    def test_extra_shadowing_config_key_rejected(
        self, runner: AntiFoldRunner, tmp_path: Path, sample_pdb: Path
    ) -> None:
        """extra containing a runner-derived config key (mode) raises."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = _make_design_input(sample_pdb, mode="score")
        with pytest.raises(AutobioError, match="collide"):
            runner.prepare_workspace(input_data, workspace)


# ---------------------------------------------------------------------------
# TestAntiFoldParseOutput (design mode)
# ---------------------------------------------------------------------------

_SINGLE_SEQ_RESULT = {
    "designed_sequences": [
        {
            "rank": 1,
            "sequence": {"H": "EVQLVESGGGLVQPGGSLRLSCAASGFNIKD"},
            "score": -1.234,
            "recovery": 0.85,
        }
    ],
    "native_sequence": {"H": "EVQLVESGGGLVQPGGSLRLSCAASGFTFSD"},
}

_MULTI_SEQ_RESULT = {
    "designed_sequences": [
        {
            "rank": 1,
            "sequence": {"H": "EVQLVESGG", "L": "DIQMTQSPS"},
            "score": -0.95,
            "recovery": 0.88,
        },
        {
            "rank": 2,
            "sequence": {"H": "EVQLVEAGG", "L": "DIQMTQAPS"},
            "score": -1.12,
            "recovery": 0.75,
        },
        {
            "rank": 3,
            "sequence": {"H": "EVQLVESGK", "L": "DIQMTQSPK"},
            "score": -1.45,
            "recovery": 0.62,
        },
    ],
    "native_sequence": {"H": "EVQLVESGG", "L": "DIQMTQSPS"},
}


class TestAntiFoldParseOutput:
    """Tests for AntiFoldRunner.parse_output in design mode."""

    def test_parse_single_sequence(self, runner: AntiFoldRunner, tmp_path: Path) -> None:
        workspace = Workspace.create(tmp_path / "ws")
        (workspace.std_output_dir / "result_data.json").write_text(json.dumps(_SINGLE_SEQ_RESULT))

        output = runner.parse_output(workspace)
        assert isinstance(output, InverseFoldingOutput)
        assert len(output.designed_sequences) == 1
        seq = output.designed_sequences[0]
        assert seq.rank == 1
        assert seq.sequence["H"].startswith("EVQLVES")
        assert seq.recovery == pytest.approx(0.85)

    def test_parse_multiple_sequences(self, runner: AntiFoldRunner, tmp_path: Path) -> None:
        workspace = Workspace.create(tmp_path / "ws")
        (workspace.std_output_dir / "result_data.json").write_text(json.dumps(_MULTI_SEQ_RESULT))

        output = runner.parse_output(workspace)
        assert isinstance(output, InverseFoldingOutput)
        assert len(output.designed_sequences) == 3
        assert output.designed_sequences[0].rank == 1
        assert output.designed_sequences[2].rank == 3

    def test_parse_multi_chain(self, runner: AntiFoldRunner, tmp_path: Path) -> None:
        workspace = Workspace.create(tmp_path / "ws")
        (workspace.std_output_dir / "result_data.json").write_text(json.dumps(_MULTI_SEQ_RESULT))

        output = runner.parse_output(workspace)
        assert isinstance(output, InverseFoldingOutput)
        seq = output.designed_sequences[0]
        assert "H" in seq.sequence
        assert "L" in seq.sequence

    def test_parse_with_score(self, runner: AntiFoldRunner, tmp_path: Path) -> None:
        """AntiFold populates score (unlike ESM-IF1 where it is None)."""
        workspace = Workspace.create(tmp_path / "ws")
        (workspace.std_output_dir / "result_data.json").write_text(json.dumps(_SINGLE_SEQ_RESULT))

        output = runner.parse_output(workspace)
        assert isinstance(output, InverseFoldingOutput)
        seq = output.designed_sequences[0]
        assert seq.score is not None
        assert seq.score == pytest.approx(-1.234)

    def test_parse_with_native_sequence(self, runner: AntiFoldRunner, tmp_path: Path) -> None:
        workspace = Workspace.create(tmp_path / "ws")
        (workspace.std_output_dir / "result_data.json").write_text(json.dumps(_SINGLE_SEQ_RESULT))

        output = runner.parse_output(workspace)
        assert isinstance(output, InverseFoldingOutput)
        assert output.native_sequence is not None
        assert "H" in output.native_sequence

    def test_output_type(self, runner: AntiFoldRunner, tmp_path: Path) -> None:
        workspace = Workspace.create(tmp_path / "ws")
        (workspace.std_output_dir / "result_data.json").write_text(json.dumps(_SINGLE_SEQ_RESULT))

        output = runner.parse_output(workspace)
        assert isinstance(output, InverseFoldingOutput)

    def test_raw_output_path(self, runner: AntiFoldRunner, tmp_path: Path) -> None:
        workspace = Workspace.create(tmp_path / "ws")
        (workspace.std_output_dir / "result_data.json").write_text(json.dumps(_SINGLE_SEQ_RESULT))

        output = runner.parse_output(workspace)
        assert output.raw_output_path == workspace.raw_output_dir


# ---------------------------------------------------------------------------
# TestAntiFoldScorePrepareWorkspace (score mode)
# ---------------------------------------------------------------------------


class TestAntiFoldScorePrepareWorkspace:
    """Tests for AntiFoldRunner.prepare_workspace in score mode."""

    def test_mode_is_score(
        self, score_runner: AntiFoldRunner, tmp_path: Path, sample_pdb: Path
    ) -> None:
        workspace = Workspace.create(tmp_path / "ws")
        input_data = _make_score_input(sample_pdb, sequences={"H": "EVQLVES"})
        score_runner.prepare_workspace(input_data, workspace)

        cfg = json.loads(workspace.config_path.read_text())
        assert cfg["mode"] == "score"

    def test_structure_file_copied(
        self, score_runner: AntiFoldRunner, tmp_path: Path, sample_pdb: Path
    ) -> None:
        workspace = Workspace.create(tmp_path / "ws")
        input_data = _make_score_input(sample_pdb, sequences={"H": "EVQLVES"})
        score_runner.prepare_workspace(input_data, workspace)

        copied = workspace.inputs_dir / "test.pdb"
        assert copied.exists()

        cfg = json.loads(workspace.config_path.read_text())
        assert cfg["structure_path"] == "/workspace/inputs/test.pdb"

    def test_sequences_none_passthrough(
        self, score_runner: AntiFoldRunner, tmp_path: Path, sample_pdb: Path
    ) -> None:
        """sequences=None is passed through (score native)."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = _make_score_input(sample_pdb, sequences=None)
        score_runner.prepare_workspace(input_data, workspace)

        cfg = json.loads(workspace.config_path.read_text())
        assert cfg["sequences"] is None

    def test_sequences_dict_passthrough(
        self, score_runner: AntiFoldRunner, tmp_path: Path, sample_pdb: Path
    ) -> None:
        workspace = Workspace.create(tmp_path / "ws")
        sequences = {"H": "EVQLVES", "L": "DIQMTQS"}
        input_data = _make_score_input(sample_pdb, sequences=sequences)
        score_runner.prepare_workspace(input_data, workspace)

        cfg = json.loads(workspace.config_path.read_text())
        assert cfg["sequences"] == sequences

    def test_extra_dict_merged(
        self, score_runner: AntiFoldRunner, tmp_path: Path, sample_pdb: Path
    ) -> None:
        workspace = Workspace.create(tmp_path / "ws")
        input_data = _make_score_input(sample_pdb, sequences={"H": "EVQLVES"}, regions=["CDRH3"])
        score_runner.prepare_workspace(input_data, workspace)

        cfg = json.loads(workspace.config_path.read_text())
        assert cfg["regions"] == ["CDRH3"]

    def test_validation_requires_chain_id(
        self, score_runner: AntiFoldRunner, tmp_path: Path, sample_pdb: Path
    ) -> None:
        workspace = Workspace.create(tmp_path / "ws")
        input_data = ScoringInput(structure_path=sample_pdb, sequences={"H": "EVQLVES"}, extra={})
        with pytest.raises(AutobioError, match="heavy_chain.*light_chain"):
            score_runner.prepare_workspace(input_data, workspace)

    def test_extra_shadowing_typed_field_rejected(
        self, score_runner: AntiFoldRunner, tmp_path: Path, sample_pdb: Path
    ) -> None:
        """extra containing a typed field name (sequences) raises."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = ScoringInput(
            structure_path=sample_pdb,
            sequences={"H": "EVQLVES"},
            extra={
                "heavy_chain": "H",
                "light_chain": "L",
                "sequences": {"H": "GVSEKL"},
            },
        )
        with pytest.raises(AutobioError, match="collide with typed input fields"):
            score_runner.prepare_workspace(input_data, workspace)

    def test_extra_shadowing_config_key_rejected(
        self, score_runner: AntiFoldRunner, tmp_path: Path, sample_pdb: Path
    ) -> None:
        """extra containing a runner-derived config key (mode) raises."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = _make_score_input(sample_pdb, sequences={"H": "EVQLVES"}, mode="design")
        with pytest.raises(AutobioError, match="collide"):
            score_runner.prepare_workspace(input_data, workspace)


# ---------------------------------------------------------------------------
# TestAntiFoldScoreParseOutput (score mode)
# ---------------------------------------------------------------------------

_SCORE_RESULT = {
    "scores": [
        {
            "total_score": -0.85,
            "per_residue_scores": [-0.5, -1.2, -0.8, -0.9],
            "score_breakdown": {
                "H_mean_ll": -0.90,
                "L_mean_ll": -0.80,
                "H_perplexity": 2.46,
                "L_perplexity": 2.23,
                "perplexity": 2.34,
            },
            "units": "avg_nll",
            "structure_path": None,
            "ddg": None,
            "mutations": None,
        }
    ]
}


class TestAntiFoldScoreParseOutput:
    """Tests for AntiFoldRunner.parse_output in score mode."""

    def test_parse_score(self, score_runner: AntiFoldRunner, tmp_path: Path) -> None:
        workspace = Workspace.create(tmp_path / "ws")
        (workspace.std_output_dir / "result_data.json").write_text(json.dumps(_SCORE_RESULT))

        output = score_runner.parse_output(workspace)
        assert isinstance(output, ScoringOutput)
        assert len(output.scores) == 1
        score = output.scores[0]
        assert score.total_score == pytest.approx(-0.85)
        assert score.units == "avg_nll"

    def test_parse_with_per_residue_scores(
        self, score_runner: AntiFoldRunner, tmp_path: Path
    ) -> None:
        workspace = Workspace.create(tmp_path / "ws")
        (workspace.std_output_dir / "result_data.json").write_text(json.dumps(_SCORE_RESULT))

        output = score_runner.parse_output(workspace)
        assert isinstance(output, ScoringOutput)
        score = output.scores[0]
        assert score.per_residue_scores is not None
        assert len(score.per_residue_scores) == 4

    def test_parse_with_breakdown(self, score_runner: AntiFoldRunner, tmp_path: Path) -> None:
        workspace = Workspace.create(tmp_path / "ws")
        (workspace.std_output_dir / "result_data.json").write_text(json.dumps(_SCORE_RESULT))

        output = score_runner.parse_output(workspace)
        assert isinstance(output, ScoringOutput)
        score = output.scores[0]
        assert score.score_breakdown is not None
        assert "perplexity" in score.score_breakdown
        assert "H_mean_ll" in score.score_breakdown

    def test_output_type(self, score_runner: AntiFoldRunner, tmp_path: Path) -> None:
        workspace = Workspace.create(tmp_path / "ws")
        (workspace.std_output_dir / "result_data.json").write_text(json.dumps(_SCORE_RESULT))

        output = score_runner.parse_output(workspace)
        assert isinstance(output, ScoringOutput)

    def test_raw_output_path(self, score_runner: AntiFoldRunner, tmp_path: Path) -> None:
        workspace = Workspace.create(tmp_path / "ws")
        (workspace.std_output_dir / "result_data.json").write_text(json.dumps(_SCORE_RESULT))

        output = score_runner.parse_output(workspace)
        assert output.raw_output_path == workspace.raw_output_dir


# ---------------------------------------------------------------------------
# TestAntiFoldByteCompatConfig — full-dict config.json equality, per mode
# ---------------------------------------------------------------------------


class TestAntiFoldByteCompatConfig:
    """Full-dict ``config.json`` equality tests, pinning key order per mode."""

    def test_design_full_config_minimal(
        self, config: AutobioConfig, tmp_path: Path, sample_pdb: Path
    ) -> None:
        """Minimal design input: only the fixed keys, no chains_to_design/fixed_positions."""
        r = _make_runner("design", config)
        workspace = Workspace.create(tmp_path / "ws")
        r.prepare_workspace(_make_design_input(sample_pdb), workspace)

        cfg = json.loads(workspace.config_path.read_text())
        expected = {
            "mode": "design",
            "structure_path": f"/workspace/inputs/{sample_pdb.name}",
            "num_sequences": 1,
            "temperature": 0.1,
            "heavy_chain": "H",
            "light_chain": "L",
        }
        assert cfg == expected
        assert list(cfg.keys()) == list(expected.keys())

    def test_design_full_config_with_antibody_extra(
        self, config: AutobioConfig, tmp_path: Path, sample_pdb: Path
    ) -> None:
        """Design input with antibody params + tool-specific extra — full key order.

        Antibody params (heavy_chain/light_chain/antigen_chain/regions) and any other
        extra keys must land AFTER the fixed keys (mode/structure_path/num_sequences/
        temperature) since they flow through ``_apply_extra`` at the end. No
        chains_to_design/fixed_positions keys should ever appear.
        """
        r = _make_runner("design", config)
        workspace = Workspace.create(tmp_path / "ws")
        r.prepare_workspace(
            InverseFoldingInput(
                structure_path=sample_pdb,
                num_sequences=3,
                temperature=0.5,
                extra={
                    "heavy_chain": "H",
                    "light_chain": "L",
                    "antigen_chain": "C",
                    "regions": ["CDRH3"],
                    "seed": 42,
                },
            ),
            workspace,
        )

        cfg = json.loads(workspace.config_path.read_text())
        expected = {
            "mode": "design",
            "structure_path": f"/workspace/inputs/{sample_pdb.name}",
            "num_sequences": 3,
            "temperature": 0.5,
            "heavy_chain": "H",
            "light_chain": "L",
            "antigen_chain": "C",
            "regions": ["CDRH3"],
            "seed": 42,
        }
        assert cfg == expected
        assert list(cfg.keys()) == list(expected.keys())
        assert "chains_to_design" not in cfg
        assert "fixed_positions" not in cfg

    def test_score_full_config(
        self, config: AutobioConfig, tmp_path: Path, sample_pdb: Path
    ) -> None:
        r = _make_runner("score", config)
        workspace = Workspace.create(tmp_path / "ws")
        r.prepare_workspace(
            ScoringInput(
                structure_path=sample_pdb,
                sequences={"H": "EVQLVES"},
                extra={"heavy_chain": "H", "light_chain": "L", "regions": ["CDRH3"]},
            ),
            workspace,
        )

        cfg = json.loads(workspace.config_path.read_text())
        expected = {
            "mode": "score",
            "structure_path": f"/workspace/inputs/{sample_pdb.name}",
            "sequences": {"H": "EVQLVES"},
            "heavy_chain": "H",
            "light_chain": "L",
            "regions": ["CDRH3"],
        }
        assert cfg == expected
        assert list(cfg.keys()) == list(expected.keys())


# ---------------------------------------------------------------------------
# TestAntiFoldRegistration
# ---------------------------------------------------------------------------


class TestAntiFoldRegistration:
    """Tests for the catalog Tool + runner registration."""

    def test_antifold_registered_as_single_tool(self) -> None:
        import autobio.tools  # noqa: F401 - populate registries

        tool = get_tool("antifold")
        assert set(tool.modes) == {"design", "score"}
        assert tool.default_mode == "design"
        assert tool.category == ToolCategory.INVERSE_FOLDING
        assert tool.requires_gpu is True
        assert tool.gpu_count == 1
        assert tool.image_tag == "antifold:1.0.0"

    @pytest.mark.parametrize("flat_name", ("antifold", *_OLD_FLAT_NAMES))
    def test_old_flat_names_absent_from_tool_registry(self, flat_name: str) -> None:
        import autobio.tools  # noqa: F401

        assert flat_name not in TOOL_REGISTRY

    @pytest.mark.parametrize("flat_name", _OLD_FLAT_NAMES)
    def test_old_flat_names_absent_from_tool_runners(self, flat_name: str) -> None:
        import autobio.tools  # noqa: F401

        assert flat_name not in TOOL_RUNNERS

    def test_antifold_in_tool_runners(self) -> None:
        import autobio.tools  # noqa: F401

        assert "antifold" in TOOL_RUNNERS
        assert TOOL_RUNNERS["antifold"] is AntiFoldRunner

    def test_antifold_score_runner_class_removed(self) -> None:
        import autobio.tools.antifold as antifold_module

        assert not hasattr(antifold_module, "AntiFoldScoreRunner")

    def test_both_share_image_tag(self) -> None:
        """Neither mode overrides image_tag — both fall back to the Tool's."""
        tool = get_tool("antifold")
        assert tool.image_tag == "antifold:1.0.0"
        assert tool.modes["design"].image_tag is None
        assert tool.modes["score"].image_tag is None

    @pytest.mark.parametrize(
        ("mode_name", "timeout"),
        [("design", 600), ("score", 300)],
    )
    def test_modes_have_per_mode_timeout(self, mode_name: str, timeout: int) -> None:
        assert get_tool("antifold").modes[mode_name].default_timeout == timeout

    def test_mode_schemas(self) -> None:
        tool = get_tool("antifold")
        assert tool.modes["design"].input_schema is InverseFoldingInput
        assert tool.modes["design"].output_schema is InverseFoldingOutput
        assert tool.modes["score"].input_schema is ScoringInput
        assert tool.modes["score"].output_schema is ScoringOutput

    def test_get_runner_returns_correct_class(self, config: AutobioConfig) -> None:
        with (
            patch("autobio.tools.base.ContainerManager"),
            patch("autobio.tools.base.GPUManager"),
        ):
            r = get_runner("antifold", config)
        assert isinstance(r, AntiFoldRunner)
        assert r.tool_name == "antifold"

    def test_get_runner_removed_flat_name_raises(self, config: AutobioConfig) -> None:
        with pytest.raises(KeyError, match="antifold_score"):
            get_runner("antifold_score", config)


# ---------------------------------------------------------------------------
# TestAntiFoldCrossCategory — cross-category catalog Tool
# ---------------------------------------------------------------------------


class TestAntiFoldCrossCategory:
    """antifold's modes span two categories (design=INVERSE_FOLDING, score=SCORING)."""

    def test_tool_categories_union(self) -> None:
        import autobio.tools  # noqa: F401

        tool = get_tool("antifold")
        assert tool_categories(tool) == (ToolCategory.INVERSE_FOLDING, ToolCategory.SCORING)

    def test_listed_under_inverse_folding(self) -> None:
        import autobio.tools  # noqa: F401
        from autobio.core.catalog import list_tools

        assert "antifold" in list_tools(category=ToolCategory.INVERSE_FOLDING)

    def test_listed_under_scoring(self) -> None:
        import autobio.tools  # noqa: F401
        from autobio.core.catalog import list_tools

        assert "antifold" in list_tools(category=ToolCategory.SCORING)


# ---------------------------------------------------------------------------
# TestAntiFoldInfoSnapshot
# ---------------------------------------------------------------------------


class TestAntiFoldInfoSnapshot:
    """``autobio info antifold`` output — per-mode notes, output_schema, category."""

    def test_info_snapshot(self) -> None:
        import autobio.tools  # noqa: F401
        from autobio.cli.formatters import OutputFormat, format_tool_info_catalog

        parsed = json.loads(format_tool_info_catalog(get_tool("antifold"), OutputFormat.JSON))
        assert [m["name"] for m in parsed["modes"]] == ["design", "score"]

        design_mode = parsed["modes"][0]
        assert len(design_mode["notes"]) > 0
        assert "output_schema" in design_mode

        score_mode = parsed["modes"][1]
        assert len(score_mode["notes"]) > 0
        assert "output_schema" in score_mode
        assert score_mode["category"] == "scoring"


# ---------------------------------------------------------------------------
# TestAntiFoldRunMetadataMode — full run() lifecycle threads mode into metadata
# ---------------------------------------------------------------------------

_MIN_DESIGN_RESULT = {
    "designed_sequences": [
        {"rank": 1, "sequence": {"H": "EVQLVES"}, "score": -1.0, "recovery": 0.5}
    ],
    "native_sequence": {"H": "EVQLVES"},
}

_MIN_SCORE_RESULT = {
    "scores": [
        {
            "total_score": -1.0,
            "per_residue_scores": None,
            "score_breakdown": None,
            "units": "avg_nll",
            "structure_path": None,
            "ddg": None,
            "mutations": None,
        }
    ]
}


class TestAntiFoldRunMetadataMode:
    """``run(...).metadata.mode`` reflects the selected mode for each mode."""

    @pytest.mark.parametrize(
        ("mode_name", "result_data"),
        [("design", _MIN_DESIGN_RESULT), ("score", _MIN_SCORE_RESULT)],
    )
    def test_run_metadata_mode(
        self,
        mode_name: str,
        result_data: dict,
        config: AutobioConfig,
        tmp_path: Path,
        sample_pdb: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import autobio.tools  # noqa: F401

        output_dir = tmp_path / "ws"
        std_dir = output_dir / "outputs" / "standardized"
        std_dir.mkdir(parents=True)
        (std_dir / "result_data.json").write_text(json.dumps(result_data))

        monkeypatch.setattr(
            "autobio.core.workspace.Workspace.read_result",
            lambda self: SimpleNamespace(
                status="success", phase="run", exit_code=0, error_message=None
            ),
        )

        with (
            patch("autobio.tools.base.ContainerManager"),
            patch("autobio.tools.base.GPUManager"),
        ):
            r = AntiFoldRunner("antifold", config)

        if mode_name == "design":
            input_data: InverseFoldingInput | ScoringInput = _make_design_input(sample_pdb)
        else:
            input_data = _make_score_input(sample_pdb, sequences={"H": "EVQLVES"})

        out = r.run(input_data, gpu="none", output_dir=output_dir, mode=mode_name)
        assert out.metadata.mode == mode_name
        assert out.metadata.tool_name == "antifold"
