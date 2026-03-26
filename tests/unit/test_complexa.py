"""Tests for ComplexaRunner — prepare_workspace, parse_output, host validation, and registration."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

from autobio.core.config import AutobioConfig
from autobio.core.registry import TOOL_REGISTRY, ToolCategory
from autobio.core.result import AutobioError
from autobio.core.workspace import Workspace
from autobio.schemas.structure_design import (
    StructureDesignInput,
    StructureDesignOutput,
)
from autobio.tools import TOOL_RUNNERS, get_runner
from autobio.tools.complexa import ComplexaRunner

if TYPE_CHECKING:
    from pathlib import Path


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def config() -> AutobioConfig:
    return AutobioConfig.resolve()


@pytest.fixture()
def runner(config: AutobioConfig) -> ComplexaRunner:
    """Create a ComplexaRunner (protein binder variant) with mocked infra."""
    with (
        patch("autobio.tools.base.ContainerManager"),
        patch("autobio.tools.base.GPUManager"),
    ):
        return ComplexaRunner("complexa", config)


@pytest.fixture()
def ligand_runner(config: AutobioConfig) -> ComplexaRunner:
    """Create a ComplexaRunner (ligand binder variant) with mocked infra."""
    with (
        patch("autobio.tools.base.ContainerManager"),
        patch("autobio.tools.base.GPUManager"),
    ):
        return ComplexaRunner("complexa_ligand", config)


@pytest.fixture()
def ame_runner(config: AutobioConfig) -> ComplexaRunner:
    """Create a ComplexaRunner (AME variant) with mocked infra."""
    with (
        patch("autobio.tools.base.ContainerManager"),
        patch("autobio.tools.base.GPUManager"),
    ):
        return ComplexaRunner("complexa_ame", config)


@pytest.fixture()
def sample_pdb(tmp_path: Path) -> Path:
    """Write a minimal PDB file for testing."""
    pdb_path = tmp_path / "target.pdb"
    pdb_path.write_text("ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00\nEND\n")
    return pdb_path


# ---------------------------------------------------------------------------
# TestComplexaPrepareWorkspace
# ---------------------------------------------------------------------------


class TestComplexaPrepareWorkspace:
    """Tests for ComplexaRunner.prepare_workspace."""

    def test_design_specs_written_to_config(
        self, runner: ComplexaRunner, tmp_path: Path, sample_pdb: Path
    ) -> None:
        workspace = Workspace.create(tmp_path / "ws")
        input_data = StructureDesignInput(
            input_structures=[sample_pdb],
            design_specs={
                "pdl1": {
                    "input": str(sample_pdb),
                    "target_input": "A1-115",
                    "hotspot_residues": ["A37", "A39"],
                    "binder_length": [64, 155],
                },
            },
        )
        runner.prepare_workspace(input_data, workspace)

        cfg = json.loads(workspace.config_path.read_text())
        assert "design_specs" in cfg
        assert "pdl1" in cfg["design_specs"]
        assert cfg["design_specs"]["pdl1"]["target_input"] == "A1-115"
        assert cfg["design_specs"]["pdl1"]["hotspot_residues"] == ["A37", "A39"]
        assert cfg["design_specs"]["pdl1"]["binder_length"] == [64, 155]

    def test_variant_config_protein_binder(self, runner: ComplexaRunner, tmp_path: Path) -> None:
        workspace = Workspace.create(tmp_path / "ws")
        input_data = StructureDesignInput(
            design_specs={"test": {"target_input": "A1-50"}},
        )
        runner.prepare_workspace(input_data, workspace)

        cfg = json.loads(workspace.config_path.read_text())
        assert cfg["variant"] == "protein_binder"
        assert cfg["ckpt_name"] == "complexa.ckpt"
        assert cfg["ae_ckpt_name"] == "complexa_ae.ckpt"
        assert cfg["pipeline_config"] == "search_binder_local_pipeline"

    def test_variant_config_ligand_binder(
        self, ligand_runner: ComplexaRunner, tmp_path: Path
    ) -> None:
        workspace = Workspace.create(tmp_path / "ws")
        input_data = StructureDesignInput(
            design_specs={"test": {"target_input": "A1-50"}},
        )
        ligand_runner.prepare_workspace(input_data, workspace)

        cfg = json.loads(workspace.config_path.read_text())
        assert cfg["variant"] == "ligand_binder"
        assert cfg["ckpt_name"] == "complexa_ligand.ckpt"
        assert cfg["ae_ckpt_name"] == "complexa_ligand_ae.ckpt"

    def test_variant_config_ame(self, ame_runner: ComplexaRunner, tmp_path: Path) -> None:
        workspace = Workspace.create(tmp_path / "ws")
        input_data = StructureDesignInput(
            design_specs={"test": {"target_input": "A1-50"}},
        )
        ame_runner.prepare_workspace(input_data, workspace)

        cfg = json.loads(workspace.config_path.read_text())
        assert cfg["variant"] == "ame"
        assert cfg["ckpt_name"] == "complexa_ame.ckpt"
        assert cfg["ae_ckpt_name"] == "complexa_ame_ae.ckpt"

    def test_input_structures_copied(
        self, runner: ComplexaRunner, tmp_path: Path, sample_pdb: Path
    ) -> None:
        workspace = Workspace.create(tmp_path / "ws")
        input_data = StructureDesignInput(
            input_structures=[sample_pdb],
            design_specs={"test": {"input": str(sample_pdb), "target_input": "A1-50"}},
        )
        runner.prepare_workspace(input_data, workspace)

        copied = workspace.inputs_dir / "target.pdb"
        assert copied.exists()
        assert copied.read_text() == sample_pdb.read_text()

    def test_input_paths_rewritten(
        self, runner: ComplexaRunner, tmp_path: Path, sample_pdb: Path
    ) -> None:
        """'input' values in specs are rewritten to container-internal paths."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = StructureDesignInput(
            input_structures=[sample_pdb],
            design_specs={"test": {"input": str(sample_pdb), "target_input": "A1-50"}},
        )
        runner.prepare_workspace(input_data, workspace)

        cfg = json.loads(workspace.config_path.read_text())
        assert cfg["design_specs"]["test"]["input"] == "/workspace/inputs/target.pdb"

    def test_n_batches_in_config(self, runner: ComplexaRunner, tmp_path: Path) -> None:
        workspace = Workspace.create(tmp_path / "ws")
        input_data = StructureDesignInput(
            design_specs={"test": {"target_input": "A1-50"}},
            n_batches=3,
        )
        runner.prepare_workspace(input_data, workspace)

        cfg = json.loads(workspace.config_path.read_text())
        assert cfg["n_batches"] == 3

    def test_out_dir_set(self, runner: ComplexaRunner, tmp_path: Path) -> None:
        workspace = Workspace.create(tmp_path / "ws")
        input_data = StructureDesignInput(
            design_specs={"test": {"target_input": "A1-50"}},
        )
        runner.prepare_workspace(input_data, workspace)

        cfg = json.loads(workspace.config_path.read_text())
        assert cfg["out_dir"] == "/workspace/outputs/raw"

    def test_weights_dir_set(self, runner: ComplexaRunner, tmp_path: Path) -> None:
        workspace = Workspace.create(tmp_path / "ws")
        input_data = StructureDesignInput(
            design_specs={"test": {"target_input": "A1-50"}},
        )
        runner.prepare_workspace(input_data, workspace)

        cfg = json.loads(workspace.config_path.read_text())
        assert cfg["weights_dir"] == "/app/proteina-complexa/ckpts"

    def test_extra_dict_merged(self, runner: ComplexaRunner, tmp_path: Path) -> None:
        """Extra dict keys appear at the top level of config.json."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = StructureDesignInput(
            design_specs={"test": {"target_input": "A1-50"}},
            extra={"batch_size": 8, "seed": 123, "search_algorithm": "best-of-n"},
        )
        runner.prepare_workspace(input_data, workspace)

        cfg = json.loads(workspace.config_path.read_text())
        assert cfg["batch_size"] == 8
        assert cfg["seed"] == 123
        assert cfg["search_algorithm"] == "best-of-n"

    def test_multiple_specs(self, runner: ComplexaRunner, tmp_path: Path, sample_pdb: Path) -> None:
        """Multiple named specs are preserved in config."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = StructureDesignInput(
            input_structures=[sample_pdb],
            design_specs={
                "target_a": {
                    "input": str(sample_pdb),
                    "target_input": "A1-100",
                    "hotspot_residues": ["A10"],
                },
                "target_b": {
                    "input": str(sample_pdb),
                    "target_input": "B1-50",
                    "hotspot_residues": ["B20", "B30"],
                },
            },
        )
        runner.prepare_workspace(input_data, workspace)

        cfg = json.loads(workspace.config_path.read_text())
        assert len(cfg["design_specs"]) == 2
        assert "target_a" in cfg["design_specs"]
        assert "target_b" in cfg["design_specs"]

    def test_original_specs_not_mutated(
        self, runner: ComplexaRunner, tmp_path: Path, sample_pdb: Path
    ) -> None:
        """prepare_workspace deep-copies specs; original input_data is unchanged."""
        workspace = Workspace.create(tmp_path / "ws")
        original_path = str(sample_pdb)
        input_data = StructureDesignInput(
            input_structures=[sample_pdb],
            design_specs={"test": {"input": original_path, "target_input": "A1-50"}},
        )
        runner.prepare_workspace(input_data, workspace)

        # Original input_data should still reference host path
        assert input_data.design_specs["test"]["input"] == original_path

    def test_mode_design_passed_through(self, runner: ComplexaRunner, tmp_path: Path) -> None:
        """extra={'mode': 'design'} appears in config.json."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = StructureDesignInput(
            design_specs={"test": {"target_input": "A1-50"}},
            extra={"mode": "design"},
        )
        runner.prepare_workspace(input_data, workspace)

        cfg = json.loads(workspace.config_path.read_text())
        assert cfg["mode"] == "design"

    def test_mode_generate_default(self, runner: ComplexaRunner, tmp_path: Path) -> None:
        """No mode in extra → no mode key in config (container defaults to generate)."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = StructureDesignInput(
            design_specs={"test": {"target_input": "A1-50"}},
        )
        runner.prepare_workspace(input_data, workspace)

        cfg = json.loads(workspace.config_path.read_text())
        assert "mode" not in cfg

    def test_design_mode_eval_njobs(self, runner: ComplexaRunner, tmp_path: Path) -> None:
        """Design mode extra keys (eval_njobs, gen_njobs) pass through to config."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = StructureDesignInput(
            design_specs={"test": {"target_input": "A1-50"}},
            extra={"mode": "design", "eval_njobs": 4, "gen_njobs": 2},
        )
        runner.prepare_workspace(input_data, workspace)

        cfg = json.loads(workspace.config_path.read_text())
        assert cfg["mode"] == "design"
        assert cfg["eval_njobs"] == 4
        assert cfg["gen_njobs"] == 2


# ---------------------------------------------------------------------------
# TestComplexaHostValidation
# ---------------------------------------------------------------------------


class TestComplexaHostValidation:
    """Tests for host-side input validation in prepare_workspace."""

    def test_empty_design_specs_raises(self, runner: ComplexaRunner, tmp_path: Path) -> None:
        workspace = Workspace.create(tmp_path / "ws")
        input_data = StructureDesignInput(design_specs={})
        with pytest.raises(AutobioError, match="at least one specification"):
            runner.prepare_workspace(input_data, workspace)

    def test_n_batches_zero_raises(self, runner: ComplexaRunner, tmp_path: Path) -> None:
        workspace = Workspace.create(tmp_path / "ws")
        input_data = StructureDesignInput(
            design_specs={"test": {"target_input": "A1-50"}},
            n_batches=0,
        )
        with pytest.raises(AutobioError, match="n_batches must be at least 1"):
            runner.prepare_workspace(input_data, workspace)

    def test_negative_n_batches_raises(self, runner: ComplexaRunner, tmp_path: Path) -> None:
        workspace = Workspace.create(tmp_path / "ws")
        input_data = StructureDesignInput(
            design_specs={"test": {"target_input": "A1-50"}},
            n_batches=-1,
        )
        with pytest.raises(AutobioError, match="n_batches must be at least 1"):
            runner.prepare_workspace(input_data, workspace)

    def test_missing_input_file_raises(self, runner: ComplexaRunner, tmp_path: Path) -> None:
        workspace = Workspace.create(tmp_path / "ws")
        nonexistent = tmp_path / "nonexistent.pdb"
        input_data = StructureDesignInput(
            input_structures=[nonexistent],
            design_specs={"test": {"input": str(nonexistent), "target_input": "A1-50"}},
        )
        with pytest.raises(AutobioError, match="does not exist"):
            runner.prepare_workspace(input_data, workspace)

    def test_unreferenced_spec_input_raises(self, runner: ComplexaRunner, tmp_path: Path) -> None:
        """Spec references a file not in input_structures."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = StructureDesignInput(
            input_structures=[],
            design_specs={"test": {"input": "/some/file.pdb", "target_input": "A1-50"}},
        )
        with pytest.raises(AutobioError, match="no matching file"):
            runner.prepare_workspace(input_data, workspace)


# ---------------------------------------------------------------------------
# TestComplexaParseOutput
# ---------------------------------------------------------------------------

_SINGLE_DESIGN_RESULT = {
    "designs": [
        {
            "spec_name": "pdl1",
            "batch_index": 0,
            "design_index": 0,
            "structure_path": "/workspace/outputs/standardized/pdl1_b0_d0.pdb",
            "diffusion_metadata": {
                "binder_length": 80,
                "rewards": {"total_reward": "2.5"},
            },
        },
    ],
    "spec_summary": {"pdl1": 1},
}

_DESIGN_MODE_RESULT = {
    "designs": [
        {
            "spec_name": "pdl1",
            "batch_index": 0,
            "design_index": 0,
            "structure_path": "/workspace/outputs/standardized/pdl1_b0_d0.pdb",
            "diffusion_metadata": {"binder_length": 80},
            "evaluation_metrics": {
                "self_complex_i_pAE": 5.2,
                "self_binder_pLDDT": 0.92,
                "self_binder_scRMSD_ca": 1.1,
                "mpnn_complex_i_pAE": 4.8,
                "mpnn_binder_scRMSD_ca": 1.3,
            },
        },
    ],
    "spec_summary": {"pdl1": 1},
}

_MULTI_DESIGN_RESULT = {
    "designs": [
        {
            "spec_name": "target_a",
            "batch_index": 0,
            "design_index": 0,
            "structure_path": "/workspace/outputs/standardized/target_a_b0_d0.pdb",
            "diffusion_metadata": {"binder_length": 64},
        },
        {
            "spec_name": "target_a",
            "batch_index": 0,
            "design_index": 1,
            "structure_path": "/workspace/outputs/standardized/target_a_b0_d1.pdb",
            "diffusion_metadata": {"binder_length": 80},
        },
        {
            "spec_name": "target_b",
            "batch_index": 0,
            "design_index": 0,
            "structure_path": "/workspace/outputs/standardized/target_b_b0_d0.pdb",
            "diffusion_metadata": None,
        },
    ],
    "spec_summary": {"target_a": 2, "target_b": 1},
}


class TestComplexaParseOutput:
    """Tests for ComplexaRunner.parse_output."""

    def test_parse_single_design(self, runner: ComplexaRunner, tmp_path: Path) -> None:
        workspace = Workspace.create(tmp_path / "ws")
        (workspace.std_output_dir / "result_data.json").write_text(
            json.dumps(_SINGLE_DESIGN_RESULT)
        )

        output = runner.parse_output(workspace)
        assert len(output.designs) == 1
        d = output.designs[0]
        assert d.spec_name == "pdl1"
        assert d.batch_index == 0
        assert d.design_index == 0
        assert d.diffusion_metadata is not None
        assert d.diffusion_metadata["binder_length"] == 80

    def test_parse_multiple_designs(self, runner: ComplexaRunner, tmp_path: Path) -> None:
        workspace = Workspace.create(tmp_path / "ws")
        (workspace.std_output_dir / "result_data.json").write_text(json.dumps(_MULTI_DESIGN_RESULT))

        output = runner.parse_output(workspace)
        assert len(output.designs) == 3

    def test_spec_summary_correct(self, runner: ComplexaRunner, tmp_path: Path) -> None:
        workspace = Workspace.create(tmp_path / "ws")
        (workspace.std_output_dir / "result_data.json").write_text(json.dumps(_MULTI_DESIGN_RESULT))

        output = runner.parse_output(workspace)
        assert output.spec_summary == {"target_a": 2, "target_b": 1}

    def test_spec_summary_computed_from_designs(
        self, runner: ComplexaRunner, tmp_path: Path
    ) -> None:
        """If spec_summary is missing from JSON, compute it from designs."""
        data = {**_MULTI_DESIGN_RESULT, "spec_summary": {}}
        workspace = Workspace.create(tmp_path / "ws")
        (workspace.std_output_dir / "result_data.json").write_text(json.dumps(data))

        output = runner.parse_output(workspace)
        assert output.spec_summary == {"target_a": 2, "target_b": 1}

    def test_diffusion_metadata_preserved(self, runner: ComplexaRunner, tmp_path: Path) -> None:
        workspace = Workspace.create(tmp_path / "ws")
        (workspace.std_output_dir / "result_data.json").write_text(
            json.dumps(_SINGLE_DESIGN_RESULT)
        )

        output = runner.parse_output(workspace)
        d = output.designs[0]
        assert d.diffusion_metadata is not None
        assert d.diffusion_metadata["rewards"]["total_reward"] == "2.5"

    def test_output_type(self, runner: ComplexaRunner, tmp_path: Path) -> None:
        workspace = Workspace.create(tmp_path / "ws")
        (workspace.std_output_dir / "result_data.json").write_text(
            json.dumps(_SINGLE_DESIGN_RESULT)
        )

        output = runner.parse_output(workspace)
        assert isinstance(output, StructureDesignOutput)

    def test_raw_output_path(self, runner: ComplexaRunner, tmp_path: Path) -> None:
        workspace = Workspace.create(tmp_path / "ws")
        (workspace.std_output_dir / "result_data.json").write_text(
            json.dumps(_SINGLE_DESIGN_RESULT)
        )

        output = runner.parse_output(workspace)
        assert output.raw_output_path == workspace.raw_output_dir

    def test_parse_output_with_evaluation_metrics(
        self, runner: ComplexaRunner, tmp_path: Path
    ) -> None:
        """Design mode result_data.json with evaluation_metrics is parsed correctly."""
        workspace = Workspace.create(tmp_path / "ws")
        (workspace.std_output_dir / "result_data.json").write_text(json.dumps(_DESIGN_MODE_RESULT))

        output = runner.parse_output(workspace)
        assert len(output.designs) == 1
        d = output.designs[0]
        assert d.evaluation_metrics is not None
        assert d.evaluation_metrics["self_complex_i_pAE"] == 5.2
        assert d.evaluation_metrics["self_binder_pLDDT"] == 0.92
        assert d.evaluation_metrics["mpnn_complex_i_pAE"] == 4.8

    def test_parse_output_without_evaluation_metrics(
        self, runner: ComplexaRunner, tmp_path: Path
    ) -> None:
        """Generate mode results (no evaluation_metrics key) parse with None."""
        workspace = Workspace.create(tmp_path / "ws")
        (workspace.std_output_dir / "result_data.json").write_text(
            json.dumps(_SINGLE_DESIGN_RESULT)
        )

        output = runner.parse_output(workspace)
        d = output.designs[0]
        assert d.evaluation_metrics is None


# ---------------------------------------------------------------------------
# TestComplexaRegistration
# ---------------------------------------------------------------------------


class TestComplexaRegistration:
    """Tests for tool and runner registration across all three variants."""

    @pytest.mark.parametrize("tool_name", ["complexa", "complexa_ligand", "complexa_ame"])
    def test_in_registry(self, tool_name: str) -> None:
        assert tool_name in TOOL_REGISTRY
        entry = TOOL_REGISTRY[tool_name]
        assert entry.category == ToolCategory.STRUCTURE_DESIGN
        assert entry.input_schema is StructureDesignInput
        assert entry.output_schema is StructureDesignOutput
        assert entry.requires_gpu is True
        assert entry.gpu_count == 1

    @pytest.mark.parametrize("tool_name", ["complexa", "complexa_ligand", "complexa_ame"])
    def test_tool_runner_registered(self, tool_name: str) -> None:
        assert tool_name in TOOL_RUNNERS
        assert TOOL_RUNNERS[tool_name] is ComplexaRunner

    @pytest.mark.parametrize("tool_name", ["complexa", "complexa_ligand", "complexa_ame"])
    def test_get_runner_returns_complexa_runner(
        self, config: AutobioConfig, tool_name: str
    ) -> None:
        with (
            patch("autobio.tools.base.ContainerManager"),
            patch("autobio.tools.base.GPUManager"),
        ):
            r = get_runner(tool_name, config)
        assert isinstance(r, ComplexaRunner)
        assert r.tool_name == tool_name

    @pytest.mark.parametrize("tool_name", ["complexa", "complexa_ligand", "complexa_ame"])
    def test_notes_populated(self, tool_name: str) -> None:
        entry = TOOL_REGISTRY[tool_name]
        assert len(entry.notes) > 0
        all_notes = " ".join(entry.notes).lower()
        assert "gpu" in all_notes or "search" in all_notes

    @pytest.mark.parametrize("tool_name", ["complexa", "complexa_ligand", "complexa_ame"])
    def test_input_format_populated(self, tool_name: str) -> None:
        entry = TOOL_REGISTRY[tool_name]
        assert len(entry.input_format) > 0
        all_fmt = " ".join(entry.input_format).lower()
        assert "design_specs" in all_fmt or "input" in all_fmt

    @pytest.mark.parametrize("tool_name", ["complexa", "complexa_ligand", "complexa_ame"])
    def test_shared_image_tag(self, tool_name: str) -> None:
        """All three variants share the same container image."""
        entry = TOOL_REGISTRY[tool_name]
        assert entry.image_tag == "complexa:2.0.0"

    @pytest.mark.parametrize("tool_name", ["complexa", "complexa_ligand", "complexa_ame"])
    def test_default_timeout(self, tool_name: str) -> None:
        entry = TOOL_REGISTRY[tool_name]
        assert entry.default_timeout == 43200

    @pytest.mark.parametrize("tool_name", ["complexa", "complexa_ligand", "complexa_ame"])
    def test_supports_batch(self, tool_name: str) -> None:
        entry = TOOL_REGISTRY[tool_name]
        assert entry.supports_batch is True

    @pytest.mark.parametrize("tool_name", ["complexa", "complexa_ligand", "complexa_ame"])
    def test_design_mode_documented_in_notes(self, tool_name: str) -> None:
        """All variants document design mode in their notes."""
        entry = TOOL_REGISTRY[tool_name]
        all_notes = " ".join(entry.notes).lower()
        assert "design" in all_notes
        assert "mode" in all_notes
