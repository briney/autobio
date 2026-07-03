"""Tests for the migrated complexa Tool (modes: protein_binder, ligand_binder, ame)."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

from autobio.core.catalog import get_tool
from autobio.core.config import AutobioConfig
from autobio.core.registry import ToolCategory
from autobio.core.result import AutobioError
from autobio.core.workspace import Workspace
from autobio.schemas.structure_design import ComplexaInput, StructureDesignOutput
from autobio.tools import TOOL_RUNNERS, get_runner
from autobio.tools.complexa import ComplexaRunner

if TYPE_CHECKING:
    from pathlib import Path

# TOOL_RUNNERS keys retired by the collapse ("complexa" itself is reused,
# now dispatching via Modes instead of a flat per-name entry).
_RETIRED_RUNNER_KEYS = ("complexa_ligand", "complexa_ame")

_MODE_NAMES = ("protein_binder", "ligand_binder", "ame")

_MODE_CONFIG_EXPECTED = {
    "protein_binder": {
        "pipeline_config": "search_binder_local_pipeline",
        "ckpt_name": "complexa.ckpt",
        "ae_ckpt_name": "complexa_ae.ckpt",
    },
    "ligand_binder": {
        "pipeline_config": "search_ligand_binder_local_pipeline",
        "ckpt_name": "complexa_ligand.ckpt",
        "ae_ckpt_name": "complexa_ligand_ae.ckpt",
    },
    "ame": {
        "pipeline_config": "search_ame_local_pipeline",
        "ckpt_name": "complexa_ame.ckpt",
        "ae_ckpt_name": "complexa_ame_ae.ckpt",
    },
}


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


@pytest.fixture()
def config() -> AutobioConfig:
    return AutobioConfig.resolve()


def _make_runner(mode_name: str, config: AutobioConfig) -> ComplexaRunner:
    """Create a ComplexaRunner with mocked infra, current_mode pinned to *mode_name*."""
    with (
        patch("autobio.tools.base.ContainerManager"),
        patch("autobio.tools.base.GPUManager"),
    ):
        runner = ComplexaRunner("complexa", config)
    runner.current_mode = get_tool("complexa").modes[mode_name]
    return runner


@pytest.fixture()
def runner(config: AutobioConfig) -> ComplexaRunner:
    """Create a ComplexaRunner pinned to the protein_binder mode."""
    return _make_runner("protein_binder", config)


@pytest.fixture()
def ligand_runner(config: AutobioConfig) -> ComplexaRunner:
    """Create a ComplexaRunner pinned to the ligand_binder mode."""
    return _make_runner("ligand_binder", config)


@pytest.fixture()
def ame_runner(config: AutobioConfig) -> ComplexaRunner:
    """Create a ComplexaRunner pinned to the ame mode."""
    return _make_runner("ame", config)


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
        input_data = ComplexaInput(
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

    @pytest.mark.parametrize("mode_name", _MODE_NAMES)
    def test_mode_config(self, mode_name: str, config: AutobioConfig, tmp_path: Path) -> None:
        """variant/pipeline_config/ckpt_name/ae_ckpt_name differ per mode."""
        r = _make_runner(mode_name, config)
        workspace = Workspace.create(tmp_path / "ws")
        input_data = ComplexaInput(design_specs={"test": {"target_input": "A1-50"}})
        r.prepare_workspace(input_data, workspace)

        cfg = json.loads(workspace.config_path.read_text())
        expected = _MODE_CONFIG_EXPECTED[mode_name]
        assert cfg["variant"] == mode_name
        assert cfg["pipeline_config"] == expected["pipeline_config"]
        assert cfg["ckpt_name"] == expected["ckpt_name"]
        assert cfg["ae_ckpt_name"] == expected["ae_ckpt_name"]

    def test_input_structures_copied(
        self, runner: ComplexaRunner, tmp_path: Path, sample_pdb: Path
    ) -> None:
        workspace = Workspace.create(tmp_path / "ws")
        input_data = ComplexaInput(
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
        input_data = ComplexaInput(
            input_structures=[sample_pdb],
            design_specs={"test": {"input": str(sample_pdb), "target_input": "A1-50"}},
        )
        runner.prepare_workspace(input_data, workspace)

        cfg = json.loads(workspace.config_path.read_text())
        assert cfg["design_specs"]["test"]["input"] == "/workspace/inputs/target.pdb"

    def test_n_batches_in_config(self, runner: ComplexaRunner, tmp_path: Path) -> None:
        workspace = Workspace.create(tmp_path / "ws")
        input_data = ComplexaInput(
            design_specs={"test": {"target_input": "A1-50"}},
            n_batches=3,
        )
        runner.prepare_workspace(input_data, workspace)

        cfg = json.loads(workspace.config_path.read_text())
        assert cfg["n_batches"] == 3

    def test_out_dir_set(self, runner: ComplexaRunner, tmp_path: Path) -> None:
        workspace = Workspace.create(tmp_path / "ws")
        input_data = ComplexaInput(design_specs={"test": {"target_input": "A1-50"}})
        runner.prepare_workspace(input_data, workspace)

        cfg = json.loads(workspace.config_path.read_text())
        assert cfg["out_dir"] == "/workspace/outputs/raw"

    def test_weights_dir_set(self, runner: ComplexaRunner, tmp_path: Path) -> None:
        workspace = Workspace.create(tmp_path / "ws")
        input_data = ComplexaInput(design_specs={"test": {"target_input": "A1-50"}})
        runner.prepare_workspace(input_data, workspace)

        cfg = json.loads(workspace.config_path.read_text())
        assert cfg["weights_dir"] == "/app/proteina-complexa/ckpts"

    def test_extra_dict_merged(self, runner: ComplexaRunner, tmp_path: Path) -> None:
        """Extra dict keys appear at the top level of config.json."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = ComplexaInput(
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
        input_data = ComplexaInput(
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
        input_data = ComplexaInput(
            input_structures=[sample_pdb],
            design_specs={"test": {"input": original_path, "target_input": "A1-50"}},
        )
        runner.prepare_workspace(input_data, workspace)

        # Original input_data should still reference host path
        assert input_data.design_specs["test"]["input"] == original_path

    def test_mode_design_passed_through(self, runner: ComplexaRunner, tmp_path: Path) -> None:
        """extra={'mode': 'design'} appears in config.json (container-level pipeline switch)."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = ComplexaInput(
            design_specs={"test": {"target_input": "A1-50"}},
            extra={"mode": "design"},
        )
        runner.prepare_workspace(input_data, workspace)

        cfg = json.loads(workspace.config_path.read_text())
        assert cfg["mode"] == "design"

    def test_mode_generate_default(self, runner: ComplexaRunner, tmp_path: Path) -> None:
        """No mode in extra → no mode key in config (container defaults to generate)."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = ComplexaInput(design_specs={"test": {"target_input": "A1-50"}})
        runner.prepare_workspace(input_data, workspace)

        cfg = json.loads(workspace.config_path.read_text())
        assert "mode" not in cfg

    def test_design_mode_eval_njobs(self, runner: ComplexaRunner, tmp_path: Path) -> None:
        """Design mode extra keys (eval_njobs, gen_njobs) pass through to config."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = ComplexaInput(
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
        input_data = ComplexaInput(design_specs={})
        with pytest.raises(AutobioError, match="at least one specification"):
            runner.prepare_workspace(input_data, workspace)

    def test_n_batches_zero_raises(self, runner: ComplexaRunner, tmp_path: Path) -> None:
        workspace = Workspace.create(tmp_path / "ws")
        input_data = ComplexaInput(
            design_specs={"test": {"target_input": "A1-50"}},
            n_batches=0,
        )
        with pytest.raises(AutobioError, match="n_batches must be at least 1"):
            runner.prepare_workspace(input_data, workspace)

    def test_negative_n_batches_raises(self, runner: ComplexaRunner, tmp_path: Path) -> None:
        workspace = Workspace.create(tmp_path / "ws")
        input_data = ComplexaInput(
            design_specs={"test": {"target_input": "A1-50"}},
            n_batches=-1,
        )
        with pytest.raises(AutobioError, match="n_batches must be at least 1"):
            runner.prepare_workspace(input_data, workspace)

    def test_missing_input_file_raises(self, runner: ComplexaRunner, tmp_path: Path) -> None:
        workspace = Workspace.create(tmp_path / "ws")
        nonexistent = tmp_path / "nonexistent.pdb"
        input_data = ComplexaInput(
            input_structures=[nonexistent],
            design_specs={"test": {"input": str(nonexistent), "target_input": "A1-50"}},
        )
        with pytest.raises(AutobioError, match="does not exist"):
            runner.prepare_workspace(input_data, workspace)

    def test_unreferenced_spec_input_raises(self, runner: ComplexaRunner, tmp_path: Path) -> None:
        """Spec references a file not in input_structures."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = ComplexaInput(
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
# TestComplexaByteCompatConfig — full-dict config.json equality, per mode
# ---------------------------------------------------------------------------


class TestComplexaByteCompatConfig:
    """Full-dict ``config.json`` equality tests, pinning key order per mode."""

    def test_protein_binder_full_config(
        self, config: AutobioConfig, sample_pdb: Path, tmp_path: Path
    ) -> None:
        r = _make_runner("protein_binder", config)
        workspace = Workspace.create(tmp_path / "ws")
        input_data = ComplexaInput(
            input_structures=[sample_pdb],
            design_specs={
                "pdl1": {"input": str(sample_pdb), "target_input": "A1-115"},
            },
            n_batches=2,
            extra={"mode": "design", "batch_size": 8},
        )
        r.prepare_workspace(input_data, workspace)

        cfg = json.loads(workspace.config_path.read_text())
        expected = {
            "variant": "protein_binder",
            "pipeline_config": "search_binder_local_pipeline",
            "ckpt_name": "complexa.ckpt",
            "ae_ckpt_name": "complexa_ae.ckpt",
            "weights_dir": "/app/proteina-complexa/ckpts",
            "design_specs": {
                "pdl1": {
                    "input": "/workspace/inputs/target.pdb",
                    "target_input": "A1-115",
                },
            },
            "n_batches": 2,
            "out_dir": "/workspace/outputs/raw",
            "mode": "design",
            "batch_size": 8,
        }
        assert cfg == expected
        assert list(cfg.keys()) == list(expected.keys())

    def test_ligand_binder_full_config(
        self, config: AutobioConfig, sample_pdb: Path, tmp_path: Path
    ) -> None:
        r = _make_runner("ligand_binder", config)
        workspace = Workspace.create(tmp_path / "ws")
        input_data = ComplexaInput(
            input_structures=[sample_pdb],
            design_specs={
                "trypsin": {
                    "input": str(sample_pdb),
                    "target_input": "A1-223",
                    "ligand": "BEN",
                },
            },
            n_batches=1,
            extra={"seed": 42},
        )
        r.prepare_workspace(input_data, workspace)

        cfg = json.loads(workspace.config_path.read_text())
        expected = {
            "variant": "ligand_binder",
            "pipeline_config": "search_ligand_binder_local_pipeline",
            "ckpt_name": "complexa_ligand.ckpt",
            "ae_ckpt_name": "complexa_ligand_ae.ckpt",
            "weights_dir": "/app/proteina-complexa/ckpts",
            "design_specs": {
                "trypsin": {
                    "input": "/workspace/inputs/target.pdb",
                    "target_input": "A1-223",
                    "ligand": "BEN",
                },
            },
            "n_batches": 1,
            "out_dir": "/workspace/outputs/raw",
            "seed": 42,
        }
        assert cfg == expected
        assert list(cfg.keys()) == list(expected.keys())

    def test_ame_full_config(self, config: AutobioConfig, sample_pdb: Path, tmp_path: Path) -> None:
        r = _make_runner("ame", config)
        workspace = Workspace.create(tmp_path / "ws")
        input_data = ComplexaInput(
            input_structures=[sample_pdb],
            design_specs={
                "motif": {
                    "input": str(sample_pdb),
                    "contig_atoms": "A5: [N, CA, C, O, CB]",
                },
            },
            n_batches=1,
        )
        r.prepare_workspace(input_data, workspace)

        cfg = json.loads(workspace.config_path.read_text())
        expected = {
            "variant": "ame",
            "pipeline_config": "search_ame_local_pipeline",
            "ckpt_name": "complexa_ame.ckpt",
            "ae_ckpt_name": "complexa_ame_ae.ckpt",
            "weights_dir": "/app/proteina-complexa/ckpts",
            "design_specs": {
                "motif": {
                    "input": "/workspace/inputs/target.pdb",
                    "contig_atoms": "A5: [N, CA, C, O, CB]",
                },
            },
            "n_batches": 1,
            "out_dir": "/workspace/outputs/raw",
        }
        assert cfg == expected
        assert list(cfg.keys()) == list(expected.keys())

    def test_extra_shadowing_typed_field_rejected(
        self, runner: ComplexaRunner, tmp_path: Path
    ) -> None:
        """extra containing a typed field name (n_batches) raises."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = ComplexaInput(
            design_specs={"test": {"target_input": "A1-50"}},
            extra={"n_batches": 5},
        )
        with pytest.raises(AutobioError, match="collide with typed input fields"):
            runner.prepare_workspace(input_data, workspace)

    def test_extra_unknown_key_passed_through(self, runner: ComplexaRunner, tmp_path: Path) -> None:
        workspace = Workspace.create(tmp_path / "ws")
        input_data = ComplexaInput(
            design_specs={"test": {"target_input": "A1-50"}},
            extra={"custom_flag": True},
        )
        runner.prepare_workspace(input_data, workspace)

        cfg = json.loads(workspace.config_path.read_text())
        assert cfg["custom_flag"] is True


# ---------------------------------------------------------------------------
# TestComplexaRegistration
# ---------------------------------------------------------------------------


class TestComplexaRegistration:
    """Tests for the catalog Tool + runner registration."""

    def test_complexa_registered_as_single_tool(self) -> None:
        import autobio.tools  # noqa: F401 - populate registries

        tool = get_tool("complexa")
        assert set(tool.modes) == {"protein_binder", "ligand_binder", "ame"}
        assert tool.default_mode == "protein_binder"
        assert tool.category == ToolCategory.STRUCTURE_DESIGN
        assert tool.requires_gpu is True
        assert tool.gpu_count == 1

    @pytest.mark.parametrize("flat_name", _RETIRED_RUNNER_KEYS)
    def test_retired_flat_names_absent_from_tool_runners(self, flat_name: str) -> None:
        import autobio.tools  # noqa: F401

        assert flat_name not in TOOL_RUNNERS

    def test_complexa_in_tool_runners(self) -> None:
        import autobio.tools  # noqa: F401

        assert "complexa" in TOOL_RUNNERS
        assert TOOL_RUNNERS["complexa"] is ComplexaRunner

    def test_get_runner_complexa_resolves_catalog_tool(self, config: AutobioConfig) -> None:
        with (
            patch("autobio.tools.base.ContainerManager"),
            patch("autobio.tools.base.GPUManager"),
        ):
            r = get_runner("complexa", config)
        assert isinstance(r, ComplexaRunner)
        assert r.tool_name == "complexa"
        assert r.tool is not None and r.tool.name == "complexa"

    @pytest.mark.parametrize("flat_name", _RETIRED_RUNNER_KEYS)
    def test_get_runner_removed_flat_name_raises(
        self, flat_name: str, config: AutobioConfig
    ) -> None:
        with pytest.raises(KeyError, match=flat_name):
            get_runner(flat_name, config)

    @pytest.mark.parametrize(
        ("mode_name", "timeout"),
        [("protein_binder", 43200), ("ligand_binder", 43200), ("ame", 43200)],
    )
    def test_modes_have_uniform_timeout(self, mode_name: str, timeout: int) -> None:
        import autobio.tools  # noqa: F401

        assert get_tool("complexa").modes[mode_name].default_timeout == timeout

    def test_modes_share_single_image(self) -> None:
        import autobio.tools  # noqa: F401

        tool = get_tool("complexa")
        assert tool.image_tag == "complexa:2.0.0"
        for mode in tool.modes.values():
            assert mode.image_tag is None  # falls back to Tool.image_tag

    @pytest.mark.parametrize("mode_name", _MODE_NAMES)
    def test_supports_batch(self, mode_name: str) -> None:
        import autobio.tools  # noqa: F401

        assert get_tool("complexa").modes[mode_name].supports_batch is True

    @pytest.mark.parametrize("mode_name", _MODE_NAMES)
    def test_notes_populated(self, mode_name: str) -> None:
        import autobio.tools  # noqa: F401

        notes = get_tool("complexa").modes[mode_name].notes
        assert len(notes) > 0
        all_notes = " ".join(notes).lower()
        assert "gpu" in all_notes or "search" in all_notes

    @pytest.mark.parametrize("mode_name", _MODE_NAMES)
    def test_design_mode_documented_in_notes(self, mode_name: str) -> None:
        """All modes document the container-level design/generate pipeline switch."""
        notes = get_tool("complexa").modes[mode_name].notes
        all_notes = " ".join(notes).lower()
        assert "design" in all_notes
        assert "mode" in all_notes


# ---------------------------------------------------------------------------
# TestComplexaInfoSnapshot
# ---------------------------------------------------------------------------


class TestComplexaInfoSnapshot:
    """``autobio info complexa`` output — per-mode notes, hints, output_schema."""

    def test_info_snapshot(self) -> None:
        import autobio.tools  # noqa: F401
        from autobio.cli.formatters import OutputFormat, format_tool_info_catalog

        parsed = json.loads(format_tool_info_catalog(get_tool("complexa"), OutputFormat.JSON))
        assert [m["name"] for m in parsed["modes"]] == ["protein_binder", "ligand_binder", "ame"]

        for mode in parsed["modes"]:
            assert len(mode["notes"]) > 0
            assert "output_schema" in mode
            design_specs_prop = mode["input_schema"]["properties"]["design_specs"]
            assert design_specs_prop["x-autobio"]["widget"] == "textarea"

        # Only the output-format note text differs per mode.
        assert parsed["modes"][0]["notes"] != parsed["modes"][1]["notes"]
        assert parsed["modes"][0]["notes"] != parsed["modes"][2]["notes"]
