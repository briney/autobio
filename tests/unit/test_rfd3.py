"""Tests for RFD3Runner — prepare_workspace, parse_output, host validation, and registration."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

from autobio.core.catalog import CATALOG, get_tool
from autobio.core.config import AutobioConfig
from autobio.core.registry import TOOL_REGISTRY, ToolCategory
from autobio.core.result import AutobioError
from autobio.core.workspace import Workspace
from autobio.schemas.structure_design import RFD3Input, StructureDesignOutput
from autobio.tools import TOOL_RUNNERS, get_runner
from autobio.tools.rfd3 import RFD3_TOOL, RFD3Runner

if TYPE_CHECKING:
    from pathlib import Path


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def config() -> AutobioConfig:
    return AutobioConfig.resolve()


def _make_runner(config: AutobioConfig) -> RFD3Runner:
    """Create an RFD3Runner with mocked ContainerManager/GPUManager and current_mode set.

    ``current_mode`` is set directly (rather than via ``run()``) so that
    ``prepare_workspace`` — which calls ``_apply_extra`` — can be exercised
    in isolation.
    """
    with (
        patch("autobio.tools.base.ContainerManager"),
        patch("autobio.tools.base.GPUManager"),
    ):
        runner = RFD3Runner("rfd3", config)
    runner.current_mode = get_tool("rfd3").modes["generate"]
    return runner


@pytest.fixture()
def runner(config: AutobioConfig) -> RFD3Runner:
    """Create an RFD3Runner with mocked ContainerManager and GPUManager."""
    return _make_runner(config)


@pytest.fixture()
def sample_pdb(tmp_path: Path) -> Path:
    """Write a minimal PDB file for testing."""
    pdb_path = tmp_path / "target.pdb"
    pdb_path.write_text("ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00\nEND\n")
    return pdb_path


@pytest.fixture()
def sample_cif(tmp_path: Path) -> Path:
    """Write a minimal mmCIF file for testing."""
    cif_path = tmp_path / "target.cif"
    cif_path.write_text("data_test\n_entry.id test\n")
    return cif_path


# ---------------------------------------------------------------------------
# TestRFD3PrepareWorkspace
# ---------------------------------------------------------------------------


class TestRFD3PrepareWorkspace:
    """Tests for RFD3Runner.prepare_workspace."""

    def test_design_specs_written_to_config(
        self, runner: RFD3Runner, tmp_path: Path, sample_pdb: Path
    ) -> None:
        workspace = Workspace.create(tmp_path / "ws")
        input_data = RFD3Input(
            input_structures=[sample_pdb],
            design_specs={"binder": {"input": str(sample_pdb), "contig": "40-80"}},
        )
        runner.prepare_workspace(input_data, workspace)

        cfg = json.loads(workspace.config_path.read_text())
        assert "design_specs" in cfg
        assert "binder" in cfg["design_specs"]
        assert cfg["design_specs"]["binder"]["contig"] == "40-80"

    def test_input_structures_copied(
        self, runner: RFD3Runner, tmp_path: Path, sample_pdb: Path
    ) -> None:
        workspace = Workspace.create(tmp_path / "ws")
        input_data = RFD3Input(
            input_structures=[sample_pdb],
            design_specs={"test": {"input": str(sample_pdb), "length": "50"}},
        )
        runner.prepare_workspace(input_data, workspace)

        copied = workspace.inputs_dir / "target.pdb"
        assert copied.exists()
        assert copied.read_text() == sample_pdb.read_text()

    def test_input_paths_rewritten(
        self, runner: RFD3Runner, tmp_path: Path, sample_pdb: Path
    ) -> None:
        """'input' values in specs are rewritten to container-internal paths."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = RFD3Input(
            input_structures=[sample_pdb],
            design_specs={"test": {"input": str(sample_pdb), "length": "50"}},
        )
        runner.prepare_workspace(input_data, workspace)

        cfg = json.loads(workspace.config_path.read_text())
        assert cfg["design_specs"]["test"]["input"] == "/workspace/inputs/target.pdb"

    def test_n_batches_in_config(self, runner: RFD3Runner, tmp_path: Path) -> None:
        workspace = Workspace.create(tmp_path / "ws")
        input_data = RFD3Input(
            design_specs={"test": {"length": "50"}},
            n_batches=3,
        )
        runner.prepare_workspace(input_data, workspace)

        cfg = json.loads(workspace.config_path.read_text())
        assert cfg["n_batches"] == 3

    def test_out_dir_set(self, runner: RFD3Runner, tmp_path: Path) -> None:
        workspace = Workspace.create(tmp_path / "ws")
        input_data = RFD3Input(
            design_specs={"test": {"length": "50"}},
        )
        runner.prepare_workspace(input_data, workspace)

        cfg = json.loads(workspace.config_path.read_text())
        assert cfg["out_dir"] == "/workspace/outputs/raw"

    def test_extra_dict_merged(self, runner: RFD3Runner, tmp_path: Path) -> None:
        """Extra dict keys appear at the top level of config.json."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = RFD3Input(
            design_specs={"test": {"length": "50"}},
            extra={"step_scale": 3.0, "gamma_0": 0.2, "low_memory_mode": True},
        )
        runner.prepare_workspace(input_data, workspace)

        cfg = json.loads(workspace.config_path.read_text())
        assert cfg["step_scale"] == 3.0
        assert cfg["gamma_0"] == 0.2
        assert cfg["low_memory_mode"] is True

    def test_multiple_specs(self, runner: RFD3Runner, tmp_path: Path, sample_pdb: Path) -> None:
        """Multiple named specs are preserved in config."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = RFD3Input(
            input_structures=[sample_pdb],
            design_specs={
                "binder_a": {"input": str(sample_pdb), "contig": "40-80,/0,A1-50"},
                "binder_b": {"input": str(sample_pdb), "contig": "60-100,/0,A1-50"},
            },
        )
        runner.prepare_workspace(input_data, workspace)

        cfg = json.loads(workspace.config_path.read_text())
        assert len(cfg["design_specs"]) == 2
        assert "binder_a" in cfg["design_specs"]
        assert "binder_b" in cfg["design_specs"]

    def test_unconditioned_design_no_input_structures(
        self, runner: RFD3Runner, tmp_path: Path
    ) -> None:
        """Unconditioned design requires no input structures."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = RFD3Input(
            design_specs={"uncond": {"length": "100", "is_non_loopy": True}},
        )
        runner.prepare_workspace(input_data, workspace)

        cfg = json.loads(workspace.config_path.read_text())
        assert "input" not in cfg["design_specs"]["uncond"]

    def test_original_specs_not_mutated(
        self, runner: RFD3Runner, tmp_path: Path, sample_pdb: Path
    ) -> None:
        """prepare_workspace deep-copies specs; original input_data is unchanged."""
        workspace = Workspace.create(tmp_path / "ws")
        original_path = str(sample_pdb)
        input_data = RFD3Input(
            input_structures=[sample_pdb],
            design_specs={"test": {"input": original_path, "length": "50"}},
        )
        runner.prepare_workspace(input_data, workspace)

        # Original input_data should still reference host path
        assert input_data.design_specs["test"]["input"] == original_path

    def test_config_full_dict_equality_minimal(self, runner: RFD3Runner, tmp_path: Path) -> None:
        """Full config.json dict is byte-compat with the pre-migration output (minimal input)."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = RFD3Input(design_specs={"uncond": {"length": "100"}})
        runner.prepare_workspace(input_data, workspace)

        cfg = json.loads(workspace.config_path.read_text())
        assert cfg == {
            "design_specs": {"uncond": {"length": "100"}},
            "n_batches": 1,
            "out_dir": "/workspace/outputs/raw",
        }

    def test_config_full_dict_equality_with_overrides(
        self, runner: RFD3Runner, tmp_path: Path, sample_pdb: Path
    ) -> None:
        """Full config.json dict with a rewritten input path, n_batches, and extra CLI knobs."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = RFD3Input(
            input_structures=[sample_pdb],
            design_specs={"binder": {"input": str(sample_pdb), "contig": "40-80"}},
            n_batches=2,
            extra={"step_scale": 3.0, "gamma_0": 0.2},
        )
        runner.prepare_workspace(input_data, workspace)

        cfg = json.loads(workspace.config_path.read_text())
        assert cfg == {
            "design_specs": {
                "binder": {"input": "/workspace/inputs/target.pdb", "contig": "40-80"}
            },
            "n_batches": 2,
            "out_dir": "/workspace/outputs/raw",
            "step_scale": 3.0,
            "gamma_0": 0.2,
        }


# ---------------------------------------------------------------------------
# TestRFD3HostValidation
# ---------------------------------------------------------------------------


class TestRFD3HostValidation:
    """Tests for host-side input validation in prepare_workspace."""

    def test_empty_design_specs_raises(self, runner: RFD3Runner, tmp_path: Path) -> None:
        workspace = Workspace.create(tmp_path / "ws")
        input_data = RFD3Input(design_specs={})
        with pytest.raises(AutobioError, match="at least one specification"):
            runner.prepare_workspace(input_data, workspace)

    def test_non_dict_spec_value_raises(self, runner: RFD3Runner, tmp_path: Path) -> None:
        """Pydantic rejects non-dict spec values at schema validation time."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="valid dictionary"):
            RFD3Input(
                design_specs={"bad": "not a dict"},  # type: ignore[dict-item]
            )

    def test_n_batches_zero_raises(self, runner: RFD3Runner, tmp_path: Path) -> None:
        workspace = Workspace.create(tmp_path / "ws")
        input_data = RFD3Input(
            design_specs={"test": {"length": "50"}},
            n_batches=0,
        )
        with pytest.raises(AutobioError, match="n_batches must be at least 1"):
            runner.prepare_workspace(input_data, workspace)

    def test_negative_n_batches_raises(self, runner: RFD3Runner, tmp_path: Path) -> None:
        workspace = Workspace.create(tmp_path / "ws")
        input_data = RFD3Input(
            design_specs={"test": {"length": "50"}},
            n_batches=-1,
        )
        with pytest.raises(AutobioError, match="n_batches must be at least 1"):
            runner.prepare_workspace(input_data, workspace)

    def test_missing_input_file_raises(self, runner: RFD3Runner, tmp_path: Path) -> None:
        workspace = Workspace.create(tmp_path / "ws")
        nonexistent = tmp_path / "nonexistent.pdb"
        input_data = RFD3Input(
            input_structures=[nonexistent],
            design_specs={"test": {"input": str(nonexistent), "length": "50"}},
        )
        with pytest.raises(AutobioError, match="does not exist"):
            runner.prepare_workspace(input_data, workspace)

    def test_unreferenced_spec_input_raises(self, runner: RFD3Runner, tmp_path: Path) -> None:
        """Spec references a file not in input_structures."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = RFD3Input(
            input_structures=[],  # no files provided
            design_specs={"test": {"input": "/some/file.pdb", "length": "50"}},
        )
        with pytest.raises(AutobioError, match="no matching file"):
            runner.prepare_workspace(input_data, workspace)


# ---------------------------------------------------------------------------
# TestRFD3ExtraShadowRejection
# ---------------------------------------------------------------------------


class TestRFD3ExtraShadowRejection:
    """Promoted keys must not be accepted via extra — _apply_extra rejects them."""

    def test_n_batches_via_extra_rejected(self, runner: RFD3Runner, tmp_path: Path) -> None:
        """n_batches is a typed field — passing it via extra collides."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = RFD3Input(
            design_specs={"test": {"length": "50"}},
            extra={"n_batches": 5},
        )
        with pytest.raises(AutobioError, match="collide"):
            runner.prepare_workspace(input_data, workspace)

    def test_out_dir_via_extra_rejected(self, runner: RFD3Runner, tmp_path: Path) -> None:
        """out_dir is a runner-derived config key — passing it via extra collides."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = RFD3Input(
            design_specs={"test": {"length": "50"}},
            extra={"out_dir": "/workspace/outputs/other"},
        )
        with pytest.raises(AutobioError, match="collide"):
            runner.prepare_workspace(input_data, workspace)


# ---------------------------------------------------------------------------
# TestRFD3ParseOutput
# ---------------------------------------------------------------------------

_SINGLE_DESIGN_RESULT = {
    "designs": [
        {
            "spec_name": "test",
            "batch_index": 0,
            "design_index": 0,
            "structure_path": "/workspace/outputs/standardized/test_b0_d0.cif",
            "diffusion_metadata": {"timing": {"total_seconds": 42.5}},
        },
    ],
    "spec_summary": {"test": 1},
}

_MULTI_DESIGN_RESULT = {
    "designs": [
        {
            "spec_name": "binder",
            "batch_index": 0,
            "design_index": 0,
            "structure_path": "/workspace/outputs/standardized/binder_b0_d0.cif",
            "diffusion_metadata": None,
        },
        {
            "spec_name": "binder",
            "batch_index": 0,
            "design_index": 1,
            "structure_path": "/workspace/outputs/standardized/binder_b0_d1.cif",
            "diffusion_metadata": None,
        },
        {
            "spec_name": "enzyme",
            "batch_index": 0,
            "design_index": 0,
            "structure_path": "/workspace/outputs/standardized/enzyme_b0_d0.cif",
            "diffusion_metadata": {"residue_mapping": {"designed": [1, 2, 3]}},
        },
    ],
    "spec_summary": {"binder": 2, "enzyme": 1},
}


class TestRFD3ParseOutput:
    """Tests for RFD3Runner.parse_output."""

    def test_parse_single_design(self, runner: RFD3Runner, tmp_path: Path) -> None:
        workspace = Workspace.create(tmp_path / "ws")
        (workspace.std_output_dir / "result_data.json").write_text(
            json.dumps(_SINGLE_DESIGN_RESULT)
        )

        output = runner.parse_output(workspace)
        assert len(output.designs) == 1
        d = output.designs[0]
        assert d.spec_name == "test"
        assert d.batch_index == 0
        assert d.design_index == 0
        assert d.diffusion_metadata is not None
        assert d.diffusion_metadata["timing"]["total_seconds"] == 42.5

    def test_parse_multiple_designs(self, runner: RFD3Runner, tmp_path: Path) -> None:
        workspace = Workspace.create(tmp_path / "ws")
        (workspace.std_output_dir / "result_data.json").write_text(json.dumps(_MULTI_DESIGN_RESULT))

        output = runner.parse_output(workspace)
        assert len(output.designs) == 3

    def test_parse_multi_spec(self, runner: RFD3Runner, tmp_path: Path) -> None:
        workspace = Workspace.create(tmp_path / "ws")
        (workspace.std_output_dir / "result_data.json").write_text(json.dumps(_MULTI_DESIGN_RESULT))

        output = runner.parse_output(workspace)
        spec_names = {d.spec_name for d in output.designs}
        assert spec_names == {"binder", "enzyme"}

    def test_spec_summary_correct(self, runner: RFD3Runner, tmp_path: Path) -> None:
        workspace = Workspace.create(tmp_path / "ws")
        (workspace.std_output_dir / "result_data.json").write_text(json.dumps(_MULTI_DESIGN_RESULT))

        output = runner.parse_output(workspace)
        assert output.spec_summary == {"binder": 2, "enzyme": 1}

    def test_spec_summary_computed_from_designs(self, runner: RFD3Runner, tmp_path: Path) -> None:
        """If spec_summary is missing from JSON, compute it from designs."""
        data = {**_MULTI_DESIGN_RESULT, "spec_summary": {}}
        workspace = Workspace.create(tmp_path / "ws")
        (workspace.std_output_dir / "result_data.json").write_text(json.dumps(data))

        output = runner.parse_output(workspace)
        assert output.spec_summary == {"binder": 2, "enzyme": 1}

    def test_diffusion_metadata_preserved(self, runner: RFD3Runner, tmp_path: Path) -> None:
        workspace = Workspace.create(tmp_path / "ws")
        (workspace.std_output_dir / "result_data.json").write_text(json.dumps(_MULTI_DESIGN_RESULT))

        output = runner.parse_output(workspace)
        enzyme = [d for d in output.designs if d.spec_name == "enzyme"][0]
        assert enzyme.diffusion_metadata is not None
        assert enzyme.diffusion_metadata["residue_mapping"]["designed"] == [1, 2, 3]

    def test_output_type(self, runner: RFD3Runner, tmp_path: Path) -> None:
        workspace = Workspace.create(tmp_path / "ws")
        (workspace.std_output_dir / "result_data.json").write_text(
            json.dumps(_SINGLE_DESIGN_RESULT)
        )

        output = runner.parse_output(workspace)
        assert isinstance(output, StructureDesignOutput)

    def test_raw_output_path(self, runner: RFD3Runner, tmp_path: Path) -> None:
        workspace = Workspace.create(tmp_path / "ws")
        (workspace.std_output_dir / "result_data.json").write_text(
            json.dumps(_SINGLE_DESIGN_RESULT)
        )

        output = runner.parse_output(workspace)
        assert output.raw_output_path == workspace.raw_output_dir


# ---------------------------------------------------------------------------
# TestRFD3Registration
# ---------------------------------------------------------------------------


class TestRFD3Registration:
    """Tests for catalog Tool and runner registration."""

    def test_rfd3_registered_as_catalog_tool(self) -> None:
        import autobio.tools  # noqa: F401 - importing populates the catalog

        assert "rfd3" in CATALOG
        assert set(get_tool("rfd3").modes) == {"generate"}
        assert get_tool("rfd3").default_mode == "generate"
        assert get_tool("rfd3").category == ToolCategory.STRUCTURE_DESIGN
        assert get_tool("rfd3").requires_gpu is True
        assert get_tool("rfd3").gpu_count == 1
        assert "rfd3" not in TOOL_REGISTRY

    def test_rfd3_tool_constant_registered(self) -> None:
        import autobio.tools  # noqa: F401 - importing populates the catalog

        assert RFD3_TOOL.name == "rfd3"
        assert get_tool("rfd3") is RFD3_TOOL

    def test_tool_runner_registered(self) -> None:
        assert "rfd3" in TOOL_RUNNERS
        assert TOOL_RUNNERS["rfd3"] is RFD3Runner

    def test_get_runner_returns_rfd3_runner(self, config: AutobioConfig) -> None:
        with (
            patch("autobio.tools.base.ContainerManager"),
            patch("autobio.tools.base.GPUManager"),
        ):
            r = get_runner("rfd3", config)
        assert isinstance(r, RFD3Runner)
        assert r.tool_name == "rfd3"

    def test_notes_populated(self) -> None:
        """Notes contain key operational topics for agent guidance."""
        notes = " ".join(get_tool("rfd3").modes["generate"].notes)
        assert "designability" in notes.lower() or "diversity" in notes.lower()

    def test_default_timeout(self) -> None:
        assert get_tool("rfd3").modes["generate"].default_timeout == 3600

    def test_supports_batch(self) -> None:
        assert get_tool("rfd3").modes["generate"].supports_batch is True


def test_info_snapshot_rfd3() -> None:
    import autobio.tools  # noqa: F401 - importing populates the catalog
    from autobio.cli.formatters import OutputFormat, format_tool_info_catalog

    parsed = json.loads(format_tool_info_catalog(get_tool("rfd3"), OutputFormat.JSON))
    assert parsed["modes"][0]["name"] == "generate"
    props = parsed["modes"][0]["input_schema"]["properties"]
    assert props["design_specs"]["x-autobio"]["widget"] == "textarea"
    assert "output_schema" in parsed["modes"][0]
    assert parsed["modes"][0]["notes"]
