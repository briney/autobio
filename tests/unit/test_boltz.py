"""Tests for BoltzRunner — prepare_workspace, parse_output, host validation, and registration."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest
import yaml

from autobio.core.catalog import CATALOG, get_tool
from autobio.core.config import AutobioConfig
from autobio.core.registry import ToolCategory
from autobio.core.result import AutobioError
from autobio.core.workspace import Workspace
from autobio.schemas.structure_prediction import BoltzInput, StructurePredictionOutput
from autobio.tools import TOOL_RUNNERS, get_runner
from autobio.tools.boltz import BOLTZ1_TOOL, BOLTZ2_TOOL, BoltzRunner

if TYPE_CHECKING:
    from pathlib import Path


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def config() -> AutobioConfig:
    return AutobioConfig.resolve()


def _make_runner(tool_name: str, config: AutobioConfig) -> BoltzRunner:
    """Create a BoltzRunner with mocked ContainerManager/GPUManager and current_mode set.

    ``current_mode`` is set directly (rather than via ``run()``) so that
    ``prepare_workspace`` — which calls ``_apply_extra`` — can be exercised
    in isolation.
    """
    with (
        patch("autobio.tools.base.ContainerManager"),
        patch("autobio.tools.base.GPUManager"),
    ):
        runner = BoltzRunner(tool_name, config)
    runner.current_mode = get_tool(tool_name).modes["predict"]
    return runner


@pytest.fixture()
def runner(config: AutobioConfig) -> BoltzRunner:
    """Create a BoltzRunner (boltz2) with mocked ContainerManager and GPUManager."""
    return _make_runner("boltz2", config)


@pytest.fixture()
def boltz1_runner(config: AutobioConfig) -> BoltzRunner:
    """Create a BoltzRunner for boltz1."""
    return _make_runner("boltz1", config)


# ---------------------------------------------------------------------------
# TestBoltzPrepareWorkspace
# ---------------------------------------------------------------------------


class TestBoltzPrepareWorkspace:
    """Tests for BoltzRunner.prepare_workspace."""

    @pytest.mark.parametrize(
        ("tool_name", "expected_model"),
        [
            ("boltz1", "boltz1"),
            ("boltz2", "boltz2"),
        ],
    )
    def test_model_config_per_tool(
        self,
        tool_name: str,
        expected_model: str,
        config: AutobioConfig,
        tmp_path: Path,
    ) -> None:
        """Config contains correct model flag per tool name."""
        r = _make_runner(tool_name, config)
        workspace = Workspace.create(tmp_path / "ws")
        input_data = BoltzInput(sequences={"A": "MVLSPADKTNVKAAWGKVGA"})
        r.prepare_workspace(input_data, workspace)

        cfg = json.loads(workspace.config_path.read_text())
        assert cfg["model"] == expected_model

    def test_yaml_generated_from_sequences(self, runner: BoltzRunner, tmp_path: Path) -> None:
        """Sequences dict is translated into a Boltz YAML input file."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = BoltzInput(sequences={"A": "MKWVTFIS", "B": "GVSEKL"})
        runner.prepare_workspace(input_data, workspace)

        yaml_path = workspace.inputs_dir / "input.yaml"
        assert yaml_path.exists()
        yaml_data = yaml.safe_load(yaml_path.read_text())
        assert yaml_data["version"] == 1
        assert len(yaml_data["sequences"]) == 2

        # Check that both chains appear as protein entities
        ids = set()
        for entry in yaml_data["sequences"]:
            assert "protein" in entry
            ids.add(entry["protein"]["id"])
        assert ids == {"A", "B"}

    def test_num_models_maps_to_diffusion_samples(
        self, runner: BoltzRunner, tmp_path: Path
    ) -> None:
        workspace = Workspace.create(tmp_path / "ws")
        input_data = BoltzInput(sequences={"A": "MVLSPADKTNVKAAWGKVGA"}, num_models=5)
        runner.prepare_workspace(input_data, workspace)

        cfg = json.loads(workspace.config_path.read_text())
        assert cfg["diffusion_samples"] == 5

    def test_num_models_default_omits_diffusion_samples(
        self, runner: BoltzRunner, tmp_path: Path
    ) -> None:
        """When num_models=1 (default), diffusion_samples is not in config."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = BoltzInput(sequences={"A": "MVLSPADKTNVKAAWGKVGA"})
        runner.prepare_workspace(input_data, workspace)

        cfg = json.loads(workspace.config_path.read_text())
        assert "diffusion_samples" not in cfg

    def test_use_msa_server_default_true(self, runner: BoltzRunner, tmp_path: Path) -> None:
        """use_msa_server defaults to True in config."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = BoltzInput(sequences={"A": "MVLSPADKTNVKAAWGKVGA"})
        runner.prepare_workspace(input_data, workspace)

        cfg = json.loads(workspace.config_path.read_text())
        assert cfg["use_msa_server"] is True

    def test_use_msa_server_can_be_disabled(self, runner: BoltzRunner, tmp_path: Path) -> None:
        """The top-level use_msa_server field overrides the default."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = BoltzInput(
            sequences={"A": "MVLSPADKTNVKAAWGKVGA"},
            use_msa_server=False,
        )
        runner.prepare_workspace(input_data, workspace)

        cfg = json.loads(workspace.config_path.read_text())
        assert cfg["use_msa_server"] is False

    def test_templates_copied(self, runner: BoltzRunner, tmp_path: Path) -> None:
        """Template files are copied into workspace/inputs/."""
        tmpl = tmp_path / "template.cif"
        tmpl.write_text("data_test\n_entry.id test\n")

        workspace = Workspace.create(tmp_path / "ws")
        input_data = BoltzInput(
            sequences={"A": "MVLSPADKTNVKAAWGKVGA"},
            templates=[tmpl],
        )
        runner.prepare_workspace(input_data, workspace)

        copied = workspace.inputs_dir / "template.cif"
        assert copied.exists()

        # YAML should reference the template with container path
        yaml_data = yaml.safe_load((workspace.inputs_dir / "input.yaml").read_text())
        assert "templates" in yaml_data
        assert yaml_data["templates"][0]["cif"] == "/workspace/inputs/template.cif"

    def test_entity_types_override(self, runner: BoltzRunner, tmp_path: Path) -> None:
        """Non-protein entity types are correctly tagged in the YAML."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = BoltzInput(
            sequences={"A": "MVLSPADKTNVKAAWGKVGA", "B": "ATCGATCG"},
            entity_types={"B": "dna"},
        )
        runner.prepare_workspace(input_data, workspace)

        yaml_data = yaml.safe_load((workspace.inputs_dir / "input.yaml").read_text())
        types = {}
        for entry in yaml_data["sequences"]:
            for etype, edata in entry.items():
                types[edata["id"]] = etype
        assert types["A"] == "protein"
        assert types["B"] == "dna"

    def test_ligand_smiles_entity(self, runner: BoltzRunner, tmp_path: Path) -> None:
        """Ligand entities with SMILES are correctly constructed."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = BoltzInput(
            sequences={"A": "MVLSPADKTNVKAAWGKVGA", "L": ""},
            entity_types={"L": {"smiles": "CC(=O)NC1=CC=C(O)C=C1"}},
        )
        runner.prepare_workspace(input_data, workspace)

        yaml_data = yaml.safe_load((workspace.inputs_dir / "input.yaml").read_text())
        ligand_entry = None
        for entry in yaml_data["sequences"]:
            if "ligand" in entry:
                ligand_entry = entry["ligand"]
        assert ligand_entry is not None
        assert ligand_entry["id"] == "L"
        assert ligand_entry["smiles"] == "CC(=O)NC1=CC=C(O)C=C1"

    def test_ligand_ccd_entity(self, runner: BoltzRunner, tmp_path: Path) -> None:
        """Ligand entities with CCD codes are correctly constructed."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = BoltzInput(
            sequences={"A": "MVLSPADKTNVKAAWGKVGA", "L": ""},
            entity_types={"L": {"ccd": "ATP"}},
        )
        runner.prepare_workspace(input_data, workspace)

        yaml_data = yaml.safe_load((workspace.inputs_dir / "input.yaml").read_text())
        ligand_entry = None
        for entry in yaml_data["sequences"]:
            if "ligand" in entry:
                ligand_entry = entry["ligand"]
        assert ligand_entry is not None
        assert ligand_entry["ccd"] == "ATP"

    def test_raw_yaml_passthrough(self, runner: BoltzRunner, tmp_path: Path) -> None:
        """boltz_yaml bypasses automatic YAML generation."""
        custom_yaml = {
            "version": 1,
            "sequences": [
                {"protein": {"id": "X", "sequence": "MKWVTFIS"}},
            ],
            "constraints": [{"bond": {"atom1": ["X", 1, "CA"], "atom2": ["X", 5, "CA"]}}],
        }
        workspace = Workspace.create(tmp_path / "ws")
        input_data = BoltzInput(
            sequences={},
            boltz_yaml=custom_yaml,
        )
        runner.prepare_workspace(input_data, workspace)

        yaml_data = yaml.safe_load((workspace.inputs_dir / "input.yaml").read_text())
        assert yaml_data["version"] == 1
        assert yaml_data["sequences"][0]["protein"]["id"] == "X"
        assert "constraints" in yaml_data

    def test_extra_dict_merged(self, runner: BoltzRunner, tmp_path: Path) -> None:
        """CLI-level extra keys appear in config.json."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = BoltzInput(
            sequences={"A": "MVLSPADKTNVKAAWGKVGA"},
            entity_types={"A": "protein"},
            extra={
                "sampling_steps": 100,
                "seed": 42,
                "step_scale": 1.5,
            },
        )
        runner.prepare_workspace(input_data, workspace)

        cfg = json.loads(workspace.config_path.read_text())
        assert cfg["sampling_steps"] == 100
        assert cfg["seed"] == 42
        assert cfg["step_scale"] == 1.5
        # entity_types is a typed field, not a config.json key at all
        assert "entity_types" not in cfg

    def test_boltz2_affinity_extras_merged(self, runner: BoltzRunner, tmp_path: Path) -> None:
        """Boltz-2-only affinity CLI knobs flat-merge through extra like any other."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = BoltzInput(
            sequences={"A": "MVLSPADKTNVKAAWGKVGA"},
            extra={
                "sampling_steps_affinity": 100,
                "diffusion_samples_affinity": 3,
                "method": "crystal",
            },
        )
        runner.prepare_workspace(input_data, workspace)

        cfg = json.loads(workspace.config_path.read_text())
        assert cfg["sampling_steps_affinity"] == 100
        assert cfg["diffusion_samples_affinity"] == 3
        assert cfg["method"] == "crystal"

    def test_defaults_applied(self, runner: BoltzRunner, tmp_path: Path) -> None:
        """Minimal input produces sensible defaults."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = BoltzInput(sequences={"A": "MVLSPADKTNVKAAWGKVGA"})
        runner.prepare_workspace(input_data, workspace)

        cfg = json.loads(workspace.config_path.read_text())
        assert cfg["model"] == "boltz2"
        assert cfg["input_path"] == "/workspace/inputs/input.yaml"
        assert cfg["output_dir"] == "/workspace/outputs/raw"
        assert cfg["cache_dir"] == "/app/boltz/cache"
        assert cfg["use_msa_server"] is True

    def test_cache_dir_in_config(self, runner: BoltzRunner, tmp_path: Path) -> None:
        workspace = Workspace.create(tmp_path / "ws")
        input_data = BoltzInput(sequences={"A": "MVLSPADKTNVKAAWGKVGA"})
        runner.prepare_workspace(input_data, workspace)

        cfg = json.loads(workspace.config_path.read_text())
        assert cfg["cache_dir"] == "/app/boltz/cache"

    def test_msa_paths_copied(self, runner: BoltzRunner, tmp_path: Path) -> None:
        """MSA files are copied and paths rewritten in the YAML."""
        msa_file = tmp_path / "A.a3m"
        msa_file.write_text(">A\nMVLSPADKTNVKAAWGKVGA\n")

        workspace = Workspace.create(tmp_path / "ws")
        input_data = BoltzInput(
            sequences={"A": "MVLSPADKTNVKAAWGKVGA"},
            msa_paths=[str(msa_file)],
        )
        runner.prepare_workspace(input_data, workspace)

        # File copied
        assert (workspace.inputs_dir / "A.a3m").exists()

        # YAML references container path
        yaml_data = yaml.safe_load((workspace.inputs_dir / "input.yaml").read_text())
        protein_entry = yaml_data["sequences"][0]["protein"]
        assert protein_entry["msa"] == "/workspace/inputs/A.a3m"

    def test_constraints_in_yaml(self, runner: BoltzRunner, tmp_path: Path) -> None:
        """Constraints from the typed field appear in the generated YAML."""
        workspace = Workspace.create(tmp_path / "ws")
        constraints = [{"pocket": {"binder": "A", "contacts": [["B", 10]]}}]
        input_data = BoltzInput(
            sequences={"A": "MVLSPADKTNVKAAWGKVGA", "B": "GVSEKL"},
            constraints=constraints,
        )
        runner.prepare_workspace(input_data, workspace)

        yaml_data = yaml.safe_load((workspace.inputs_dir / "input.yaml").read_text())
        assert "constraints" in yaml_data
        assert yaml_data["constraints"] == constraints

    def test_config_full_dict_equality_num_models_1(
        self, runner: BoltzRunner, tmp_path: Path
    ) -> None:
        """Full config.json dict is byte-compat with the pre-migration output (num_models=1)."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = BoltzInput(sequences={"A": "MVLSPADKTNVKAAWGKVGA"})
        runner.prepare_workspace(input_data, workspace)

        cfg = json.loads(workspace.config_path.read_text())
        assert cfg == {
            "model": "boltz2",
            "input_path": "/workspace/inputs/input.yaml",
            "output_dir": "/workspace/outputs/raw",
            "cache_dir": "/app/boltz/cache",
            "use_msa_server": True,
        }

    def test_config_full_dict_equality_with_overrides(
        self, runner: BoltzRunner, tmp_path: Path
    ) -> None:
        """Full config.json dict with num_models>1, use_msa_server=False, and extra CLI knobs."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = BoltzInput(
            sequences={"A": "MVLSPADKTNVKAAWGKVGA"},
            num_models=3,
            use_msa_server=False,
            extra={"sampling_steps": 100, "seed": 42},
        )
        runner.prepare_workspace(input_data, workspace)

        cfg = json.loads(workspace.config_path.read_text())
        assert cfg == {
            "model": "boltz2",
            "input_path": "/workspace/inputs/input.yaml",
            "output_dir": "/workspace/outputs/raw",
            "cache_dir": "/app/boltz/cache",
            "use_msa_server": False,
            "diffusion_samples": 3,
            "sampling_steps": 100,
            "seed": 42,
        }

    def test_generated_yaml_equality_protein_ligand(
        self, runner: BoltzRunner, tmp_path: Path
    ) -> None:
        """The generated input.yaml is unchanged for a representative protein+ligand input."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = BoltzInput(
            sequences={"A": "MVLSPADKTNVKAAWGKVGA", "L": ""},
            entity_types={"L": {"smiles": "CC(=O)NC1=CC=C(O)C=C1"}},
            constraints=[{"pocket": {"binder": "L", "contacts": [["A", 10]]}}],
        )
        runner.prepare_workspace(input_data, workspace)

        yaml_content = (workspace.inputs_dir / "input.yaml").read_text()
        expected = yaml.dump(
            {
                "version": 1,
                "sequences": [
                    {"protein": {"id": "A", "sequence": "MVLSPADKTNVKAAWGKVGA"}},
                    {"ligand": {"id": "L", "smiles": "CC(=O)NC1=CC=C(O)C=C1"}},
                ],
                "constraints": [{"pocket": {"binder": "L", "contacts": [["A", 10]]}}],
            },
            default_flow_style=False,
            sort_keys=False,
        )
        assert yaml_content == expected


# ---------------------------------------------------------------------------
# TestBoltzHostValidation
# ---------------------------------------------------------------------------


class TestBoltzHostValidation:
    """Tests for host-side input validation."""

    def test_empty_sequences_raises(self, runner: BoltzRunner, tmp_path: Path) -> None:
        workspace = Workspace.create(tmp_path / "ws")
        input_data = BoltzInput(sequences={})
        with pytest.raises(
            AutobioError,
            match="sequences must be non-empty, or provide a raw Boltz YAML via the boltz_yaml "
            "field",
        ):
            runner.prepare_workspace(input_data, workspace)

    def test_empty_sequences_ok_with_boltz_yaml(self, runner: BoltzRunner, tmp_path: Path) -> None:
        """Empty sequences is allowed when boltz_yaml is provided."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = BoltzInput(
            sequences={},
            boltz_yaml={"version": 1, "sequences": []},
        )
        # Should not raise
        runner.prepare_workspace(input_data, workspace)

    def test_missing_template_file_raises(self, runner: BoltzRunner, tmp_path: Path) -> None:
        workspace = Workspace.create(tmp_path / "ws")
        input_data = BoltzInput(
            sequences={"A": "MVLSPADKTNVKAAWGKVGA"},
            templates=[tmp_path / "nonexistent.cif"],
        )
        with pytest.raises(AutobioError, match="Template file does not exist"):
            runner.prepare_workspace(input_data, workspace)

    def test_missing_msa_file_raises(self, runner: BoltzRunner, tmp_path: Path) -> None:
        workspace = Workspace.create(tmp_path / "ws")
        input_data = BoltzInput(
            sequences={"A": "MVLSPADKTNVKAAWGKVGA"},
            msa_paths=[str(tmp_path / "nonexistent.a3m")],
        )
        with pytest.raises(AutobioError, match="MSA file does not exist"):
            runner.prepare_workspace(input_data, workspace)

    def test_entity_types_unknown_chain_raises(self, runner: BoltzRunner, tmp_path: Path) -> None:
        workspace = Workspace.create(tmp_path / "ws")
        input_data = BoltzInput(
            sequences={"A": "MVLSPADKTNVKAAWGKVGA"},
            entity_types={"Z": "dna"},
        )
        with pytest.raises(AutobioError, match="unknown chain IDs"):
            runner.prepare_workspace(input_data, workspace)

    def test_invalid_entity_type_dict_raises(self, runner: BoltzRunner, tmp_path: Path) -> None:
        workspace = Workspace.create(tmp_path / "ws")
        input_data = BoltzInput(
            sequences={"A": "MVLSPADKTNVKAAWGKVGA"},
            entity_types={"A": {"invalid_key": "value"}},
        )
        with pytest.raises(AutobioError, match="Unknown entity type dict"):
            runner.prepare_workspace(input_data, workspace)


# ---------------------------------------------------------------------------
# TestBoltzExtraShadowRejection
# ---------------------------------------------------------------------------


class TestBoltzExtraShadowRejection:
    """Promoted keys must no longer be accepted via extra — _apply_extra rejects them."""

    def test_entity_types_via_extra_rejected(self, runner: BoltzRunner, tmp_path: Path) -> None:
        workspace = Workspace.create(tmp_path / "ws")
        input_data = BoltzInput(
            sequences={"A": "MVLSPADKTNVKAAWGKVGA"},
            extra={"entity_types": {"A": "protein"}},
        )
        with pytest.raises(AutobioError, match="collide with typed input fields"):
            runner.prepare_workspace(input_data, workspace)

    def test_use_msa_server_via_extra_rejected(self, runner: BoltzRunner, tmp_path: Path) -> None:
        workspace = Workspace.create(tmp_path / "ws")
        input_data = BoltzInput(
            sequences={"A": "MVLSPADKTNVKAAWGKVGA"},
            extra={"use_msa_server": False},
        )
        with pytest.raises(AutobioError, match="collide with typed input fields"):
            runner.prepare_workspace(input_data, workspace)

    def test_boltz_yaml_via_extra_rejected(self, runner: BoltzRunner, tmp_path: Path) -> None:
        workspace = Workspace.create(tmp_path / "ws")
        input_data = BoltzInput(
            sequences={"A": "MVLSPADKTNVKAAWGKVGA"},
            extra={"boltz_yaml": {"version": 1, "sequences": []}},
        )
        with pytest.raises(AutobioError, match="collide with typed input fields"):
            runner.prepare_workspace(input_data, workspace)

    def test_msa_paths_via_extra_rejected(self, runner: BoltzRunner, tmp_path: Path) -> None:
        workspace = Workspace.create(tmp_path / "ws")
        input_data = BoltzInput(
            sequences={"A": "MVLSPADKTNVKAAWGKVGA"},
            extra={"msa_paths": ["A.a3m"]},
        )
        with pytest.raises(AutobioError, match="collide with typed input fields"):
            runner.prepare_workspace(input_data, workspace)


# ---------------------------------------------------------------------------
# TestBoltzParseOutput
# ---------------------------------------------------------------------------

_SINGLE_MODEL_RESULT = {
    "structures": [
        {
            "model_rank": 1,
            "structure_path": "/workspace/outputs/standardized/model_1.cif",
            "plddt_per_residue": [92.1, 88.4, 95.0, 91.2],
            "plddt_mean": 91.675,
            "ptm": 0.89,
            "iptm": 0.85,
            "chain_mapping": None,
        }
    ],
    "confidence": {
        "best_plddt_mean": 91.675,
        "best_ptm": 0.89,
        "best_iptm": 0.85,
    },
}

_MULTI_MODEL_RESULT = {
    "structures": [
        {
            "model_rank": 1,
            "structure_path": "/workspace/outputs/standardized/model_1.cif",
            "plddt_per_residue": [92.0, 90.0],
            "plddt_mean": 91.0,
            "ptm": 0.92,
            "iptm": 0.88,
            "chain_mapping": None,
        },
        {
            "model_rank": 2,
            "structure_path": "/workspace/outputs/standardized/model_2.cif",
            "plddt_per_residue": [85.0, 82.0],
            "plddt_mean": 83.5,
            "ptm": 0.80,
            "iptm": 0.75,
            "chain_mapping": None,
        },
        {
            "model_rank": 3,
            "structure_path": "/workspace/outputs/standardized/model_3.cif",
            "plddt_per_residue": [78.0, 76.0],
            "plddt_mean": 77.0,
            "ptm": 0.72,
            "iptm": 0.68,
            "chain_mapping": None,
        },
    ],
    "confidence": {
        "best_plddt_mean": 91.0,
        "best_ptm": 0.92,
        "best_iptm": 0.88,
    },
}

_AFFINITY_RESULT = {
    "structures": [
        {
            "model_rank": 1,
            "structure_path": "/workspace/outputs/standardized/model_1.cif",
            "plddt_per_residue": [90.0],
            "plddt_mean": 90.0,
            "ptm": 0.88,
            "iptm": 0.84,
            "chain_mapping": None,
            "affinity_probability": 0.92,
            "affinity_value": -1.5,
        }
    ],
    "confidence": {
        "best_plddt_mean": 90.0,
        "best_ptm": 0.88,
        "best_iptm": 0.84,
    },
}


class TestBoltzParseOutput:
    """Tests for BoltzRunner.parse_output."""

    def test_parse_single_model(self, runner: BoltzRunner, tmp_path: Path) -> None:
        workspace = Workspace.create(tmp_path / "ws")
        (workspace.std_output_dir / "result_data.json").write_text(json.dumps(_SINGLE_MODEL_RESULT))

        output = runner.parse_output(workspace)
        assert len(output.structures) == 1
        s = output.structures[0]
        assert s.model_rank == 1
        assert s.plddt_mean == pytest.approx(91.675)
        assert s.ptm == pytest.approx(0.89)
        assert s.iptm == pytest.approx(0.85)

    def test_parse_multiple_models(self, runner: BoltzRunner, tmp_path: Path) -> None:
        workspace = Workspace.create(tmp_path / "ws")
        (workspace.std_output_dir / "result_data.json").write_text(json.dumps(_MULTI_MODEL_RESULT))

        output = runner.parse_output(workspace)
        assert len(output.structures) == 3
        assert output.structures[0].model_rank == 1
        assert output.structures[2].model_rank == 3

    def test_confidence_metrics(self, runner: BoltzRunner, tmp_path: Path) -> None:
        workspace = Workspace.create(tmp_path / "ws")
        (workspace.std_output_dir / "result_data.json").write_text(json.dumps(_MULTI_MODEL_RESULT))

        output = runner.parse_output(workspace)
        assert output.confidence.best_plddt_mean == pytest.approx(91.0)
        assert output.confidence.best_ptm == pytest.approx(0.92)
        assert output.confidence.best_iptm == pytest.approx(0.88)

    def test_plddt_per_residue(self, runner: BoltzRunner, tmp_path: Path) -> None:
        workspace = Workspace.create(tmp_path / "ws")
        (workspace.std_output_dir / "result_data.json").write_text(json.dumps(_SINGLE_MODEL_RESULT))

        output = runner.parse_output(workspace)
        assert output.structures[0].plddt_per_residue is not None
        assert len(output.structures[0].plddt_per_residue) == 4

    def test_affinity_data(self, runner: BoltzRunner, tmp_path: Path) -> None:
        workspace = Workspace.create(tmp_path / "ws")
        (workspace.std_output_dir / "result_data.json").write_text(json.dumps(_AFFINITY_RESULT))

        output = runner.parse_output(workspace)
        s = output.structures[0]
        assert s.affinity_probability == pytest.approx(0.92)
        assert s.affinity_value == pytest.approx(-1.5)

    def test_affinity_data_absent(self, runner: BoltzRunner, tmp_path: Path) -> None:
        """When no affinity data, fields are None."""
        workspace = Workspace.create(tmp_path / "ws")
        (workspace.std_output_dir / "result_data.json").write_text(json.dumps(_SINGLE_MODEL_RESULT))

        output = runner.parse_output(workspace)
        assert output.structures[0].affinity_probability is None
        assert output.structures[0].affinity_value is None

    def test_output_type(self, runner: BoltzRunner, tmp_path: Path) -> None:
        workspace = Workspace.create(tmp_path / "ws")
        (workspace.std_output_dir / "result_data.json").write_text(json.dumps(_SINGLE_MODEL_RESULT))

        output = runner.parse_output(workspace)
        assert isinstance(output, StructurePredictionOutput)

    def test_raw_output_path(self, runner: BoltzRunner, tmp_path: Path) -> None:
        workspace = Workspace.create(tmp_path / "ws")
        (workspace.std_output_dir / "result_data.json").write_text(json.dumps(_SINGLE_MODEL_RESULT))

        output = runner.parse_output(workspace)
        assert output.raw_output_path == workspace.raw_output_dir

    def test_container_paths_resolved_to_host(self, runner: BoltzRunner, tmp_path: Path) -> None:
        """Container-internal /workspace/... paths are remapped to host workspace."""
        workspace = Workspace.create(tmp_path / "ws")
        (workspace.std_output_dir / "result_data.json").write_text(json.dumps(_SINGLE_MODEL_RESULT))

        output = runner.parse_output(workspace)
        # The path should be under the host workspace, not /workspace/
        expected = workspace.root / "outputs" / "standardized" / "model_1.cif"
        assert output.structures[0].structure_path == expected


# ---------------------------------------------------------------------------
# TestBoltzRegistration
# ---------------------------------------------------------------------------


class TestBoltzRegistration:
    """Tests for catalog Tool and runner registration."""

    def test_boltz1_registered_as_catalog_tool(self) -> None:
        assert "boltz1" in CATALOG
        assert set(get_tool("boltz1").modes) == {"predict"}
        assert get_tool("boltz1").default_mode == "predict"
        assert get_tool("boltz1").category == ToolCategory.STRUCTURE_PREDICTION

    def test_boltz2_registered_as_catalog_tool(self) -> None:
        assert "boltz2" in CATALOG
        assert set(get_tool("boltz2").modes) == {"predict"}
        assert get_tool("boltz2").default_mode == "predict"
        assert get_tool("boltz2").category == ToolCategory.STRUCTURE_PREDICTION
        assert "affinity" in get_tool("boltz2").description.lower()

    def test_both_share_image_tag(self) -> None:
        assert get_tool("boltz1").image_tag == get_tool("boltz2").image_tag

    def test_boltz2_longer_timeout(self) -> None:
        boltz1_timeout = get_tool("boltz1").modes["predict"].default_timeout
        boltz2_timeout = get_tool("boltz2").modes["predict"].default_timeout
        assert boltz2_timeout > boltz1_timeout

    def test_tool_runners_registered(self) -> None:
        assert "boltz1" in TOOL_RUNNERS
        assert "boltz2" in TOOL_RUNNERS
        assert TOOL_RUNNERS["boltz1"] is BoltzRunner
        assert TOOL_RUNNERS["boltz2"] is BoltzRunner

    def test_get_runner_returns_boltz_runner(self, config: AutobioConfig) -> None:
        with (
            patch("autobio.tools.base.ContainerManager"),
            patch("autobio.tools.base.GPUManager"),
        ):
            r = get_runner("boltz2", config)
        assert isinstance(r, BoltzRunner)
        assert r.tool_name == "boltz2"

    def test_config_model_boltz1_vs_boltz2(self, config: AutobioConfig, tmp_path: Path) -> None:
        """get_runner('boltz1'/'boltz2') writes the matching config['model']."""
        input_data = BoltzInput(sequences={"A": "MVLSPADKTNVKAAWGKVGA"})
        for tool_name in ("boltz1", "boltz2"):
            r = _make_runner(tool_name, config)
            workspace = Workspace.create(tmp_path / f"ws-{tool_name}")
            r.prepare_workspace(input_data, workspace)
            cfg = json.loads(workspace.config_path.read_text())
            assert cfg["model"] == tool_name

    def test_notes_populated(self) -> None:
        """Notes contain key operational topics for agent guidance."""
        boltz2_notes = " ".join(get_tool("boltz2").modes["predict"].notes)
        assert "affinity" in boltz2_notes.lower()
        assert "msa" in boltz2_notes.lower()

    def test_boltz_tool_constants_registered(self) -> None:
        assert BOLTZ1_TOOL.name == "boltz1"
        assert BOLTZ2_TOOL.name == "boltz2"
        assert get_tool("boltz1") is BOLTZ1_TOOL
        assert get_tool("boltz2") is BOLTZ2_TOOL


def test_info_snapshot_boltz1() -> None:
    from autobio.cli.formatters import OutputFormat, format_tool_info_catalog

    parsed = json.loads(format_tool_info_catalog(get_tool("boltz1"), OutputFormat.JSON))
    assert parsed["modes"][0]["name"] == "predict"
    props = parsed["modes"][0]["input_schema"]["properties"]
    assert props["use_msa_server"]["x-autobio"]["widget"] == "toggle"
    assert props["entity_types"]["x-autobio"]["tier"] == "advanced"
    assert "output_schema" in parsed["modes"][0]


def test_info_snapshot_boltz2() -> None:
    from autobio.cli.formatters import OutputFormat, format_tool_info_catalog

    parsed = json.loads(format_tool_info_catalog(get_tool("boltz2"), OutputFormat.JSON))
    assert parsed["modes"][0]["name"] == "predict"
    props = parsed["modes"][0]["input_schema"]["properties"]
    assert props["boltz_yaml"]["x-autobio"]["tier"] == "advanced"
    assert "output_schema" in parsed["modes"][0]
