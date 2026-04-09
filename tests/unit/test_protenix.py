"""Tests for ProtenixRunner — prepare_workspace, parse_output, validation, registration."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

from autobio.core.config import AutobioConfig
from autobio.core.registry import TOOL_REGISTRY, ToolCategory
from autobio.core.result import AutobioError
from autobio.core.workspace import Workspace
from autobio.schemas.structure_prediction import (
    StructurePredictionInput,
    StructurePredictionOutput,
)
from autobio.tools import TOOL_RUNNERS, get_runner
from autobio.tools.protenix import ProtenixRunner

if TYPE_CHECKING:
    from pathlib import Path


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def config() -> AutobioConfig:
    return AutobioConfig.resolve()


@pytest.fixture()
def runner(config: AutobioConfig) -> ProtenixRunner:
    """Create a ProtenixRunner with mocked ContainerManager and GPUManager."""
    with (
        patch("autobio.tools.base.ContainerManager"),
        patch("autobio.tools.base.GPUManager"),
    ):
        return ProtenixRunner("protenix_v2", config)


# ---------------------------------------------------------------------------
# TestProtenixPrepareWorkspace
# ---------------------------------------------------------------------------


class TestProtenixPrepareWorkspace:
    """Tests for ProtenixRunner.prepare_workspace."""

    def test_protenix_json_generated_from_sequences(
        self, runner: ProtenixRunner, tmp_path: Path
    ) -> None:
        """Sequences dict is translated into a Protenix input JSON file."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = StructurePredictionInput(sequences={"A": "MKWVTFIS", "B": "GVSEKL"})
        runner.prepare_workspace(input_data, workspace)

        input_path = workspace.inputs_dir / "input.json"
        assert input_path.exists()
        query = json.loads(input_path.read_text())
        assert isinstance(query, list)
        assert len(query) == 1

        job = query[0]
        assert job["name"] == "prediction"
        assert len(job["sequences"]) == 2

        # Both should be proteinChain by default
        for entity in job["sequences"]:
            assert "proteinChain" in entity

        chain_ids = {entity["proteinChain"]["id"][0] for entity in job["sequences"]}
        assert chain_ids == {"A", "B"}

        sequences = {
            entity["proteinChain"]["id"][0]: entity["proteinChain"]["sequence"]
            for entity in job["sequences"]
        }
        assert sequences["A"] == "MKWVTFIS"
        assert sequences["B"] == "GVSEKL"

    def test_num_models_maps_to_diffusion_samples(
        self, runner: ProtenixRunner, tmp_path: Path
    ) -> None:
        workspace = Workspace.create(tmp_path / "ws")
        input_data = StructurePredictionInput(sequences={"A": "MVLSPADKTNVKAAWGKVGA"}, num_models=5)
        runner.prepare_workspace(input_data, workspace)

        cfg = json.loads(workspace.config_path.read_text())
        assert cfg["diffusion_samples"] == 5

    def test_num_models_default_sets_one(self, runner: ProtenixRunner, tmp_path: Path) -> None:
        """When num_models=1 (default), diffusion_samples=1."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = StructurePredictionInput(sequences={"A": "MVLSPADKTNVKAAWGKVGA"})
        runner.prepare_workspace(input_data, workspace)

        cfg = json.loads(workspace.config_path.read_text())
        assert cfg["diffusion_samples"] == 1

    def test_defaults_applied(self, runner: ProtenixRunner, tmp_path: Path) -> None:
        """Minimal input produces sensible defaults."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = StructurePredictionInput(sequences={"A": "MVLSPADKTNVKAAWGKVGA"})
        runner.prepare_workspace(input_data, workspace)

        cfg = json.loads(workspace.config_path.read_text())
        assert cfg["input_json_path"] == "/workspace/inputs/input.json"
        assert cfg["output_dir"] == "/workspace/outputs/raw"
        assert cfg["model_name"] == "protenix_base_default_v1.0.0"
        assert cfg["use_msa"] is True
        assert cfg["use_template"] is False
        assert cfg["dtype"] == "bf16"
        assert cfg["seeds"] == "101"
        assert cfg["pairformer_cycles"] == 10
        assert cfg["diffusion_steps"] == 200

    def test_use_msa_default_true(self, runner: ProtenixRunner, tmp_path: Path) -> None:
        workspace = Workspace.create(tmp_path / "ws")
        input_data = StructurePredictionInput(sequences={"A": "MVLSPADKTNVKAAWGKVGA"})
        runner.prepare_workspace(input_data, workspace)

        cfg = json.loads(workspace.config_path.read_text())
        assert cfg["use_msa"] is True

    def test_use_msa_can_be_disabled(self, runner: ProtenixRunner, tmp_path: Path) -> None:
        workspace = Workspace.create(tmp_path / "ws")
        input_data = StructurePredictionInput(
            sequences={"A": "MVLSPADKTNVKAAWGKVGA"},
            extra={"use_msa": False},
        )
        runner.prepare_workspace(input_data, workspace)

        cfg = json.loads(workspace.config_path.read_text())
        assert cfg["use_msa"] is False

    def test_entity_types_dna(self, runner: ProtenixRunner, tmp_path: Path) -> None:
        workspace = Workspace.create(tmp_path / "ws")
        input_data = StructurePredictionInput(
            sequences={"A": "MVLSPADKTNVKAAWGKVGA", "B": "ATCGATCG"},
            extra={"entity_types": {"B": "dna"}},
        )
        runner.prepare_workspace(input_data, workspace)

        query = json.loads((workspace.inputs_dir / "input.json").read_text())
        entities = query[0]["sequences"]
        dna_entity = next(e for e in entities if "dnaSequence" in e)
        assert dna_entity["dnaSequence"]["sequence"] == "ATCGATCG"

    def test_entity_types_rna(self, runner: ProtenixRunner, tmp_path: Path) -> None:
        workspace = Workspace.create(tmp_path / "ws")
        input_data = StructurePredictionInput(
            sequences={"A": "MVLSPADKTNVKAAWGKVGA", "R": "ACGUACGU"},
            extra={"entity_types": {"R": "rna"}},
        )
        runner.prepare_workspace(input_data, workspace)

        query = json.loads((workspace.inputs_dir / "input.json").read_text())
        entities = query[0]["sequences"]
        rna_entity = next(e for e in entities if "rnaSequence" in e)
        assert rna_entity["rnaSequence"]["sequence"] == "ACGUACGU"

    def test_entity_types_ion(self, runner: ProtenixRunner, tmp_path: Path) -> None:
        """Ion entity type is unique to Protenix."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = StructurePredictionInput(
            sequences={"A": "MVLSPADKTNVKAAWGKVGA", "I": "MG"},
            extra={"entity_types": {"I": "ion"}},
        )
        runner.prepare_workspace(input_data, workspace)

        query = json.loads((workspace.inputs_dir / "input.json").read_text())
        entities = query[0]["sequences"]
        ion_entity = next(e for e in entities if "ion" in e)
        assert ion_entity["ion"]["ion"] == "MG"
        assert ion_entity["ion"]["count"] == 1

    def test_ligand_ccd_dict(self, runner: ProtenixRunner, tmp_path: Path) -> None:
        """Ligand via CCD code generates correct entity with CCD_ prefix."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = StructurePredictionInput(
            sequences={"A": "MVLSPADKTNVKAAWGKVGA", "L": ""},
            extra={"entity_types": {"L": {"ccd": "ATP"}}},
        )
        runner.prepare_workspace(input_data, workspace)

        query = json.loads((workspace.inputs_dir / "input.json").read_text())
        entities = query[0]["sequences"]
        lig_entity = next(e for e in entities if "ligand" in e)
        assert lig_entity["ligand"]["ligand"] == "CCD_ATP"

    def test_ligand_smiles_dict(self, runner: ProtenixRunner, tmp_path: Path) -> None:
        """Ligand via SMILES dict generates correct entity."""
        smiles = "CC(=O)NC1=CC=C(O)C=C1"
        workspace = Workspace.create(tmp_path / "ws")
        input_data = StructurePredictionInput(
            sequences={"A": "MVLSPADKTNVKAAWGKVGA", "L": ""},
            extra={"entity_types": {"L": {"smiles": smiles}}},
        )
        runner.prepare_workspace(input_data, workspace)

        query = json.loads((workspace.inputs_dir / "input.json").read_text())
        entities = query[0]["sequences"]
        lig_entity = next(e for e in entities if "ligand" in e)
        assert lig_entity["ligand"]["ligand"] == smiles

    def test_ligand_string_uses_sequence_as_specifier(
        self, runner: ProtenixRunner, tmp_path: Path
    ) -> None:
        """entity_types: {"L": "ligand"} uses sequence value as ligand specifier."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = StructurePredictionInput(
            sequences={"A": "MVLSPADKTNVKAAWGKVGA", "L": "CCD_ATP"},
            extra={"entity_types": {"L": "ligand"}},
        )
        runner.prepare_workspace(input_data, workspace)

        query = json.loads((workspace.inputs_dir / "input.json").read_text())
        entities = query[0]["sequences"]
        lig_entity = next(e for e in entities if "ligand" in e)
        assert lig_entity["ligand"]["ligand"] == "CCD_ATP"

    def test_covalent_bonds_passthrough(self, runner: ProtenixRunner, tmp_path: Path) -> None:
        """extra['covalent_bonds'] appears in the output JSON."""
        bonds = [{"entity1": "A", "entity2": "B", "atom1": "SG", "atom2": "C1"}]
        workspace = Workspace.create(tmp_path / "ws")
        input_data = StructurePredictionInput(
            sequences={"A": "MVLSPADKTNVKAAWGKVGA"},
            extra={"covalent_bonds": bonds},
        )
        runner.prepare_workspace(input_data, workspace)

        query = json.loads((workspace.inputs_dir / "input.json").read_text())
        assert query[0]["covalent_bonds"] == bonds

    def test_constraints_passthrough(self, runner: ProtenixRunner, tmp_path: Path) -> None:
        """extra['constraints'] appears in the output JSON."""
        constraints = {"contact": [{"chain1": "A", "chain2": "B"}]}
        workspace = Workspace.create(tmp_path / "ws")
        input_data = StructurePredictionInput(
            sequences={"A": "MVLSPADKTNVKAAWGKVGA"},
            extra={"constraints": constraints},
        )
        runner.prepare_workspace(input_data, workspace)

        query = json.loads((workspace.inputs_dir / "input.json").read_text())
        assert query[0]["constraint"] == constraints

    def test_templates_copied(self, runner: ProtenixRunner, tmp_path: Path) -> None:
        tmpl = tmp_path / "template.cif"
        tmpl.write_text("data_test\n_entry.id test\n")

        workspace = Workspace.create(tmp_path / "ws")
        input_data = StructurePredictionInput(
            sequences={"A": "MVLSPADKTNVKAAWGKVGA"},
            templates=[tmpl],
        )
        runner.prepare_workspace(input_data, workspace)

        assert (workspace.inputs_dir / "template.cif").exists()

    def test_msa_dir_copied(self, runner: ProtenixRunner, tmp_path: Path) -> None:
        """MSA directories from extra['msa_paths'] are copied into workspace."""
        msa_dir = tmp_path / "chain_0"
        msa_dir.mkdir()
        (msa_dir / "pairing.a3m").write_text(">query\nMVLSPADK\n")
        (msa_dir / "non_pairing.a3m").write_text(">query\nMVLSPADK\n")

        workspace = Workspace.create(tmp_path / "ws")
        input_data = StructurePredictionInput(
            sequences={"A": "MVLSPADKTNVKAAWGKVGA"},
            extra={"msa_paths": [str(msa_dir)]},
        )
        runner.prepare_workspace(input_data, workspace)

        assert (workspace.inputs_dir / "msa" / "chain_0" / "pairing.a3m").exists()

    def test_raw_query_json_passthrough_dict(self, runner: ProtenixRunner, tmp_path: Path) -> None:
        """extra['query_json'] dict bypasses automatic generation."""
        custom_query = [
            {
                "name": "my_prediction",
                "sequences": [{"proteinChain": {"sequence": "MKWVTFIS", "count": 1, "id": ["X"]}}],
                "covalent_bonds": [],
                "constraint": {},
            }
        ]
        workspace = Workspace.create(tmp_path / "ws")
        input_data = StructurePredictionInput(
            sequences={},
            extra={"query_json": custom_query},
        )
        runner.prepare_workspace(input_data, workspace)

        query = json.loads((workspace.inputs_dir / "input.json").read_text())
        assert query[0]["name"] == "my_prediction"

    def test_raw_query_json_passthrough_string(
        self, runner: ProtenixRunner, tmp_path: Path
    ) -> None:
        """extra['query_json'] string is written directly."""
        custom_json = '[{"name": "q1", "sequences": [], "covalent_bonds": [], "constraint": {}}]'
        workspace = Workspace.create(tmp_path / "ws")
        input_data = StructurePredictionInput(
            sequences={},
            extra={"query_json": custom_json},
        )
        runner.prepare_workspace(input_data, workspace)

        content = (workspace.inputs_dir / "input.json").read_text()
        assert content == custom_json

    def test_extra_dict_merged(self, runner: ProtenixRunner, tmp_path: Path) -> None:
        """Pass-through extra keys appear in config.json, consumed keys do not."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = StructurePredictionInput(
            sequences={"A": "MVLSPADKTNVKAAWGKVGA"},
            extra={
                "seeds": "101,42,123",
                "use_tfg_guidance": True,
                "msa_server_url": "https://my-server.com",
                "entity_types": {"A": "protein"},  # consumed — should NOT appear
            },
        )
        runner.prepare_workspace(input_data, workspace)

        cfg = json.loads(workspace.config_path.read_text())
        assert cfg["seeds"] == "101,42,123"
        assert cfg["use_tfg_guidance"] is True
        assert cfg["msa_server_url"] == "https://my-server.com"
        assert "entity_types" not in cfg

    def test_consumed_keys_excluded(self, runner: ProtenixRunner, tmp_path: Path) -> None:
        """All consumed keys are excluded from config.json."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = StructurePredictionInput(
            sequences={"A": "MVLSPADKTNVKAAWGKVGA"},
            extra={
                "entity_types": {"A": "protein"},
                "query_json": None,
                "msa_paths": None,
                "covalent_bonds": [],
                "constraints": {},
            },
        )
        runner.prepare_workspace(input_data, workspace)

        cfg = json.loads(workspace.config_path.read_text())
        for key in ("entity_types", "query_json", "msa_paths", "covalent_bonds", "constraints"):
            assert key not in cfg

    def test_multi_entity_complex(self, runner: ProtenixRunner, tmp_path: Path) -> None:
        """Complex with protein, DNA, RNA, ligand, and ion generates correct JSON."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = StructurePredictionInput(
            sequences={
                "A": "MVLSPADKTNVKAAWGKVGA",
                "B": "ATCGATCG",
                "C": "ACGUACGU",
                "D": "CC(=O)O",
                "I": "MG",
            },
            extra={
                "entity_types": {
                    "B": "dna",
                    "C": "rna",
                    "D": "ligand",
                    "I": "ion",
                },
            },
        )
        runner.prepare_workspace(input_data, workspace)

        query = json.loads((workspace.inputs_dir / "input.json").read_text())
        entities = query[0]["sequences"]
        assert len(entities) == 5

        entity_keys = set()
        for e in entities:
            entity_keys.update(e.keys())
        assert entity_keys == {"proteinChain", "dnaSequence", "rnaSequence", "ligand", "ion"}

    def test_custom_seeds(self, runner: ProtenixRunner, tmp_path: Path) -> None:
        """Custom seeds via extra are passed through to config."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = StructurePredictionInput(
            sequences={"A": "MVLSPADKTNVKAAWGKVGA"},
            extra={"seeds": "101,42,123"},
        )
        runner.prepare_workspace(input_data, workspace)

        cfg = json.loads(workspace.config_path.read_text())
        assert cfg["seeds"] == "101,42,123"

    def test_custom_msa_server_url(self, runner: ProtenixRunner, tmp_path: Path) -> None:
        """Custom MSA server URL via extra is passed through to config."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = StructurePredictionInput(
            sequences={"A": "MVLSPADKTNVKAAWGKVGA"},
            extra={"msa_server_url": "https://msa.internal.net"},
        )
        runner.prepare_workspace(input_data, workspace)

        cfg = json.loads(workspace.config_path.read_text())
        assert cfg["msa_server_url"] == "https://msa.internal.net"


# ---------------------------------------------------------------------------
# TestProtenixHostValidation
# ---------------------------------------------------------------------------


class TestProtenixHostValidation:
    """Tests for host-side input validation."""

    def test_empty_sequences_raises(self, runner: ProtenixRunner, tmp_path: Path) -> None:
        workspace = Workspace.create(tmp_path / "ws")
        input_data = StructurePredictionInput(sequences={})
        with pytest.raises(AutobioError, match="sequences must be non-empty"):
            runner.prepare_workspace(input_data, workspace)

    def test_empty_sequences_ok_with_query_json(
        self, runner: ProtenixRunner, tmp_path: Path
    ) -> None:
        """Empty sequences is allowed when query_json is provided."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = StructurePredictionInput(
            sequences={},
            extra={"query_json": [{"name": "q", "sequences": [], "covalent_bonds": []}]},
        )
        # Should not raise
        runner.prepare_workspace(input_data, workspace)

    def test_missing_template_file_raises(self, runner: ProtenixRunner, tmp_path: Path) -> None:
        workspace = Workspace.create(tmp_path / "ws")
        input_data = StructurePredictionInput(
            sequences={"A": "MVLSPADKTNVKAAWGKVGA"},
            templates=[tmp_path / "nonexistent.cif"],
        )
        with pytest.raises(AutobioError, match="Template file does not exist"):
            runner.prepare_workspace(input_data, workspace)

    def test_missing_msa_file_raises(self, runner: ProtenixRunner, tmp_path: Path) -> None:
        workspace = Workspace.create(tmp_path / "ws")
        input_data = StructurePredictionInput(
            sequences={"A": "MVLSPADKTNVKAAWGKVGA"},
            extra={"msa_paths": [str(tmp_path / "nonexistent_msa")]},
        )
        with pytest.raises(AutobioError, match="MSA file does not exist"):
            runner.prepare_workspace(input_data, workspace)

    def test_entity_types_unknown_chain_raises(
        self, runner: ProtenixRunner, tmp_path: Path
    ) -> None:
        workspace = Workspace.create(tmp_path / "ws")
        input_data = StructurePredictionInput(
            sequences={"A": "MVLSPADKTNVKAAWGKVGA"},
            extra={"entity_types": {"Z": "dna"}},
        )
        with pytest.raises(AutobioError, match="unknown chain IDs"):
            runner.prepare_workspace(input_data, workspace)

    def test_invalid_entity_type_string_raises(
        self, runner: ProtenixRunner, tmp_path: Path
    ) -> None:
        workspace = Workspace.create(tmp_path / "ws")
        input_data = StructurePredictionInput(
            sequences={"A": "MVLSPADKTNVKAAWGKVGA"},
            extra={"entity_types": {"A": "peptide"}},
        )
        with pytest.raises(AutobioError, match="Invalid entity type"):
            runner.prepare_workspace(input_data, workspace)

    def test_invalid_entity_type_dict_raises(self, runner: ProtenixRunner, tmp_path: Path) -> None:
        workspace = Workspace.create(tmp_path / "ws")
        input_data = StructurePredictionInput(
            sequences={"A": "MVLSPADKTNVKAAWGKVGA"},
            extra={"entity_types": {"A": {"invalid_key": "value"}}},
        )
        with pytest.raises(AutobioError, match="must contain 'smiles' or 'ccd' key"):
            runner.prepare_workspace(input_data, workspace)

    def test_covalent_bonds_not_list_raises(self, runner: ProtenixRunner, tmp_path: Path) -> None:
        workspace = Workspace.create(tmp_path / "ws")
        input_data = StructurePredictionInput(
            sequences={"A": "MVLSPADKTNVKAAWGKVGA"},
            extra={"covalent_bonds": "invalid"},
        )
        with pytest.raises(AutobioError, match="covalent_bonds must be a list"):
            runner.prepare_workspace(input_data, workspace)

    def test_constraints_not_dict_raises(self, runner: ProtenixRunner, tmp_path: Path) -> None:
        workspace = Workspace.create(tmp_path / "ws")
        input_data = StructurePredictionInput(
            sequences={"A": "MVLSPADKTNVKAAWGKVGA"},
            extra={"constraints": ["invalid"]},
        )
        with pytest.raises(AutobioError, match="constraints must be a dict"):
            runner.prepare_workspace(input_data, workspace)


# ---------------------------------------------------------------------------
# TestProtenixParseOutput
# ---------------------------------------------------------------------------

_SINGLE_MODEL_RESULT = {
    "structures": [
        {
            "model_rank": 1,
            "structure_path": "/workspace/outputs/standardized/model_1.cif",
            "plddt_per_residue": None,
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
            "plddt_per_residue": None,
            "plddt_mean": 91.0,
            "ptm": 0.92,
            "iptm": 0.88,
            "chain_mapping": None,
        },
        {
            "model_rank": 2,
            "structure_path": "/workspace/outputs/standardized/model_2.cif",
            "plddt_per_residue": None,
            "plddt_mean": 83.5,
            "ptm": 0.80,
            "iptm": 0.75,
            "chain_mapping": None,
        },
        {
            "model_rank": 3,
            "structure_path": "/workspace/outputs/standardized/model_3.cif",
            "plddt_per_residue": None,
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


class TestProtenixParseOutput:
    """Tests for ProtenixRunner.parse_output."""

    def test_parse_single_model(self, runner: ProtenixRunner, tmp_path: Path) -> None:
        workspace = Workspace.create(tmp_path / "ws")
        (workspace.std_output_dir / "result_data.json").write_text(json.dumps(_SINGLE_MODEL_RESULT))

        output = runner.parse_output(workspace)
        assert len(output.structures) == 1
        s = output.structures[0]
        assert s.model_rank == 1
        assert s.plddt_mean == pytest.approx(91.675)
        assert s.ptm == pytest.approx(0.89)
        assert s.iptm == pytest.approx(0.85)

    def test_parse_multiple_models(self, runner: ProtenixRunner, tmp_path: Path) -> None:
        workspace = Workspace.create(tmp_path / "ws")
        (workspace.std_output_dir / "result_data.json").write_text(json.dumps(_MULTI_MODEL_RESULT))

        output = runner.parse_output(workspace)
        assert len(output.structures) == 3
        assert output.structures[0].model_rank == 1
        assert output.structures[2].model_rank == 3

    def test_confidence_metrics(self, runner: ProtenixRunner, tmp_path: Path) -> None:
        workspace = Workspace.create(tmp_path / "ws")
        (workspace.std_output_dir / "result_data.json").write_text(json.dumps(_MULTI_MODEL_RESULT))

        output = runner.parse_output(workspace)
        assert output.confidence.best_plddt_mean == pytest.approx(91.0)
        assert output.confidence.best_ptm == pytest.approx(0.92)
        assert output.confidence.best_iptm == pytest.approx(0.88)

    def test_output_type(self, runner: ProtenixRunner, tmp_path: Path) -> None:
        workspace = Workspace.create(tmp_path / "ws")
        (workspace.std_output_dir / "result_data.json").write_text(json.dumps(_SINGLE_MODEL_RESULT))

        output = runner.parse_output(workspace)
        assert isinstance(output, StructurePredictionOutput)

    def test_raw_output_path(self, runner: ProtenixRunner, tmp_path: Path) -> None:
        workspace = Workspace.create(tmp_path / "ws")
        (workspace.std_output_dir / "result_data.json").write_text(json.dumps(_SINGLE_MODEL_RESULT))

        output = runner.parse_output(workspace)
        assert output.raw_output_path == workspace.raw_output_dir

    def test_container_paths_resolved_to_host(self, runner: ProtenixRunner, tmp_path: Path) -> None:
        """Container-internal /workspace/... paths are remapped to host workspace."""
        workspace = Workspace.create(tmp_path / "ws")
        (workspace.std_output_dir / "result_data.json").write_text(json.dumps(_SINGLE_MODEL_RESULT))

        output = runner.parse_output(workspace)
        expected = workspace.root / "outputs" / "standardized" / "model_1.cif"
        assert output.structures[0].structure_path == expected

    def test_null_confidence_values(self, runner: ProtenixRunner, tmp_path: Path) -> None:
        """Handles null confidence values gracefully."""
        result = {
            "structures": [
                {
                    "model_rank": 1,
                    "structure_path": "/workspace/outputs/standardized/model_1.cif",
                    "plddt_per_residue": None,
                    "plddt_mean": 75.0,
                    "ptm": None,
                    "iptm": None,
                    "chain_mapping": None,
                }
            ],
            "confidence": {
                "best_plddt_mean": 75.0,
                "best_ptm": None,
                "best_iptm": None,
            },
        }
        workspace = Workspace.create(tmp_path / "ws")
        (workspace.std_output_dir / "result_data.json").write_text(json.dumps(result))

        output = runner.parse_output(workspace)
        s = output.structures[0]
        assert s.ptm is None
        assert s.iptm is None
        assert s.plddt_mean == pytest.approx(75.0)


# ---------------------------------------------------------------------------
# TestProtenixRegistration
# ---------------------------------------------------------------------------


class TestProtenixRegistration:
    """Tests for tool and runner registration."""

    def test_protenix_v2_in_registry(self) -> None:
        assert "protenix_v2" in TOOL_REGISTRY
        entry = TOOL_REGISTRY["protenix_v2"]
        assert entry.category == ToolCategory.STRUCTURE_PREDICTION
        assert entry.input_schema is StructurePredictionInput
        assert entry.output_schema is StructurePredictionOutput
        assert entry.requires_gpu is True

    def test_tool_runners_registered(self) -> None:
        assert "protenix_v2" in TOOL_RUNNERS
        assert TOOL_RUNNERS["protenix_v2"] is ProtenixRunner

    def test_get_runner_returns_protenix_runner(self, config: AutobioConfig) -> None:
        with (
            patch("autobio.tools.base.ContainerManager"),
            patch("autobio.tools.base.GPUManager"),
        ):
            r = get_runner("protenix_v2", config)
        assert isinstance(r, ProtenixRunner)
        assert r.tool_name == "protenix_v2"

    def test_notes_populated(self) -> None:
        """Notes contain key operational topics for agent guidance."""
        notes = " ".join(TOOL_REGISTRY["protenix_v2"].notes)
        assert "msa" in notes.lower()
        assert "ion" in notes.lower()
        assert "msa_server_url" in notes

    def test_input_format_populated(self) -> None:
        """Input format contains entity construction and native format info."""
        fmt = " ".join(TOOL_REGISTRY["protenix_v2"].input_format)
        assert "entity_types" in fmt
        assert "ligand" in fmt.lower()
        assert "query_json" in fmt
        assert "ion" in fmt.lower()
        assert "json" in fmt.lower()
