"""Tests for ChaiRunner — prepare_workspace, parse_output, host validation, and registration."""

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
from autobio.tools.chai import ChaiRunner

if TYPE_CHECKING:
    from pathlib import Path


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def config() -> AutobioConfig:
    return AutobioConfig.resolve()


@pytest.fixture()
def runner(config: AutobioConfig) -> ChaiRunner:
    """Create a ChaiRunner with mocked ContainerManager and GPUManager."""
    with (
        patch("autobio.tools.base.ContainerManager"),
        patch("autobio.tools.base.GPUManager"),
    ):
        return ChaiRunner("chai1", config)


# ---------------------------------------------------------------------------
# TestChaiPrepareWorkspace
# ---------------------------------------------------------------------------


class TestChaiPrepareWorkspace:
    """Tests for ChaiRunner.prepare_workspace."""

    def test_fasta_generated_from_sequences(self, runner: ChaiRunner, tmp_path: Path) -> None:
        """Sequences dict is translated into a Chai-1 FASTA input file."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = StructurePredictionInput(sequences={"A": "MKWVTFIS", "B": "GVSEKL"})
        runner.prepare_workspace(input_data, workspace)

        fasta_path = workspace.inputs_dir / "input.fasta"
        assert fasta_path.exists()
        content = fasta_path.read_text()
        assert ">protein|name=A" in content
        assert ">protein|name=B" in content
        assert "MKWVTFIS" in content
        assert "GVSEKL" in content

    def test_chain_ordering_sorted(self, runner: ChaiRunner, tmp_path: Path) -> None:
        """Chains are written in sorted order so Chai-1's alphabetical assignment aligns."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = StructurePredictionInput(sequences={"C": "AAA", "A": "GGG", "B": "TTT"})
        runner.prepare_workspace(input_data, workspace)

        content = (workspace.inputs_dir / "input.fasta").read_text()
        lines = [line for line in content.strip().split("\n") if line.startswith(">")]
        assert lines == [">protein|name=A", ">protein|name=B", ">protein|name=C"]

    def test_num_models_maps_to_num_diffn_samples(self, runner: ChaiRunner, tmp_path: Path) -> None:
        workspace = Workspace.create(tmp_path / "ws")
        input_data = StructurePredictionInput(sequences={"A": "MVLSPADKTNVKAAWGKVGA"}, num_models=5)
        runner.prepare_workspace(input_data, workspace)

        cfg = json.loads(workspace.config_path.read_text())
        assert cfg["num_diffn_samples"] == 5

    def test_num_models_default_sets_one(
        self, runner: ChaiRunner, tmp_path: Path
    ) -> None:
        """When num_models=1 (default), num_diffn_samples=1 to override Chai-1's default of 5."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = StructurePredictionInput(sequences={"A": "MVLSPADKTNVKAAWGKVGA"})
        runner.prepare_workspace(input_data, workspace)

        cfg = json.loads(workspace.config_path.read_text())
        assert cfg["num_diffn_samples"] == 1

    def test_use_msa_server_default_true(self, runner: ChaiRunner, tmp_path: Path) -> None:
        """use_msa_server defaults to True in config."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = StructurePredictionInput(sequences={"A": "MVLSPADKTNVKAAWGKVGA"})
        runner.prepare_workspace(input_data, workspace)

        cfg = json.loads(workspace.config_path.read_text())
        assert cfg["use_msa_server"] is True

    def test_use_msa_server_can_be_disabled(self, runner: ChaiRunner, tmp_path: Path) -> None:
        """extra['use_msa_server'] = False overrides the default."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = StructurePredictionInput(
            sequences={"A": "MVLSPADKTNVKAAWGKVGA"},
            extra={"use_msa_server": False},
        )
        runner.prepare_workspace(input_data, workspace)

        cfg = json.loads(workspace.config_path.read_text())
        assert cfg["use_msa_server"] is False

    def test_use_esm_embeddings_default_false(self, runner: ChaiRunner, tmp_path: Path) -> None:
        """use_esm_embeddings defaults to False in config."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = StructurePredictionInput(sequences={"A": "MVLSPADKTNVKAAWGKVGA"})
        runner.prepare_workspace(input_data, workspace)

        cfg = json.loads(workspace.config_path.read_text())
        assert cfg["use_esm_embeddings"] is False

    def test_use_esm_embeddings_can_be_enabled(self, runner: ChaiRunner, tmp_path: Path) -> None:
        """extra['use_esm_embeddings'] = True overrides the default."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = StructurePredictionInput(
            sequences={"A": "MVLSPADKTNVKAAWGKVGA"},
            extra={"use_esm_embeddings": True},
        )
        runner.prepare_workspace(input_data, workspace)

        cfg = json.loads(workspace.config_path.read_text())
        assert cfg["use_esm_embeddings"] is True

    def test_entity_types_override(self, runner: ChaiRunner, tmp_path: Path) -> None:
        """Non-protein entity types get correct FASTA headers."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = StructurePredictionInput(
            sequences={"A": "MVLSPADKTNVKAAWGKVGA", "B": "ATCGATCG"},
            extra={"entity_types": {"B": "dna"}},
        )
        runner.prepare_workspace(input_data, workspace)

        content = (workspace.inputs_dir / "input.fasta").read_text()
        assert ">protein|name=A" in content
        assert ">dna|name=B" in content

    def test_rna_entity_type(self, runner: ChaiRunner, tmp_path: Path) -> None:
        workspace = Workspace.create(tmp_path / "ws")
        input_data = StructurePredictionInput(
            sequences={"A": "MVLSPADKTNVKAAWGKVGA", "R": "ACGUACGU"},
            extra={"entity_types": {"R": "rna"}},
        )
        runner.prepare_workspace(input_data, workspace)

        content = (workspace.inputs_dir / "input.fasta").read_text()
        assert ">rna|name=R" in content

    def test_ligand_entity_string(self, runner: ChaiRunner, tmp_path: Path) -> None:
        """entity_types: {"L": "ligand"} uses sequence value as SMILES."""
        workspace = Workspace.create(tmp_path / "ws")
        smiles = "CC(C)Cc1ccc(cc1)C(C)C(O)=O"
        input_data = StructurePredictionInput(
            sequences={"A": "MVLSPADKTNVKAAWGKVGA", "L": smiles},
            extra={"entity_types": {"L": "ligand"}},
        )
        runner.prepare_workspace(input_data, workspace)

        content = (workspace.inputs_dir / "input.fasta").read_text()
        assert ">ligand|name=L" in content
        assert smiles in content

    def test_ligand_entity_smiles_dict(self, runner: ChaiRunner, tmp_path: Path) -> None:
        """Ligand entities with SMILES dict are correctly constructed."""
        workspace = Workspace.create(tmp_path / "ws")
        smiles = "CC(=O)NC1=CC=C(O)C=C1"
        input_data = StructurePredictionInput(
            sequences={"A": "MVLSPADKTNVKAAWGKVGA", "L": ""},
            extra={"entity_types": {"L": {"smiles": smiles}}},
        )
        runner.prepare_workspace(input_data, workspace)

        content = (workspace.inputs_dir / "input.fasta").read_text()
        assert ">ligand|name=L" in content
        assert smiles in content

    def test_constraints_csv_content(self, runner: ChaiRunner, tmp_path: Path) -> None:
        """CSV string constraints are written to inputs/restraints.csv."""
        workspace = Workspace.create(tmp_path / "ws")
        csv_content = (
            "chainA,res_idxA,chainB,res_idxB,connection_type,confidence,"
            "min_distance_angstrom,max_distance_angstrom,comment,restraint_id\n"
            "A,C387,B,Y101,contact,1.0,0.0,5.5,test,r1"
        )
        input_data = StructurePredictionInput(
            sequences={"A": "MVLSPADKTNVKAAWGKVGA", "B": "GVSEKL"},
            extra={"constraints": csv_content},
        )
        runner.prepare_workspace(input_data, workspace)

        restraints_path = workspace.inputs_dir / "restraints.csv"
        assert restraints_path.exists()
        assert restraints_path.read_text() == csv_content

        cfg = json.loads(workspace.config_path.read_text())
        assert cfg["constraint_path"] == "/workspace/inputs/restraints.csv"

    def test_constraints_file_path(self, runner: ChaiRunner, tmp_path: Path) -> None:
        """File path constraints are copied to workspace."""
        constraint_file = tmp_path / "my_restraints.csv"
        constraint_file.write_text("chainA,res_idxA,chainB,res_idxB\nA,C387,B,Y101\n")

        workspace = Workspace.create(tmp_path / "ws")
        input_data = StructurePredictionInput(
            sequences={"A": "MVLSPADKTNVKAAWGKVGA", "B": "GVSEKL"},
            extra={"constraints": str(constraint_file)},
        )
        runner.prepare_workspace(input_data, workspace)

        copied = workspace.inputs_dir / "restraints.csv"
        assert copied.exists()

    def test_msa_directory_copied(self, runner: ChaiRunner, tmp_path: Path) -> None:
        """MSA directory contents are copied into workspace."""
        msa_dir = tmp_path / "msas"
        msa_dir.mkdir()
        (msa_dir / "A.aligned.pqt").write_bytes(b"fake parquet data")

        workspace = Workspace.create(tmp_path / "ws")
        input_data = StructurePredictionInput(
            sequences={"A": "MVLSPADKTNVKAAWGKVGA"},
            extra={"msa_directory": str(msa_dir)},
        )
        runner.prepare_workspace(input_data, workspace)

        assert (workspace.inputs_dir / "msa" / "A.aligned.pqt").exists()

        cfg = json.loads(workspace.config_path.read_text())
        assert cfg["msa_directory"] == "/workspace/inputs/msa"

    def test_templates_copied(self, runner: ChaiRunner, tmp_path: Path) -> None:
        """Template files are copied into workspace/inputs/."""
        tmpl = tmp_path / "template.cif"
        tmpl.write_text("data_test\n_entry.id test\n")

        workspace = Workspace.create(tmp_path / "ws")
        input_data = StructurePredictionInput(
            sequences={"A": "MVLSPADKTNVKAAWGKVGA"},
            templates=[tmpl],
        )
        runner.prepare_workspace(input_data, workspace)

        assert (workspace.inputs_dir / "template.cif").exists()

    def test_raw_fasta_passthrough(self, runner: ChaiRunner, tmp_path: Path) -> None:
        """extra['chai_fasta'] bypasses automatic FASTA generation."""
        custom_fasta = ">protein|name=X\nMKWVTFIS\n>ligand|name=Y\nCC(=O)O\n"
        workspace = Workspace.create(tmp_path / "ws")
        input_data = StructurePredictionInput(
            sequences={},
            extra={"chai_fasta": custom_fasta},
        )
        runner.prepare_workspace(input_data, workspace)

        content = (workspace.inputs_dir / "input.fasta").read_text()
        assert content == custom_fasta

    def test_extra_dict_merged(self, runner: ChaiRunner, tmp_path: Path) -> None:
        """CLI-level extra keys appear in config.json, consumed keys do not."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = StructurePredictionInput(
            sequences={"A": "MVLSPADKTNVKAAWGKVGA"},
            extra={
                "num_trunk_recycles": 5,
                "seed": 42,
                "num_diffn_timesteps": 100,
                "entity_types": {"A": "protein"},  # consumed — should NOT appear
            },
        )
        runner.prepare_workspace(input_data, workspace)

        cfg = json.loads(workspace.config_path.read_text())
        assert cfg["num_trunk_recycles"] == 5
        assert cfg["seed"] == 42
        assert cfg["num_diffn_timesteps"] == 100
        assert "entity_types" not in cfg

    def test_consumed_keys_excluded(self, runner: ChaiRunner, tmp_path: Path) -> None:
        """All consumed keys are excluded from config.json."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = StructurePredictionInput(
            sequences={"A": "MVLSPADKTNVKAAWGKVGA"},
            extra={
                "entity_types": {"A": "protein"},
                "chai_fasta": None,  # even if set to None, should be excluded
                "constraints": "a,b,c,d,e,f,g,h,i,j\n1,2,3,4,5,6,7,8,9,10",
            },
        )
        runner.prepare_workspace(input_data, workspace)

        cfg = json.loads(workspace.config_path.read_text())
        assert "entity_types" not in cfg
        assert "chai_fasta" not in cfg
        assert "constraints" not in cfg

    def test_defaults_applied(self, runner: ChaiRunner, tmp_path: Path) -> None:
        """Minimal input produces sensible defaults."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = StructurePredictionInput(sequences={"A": "MVLSPADKTNVKAAWGKVGA"})
        runner.prepare_workspace(input_data, workspace)

        cfg = json.loads(workspace.config_path.read_text())
        assert cfg["fasta_path"] == "/workspace/inputs/input.fasta"
        assert cfg["output_dir"] == "/workspace/outputs/raw"
        assert cfg["downloads_dir"] == "/app/chai/downloads"
        assert cfg["use_msa_server"] is True
        assert cfg["use_esm_embeddings"] is False

    def test_no_constraint_path_when_absent(self, runner: ChaiRunner, tmp_path: Path) -> None:
        """constraint_path not in config when no constraints provided."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = StructurePredictionInput(sequences={"A": "MVLSPADKTNVKAAWGKVGA"})
        runner.prepare_workspace(input_data, workspace)

        cfg = json.loads(workspace.config_path.read_text())
        assert "constraint_path" not in cfg


# ---------------------------------------------------------------------------
# TestChaiHostValidation
# ---------------------------------------------------------------------------


class TestChaiHostValidation:
    """Tests for host-side input validation."""

    def test_empty_sequences_raises(self, runner: ChaiRunner, tmp_path: Path) -> None:
        workspace = Workspace.create(tmp_path / "ws")
        input_data = StructurePredictionInput(sequences={})
        with pytest.raises(AutobioError, match="sequences must be non-empty"):
            runner.prepare_workspace(input_data, workspace)

    def test_empty_sequences_ok_with_chai_fasta(self, runner: ChaiRunner, tmp_path: Path) -> None:
        """Empty sequences is allowed when chai_fasta is provided."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = StructurePredictionInput(
            sequences={},
            extra={"chai_fasta": ">protein|name=A\nMKWVTFIS\n"},
        )
        # Should not raise
        runner.prepare_workspace(input_data, workspace)

    def test_missing_template_file_raises(self, runner: ChaiRunner, tmp_path: Path) -> None:
        workspace = Workspace.create(tmp_path / "ws")
        input_data = StructurePredictionInput(
            sequences={"A": "MVLSPADKTNVKAAWGKVGA"},
            templates=[tmp_path / "nonexistent.cif"],
        )
        with pytest.raises(AutobioError, match="Template file does not exist"):
            runner.prepare_workspace(input_data, workspace)

    def test_missing_constraint_file_raises(self, runner: ChaiRunner, tmp_path: Path) -> None:
        workspace = Workspace.create(tmp_path / "ws")
        input_data = StructurePredictionInput(
            sequences={"A": "MVLSPADKTNVKAAWGKVGA"},
            extra={"constraints": str(tmp_path / "nonexistent.csv")},
        )
        with pytest.raises(AutobioError, match="Constraint file does not exist"):
            runner.prepare_workspace(input_data, workspace)

    def test_missing_msa_directory_raises(self, runner: ChaiRunner, tmp_path: Path) -> None:
        workspace = Workspace.create(tmp_path / "ws")
        input_data = StructurePredictionInput(
            sequences={"A": "MVLSPADKTNVKAAWGKVGA"},
            extra={"msa_directory": str(tmp_path / "nonexistent_dir")},
        )
        with pytest.raises(AutobioError, match="MSA directory does not exist"):
            runner.prepare_workspace(input_data, workspace)

    def test_entity_types_unknown_chain_raises(self, runner: ChaiRunner, tmp_path: Path) -> None:
        workspace = Workspace.create(tmp_path / "ws")
        input_data = StructurePredictionInput(
            sequences={"A": "MVLSPADKTNVKAAWGKVGA"},
            extra={"entity_types": {"Z": "dna"}},
        )
        with pytest.raises(AutobioError, match="unknown chain IDs"):
            runner.prepare_workspace(input_data, workspace)

    def test_invalid_entity_type_string_raises(self, runner: ChaiRunner, tmp_path: Path) -> None:
        workspace = Workspace.create(tmp_path / "ws")
        input_data = StructurePredictionInput(
            sequences={"A": "MVLSPADKTNVKAAWGKVGA"},
            extra={"entity_types": {"A": "peptide"}},
        )
        with pytest.raises(AutobioError, match="Invalid entity type"):
            runner.prepare_workspace(input_data, workspace)

    def test_invalid_entity_type_dict_raises(self, runner: ChaiRunner, tmp_path: Path) -> None:
        workspace = Workspace.create(tmp_path / "ws")
        input_data = StructurePredictionInput(
            sequences={"A": "MVLSPADKTNVKAAWGKVGA"},
            extra={"entity_types": {"A": {"invalid_key": "value"}}},
        )
        with pytest.raises(AutobioError, match="must contain 'smiles' key"):
            runner.prepare_workspace(input_data, workspace)


# ---------------------------------------------------------------------------
# TestChaiParseOutput
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


class TestChaiParseOutput:
    """Tests for ChaiRunner.parse_output."""

    def test_parse_single_model(self, runner: ChaiRunner, tmp_path: Path) -> None:
        workspace = Workspace.create(tmp_path / "ws")
        (workspace.std_output_dir / "result_data.json").write_text(json.dumps(_SINGLE_MODEL_RESULT))

        output = runner.parse_output(workspace)
        assert len(output.structures) == 1
        s = output.structures[0]
        assert s.model_rank == 1
        assert s.plddt_mean == pytest.approx(91.675)
        assert s.ptm == pytest.approx(0.89)
        assert s.iptm == pytest.approx(0.85)

    def test_parse_multiple_models(self, runner: ChaiRunner, tmp_path: Path) -> None:
        workspace = Workspace.create(tmp_path / "ws")
        (workspace.std_output_dir / "result_data.json").write_text(json.dumps(_MULTI_MODEL_RESULT))

        output = runner.parse_output(workspace)
        assert len(output.structures) == 3
        assert output.structures[0].model_rank == 1
        assert output.structures[2].model_rank == 3

    def test_confidence_metrics(self, runner: ChaiRunner, tmp_path: Path) -> None:
        workspace = Workspace.create(tmp_path / "ws")
        (workspace.std_output_dir / "result_data.json").write_text(json.dumps(_MULTI_MODEL_RESULT))

        output = runner.parse_output(workspace)
        assert output.confidence.best_plddt_mean == pytest.approx(91.0)
        assert output.confidence.best_ptm == pytest.approx(0.92)
        assert output.confidence.best_iptm == pytest.approx(0.88)

    def test_plddt_per_residue(self, runner: ChaiRunner, tmp_path: Path) -> None:
        workspace = Workspace.create(tmp_path / "ws")
        (workspace.std_output_dir / "result_data.json").write_text(json.dumps(_SINGLE_MODEL_RESULT))

        output = runner.parse_output(workspace)
        assert output.structures[0].plddt_per_residue is not None
        assert len(output.structures[0].plddt_per_residue) == 4

    def test_output_type(self, runner: ChaiRunner, tmp_path: Path) -> None:
        workspace = Workspace.create(tmp_path / "ws")
        (workspace.std_output_dir / "result_data.json").write_text(json.dumps(_SINGLE_MODEL_RESULT))

        output = runner.parse_output(workspace)
        assert isinstance(output, StructurePredictionOutput)

    def test_raw_output_path(self, runner: ChaiRunner, tmp_path: Path) -> None:
        workspace = Workspace.create(tmp_path / "ws")
        (workspace.std_output_dir / "result_data.json").write_text(json.dumps(_SINGLE_MODEL_RESULT))

        output = runner.parse_output(workspace)
        assert output.raw_output_path == workspace.raw_output_dir

    def test_container_paths_resolved_to_host(self, runner: ChaiRunner, tmp_path: Path) -> None:
        """Container-internal /workspace/... paths are remapped to host workspace."""
        workspace = Workspace.create(tmp_path / "ws")
        (workspace.std_output_dir / "result_data.json").write_text(json.dumps(_SINGLE_MODEL_RESULT))

        output = runner.parse_output(workspace)
        expected = workspace.root / "outputs" / "standardized" / "model_1.cif"
        assert output.structures[0].structure_path == expected


# ---------------------------------------------------------------------------
# TestChaiRegistration
# ---------------------------------------------------------------------------


class TestChaiRegistration:
    """Tests for tool and runner registration."""

    def test_chai1_in_registry(self) -> None:
        assert "chai1" in TOOL_REGISTRY
        entry = TOOL_REGISTRY["chai1"]
        assert entry.category == ToolCategory.STRUCTURE_PREDICTION
        assert entry.input_schema is StructurePredictionInput
        assert entry.output_schema is StructurePredictionOutput
        assert entry.requires_gpu is True

    def test_tool_runners_registered(self) -> None:
        assert "chai1" in TOOL_RUNNERS
        assert TOOL_RUNNERS["chai1"] is ChaiRunner

    def test_get_runner_returns_chai_runner(self, config: AutobioConfig) -> None:
        with (
            patch("autobio.tools.base.ContainerManager"),
            patch("autobio.tools.base.GPUManager"),
        ):
            r = get_runner("chai1", config)
        assert isinstance(r, ChaiRunner)
        assert r.tool_name == "chai1"

    def test_notes_populated(self) -> None:
        """Notes contain key topics for agent guidance."""
        chai_notes = " ".join(TOOL_REGISTRY["chai1"].notes)
        assert "entity_types" in chai_notes
        assert "restraint" in chai_notes.lower() or "constraint" in chai_notes.lower()
        assert "msa" in chai_notes.lower()
        assert "ligand" in chai_notes.lower()
        assert "covalent" in chai_notes.lower()
        assert "chai_fasta" in chai_notes
