"""Tests for OpenFold3Runner — prepare_workspace, parse_output, validation, registration."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

from autobio.core.catalog import CATALOG, get_tool
from autobio.core.config import AutobioConfig
from autobio.core.registry import ToolCategory
from autobio.core.result import AutobioError
from autobio.core.workspace import Workspace
from autobio.schemas.structure_prediction import OpenFold3Input, StructurePredictionOutput
from autobio.tools import TOOL_RUNNERS, get_runner
from autobio.tools.openfold3 import OPENFOLD3_TOOL, OpenFold3Runner

if TYPE_CHECKING:
    from pathlib import Path


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def config() -> AutobioConfig:
    return AutobioConfig.resolve()


def _make_runner(config: AutobioConfig) -> OpenFold3Runner:
    """Create an OpenFold3Runner with mocked ContainerManager/GPUManager and current_mode set.

    ``current_mode`` is set directly (rather than via ``run()``) so that
    ``prepare_workspace`` — which calls ``_apply_extra`` — can be exercised
    in isolation.
    """
    with (
        patch("autobio.tools.base.ContainerManager"),
        patch("autobio.tools.base.GPUManager"),
    ):
        runner = OpenFold3Runner("openfold3", config)
    runner.current_mode = get_tool("openfold3").modes["predict"]
    return runner


@pytest.fixture()
def runner(config: AutobioConfig) -> OpenFold3Runner:
    """Create an OpenFold3Runner with mocked ContainerManager and GPUManager."""
    return _make_runner(config)


# ---------------------------------------------------------------------------
# TestOpenFold3PrepareWorkspace
# ---------------------------------------------------------------------------


class TestOpenFold3PrepareWorkspace:
    """Tests for OpenFold3Runner.prepare_workspace."""

    def test_query_json_generated_from_sequences(
        self, runner: OpenFold3Runner, tmp_path: Path
    ) -> None:
        """Sequences dict is translated into an OpenFold3 query JSON file."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = OpenFold3Input(sequences={"A": "MKWVTFIS", "B": "GVSEKL"})
        runner.prepare_workspace(input_data, workspace)

        query_path = workspace.inputs_dir / "query.json"
        assert query_path.exists()
        query = json.loads(query_path.read_text())
        assert "queries" in query
        assert "query_1" in query["queries"]

        chains = query["queries"]["query_1"]["chains"]
        assert len(chains) == 2

        # Both should be protein by default
        for chain in chains:
            assert chain["molecule_type"] == "protein"

        chain_ids = {c["chain_ids"] for c in chains}
        assert chain_ids == {"A", "B"}

        sequences = {c["chain_ids"]: c["sequence"] for c in chains}
        assert sequences["A"] == "MKWVTFIS"
        assert sequences["B"] == "GVSEKL"

    def test_num_models_maps_to_num_diffusion_samples(
        self, runner: OpenFold3Runner, tmp_path: Path
    ) -> None:
        workspace = Workspace.create(tmp_path / "ws")
        input_data = OpenFold3Input(sequences={"A": "MVLSPADKTNVKAAWGKVGA"}, num_models=5)
        runner.prepare_workspace(input_data, workspace)

        cfg = json.loads(workspace.config_path.read_text())
        assert cfg["num_diffusion_samples"] == 5

    def test_num_models_default_sets_one(self, runner: OpenFold3Runner, tmp_path: Path) -> None:
        """When num_models=1 (default), num_diffusion_samples=1."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = OpenFold3Input(sequences={"A": "MVLSPADKTNVKAAWGKVGA"})
        runner.prepare_workspace(input_data, workspace)

        cfg = json.loads(workspace.config_path.read_text())
        assert cfg["num_diffusion_samples"] == 1

    def test_defaults_applied(self, runner: OpenFold3Runner, tmp_path: Path) -> None:
        """Minimal input produces sensible defaults."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = OpenFold3Input(sequences={"A": "MVLSPADKTNVKAAWGKVGA"})
        runner.prepare_workspace(input_data, workspace)

        cfg = json.loads(workspace.config_path.read_text())
        assert cfg["query_json_path"] == "/workspace/inputs/query.json"
        assert cfg["output_dir"] == "/workspace/outputs/raw"
        assert cfg["checkpoint_path"] == "/app/openfold3/weights/of3-p2-155k.pt"
        assert cfg["use_msa_server"] is True
        assert cfg["use_templates"] is True
        assert cfg["pae_enabled"] is True

    def test_use_msa_server_default_true(self, runner: OpenFold3Runner, tmp_path: Path) -> None:
        workspace = Workspace.create(tmp_path / "ws")
        input_data = OpenFold3Input(sequences={"A": "MVLSPADKTNVKAAWGKVGA"})
        runner.prepare_workspace(input_data, workspace)

        cfg = json.loads(workspace.config_path.read_text())
        assert cfg["use_msa_server"] is True

    def test_use_msa_server_can_be_disabled(self, runner: OpenFold3Runner, tmp_path: Path) -> None:
        """The top-level use_msa_server field overrides the default."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = OpenFold3Input(
            sequences={"A": "MVLSPADKTNVKAAWGKVGA"},
            use_msa_server=False,
        )
        runner.prepare_workspace(input_data, workspace)

        cfg = json.loads(workspace.config_path.read_text())
        assert cfg["use_msa_server"] is False

    def test_use_templates_default_true(self, runner: OpenFold3Runner, tmp_path: Path) -> None:
        workspace = Workspace.create(tmp_path / "ws")
        input_data = OpenFold3Input(sequences={"A": "MVLSPADKTNVKAAWGKVGA"})
        runner.prepare_workspace(input_data, workspace)

        cfg = json.loads(workspace.config_path.read_text())
        assert cfg["use_templates"] is True

    def test_use_templates_can_be_disabled(self, runner: OpenFold3Runner, tmp_path: Path) -> None:
        """The top-level use_templates field overrides the default."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = OpenFold3Input(
            sequences={"A": "MVLSPADKTNVKAAWGKVGA"},
            use_templates=False,
        )
        runner.prepare_workspace(input_data, workspace)

        cfg = json.loads(workspace.config_path.read_text())
        assert cfg["use_templates"] is False

    def test_pae_enabled_default_true(self, runner: OpenFold3Runner, tmp_path: Path) -> None:
        workspace = Workspace.create(tmp_path / "ws")
        input_data = OpenFold3Input(sequences={"A": "MVLSPADKTNVKAAWGKVGA"})
        runner.prepare_workspace(input_data, workspace)

        cfg = json.loads(workspace.config_path.read_text())
        assert cfg["pae_enabled"] is True

    def test_pae_can_be_disabled(self, runner: OpenFold3Runner, tmp_path: Path) -> None:
        """The top-level pae_enabled field overrides the default."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = OpenFold3Input(
            sequences={"A": "MVLSPADKTNVKAAWGKVGA"},
            pae_enabled=False,
        )
        runner.prepare_workspace(input_data, workspace)

        cfg = json.loads(workspace.config_path.read_text())
        assert cfg["pae_enabled"] is False

    def test_entity_types_dna(self, runner: OpenFold3Runner, tmp_path: Path) -> None:
        workspace = Workspace.create(tmp_path / "ws")
        input_data = OpenFold3Input(
            sequences={"A": "MVLSPADKTNVKAAWGKVGA", "B": "ATCGATCG"},
            entity_types={"B": "dna"},
        )
        runner.prepare_workspace(input_data, workspace)

        query = json.loads((workspace.inputs_dir / "query.json").read_text())
        chains = query["queries"]["query_1"]["chains"]
        chain_b = next(c for c in chains if c["chain_ids"] == "B")
        assert chain_b["molecule_type"] == "dna"
        assert chain_b["sequence"] == "ATCGATCG"

    def test_entity_types_rna(self, runner: OpenFold3Runner, tmp_path: Path) -> None:
        workspace = Workspace.create(tmp_path / "ws")
        input_data = OpenFold3Input(
            sequences={"A": "MVLSPADKTNVKAAWGKVGA", "R": "ACGUACGU"},
            entity_types={"R": "rna"},
        )
        runner.prepare_workspace(input_data, workspace)

        query = json.loads((workspace.inputs_dir / "query.json").read_text())
        chains = query["queries"]["query_1"]["chains"]
        chain_r = next(c for c in chains if c["chain_ids"] == "R")
        assert chain_r["molecule_type"] == "rna"

    def test_ligand_smiles_dict(self, runner: OpenFold3Runner, tmp_path: Path) -> None:
        """Ligand via SMILES dict generates correct chain entry."""
        smiles = "CC(=O)NC1=CC=C(O)C=C1"
        workspace = Workspace.create(tmp_path / "ws")
        input_data = OpenFold3Input(
            sequences={"A": "MVLSPADKTNVKAAWGKVGA", "L": ""},
            entity_types={"L": {"smiles": smiles}},
        )
        runner.prepare_workspace(input_data, workspace)

        query = json.loads((workspace.inputs_dir / "query.json").read_text())
        chains = query["queries"]["query_1"]["chains"]
        chain_l = next(c for c in chains if c["chain_ids"] == "L")
        assert chain_l["molecule_type"] == "ligand"
        assert chain_l["smiles"] == smiles
        assert "sequence" not in chain_l

    def test_ligand_ccd_dict(self, runner: OpenFold3Runner, tmp_path: Path) -> None:
        """Ligand via CCD code generates correct chain entry."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = OpenFold3Input(
            sequences={"A": "MVLSPADKTNVKAAWGKVGA", "L": ""},
            entity_types={"L": {"ccd": "ATP"}},
        )
        runner.prepare_workspace(input_data, workspace)

        query = json.loads((workspace.inputs_dir / "query.json").read_text())
        chains = query["queries"]["query_1"]["chains"]
        chain_l = next(c for c in chains if c["chain_ids"] == "L")
        assert chain_l["molecule_type"] == "ligand"
        assert chain_l["ccd_codes"] == "ATP"

    def test_ligand_string_uses_sequence_as_smiles(
        self, runner: OpenFold3Runner, tmp_path: Path
    ) -> None:
        """entity_types: {"L": "ligand"} uses sequence value as SMILES."""
        smiles = "CC(C)Cc1ccc(cc1)C(C)C(O)=O"
        workspace = Workspace.create(tmp_path / "ws")
        input_data = OpenFold3Input(
            sequences={"A": "MVLSPADKTNVKAAWGKVGA", "L": smiles},
            entity_types={"L": "ligand"},
        )
        runner.prepare_workspace(input_data, workspace)

        query = json.loads((workspace.inputs_dir / "query.json").read_text())
        chains = query["queries"]["query_1"]["chains"]
        chain_l = next(c for c in chains if c["chain_ids"] == "L")
        assert chain_l["molecule_type"] == "ligand"
        assert chain_l["smiles"] == smiles

    def test_non_canonical_residues(self, runner: OpenFold3Runner, tmp_path: Path) -> None:
        """Non-canonical residues are added to the chain entry."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = OpenFold3Input(
            sequences={"A": "MVLSPADKTNVKAAWGKVGA"},
            non_canonical_residues={"A": {"3": "MHO", "5": "SEP"}},
        )
        runner.prepare_workspace(input_data, workspace)

        query = json.loads((workspace.inputs_dir / "query.json").read_text())
        chains = query["queries"]["query_1"]["chains"]
        assert chains[0]["non_canonical_residues"] == {"3": "MHO", "5": "SEP"}

    def test_templates_copied(self, runner: OpenFold3Runner, tmp_path: Path) -> None:
        tmpl = tmp_path / "template.cif"
        tmpl.write_text("data_test\n_entry.id test\n")

        workspace = Workspace.create(tmp_path / "ws")
        input_data = OpenFold3Input(
            sequences={"A": "MVLSPADKTNVKAAWGKVGA"},
            templates=[tmpl],
        )
        runner.prepare_workspace(input_data, workspace)

        assert (workspace.inputs_dir / "template.cif").exists()

        cfg = json.loads(workspace.config_path.read_text())
        assert "templates" not in cfg
        assert "template_path" not in cfg

    def test_msa_files_copied(self, runner: OpenFold3Runner, tmp_path: Path) -> None:
        """MSA files from the msa_paths field are copied into workspace."""
        msa_file = tmp_path / "A.a3m"
        msa_file.write_text(">query\nMVLSPADK\n")

        workspace = Workspace.create(tmp_path / "ws")
        input_data = OpenFold3Input(
            sequences={"A": "MVLSPADKTNVKAAWGKVGA"},
            msa_paths=[str(msa_file)],
        )
        runner.prepare_workspace(input_data, workspace)

        assert (workspace.inputs_dir / "A.a3m").exists()

    def test_raw_query_json_passthrough_dict(self, runner: OpenFold3Runner, tmp_path: Path) -> None:
        """query_json dict bypasses automatic generation."""
        custom_query = {
            "queries": {
                "my_query": {
                    "chains": [
                        {"molecule_type": "protein", "chain_ids": "X", "sequence": "MKWVTFIS"}
                    ]
                }
            }
        }
        workspace = Workspace.create(tmp_path / "ws")
        input_data = OpenFold3Input(
            sequences={},
            query_json=custom_query,
        )
        runner.prepare_workspace(input_data, workspace)

        query = json.loads((workspace.inputs_dir / "query.json").read_text())
        assert "my_query" in query["queries"]

    def test_raw_query_json_passthrough_string(
        self, runner: OpenFold3Runner, tmp_path: Path
    ) -> None:
        """query_json string is written directly."""
        custom_json = '{"queries": {"q1": {"chains": []}}}'
        workspace = Workspace.create(tmp_path / "ws")
        input_data = OpenFold3Input(
            sequences={},
            query_json=custom_json,
        )
        runner.prepare_workspace(input_data, workspace)

        content = (workspace.inputs_dir / "query.json").read_text()
        assert content == custom_json

    def test_query_json_none_builds_from_sequences(
        self, runner: OpenFold3Runner, tmp_path: Path
    ) -> None:
        """Footgun resolution: query_json=None builds query.json FROM sequences.

        Pre-migration, ``extra["query_json"] = None`` with presence-based checks
        would ``json.dumps(None)`` the literal 4-char string "null", silently
        discarding any valid ``sequences``. The typed field's ``is not None``
        check treats ``query_json=None`` as "not provided" consistently.
        """
        workspace = Workspace.create(tmp_path / "ws")
        input_data = OpenFold3Input(sequences={"A": "MKT"}, query_json=None)
        runner.prepare_workspace(input_data, workspace)

        content = (workspace.inputs_dir / "query.json").read_text()
        assert content != "null"

        query = json.loads(content)
        chains = query["queries"]["query_1"]["chains"]
        assert len(chains) == 1
        assert chains[0]["chain_ids"] == "A"
        assert chains[0]["sequence"] == "MKT"

    def test_extra_dict_merged(self, runner: OpenFold3Runner, tmp_path: Path) -> None:
        """CLI-level extra keys appear in config.json; typed fields do not."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = OpenFold3Input(
            sequences={"A": "MVLSPADKTNVKAAWGKVGA"},
            entity_types={"A": "protein"},
            extra={
                "num_model_seeds": 3,
                "seed": 42,
                "msa_server_url": "https://my-server.com",
            },
        )
        runner.prepare_workspace(input_data, workspace)

        cfg = json.loads(workspace.config_path.read_text())
        assert cfg["num_model_seeds"] == 3
        assert cfg["seed"] == 42
        assert cfg["msa_server_url"] == "https://my-server.com"
        # entity_types is a typed field, not a config.json key at all
        assert "entity_types" not in cfg

    def test_multi_entity_complex(self, runner: OpenFold3Runner, tmp_path: Path) -> None:
        """Complex with protein, DNA, RNA, and ligand generates correct query."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = OpenFold3Input(
            sequences={
                "A": "MVLSPADKTNVKAAWGKVGA",
                "B": "ATCGATCG",
                "C": "ACGUACGU",
                "D": "CC(=O)O",
            },
            entity_types={
                "B": "dna",
                "C": "rna",
                "D": "ligand",
            },
        )
        runner.prepare_workspace(input_data, workspace)

        query = json.loads((workspace.inputs_dir / "query.json").read_text())
        chains = query["queries"]["query_1"]["chains"]
        assert len(chains) == 4

        types = {c["chain_ids"]: c["molecule_type"] for c in chains}
        assert types == {"A": "protein", "B": "dna", "C": "rna", "D": "ligand"}

    def test_config_full_dict_equality_minimal(
        self, runner: OpenFold3Runner, tmp_path: Path
    ) -> None:
        """Full config.json dict is byte-compat with the pre-migration output (minimal input)."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = OpenFold3Input(sequences={"A": "MVLSPADKTNVKAAWGKVGA"})
        runner.prepare_workspace(input_data, workspace)

        cfg = json.loads(workspace.config_path.read_text())
        assert cfg == {
            "query_json_path": "/workspace/inputs/query.json",
            "output_dir": "/workspace/outputs/raw",
            "checkpoint_path": "/app/openfold3/weights/of3-p2-155k.pt",
            "use_msa_server": True,
            "use_templates": True,
            "pae_enabled": True,
            "num_diffusion_samples": 1,
        }

    def test_config_full_dict_equality_with_overrides(
        self, runner: OpenFold3Runner, tmp_path: Path
    ) -> None:
        """Full config.json dict with typed overrides and extra CLI knobs."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = OpenFold3Input(
            sequences={"A": "MVLSPADKTNVKAAWGKVGA"},
            num_models=3,
            use_msa_server=False,
            use_templates=False,
            pae_enabled=False,
            extra={"num_model_seeds": 3, "seed": 42, "output_format": "pdb"},
        )
        runner.prepare_workspace(input_data, workspace)

        cfg = json.loads(workspace.config_path.read_text())
        assert cfg == {
            "query_json_path": "/workspace/inputs/query.json",
            "output_dir": "/workspace/outputs/raw",
            "checkpoint_path": "/app/openfold3/weights/of3-p2-155k.pt",
            "use_msa_server": False,
            "use_templates": False,
            "pae_enabled": False,
            "num_diffusion_samples": 3,
            "num_model_seeds": 3,
            "seed": 42,
            "output_format": "pdb",
        }

    def test_generated_query_json_equality_protein_ligand_noncanonical(
        self, runner: OpenFold3Runner, tmp_path: Path
    ) -> None:
        """The generated query.json is unchanged for a representative protein+ligand input."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = OpenFold3Input(
            sequences={"A": "MVLSPADKTNVKAAWGKVGA", "L": "CC(=O)NC1=CC=C(O)C=C1"},
            entity_types={"L": "ligand"},
            non_canonical_residues={"A": {"3": "MHO"}},
        )
        runner.prepare_workspace(input_data, workspace)

        query = json.loads((workspace.inputs_dir / "query.json").read_text())
        expected = {
            "queries": {
                "query_1": {
                    "chains": [
                        {
                            "molecule_type": "protein",
                            "chain_ids": "A",
                            "sequence": "MVLSPADKTNVKAAWGKVGA",
                            "non_canonical_residues": {"3": "MHO"},
                        },
                        {
                            "molecule_type": "ligand",
                            "chain_ids": "L",
                            "smiles": "CC(=O)NC1=CC=C(O)C=C1",
                        },
                    ]
                }
            }
        }
        assert query == expected


# ---------------------------------------------------------------------------
# TestOpenFold3HostValidation
# ---------------------------------------------------------------------------


class TestOpenFold3HostValidation:
    """Tests for host-side input validation."""

    def test_empty_sequences_raises(self, runner: OpenFold3Runner, tmp_path: Path) -> None:
        workspace = Workspace.create(tmp_path / "ws")
        input_data = OpenFold3Input(sequences={})
        with pytest.raises(AutobioError, match="sequences must be non-empty"):
            runner.prepare_workspace(input_data, workspace)

    def test_empty_sequences_with_query_json_none_raises(
        self, runner: OpenFold3Runner, tmp_path: Path
    ) -> None:
        """Footgun resolution: query_json=None with empty sequences fails fast."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = OpenFold3Input(sequences={}, query_json=None)
        with pytest.raises(AutobioError, match="sequences must be non-empty"):
            runner.prepare_workspace(input_data, workspace)

    def test_empty_sequences_ok_with_query_json(
        self, runner: OpenFold3Runner, tmp_path: Path
    ) -> None:
        """Empty sequences is allowed when query_json is provided."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = OpenFold3Input(
            sequences={},
            query_json={"queries": {"q1": {"chains": []}}},
        )
        # Should not raise
        runner.prepare_workspace(input_data, workspace)

    def test_missing_template_file_raises(self, runner: OpenFold3Runner, tmp_path: Path) -> None:
        workspace = Workspace.create(tmp_path / "ws")
        input_data = OpenFold3Input(
            sequences={"A": "MVLSPADKTNVKAAWGKVGA"},
            templates=[tmp_path / "nonexistent.cif"],
        )
        with pytest.raises(AutobioError, match="Template file does not exist"):
            runner.prepare_workspace(input_data, workspace)

    def test_missing_msa_file_raises(self, runner: OpenFold3Runner, tmp_path: Path) -> None:
        workspace = Workspace.create(tmp_path / "ws")
        input_data = OpenFold3Input(
            sequences={"A": "MVLSPADKTNVKAAWGKVGA"},
            msa_paths=[str(tmp_path / "nonexistent.a3m")],
        )
        with pytest.raises(AutobioError, match="MSA file does not exist"):
            runner.prepare_workspace(input_data, workspace)

    def test_entity_types_unknown_chain_raises(
        self, runner: OpenFold3Runner, tmp_path: Path
    ) -> None:
        workspace = Workspace.create(tmp_path / "ws")
        input_data = OpenFold3Input(
            sequences={"A": "MVLSPADKTNVKAAWGKVGA"},
            entity_types={"Z": "dna"},
        )
        with pytest.raises(AutobioError, match="unknown chain IDs"):
            runner.prepare_workspace(input_data, workspace)

    def test_invalid_entity_type_string_raises(
        self, runner: OpenFold3Runner, tmp_path: Path
    ) -> None:
        workspace = Workspace.create(tmp_path / "ws")
        input_data = OpenFold3Input(
            sequences={"A": "MVLSPADKTNVKAAWGKVGA"},
            entity_types={"A": "peptide"},
        )
        with pytest.raises(AutobioError, match="Invalid entity type"):
            runner.prepare_workspace(input_data, workspace)

    def test_invalid_entity_type_dict_raises(self, runner: OpenFold3Runner, tmp_path: Path) -> None:
        workspace = Workspace.create(tmp_path / "ws")
        input_data = OpenFold3Input(
            sequences={"A": "MVLSPADKTNVKAAWGKVGA"},
            entity_types={"A": {"invalid_key": "value"}},
        )
        with pytest.raises(AutobioError, match="must contain 'smiles' or 'ccd' key"):
            runner.prepare_workspace(input_data, workspace)

    def test_non_canonical_residues_unknown_chain_raises(
        self, runner: OpenFold3Runner, tmp_path: Path
    ) -> None:
        workspace = Workspace.create(tmp_path / "ws")
        input_data = OpenFold3Input(
            sequences={"A": "MVLSPADKTNVKAAWGKVGA"},
            non_canonical_residues={"Z": {"1": "MHO"}},
        )
        with pytest.raises(AutobioError, match="non_canonical_residues references unknown"):
            runner.prepare_workspace(input_data, workspace)


# ---------------------------------------------------------------------------
# TestOpenFold3ExtraShadowRejection
# ---------------------------------------------------------------------------


class TestOpenFold3ExtraShadowRejection:
    """Promoted keys must no longer be accepted via extra — _apply_extra rejects them."""

    def test_entity_types_via_extra_rejected(self, runner: OpenFold3Runner, tmp_path: Path) -> None:
        workspace = Workspace.create(tmp_path / "ws")
        input_data = OpenFold3Input(
            sequences={"A": "MVLSPADKTNVKAAWGKVGA"},
            extra={"entity_types": {"A": "protein"}},
        )
        with pytest.raises(AutobioError, match="collide with typed input fields"):
            runner.prepare_workspace(input_data, workspace)

    def test_non_canonical_residues_via_extra_rejected(
        self, runner: OpenFold3Runner, tmp_path: Path
    ) -> None:
        workspace = Workspace.create(tmp_path / "ws")
        input_data = OpenFold3Input(
            sequences={"A": "MVLSPADKTNVKAAWGKVGA"},
            extra={"non_canonical_residues": {"A": {"1": "MHO"}}},
        )
        with pytest.raises(AutobioError, match="collide with typed input fields"):
            runner.prepare_workspace(input_data, workspace)

    def test_msa_paths_via_extra_rejected(self, runner: OpenFold3Runner, tmp_path: Path) -> None:
        workspace = Workspace.create(tmp_path / "ws")
        input_data = OpenFold3Input(
            sequences={"A": "MVLSPADKTNVKAAWGKVGA"},
            extra={"msa_paths": ["a3m.a3m"]},
        )
        with pytest.raises(AutobioError, match="collide with typed input fields"):
            runner.prepare_workspace(input_data, workspace)

    def test_query_json_via_extra_rejected(self, runner: OpenFold3Runner, tmp_path: Path) -> None:
        workspace = Workspace.create(tmp_path / "ws")
        input_data = OpenFold3Input(
            sequences={"A": "MVLSPADKTNVKAAWGKVGA"},
            extra={"query_json": {"queries": {}}},
        )
        with pytest.raises(AutobioError, match="collide with typed input fields"):
            runner.prepare_workspace(input_data, workspace)

    def test_use_msa_server_via_extra_rejected(
        self, runner: OpenFold3Runner, tmp_path: Path
    ) -> None:
        workspace = Workspace.create(tmp_path / "ws")
        input_data = OpenFold3Input(
            sequences={"A": "MVLSPADKTNVKAAWGKVGA"},
            extra={"use_msa_server": False},
        )
        with pytest.raises(AutobioError, match="collide with typed input fields"):
            runner.prepare_workspace(input_data, workspace)

    def test_use_templates_via_extra_rejected(
        self, runner: OpenFold3Runner, tmp_path: Path
    ) -> None:
        workspace = Workspace.create(tmp_path / "ws")
        input_data = OpenFold3Input(
            sequences={"A": "MVLSPADKTNVKAAWGKVGA"},
            extra={"use_templates": False},
        )
        with pytest.raises(AutobioError, match="collide with typed input fields"):
            runner.prepare_workspace(input_data, workspace)

    def test_pae_enabled_via_extra_rejected(self, runner: OpenFold3Runner, tmp_path: Path) -> None:
        workspace = Workspace.create(tmp_path / "ws")
        input_data = OpenFold3Input(
            sequences={"A": "MVLSPADKTNVKAAWGKVGA"},
            extra={"pae_enabled": False},
        )
        with pytest.raises(AutobioError, match="collide with typed input fields"):
            runner.prepare_workspace(input_data, workspace)


# ---------------------------------------------------------------------------
# TestOpenFold3ParseOutput
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


class TestOpenFold3ParseOutput:
    """Tests for OpenFold3Runner.parse_output."""

    def test_parse_single_model(self, runner: OpenFold3Runner, tmp_path: Path) -> None:
        workspace = Workspace.create(tmp_path / "ws")
        (workspace.std_output_dir / "result_data.json").write_text(json.dumps(_SINGLE_MODEL_RESULT))

        output = runner.parse_output(workspace)
        assert len(output.structures) == 1
        s = output.structures[0]
        assert s.model_rank == 1
        assert s.plddt_mean == pytest.approx(91.675)
        assert s.ptm == pytest.approx(0.89)
        assert s.iptm == pytest.approx(0.85)

    def test_parse_multiple_models(self, runner: OpenFold3Runner, tmp_path: Path) -> None:
        workspace = Workspace.create(tmp_path / "ws")
        (workspace.std_output_dir / "result_data.json").write_text(json.dumps(_MULTI_MODEL_RESULT))

        output = runner.parse_output(workspace)
        assert len(output.structures) == 3
        assert output.structures[0].model_rank == 1
        assert output.structures[2].model_rank == 3

    def test_confidence_metrics(self, runner: OpenFold3Runner, tmp_path: Path) -> None:
        workspace = Workspace.create(tmp_path / "ws")
        (workspace.std_output_dir / "result_data.json").write_text(json.dumps(_MULTI_MODEL_RESULT))

        output = runner.parse_output(workspace)
        assert output.confidence.best_plddt_mean == pytest.approx(91.0)
        assert output.confidence.best_ptm == pytest.approx(0.92)
        assert output.confidence.best_iptm == pytest.approx(0.88)

    def test_plddt_per_residue(self, runner: OpenFold3Runner, tmp_path: Path) -> None:
        workspace = Workspace.create(tmp_path / "ws")
        (workspace.std_output_dir / "result_data.json").write_text(json.dumps(_SINGLE_MODEL_RESULT))

        output = runner.parse_output(workspace)
        assert output.structures[0].plddt_per_residue is not None
        assert len(output.structures[0].plddt_per_residue) == 4

    def test_output_type(self, runner: OpenFold3Runner, tmp_path: Path) -> None:
        workspace = Workspace.create(tmp_path / "ws")
        (workspace.std_output_dir / "result_data.json").write_text(json.dumps(_SINGLE_MODEL_RESULT))

        output = runner.parse_output(workspace)
        assert isinstance(output, StructurePredictionOutput)

    def test_raw_output_path(self, runner: OpenFold3Runner, tmp_path: Path) -> None:
        workspace = Workspace.create(tmp_path / "ws")
        (workspace.std_output_dir / "result_data.json").write_text(json.dumps(_SINGLE_MODEL_RESULT))

        output = runner.parse_output(workspace)
        assert output.raw_output_path == workspace.raw_output_dir

    def test_container_paths_resolved_to_host(
        self, runner: OpenFold3Runner, tmp_path: Path
    ) -> None:
        """Container-internal /workspace/... paths are remapped to host workspace."""
        workspace = Workspace.create(tmp_path / "ws")
        (workspace.std_output_dir / "result_data.json").write_text(json.dumps(_SINGLE_MODEL_RESULT))

        output = runner.parse_output(workspace)
        expected = workspace.root / "outputs" / "standardized" / "model_1.cif"
        assert output.structures[0].structure_path == expected

    def test_null_confidence_values(self, runner: OpenFold3Runner, tmp_path: Path) -> None:
        """Handles null confidence values (e.g., when PAE is disabled)."""
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
# TestOpenFold3Registration
# ---------------------------------------------------------------------------


class TestOpenFold3Registration:
    """Tests for catalog Tool and runner registration."""

    def test_openfold3_registered_as_catalog_tool(self) -> None:
        import autobio.tools  # noqa: F401 - importing populates the catalog

        assert "openfold3" in CATALOG
        assert set(get_tool("openfold3").modes) == {"predict"}
        assert get_tool("openfold3").default_mode == "predict"
        assert get_tool("openfold3").category == ToolCategory.STRUCTURE_PREDICTION
        assert get_tool("openfold3").requires_gpu is True

    def test_openfold3_tool_constant_registered(self) -> None:
        import autobio.tools  # noqa: F401 - importing populates the catalog

        assert OPENFOLD3_TOOL.name == "openfold3"
        assert get_tool("openfold3") is OPENFOLD3_TOOL

    def test_tool_runners_registered(self) -> None:
        assert "openfold3" in TOOL_RUNNERS
        assert TOOL_RUNNERS["openfold3"] is OpenFold3Runner

    def test_get_runner_returns_openfold3_runner(self, config: AutobioConfig) -> None:
        with (
            patch("autobio.tools.base.ContainerManager"),
            patch("autobio.tools.base.GPUManager"),
        ):
            r = get_runner("openfold3", config)
        assert isinstance(r, OpenFold3Runner)
        assert r.tool_name == "openfold3"

    def test_notes_populated(self) -> None:
        """Notes contain key operational topics for agent guidance."""
        notes = " ".join(get_tool("openfold3").modes["predict"].notes)
        assert "msa" in notes.lower()
        assert "pae" in notes.lower()
        assert "msa_server_url" in notes


def test_info_snapshot_openfold3() -> None:
    import autobio.tools  # noqa: F401 - importing populates the catalog
    from autobio.cli.formatters import OutputFormat, format_tool_info_catalog

    parsed = json.loads(format_tool_info_catalog(get_tool("openfold3"), OutputFormat.JSON))
    assert parsed["modes"][0]["name"] == "predict"
    props = parsed["modes"][0]["input_schema"]["properties"]
    assert props["use_msa_server"]["x-autobio"]["widget"] == "toggle"
    assert props["entity_types"]["x-autobio"]["tier"] == "advanced"
    assert "output_schema" in parsed["modes"][0]
    assert parsed["modes"][0]["notes"]
