"""OpenFold3 structure prediction tool runner.

OpenFold3 is a fully open-source, trainable PyTorch reproduction of AlphaFold3.
It predicts biomolecular structures for proteins, DNA, RNA, ligands, and
non-canonical residues.

Simple protein predictions use the ``sequences`` dict on
``StructurePredictionInput``.  For multi-entity predictions (DNA, RNA, ligands)
agents specify ``extra["entity_types"]``.  For full control, provide a raw
OpenFold3 query JSON via ``extra["query_json"]``.

Parameters not directly exposed on ``StructurePredictionInput`` (MSA server URL,
PAE toggle, low memory mode, etc.) are passed through the ``extra`` dict.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

from autobio.core.registry import TOOL_REGISTRY, ToolCategory, ToolEntry
from autobio.core.result import AutobioError
from autobio.schemas.base import BaseInput  # noqa: TC001 - needed at runtime for isinstance
from autobio.schemas.structure_prediction import (
    ConfidenceMetrics,
    PredictedStructure,
    StructurePredictionInput,
    StructurePredictionOutput,
)
from autobio.tools.base import ToolRunner

if TYPE_CHECKING:
    from autobio.core.workspace import Workspace

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_CHECKPOINT_PATH = "/app/openfold3/weights/of3-p2-155k.pt"

# Valid molecule type strings for OpenFold3 query JSON.
_VALID_MOLECULE_TYPES = frozenset({"protein", "dna", "rna", "ligand"})

# Keys in ``extra`` that are consumed by the runner and should NOT be
# flat-merged into config.json.
_CONSUMED_EXTRA_KEYS = frozenset(
    {
        "entity_types",
        "query_json",
        "msa_paths",
        "non_canonical_residues",
    }
)

# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


class OpenFold3Runner(ToolRunner):
    """Runner for OpenFold3 structure prediction.

    ``prepare_workspace`` generates an OpenFold3 query JSON from the
    standardised ``StructurePredictionInput`` fields and writes
    ``config.json``.  ``parse_output`` reads the standardised
    ``result_data.json`` produced by the container's ``standardize.py``.
    """

    def prepare_workspace(self, input_data: BaseInput, workspace: Workspace) -> None:
        """Write config.json and query JSON to the workspace."""
        assert isinstance(input_data, StructurePredictionInput)

        # -- Host-side validation (fail fast before container launch) --------
        self._validate_inputs(input_data)

        # -- Generate or pass through query JSON -----------------------------
        if "query_json" in input_data.extra:
            query_json = input_data.extra["query_json"]
            if isinstance(query_json, str):
                query_content = query_json
            else:
                query_content = json.dumps(query_json, indent=2)
        else:
            query_dict = self._build_query_json(input_data)
            query_content = json.dumps(query_dict, indent=2)

        workspace.write_input_file("query.json", query_content.encode())

        # -- Copy template files into workspace -----------------------------
        if input_data.templates:
            for tmpl_path in input_data.templates:
                shutil.copy2(tmpl_path, workspace.inputs_dir / tmpl_path.name)

        # -- Copy MSA files into workspace ----------------------------------
        msa_paths = input_data.extra.get("msa_paths")
        if msa_paths:
            for msa_path_str in msa_paths:
                msa_path = Path(msa_path_str)
                shutil.copy2(msa_path, workspace.inputs_dir / msa_path.name)

        # -- Build config.json ----------------------------------------------
        config: dict[str, object] = {
            "query_json_path": "/workspace/inputs/query.json",
            "output_dir": "/workspace/outputs/raw",
            "checkpoint_path": _CHECKPOINT_PATH,
            "use_msa_server": True,
            "use_templates": True,
            "pae_enabled": True,
        }

        # Map num_models → num_diffusion_samples
        config["num_diffusion_samples"] = input_data.num_models

        # Flat-merge extra dict for pass-through parameters (excluding
        # consumed keys).  This allows extra["use_msa_server"] = False
        # to override the default.
        for key, value in input_data.extra.items():
            if key not in _CONSUMED_EXTRA_KEYS:
                config[key] = value

        workspace.write_config(config)

    def parse_output(self, workspace: Workspace) -> StructurePredictionOutput:
        """Read standardised outputs and return a ``StructurePredictionOutput``."""
        result_data_path = workspace.std_output_dir / "result_data.json"
        data = json.loads(result_data_path.read_text())

        structures = [
            PredictedStructure(
                model_rank=s["model_rank"],
                structure_path=self._resolve_container_path(s["structure_path"], workspace),
                plddt_per_residue=s.get("plddt_per_residue"),
                plddt_mean=s.get("plddt_mean"),
                ptm=s.get("ptm"),
                iptm=s.get("iptm"),
                chain_mapping=s.get("chain_mapping"),
            )
            for s in data["structures"]
        ]

        conf = data.get("confidence", {})
        confidence = ConfidenceMetrics(
            best_plddt_mean=conf.get("best_plddt_mean"),
            best_ptm=conf.get("best_ptm"),
            best_iptm=conf.get("best_iptm"),
        )

        # Placeholder metadata — overwritten by base class run()
        return StructurePredictionOutput(
            structures=structures,
            confidence=confidence,
            metadata=self._build_metadata(workspace, 0.0, [], ""),
            raw_output_path=workspace.raw_output_dir,
        )

    @staticmethod
    def _build_query_json(input_data: StructurePredictionInput) -> dict:
        """Generate an OpenFold3 query JSON dict from structured input fields.

        Each entry in ``sequences`` becomes a chain in the query.  The default
        molecule type is ``protein``; override per-chain via
        ``extra["entity_types"]``.

        Entity type values can be:
        - A string: ``"protein"``, ``"dna"``, ``"rna"``
        - A dict for ligands: ``{"smiles": "CC(=O)..."}`` or ``{"ccd": "ATP"}``

        Non-canonical residues are specified via ``extra["non_canonical_residues"]``
        as a dict mapping chain IDs to dicts of 1-based position → CCD code:
        ``{"A": {"3": "MHO", "5": "SEP"}}``.
        """
        entity_types: dict = input_data.extra.get("entity_types", {})
        non_canonical: dict = input_data.extra.get("non_canonical_residues", {})
        chains: list[dict] = []

        for chain_id, sequence in input_data.sequences.items():
            etype = entity_types.get(chain_id, "protein")

            if isinstance(etype, str) and etype in ("protein", "dna", "rna"):
                chain: dict = {
                    "molecule_type": etype,
                    "chain_ids": chain_id,
                    "sequence": sequence,
                }
                # Add non-canonical residues if specified for this chain
                if chain_id in non_canonical:
                    chain["non_canonical_residues"] = non_canonical[chain_id]
                chains.append(chain)

            elif isinstance(etype, dict):
                # Structured ligand type: {"smiles": "CC..."} or {"ccd": "ATP"}
                ligand_chain: dict = {
                    "molecule_type": "ligand",
                    "chain_ids": chain_id,
                }
                if "smiles" in etype:
                    ligand_chain["smiles"] = etype["smiles"]
                elif "ccd" in etype:
                    ligand_chain["ccd_codes"] = etype["ccd"]
                else:
                    raise AutobioError(
                        f"Unknown entity type dict for chain {chain_id!r}: {etype}. "
                        f"Expected {{'smiles': '...'}} or {{'ccd': '...'}}."
                    )
                chains.append(ligand_chain)

            elif isinstance(etype, str) and etype == "ligand":
                # String "ligand" — use sequence value as SMILES
                chains.append(
                    {
                        "molecule_type": "ligand",
                        "chain_ids": chain_id,
                        "smiles": sequence,
                    }
                )

            else:
                raise AutobioError(
                    f"Invalid entity_types value for chain {chain_id!r}: {etype!r}. "
                    f"Must be one of {sorted(_VALID_MOLECULE_TYPES)} or a dict "
                    f"({{'smiles': '...'}} or {{'ccd': '...'}})."
                )

        return {
            "queries": {
                "query_1": {
                    "chains": chains,
                }
            }
        }

    @staticmethod
    def _resolve_container_path(container_path_str: str, workspace: Workspace) -> Path:
        """Map a container-internal ``/workspace/...`` path to the host workspace."""
        container_path = Path(container_path_str)
        try:
            relative = container_path.relative_to("/workspace")
        except ValueError:
            return container_path
        return workspace.root / relative

    @staticmethod
    def _validate_inputs(input_data: StructurePredictionInput) -> None:
        """Host-side validation — catch errors before container launch."""
        has_query_json = "query_json" in input_data.extra

        if not has_query_json and not input_data.sequences:
            raise AutobioError(
                "sequences must be non-empty, or provide a raw query JSON via extra['query_json']."
            )

        # Validate template files exist
        if input_data.templates:
            for tmpl_path in input_data.templates:
                if not tmpl_path.exists():
                    raise AutobioError(f"Template file does not exist: {tmpl_path}")

        # Validate MSA files exist
        msa_paths = input_data.extra.get("msa_paths")
        if msa_paths:
            for msa_path_str in msa_paths:
                msa_path = Path(msa_path_str)
                if not msa_path.exists():
                    raise AutobioError(f"MSA file does not exist: {msa_path}")

        # Validate entity_types keys match sequences
        entity_types = input_data.extra.get("entity_types", {})
        if entity_types and not has_query_json:
            unknown_chains = set(entity_types) - set(input_data.sequences)
            if unknown_chains:
                raise AutobioError(
                    f"entity_types references unknown chain IDs: {sorted(unknown_chains)}. "
                    f"Available chains: {sorted(input_data.sequences)}"
                )

            # Validate entity type values
            for chain_id, etype in entity_types.items():
                if isinstance(etype, dict):
                    if "smiles" not in etype and "ccd" not in etype:
                        raise AutobioError(
                            f"entity_types dict for chain {chain_id!r} must contain "
                            f"'smiles' or 'ccd' key, got: {etype}"
                        )
                elif isinstance(etype, str):
                    if etype not in _VALID_MOLECULE_TYPES:
                        raise AutobioError(
                            f"Invalid entity type for chain {chain_id!r}: {etype!r}. "
                            f"Must be one of {sorted(_VALID_MOLECULE_TYPES)}."
                        )
                else:
                    raise AutobioError(
                        f"Invalid entity_types value for chain {chain_id!r}: {etype!r}. "
                        f"Must be a string or dict."
                    )

        # Validate non_canonical_residues keys match sequences
        non_canonical = input_data.extra.get("non_canonical_residues", {})
        if non_canonical and not has_query_json:
            unknown_chains = set(non_canonical) - set(input_data.sequences)
            if unknown_chains:
                raise AutobioError(
                    f"non_canonical_residues references unknown chain IDs: "
                    f"{sorted(unknown_chains)}. "
                    f"Available chains: {sorted(input_data.sequences)}"
                )


# ---------------------------------------------------------------------------
# Registry entry — populated when this module is imported
# ---------------------------------------------------------------------------

_OPENFOLD3_INPUT_FORMAT = (
    # JSON format overview
    "OpenFold3 takes a JSON query file with top-level key 'queries' containing "
    "a dict of named queries. Each query has a 'chains' list defining the "
    "molecular entities. Via the autobio API, each entry in the sequences dict "
    "becomes a chain (default molecule_type 'protein'). To specify other entity "
    "types, use extra['entity_types'] mapping chain IDs to types: "
    "{'B': 'dna', 'C': {'smiles': 'CC(=O)NC1=CC=C(O)C=C1'}, 'D': {'ccd': 'ATP'}}.",
    # Chain specification
    "JSON chain fields — Each chain requires 'molecule_type' and 'chain_ids'. "
    "Protein: {molecule_type: protein, chain_ids: A, sequence: MKLL...}. "
    "DNA: {molecule_type: dna, chain_ids: B, sequence: ATCGATCG}. "
    "RNA: {molecule_type: rna, chain_ids: C, sequence: AUCGAUCG}. "
    "Ligand (SMILES): {molecule_type: ligand, chain_ids: D, smiles: 'CC...'}. "
    "Ligand (CCD): {molecule_type: ligand, chain_ids: D, ccd_codes: ATP}. "
    "Via the API, ligands are specified in entity_types using SMILES "
    "({'smiles': 'CC...'}), CCD codes ({'ccd': 'ATP'}), or the string "
    "'ligand' (sequence value used as SMILES). chain_ids can be a string "
    "or list for multiple identical chains.",
    # Non-canonical residues
    "Non-canonical residues are specified per-chain using a dict mapping "
    "1-based residue positions to CCD codes. In JSON: "
    "{molecule_type: protein, chain_ids: A, sequence: MKLLVV, "
    "non_canonical_residues: {3: MHO, 5: SEP}}. Via the API, use "
    "extra['non_canonical_residues']: {'A': {'3': 'MHO', '5': 'SEP'}}. "
    "MSA computation uses only the primary sequence.",
    # Optional chain fields
    "Optional chain fields: 'use_msas' (bool — enable/disable MSA for this "
    "chain), 'use_main_msas'/'use_paired_msas' (bool — MSA type control), "
    "'main_msa_file_paths'/'paired_msa_file_paths' (list — precomputed MSA "
    "files), 'template_alignment_file_path' (str — template data). Multiple "
    "queries in one JSON file enables batch inference.",
    # Complete example
    "Complete example — protein-ligand JSON:\\n"
    "{\\n"
    '  "queries": {\\n'
    '    "complex_1": {\\n'
    '      "chains": [\\n'
    "        {\\n"
    '          "molecule_type": "protein",\\n'
    '          "chain_ids": "A",\\n'
    '          "sequence": "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEK"\\n'
    "        },\\n"
    "        {\\n"
    '          "molecule_type": "ligand",\\n'
    '          "chain_ids": "B",\\n'
    '          "smiles": "CC(=O)NC1=CC=C(O)C=C1"\\n'
    "        }\\n"
    "      ]\\n"
    "    }\\n"
    "  }\\n"
    "}",
    # Raw override
    "For full control over the native JSON format, provide the complete query "
    "JSON via extra['query_json'] (as a dict or JSON string). This bypasses "
    "automatic query generation. See "
    "https://openfold-3.readthedocs.io/en/latest/input_format_reference.html.",
)

_OPENFOLD3_NOTES = (
    # MSA options
    "MSA generation via ColabFold server is ENABLED BY DEFAULT "
    "(use_msa_server=true). Only protein sequences are submitted to the server. "
    "To disable, set 'use_msa_server': false in extra. To use a private "
    "ColabFold server, set extra['msa_server_url'] to the server URL. "
    "For high-throughput screening, provide pre-computed MSAs via "
    "extra['msa_paths'] (list of file paths).",
    # Template options
    "Template-based prediction is ENABLED BY DEFAULT (use_templates=true). "
    "Templates are automatically retrieved when using the ColabFold server. "
    "To provide custom template structures, use the 'templates' field on "
    "StructurePredictionInput. To disable templates entirely, set "
    "extra['use_templates'] = False.",
    # PAE and confidence
    "The PAE (Predicted Aligned Error) head is ENABLED BY DEFAULT "
    "(pae_enabled=true). This produces pTM and ipTM confidence scores and "
    "enables the sample_ranking_score for ranking predictions. Disable via "
    "extra['pae_enabled'] = False to reduce memory usage at the cost of "
    "losing pTM/ipTM metrics.",
    # Key parameters
    "Key extra parameters: 'num_model_seeds' (int, default 1 — number of "
    "independent random seeds), 'seed' (int — specific seed value), "
    "'output_format' ('cif' or 'pdb', default 'cif'), 'low_memory' (bool, "
    "default false — enables memory optimization at the cost of speed), "
    "'msa_server_url' (str — custom ColabFold server URL), "
    "'num_devices' (int — number of GPUs for multi-GPU inference). "
    "num_models on the input maps to num_diffusion_samples.",
    # GPU memory
    "OpenFold3 requires substantial GPU memory. Minimum: 32GB (A100 40GB). "
    "Recommended: A100 80GB or H100. Set extra['low_memory'] = True for "
    "constrained GPUs (enables sequential pairformer processing). For very "
    "large complexes, consider reducing num_diffusion_samples.",
)

TOOL_REGISTRY["openfold3"] = ToolEntry(
    image_tag="openfold3:1.0.0",
    category=ToolCategory.STRUCTURE_PREDICTION,
    requires_gpu=True,
    gpu_count=1,
    input_schema=StructurePredictionInput,
    output_schema=StructurePredictionOutput,
    default_timeout=3600,
    supports_batch=False,
    description=(
        "Predict biomolecular structures using OpenFold3 (open-source AlphaFold3). "
        "Supports proteins, DNA, RNA, ligands, and non-canonical residues with "
        "MSA-based and template-based prediction."
    ),
    version="1.0.0",
    notes=_OPENFOLD3_NOTES,
    input_format=_OPENFOLD3_INPUT_FORMAT,
)
