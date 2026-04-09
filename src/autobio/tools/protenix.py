"""Protenix-v2 structure prediction tool runner.

Protenix-v2 is ByteDance's 464M parameter biomolecular structure prediction
model supporting proteins, DNA, RNA, ligands, and ions.

Simple protein predictions use the ``sequences`` dict on
``StructurePredictionInput``.  For multi-entity predictions (DNA, RNA, ligands,
ions) agents specify ``extra["entity_types"]``.  For full control, provide a raw
Protenix JSON via ``extra["query_json"]``.

MSA handling is unique: this tool queries a ColabFold MSA server (configurable
URL) and converts the result to Protenix's paired/unpaired per-chain format
inside the container.  Parameters not directly exposed on
``StructurePredictionInput`` (MSA server URL, seeds, dtype, etc.) are passed
through the ``extra`` dict.
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

_DEFAULT_MODEL_NAME = "protenix_base_default_v1.0.0"

# Valid molecule type strings for Protenix input JSON.
_VALID_ENTITY_TYPES = frozenset({"protein", "dna", "rna", "ligand", "ion"})

# Keys in ``extra`` that are consumed by the runner and should NOT be
# flat-merged into config.json.
_CONSUMED_EXTRA_KEYS = frozenset(
    {
        "entity_types",
        "query_json",
        "msa_paths",
        "covalent_bonds",
        "constraints",
    }
)

# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


class ProtenixRunner(ToolRunner):
    """Runner for Protenix-v2 structure prediction.

    ``prepare_workspace`` generates a Protenix input JSON from the
    standardised ``StructurePredictionInput`` fields and writes
    ``config.json``.  ``parse_output`` reads the standardised
    ``result_data.json`` produced by the container's ``standardize.py``.
    """

    def prepare_workspace(self, input_data: BaseInput, workspace: Workspace) -> None:
        """Write config.json and input JSON to the workspace."""
        assert isinstance(input_data, StructurePredictionInput)

        # -- Host-side validation (fail fast before container launch) --------
        self._validate_inputs(input_data)

        # -- Generate or pass through Protenix JSON -------------------------
        if "query_json" in input_data.extra:
            query_json = input_data.extra["query_json"]
            if isinstance(query_json, str):
                query_content = query_json
            else:
                query_content = json.dumps(query_json, indent=2)
        else:
            query_list = self._build_protenix_json(input_data)
            query_content = json.dumps(query_list, indent=2)

        workspace.write_input_file("input.json", query_content.encode())

        # -- Copy template files into workspace -----------------------------
        if input_data.templates:
            for tmpl_path in input_data.templates:
                shutil.copy2(tmpl_path, workspace.inputs_dir / tmpl_path.name)

        # -- Copy MSA files into workspace ----------------------------------
        msa_paths = input_data.extra.get("msa_paths")
        if msa_paths:
            msa_dest = workspace.inputs_dir / "msa"
            msa_dest.mkdir(parents=True, exist_ok=True)
            for msa_path_str in msa_paths:
                msa_path = Path(msa_path_str)
                if msa_path.is_dir():
                    shutil.copytree(msa_path, msa_dest / msa_path.name)
                else:
                    shutil.copy2(msa_path, msa_dest / msa_path.name)

        # -- Build config.json ----------------------------------------------
        config: dict[str, object] = {
            "input_json_path": "/workspace/inputs/input.json",
            "output_dir": "/workspace/outputs/raw",
            "model_name": _DEFAULT_MODEL_NAME,
            "use_msa": True,
            "use_template": False,
            "dtype": "bf16",
            "seeds": "101",
            "diffusion_samples": 5,
            "diffusion_steps": 200,
            "pairformer_cycles": 10,
        }

        # Map num_models → diffusion_samples
        config["diffusion_samples"] = input_data.num_models

        # Flat-merge extra dict for pass-through parameters (excluding
        # consumed keys).  This allows extra["use_msa"] = False
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
    def _build_protenix_json(input_data: StructurePredictionInput) -> list[dict[str, object]]:
        """Generate a Protenix input JSON list from structured input fields.

        Each entry in ``sequences`` becomes an entity in the Protenix JSON.
        The default entity type is ``protein``; override per-chain via
        ``extra["entity_types"]``.

        Entity type values can be:
        - A string: ``"protein"``, ``"dna"``, ``"rna"``, ``"ligand"``, ``"ion"``
        - A dict for ligands: ``{"smiles": "CC(=O)..."}`` or ``{"ccd": "ATP"}``

        Covalent bonds and constraints are passed via ``extra["covalent_bonds"]``
        and ``extra["constraints"]``.
        """
        entity_types: dict[str, object] = input_data.extra.get("entity_types", {})
        sequences_list: list[dict[str, object]] = []

        for chain_id, sequence in input_data.sequences.items():
            etype = entity_types.get(chain_id, "protein")

            if isinstance(etype, str) and etype == "protein":
                sequences_list.append(
                    {"proteinChain": {"sequence": sequence, "count": 1, "id": [chain_id]}}
                )

            elif isinstance(etype, str) and etype == "dna":
                sequences_list.append(
                    {"dnaSequence": {"sequence": sequence, "count": 1, "id": [chain_id]}}
                )

            elif isinstance(etype, str) and etype == "rna":
                sequences_list.append(
                    {"rnaSequence": {"sequence": sequence, "count": 1, "id": [chain_id]}}
                )

            elif isinstance(etype, str) and etype == "ion":
                sequences_list.append({"ion": {"ion": sequence, "count": 1}})

            elif isinstance(etype, str) and etype == "ligand":
                # String "ligand" — use sequence value as ligand specifier
                sequences_list.append({"ligand": {"ligand": sequence, "count": 1}})

            elif isinstance(etype, dict):
                if "smiles" in etype:
                    sequences_list.append({"ligand": {"ligand": etype["smiles"], "count": 1}})
                elif "ccd" in etype:
                    sequences_list.append({"ligand": {"ligand": f"CCD_{etype['ccd']}", "count": 1}})
                else:
                    raise AutobioError(
                        f"Unknown entity type dict for chain {chain_id!r}: {etype}. "
                        f"Expected {{'smiles': '...'}} or {{'ccd': '...'}}."
                    )

            else:
                raise AutobioError(
                    f"Invalid entity_types value for chain {chain_id!r}: {etype!r}. "
                    f"Must be one of {sorted(_VALID_ENTITY_TYPES)} or a dict "
                    f"({{'smiles': '...'}} or {{'ccd': '...'}})."
                )

        return [
            {
                "name": "prediction",
                "sequences": sequences_list,
                "covalent_bonds": input_data.extra.get("covalent_bonds", []),
                "constraint": input_data.extra.get("constraints", {}),
            }
        ]

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

        # Validate MSA files/dirs exist
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
                    if etype not in _VALID_ENTITY_TYPES:
                        raise AutobioError(
                            f"Invalid entity type for chain {chain_id!r}: {etype!r}. "
                            f"Must be one of {sorted(_VALID_ENTITY_TYPES)}."
                        )
                else:
                    raise AutobioError(
                        f"Invalid entity_types value for chain {chain_id!r}: {etype!r}. "
                        f"Must be a string or dict."
                    )

        # Validate covalent_bonds is a list if present
        covalent_bonds = input_data.extra.get("covalent_bonds")
        if covalent_bonds is not None and not isinstance(covalent_bonds, list):
            raise AutobioError(
                f"covalent_bonds must be a list, got {type(covalent_bonds).__name__}"
            )

        # Validate constraints is a dict if present
        constraints = input_data.extra.get("constraints")
        if constraints is not None and not isinstance(constraints, dict):
            raise AutobioError(f"constraints must be a dict, got {type(constraints).__name__}")


# ---------------------------------------------------------------------------
# Registry entry — populated when this module is imported
# ---------------------------------------------------------------------------

_PROTENIX_INPUT_FORMAT = (
    # JSON format overview
    "Protenix takes a JSON input file containing a list of job dicts.  Each job "
    "has 'name', 'sequences' (list of entity dicts), 'covalent_bonds' (list), "
    "and 'constraint' (dict).  Via the autobio API, each entry in the sequences "
    "dict becomes an entity (default type 'protein').  To specify other entity "
    "types, use extra['entity_types'] mapping chain IDs to types: "
    "{'B': 'dna', 'C': {'smiles': 'CC(=O)...'}, 'D': {'ccd': 'ATP'}, 'I': 'ion'}.",
    # Entity specification
    "Entity types — proteinChain: {proteinChain: {sequence: MKLL..., count: 1, id: [A]}}. "
    "dnaSequence: {dnaSequence: {sequence: ATCGATCG, count: 1}}. "
    "rnaSequence: {rnaSequence: {sequence: AUCGAUCG, count: 1}}. "
    "ligand (CCD): {ligand: {ligand: CCD_ATP, count: 1}} — use 'CCD_' prefix. "
    "ligand (SMILES): {ligand: {ligand: 'CC(=O)O', count: 1}}. "
    "ion: {ion: {ion: MG, count: 1}} — bare element code, no prefix. "
    "Via the API, ligands are specified in entity_types using SMILES "
    "({'smiles': 'CC...'}), CCD codes ({'ccd': 'ATP'}), or the string "
    "'ligand' (sequence value used as ligand specifier). Ions use "
    "entity_types: {'I': 'ion'} with the ion code as sequence value.",
    # Covalent bonds and constraints
    "Covalent bonds between entities (polymer-ligand, ligand-ligand, cyclic "
    "peptides) are specified via extra['covalent_bonds'] as a list of bond "
    "dicts.  Constraints (contact and pocket) are specified via "
    "extra['constraints'] as a dict with 'contact' and/or 'pocket' keys.",
    # MSA paths
    "Pre-computed MSAs can be provided per protein chain in the native JSON "
    "using 'pairedMsaPath' and 'unpairedMsaPath' fields on proteinChain "
    "entries.  When using the autobio API, provide pre-computed MSA "
    "directories via extra['msa_paths'] (list of directory paths).",
    # Complete example
    "Complete example — protein-ligand-ion JSON:\\n"
    "[{\\n"
    '  "name": "complex_1",\\n'
    '  "sequences": [\\n'
    '    {"proteinChain": {"sequence": "MKTAYIAKQRQIS...", "count": 1, "id": ["A"]}},\\n'
    '    {"ligand": {"ligand": "CCD_ATP", "count": 1}},\\n'
    '    {"ion": {"ion": "MG", "count": 1}}\\n'
    "  ],\\n"
    '  "covalent_bonds": [],\\n'
    '  "constraint": {}\\n'
    "}]",
    # Raw override
    "For full control over the native JSON format, provide the complete input "
    "JSON via extra['query_json'] (as a dict/list or JSON string).  This "
    "bypasses automatic generation.  See "
    "https://github.com/bytedance/Protenix/blob/main/docs/infer_json_format.md.",
)

_PROTENIX_NOTES = (
    # MSA options
    "MSA generation via ColabFold server is ENABLED BY DEFAULT "
    "(use_msa=true).  Protein sequences are submitted to the ColabFold "
    "server and converted to Protenix's paired/unpaired per-chain MSA "
    "format inside the container.  To disable, set 'use_msa': false in "
    "extra.  To use a private ColabFold server, set extra['msa_server_url'] "
    "to the server URL (default: https://api.colabfold.com).  For "
    "pre-computed MSAs, provide directories via extra['msa_paths'].",
    # RNA MSA limitation
    "RNA MSA search is NOT supported via the ColabFold server.  If RNA "
    "chains require MSAs, provide pre-computed RNA MSAs via the native "
    "JSON format (extra['query_json']) with 'unpairedMsaPath' on RNA "
    "entities.",
    # Template options
    "Template-based prediction is DISABLED BY DEFAULT (use_template=false). "
    "To enable, set extra['use_template'] = True.  Provide custom template "
    "structures via the 'templates' field on StructurePredictionInput.",
    # Ion entity type
    "Protenix uniquely supports ions as first-class entities.  Specify via "
    "entity_types: {'I': 'ion'} with the ion code as the sequence value "
    "(e.g., sequences: {'I': 'MG'}).  Common ions: MG, ZN, CA, FE, NA, K, "
    "CL, MN, CO, CU.",
    # Key parameters
    "Key extra parameters: 'seeds' (comma-separated string, default '101'), "
    "'model_name' (str, default 'protenix_base_default_v1.0.0'), 'dtype' ('bf16' or 'fp32', "
    "default 'bf16'), 'pairformer_cycles' (int, default 10), "
    "'diffusion_steps' (int, default 200), 'use_tfg_guidance' (bool, "
    "default false — Training-Free Guidance for improved ligand geometry), "
    "'msa_server_url' (str — custom ColabFold server URL). "
    "num_models on the input maps to diffusion_samples.",
    # GPU memory
    "Protenix-v2 requires substantial GPU memory.  Minimum: 40GB (A100 40GB). "
    "Recommended: A100 80GB or H100.  Memory usage scales with token count: "
    "500 tokens ~6GB, 1000 tokens ~18GB, 4000 tokens ~78GB.",
)

TOOL_REGISTRY["protenix_v2"] = ToolEntry(
    image_tag="protenix:1.0.0",
    category=ToolCategory.STRUCTURE_PREDICTION,
    requires_gpu=True,
    gpu_count=1,
    input_schema=StructurePredictionInput,
    output_schema=StructurePredictionOutput,
    default_timeout=7200,
    supports_batch=False,
    description=(
        "Predict biomolecular structures using Protenix-v2 (464M parameter model). "
        "Supports proteins, DNA, RNA, ligands, and ions with MSA-based prediction "
        "and diffusion sampling."
    ),
    version="1.0.0",
    notes=_PROTENIX_NOTES,
    input_format=_PROTENIX_INPUT_FORMAT,
)
