"""OpenFold3 structure prediction tool runner.

OpenFold3 is a fully open-source, trainable PyTorch reproduction of AlphaFold3.
It predicts biomolecular structures for proteins, DNA, RNA, ligands, and
non-canonical residues.

Simple protein predictions use the ``sequences`` dict on ``OpenFold3Input``.
For multi-entity predictions (DNA, RNA, ligands) agents specify
``entity_types``. For full control, provide a raw OpenFold3 query JSON via
``query_json``.

CLI-level args (``num_model_seeds``, ``seed``, ``output_format``, etc.) are
passed through the ``extra`` dict on ``OpenFold3Input``.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

from autobio.core.catalog import Mode, Tool, register
from autobio.core.registry import ToolCategory
from autobio.core.result import AutobioError
from autobio.schemas.base import BaseInput  # noqa: TC001 - needed at runtime for isinstance
from autobio.schemas.structure_prediction import (
    ConfidenceMetrics,
    OpenFold3Input,
    PredictedStructure,
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

# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


class OpenFold3Runner(ToolRunner):
    """Runner for OpenFold3 structure prediction.

    ``prepare_workspace`` generates an OpenFold3 query JSON from the
    standardised ``OpenFold3Input`` fields and writes ``config.json``.
    ``parse_output`` reads the standardised ``result_data.json`` produced by
    the container's ``standardize.py``.
    """

    def prepare_workspace(self, input_data: BaseInput, workspace: Workspace) -> None:
        """Write config.json and query JSON to the workspace."""
        assert isinstance(input_data, OpenFold3Input)

        # -- Host-side validation (fail fast before container launch) --------
        self._validate_inputs(input_data)

        # -- Generate or pass through query JSON -----------------------------
        if input_data.query_json is not None:
            query_json = input_data.query_json
            if isinstance(query_json, str):
                query_content = query_json
            else:
                query_content = json.dumps(query_json, indent=2)
        else:
            query_dict = self._build_query_json(input_data)
            query_content = json.dumps(query_dict, indent=2)

        workspace.write_input_file("query.json", query_content.encode())

        # -- Copy MSA files into workspace ----------------------------------
        msa_paths = input_data.msa_paths
        if msa_paths:
            for msa_path_str in msa_paths:
                msa_path = Path(msa_path_str)
                shutil.copy2(msa_path, workspace.inputs_dir / msa_path.name)

        # -- Build config.json ----------------------------------------------
        config: dict[str, object] = {
            "query_json_path": "/workspace/inputs/query.json",
            "output_dir": "/workspace/outputs/raw",
            "checkpoint_path": _CHECKPOINT_PATH,
            "use_msa_server": input_data.use_msa_server,
            "use_templates": input_data.use_templates,
            "pae_enabled": input_data.pae_enabled,
        }

        # Map num_models → num_diffusion_samples
        config["num_diffusion_samples"] = input_data.num_models

        # Flat-merge extra dict for remaining CLI-level args.
        self._apply_extra(config, input_data)

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
    def _build_query_json(input_data: OpenFold3Input) -> dict[str, object]:
        """Generate an OpenFold3 query JSON dict from structured input fields.

        Each entry in ``sequences`` becomes a chain in the query.  The default
        molecule type is ``protein``; override per-chain via ``entity_types``.

        Entity type values can be:
        - A string: ``"protein"``, ``"dna"``, ``"rna"``
        - A dict for ligands: ``{"smiles": "CC(=O)..."}`` or ``{"ccd": "ATP"}``

        Non-canonical residues are specified via ``non_canonical_residues`` as a
        dict mapping chain IDs to dicts of 1-based position → CCD code:
        ``{"A": {"3": "MHO", "5": "SEP"}}``.

        Pre-computed MSAs are specified via ``msa_paths`` (list of file paths).
        The filename stem identifies the target chain ID (mirroring boltz's
        convention), e.g. ``"A.a3m"`` is wired onto chain ``"A"`` as
        ``main_msa_file_paths``. Ligand chains never receive an MSA.
        """
        entity_types: dict[str, object] = input_data.entity_types
        non_canonical: dict[str, object] = input_data.non_canonical_residues
        chains: list[dict[str, object]] = []

        msa_map: dict[str, str] = {}
        for msa_path_str in input_data.msa_paths or []:
            msa_path = Path(msa_path_str)
            msa_map[msa_path.stem] = f"/workspace/inputs/{msa_path.name}"

        for chain_id, sequence in input_data.sequences.items():
            etype = entity_types.get(chain_id, "protein")

            if isinstance(etype, str) and etype in ("protein", "dna", "rna"):
                chain: dict[str, object] = {
                    "molecule_type": etype,
                    "chain_ids": chain_id,
                    "sequence": sequence,
                }
                # Add non-canonical residues if specified for this chain
                if chain_id in non_canonical:
                    chain["non_canonical_residues"] = non_canonical[chain_id]
                # Wire pre-computed MSA file, if provided for this chain
                if chain_id in msa_map:
                    chain["main_msa_file_paths"] = [msa_map[chain_id]]
                chains.append(chain)

            elif isinstance(etype, dict):
                # Structured ligand type: {"smiles": "CC..."} or {"ccd": "ATP"}
                ligand_chain: dict[str, object] = {
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
    def _validate_inputs(input_data: OpenFold3Input) -> None:
        """Host-side validation — catch errors before container launch."""
        has_query_json = input_data.query_json is not None

        if not has_query_json and not input_data.sequences:
            raise AutobioError(
                "sequences must be non-empty, or provide a raw query JSON via the query_json field."
            )

        # Validate MSA files exist
        msa_paths = input_data.msa_paths
        if msa_paths:
            if input_data.use_msa_server:
                raise AutobioError(
                    "Cannot provide msa_paths with use_msa_server=True — set "
                    "use_msa_server=False to use precomputed MSAs."
                )
            for msa_path_str in msa_paths:
                msa_path = Path(msa_path_str)
                if not msa_path.exists():
                    raise AutobioError(f"MSA file does not exist: {msa_path}")

        # Validate entity_types keys match sequences
        entity_types = input_data.entity_types
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
        non_canonical = input_data.non_canonical_residues
        if non_canonical and not has_query_json:
            unknown_chains = set(non_canonical) - set(input_data.sequences)
            if unknown_chains:
                raise AutobioError(
                    f"non_canonical_residues references unknown chain IDs: "
                    f"{sorted(unknown_chains)}. "
                    f"Available chains: {sorted(input_data.sequences)}"
                )


# ---------------------------------------------------------------------------
# Catalog registration — populated when this module is imported
# ---------------------------------------------------------------------------

_OPENFOLD3_NOTES = (
    # MSA options
    "MSA generation via ColabFold server is ENABLED BY DEFAULT "
    "(use_msa_server=true). Only protein sequences are submitted to the server. "
    "To disable, set the 'use_msa_server' field to false. To use a private "
    "ColabFold server, set extra['msa_server_url'] to the server URL. "
    "For high-throughput screening, provide pre-computed MSAs via the "
    "'msa_paths' field (list of file paths; each filename stem must equal the "
    "target chain ID, e.g., 'A.a3m' → chain 'A'). Requires 'use_msa_server' set to false — "
    "OpenFold3's ColabFold step overwrites precomputed MSAs when the server "
    "is enabled.",
    # Template options
    "Template-based prediction is ENABLED BY DEFAULT (use_templates=true). "
    "Templates are automatically retrieved server-side by the ColabFold "
    "search step. Set 'use_templates' to false to disable template-based "
    "prediction entirely. User-supplied template structure files are not "
    "supported.",
    # PAE and confidence
    "The PAE (Predicted Aligned Error) head is ENABLED BY DEFAULT "
    "(pae_enabled=true). This produces pTM and ipTM confidence scores and "
    "enables the sample_ranking_score for ranking predictions. Disable via "
    "the 'pae_enabled' field to reduce memory usage at the cost of losing "
    "pTM/ipTM metrics.",
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

OPENFOLD3_TOOL = Tool(
    name="openfold3",
    display_name="OpenFold3",
    category=ToolCategory.STRUCTURE_PREDICTION,
    description=(
        "Predict biomolecular structures using OpenFold3 (open-source AlphaFold3). "
        "Supports proteins, DNA, RNA, ligands, and non-canonical residues with "
        "MSA-based and template-based prediction."
    ),
    version="1.0.0",
    image_tag="openfold3:1.0.0",
    requires_gpu=True,
    gpu_count=1,
    default_mode="predict",
    modes={
        "predict": Mode(
            name="predict",
            display_name="Predict structure",
            description="Predict a biomolecular complex structure.",
            input_schema=OpenFold3Input,
            output_schema=StructurePredictionOutput,
            default_timeout=3600,
            notes=_OPENFOLD3_NOTES,
        )
    },
    keywords=("openfold3", "alphafold3", "structure prediction", "complex", "ligand"),
)
"""Catalog Tool for OpenFold3."""

register(OPENFOLD3_TOOL)
