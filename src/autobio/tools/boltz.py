"""Boltz-1 and Boltz-2 structure prediction tool runners.

Both tools share a single Docker image (``autobio-boltz``) and runner class.
The ``tool_name`` (``"boltz1"`` or ``"boltz2"``) determines which model is used.

Simple protein predictions use the ``sequences`` dict on
``StructurePredictionInput``.  For multi-entity predictions (DNA, RNA, ligands)
or advanced features (constraints, modifications), agents either specify
``extra["entity_types"]`` or provide a raw Boltz YAML via
``extra["boltz_yaml"]``.

CLI-level args (``sampling_steps``, ``step_scale``, etc.) are passed through
the ``extra`` dict on ``StructurePredictionInput``.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

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
# Model configuration — maps tool name to Boltz model flag
# ---------------------------------------------------------------------------

_BOLTZ_CACHE = "/app/boltz/cache"

_MODEL_CONFIG: dict[str, str] = {
    "boltz1": "boltz1",
    "boltz2": "boltz2",
}

# Keys in ``extra`` that are consumed by the runner and should NOT be
# flat-merged into config.json as CLI args.
_CONSUMED_EXTRA_KEYS = frozenset(
    {
        "entity_types",
        "boltz_yaml",
        "msa_paths",
        "constraints",
        "templates",
        "properties",
        "modifications",
    }
)

# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


class BoltzRunner(ToolRunner):
    """Shared runner for Boltz-1 and Boltz-2 structure prediction.

    Both models use the same container image and three-phase protocol.
    ``prepare_workspace`` generates a Boltz YAML input file from the
    standardised ``StructurePredictionInput`` fields and writes
    ``config.json``.  ``parse_output`` reads the standardised
    ``result_data.json`` produced by the container's ``standardize.py``.
    """

    def prepare_workspace(self, input_data: BaseInput, workspace: Workspace) -> None:
        """Write config.json and Boltz YAML input to the workspace."""
        assert isinstance(input_data, StructurePredictionInput)

        # -- Host-side validation (fail fast before container launch) --------
        self._validate_inputs(input_data)

        # -- Generate or pass through Boltz YAML input ----------------------
        if "boltz_yaml" in input_data.extra:
            yaml_data = input_data.extra["boltz_yaml"]
            if isinstance(yaml_data, str):
                yaml_content = yaml_data
            else:
                yaml_content = yaml.dump(yaml_data, default_flow_style=False, sort_keys=False)
        else:
            yaml_dict = self._build_boltz_yaml(input_data, workspace)
            yaml_content = yaml.dump(yaml_dict, default_flow_style=False, sort_keys=False)

        workspace.write_input_file("input.yaml", yaml_content.encode())

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
            "model": _MODEL_CONFIG[self.tool_name],
            "input_path": "/workspace/inputs/input.yaml",
            "output_dir": "/workspace/outputs/raw",
            "cache_dir": _BOLTZ_CACHE,
            "use_msa_server": True,
        }

        # Map num_models → diffusion_samples
        if input_data.num_models > 1:
            config["diffusion_samples"] = input_data.num_models

        # Flat-merge extra dict for CLI-level args (excluding consumed keys).
        # This allows extra["use_msa_server"] = False to override the default.
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
                affinity_probability=s.get("affinity_probability"),
                affinity_value=s.get("affinity_value"),
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

    def _build_boltz_yaml(self, input_data: StructurePredictionInput, workspace: Workspace) -> dict:
        """Generate a Boltz YAML input dict from structured input fields.

        Each entry in ``sequences`` becomes an entity in the YAML. The default
        entity type is ``protein``; override per-chain via
        ``extra["entity_types"]``.

        Entity type values can be:
        - A string: ``"protein"``, ``"dna"``, ``"rna"``
        - A dict for ligands: ``{"smiles": "CC(=O)..."}`` or ``{"ccd": "ATP"}``
        """
        entity_types: dict = input_data.extra.get("entity_types", {})
        sequences_section: list[dict] = []

        for chain_id, sequence in input_data.sequences.items():
            etype = entity_types.get(chain_id, "protein")

            if isinstance(etype, str) and etype in ("protein", "dna", "rna"):
                sequences_section.append({etype: {"id": chain_id, "sequence": sequence}})
            elif isinstance(etype, dict):
                # Structured type: {"smiles": "CC..."} or {"ccd": "ATP"}
                if "smiles" in etype:
                    sequences_section.append(
                        {"ligand": {"id": chain_id, "smiles": etype["smiles"]}}
                    )
                elif "ccd" in etype:
                    sequences_section.append({"ligand": {"id": chain_id, "ccd": etype["ccd"]}})
                else:
                    raise AutobioError(
                        f"Unknown entity type dict for chain {chain_id!r}: {etype}. "
                        f"Expected {{'smiles': '...'}} or {{'ccd': '...'}}."
                    )
            elif isinstance(etype, str):
                # Assume it's a ligand specification format
                sequences_section.append({"ligand": {"id": chain_id, "smiles": sequence}})
            else:
                raise AutobioError(
                    f"Invalid entity_types value for chain {chain_id!r}: {etype!r}. "
                    f"Must be a string ('protein', 'dna', 'rna') or a dict "
                    f"({{'smiles': '...'}} or {{'ccd': '...'}})."
                )

        yaml_data: dict = {"version": 1, "sequences": sequences_section}

        # Add optional sections from extra
        for section in ("constraints", "properties", "modifications"):
            if section in input_data.extra:
                yaml_data[section] = input_data.extra[section]

        # Handle templates — rewrite paths to container-internal paths
        if input_data.templates:
            template_entries = []
            for tmpl_path in input_data.templates:
                template_entries.append({"cif": f"/workspace/inputs/{tmpl_path.name}"})
            yaml_data["templates"] = template_entries

        # Handle MSA paths — rewrite to container-internal paths
        msa_paths = input_data.extra.get("msa_paths")
        if msa_paths:
            # Inject MSA paths into the sequence entries
            msa_map: dict[str, str] = {}
            for msa_path_str in msa_paths:
                msa_path = Path(msa_path_str)
                # Assume MSA filename starts with chain ID, e.g., "A.a3m"
                chain_id = msa_path.stem
                msa_map[chain_id] = f"/workspace/inputs/{msa_path.name}"

            for entry in yaml_data["sequences"]:
                for _etype, edata in entry.items():
                    if isinstance(edata, dict) and edata.get("id") in msa_map:
                        edata["msa"] = msa_map[edata["id"]]

        return yaml_data

    @staticmethod
    def _resolve_container_path(container_path_str: str, workspace: Workspace) -> Path:
        """Map a container-internal ``/workspace/...`` path to the host workspace.

        The standardize.py script inside the container writes absolute paths
        rooted at ``/workspace``. This method strips that prefix and resolves
        the remainder against the host-side workspace root.
        """
        container_path = Path(container_path_str)
        try:
            relative = container_path.relative_to("/workspace")
        except ValueError:
            # Not a container path — return as-is
            return container_path
        return workspace.root / relative

    @staticmethod
    def _validate_inputs(input_data: StructurePredictionInput) -> None:
        """Host-side validation — catch errors before container launch."""
        has_boltz_yaml = "boltz_yaml" in input_data.extra

        if not has_boltz_yaml and not input_data.sequences:
            raise AutobioError(
                "sequences must be non-empty, or provide a raw Boltz YAML via extra['boltz_yaml']."
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
        if entity_types and not has_boltz_yaml:
            unknown_chains = set(entity_types) - set(input_data.sequences)
            if unknown_chains:
                raise AutobioError(
                    f"entity_types references unknown chain IDs: {sorted(unknown_chains)}. "
                    f"Available chains: {sorted(input_data.sequences)}"
                )


# ---------------------------------------------------------------------------
# Registry entries — populated when this module is imported
# ---------------------------------------------------------------------------

_BOLTZ_SHARED_NOTES = (
    # Input construction
    "Each entry in the sequences dict becomes a Boltz entity. Default entity "
    "type is 'protein'. To specify DNA, RNA, or ligand entities, use "
    "extra['entity_types'] with a dict mapping chain IDs to types: "
    "{'B': 'dna', 'C': {'smiles': 'CC(=O)NC1=CC=C(O)C=C1'}}. "
    "For protein chains, the sequence is the amino acid sequence. For DNA/RNA, "
    "the sequence is the nucleotide sequence.",
    # Ligand specification
    "Ligands are specified via entity_types using either SMILES notation "
    "({'smiles': 'CC...'}) or CCD component codes ({'ccd': 'ATP'}). The "
    "sequence value for ligand chains in the sequences dict is ignored when "
    "a SMILES or CCD code is provided in entity_types.",
    # Raw YAML override
    "For complex multi-entity predictions with constraints, modifications, or "
    "other advanced features, provide the full Boltz YAML spec via "
    "extra['boltz_yaml'] (as a dict or YAML string). This bypasses automatic "
    "YAML generation and gives full access to the Boltz input format. See "
    "https://github.com/jwohlwend/boltz/blob/main/docs/prediction.md.",
    # Templates
    "Template structures for template-based prediction are provided via the "
    "'templates' field on StructurePredictionInput. Files are copied into the "
    "workspace and paths are rewritten automatically.",
    # MSA options
    "MSA generation via ColabFold's MMSeqs2 server is ENABLED BY DEFAULT "
    "(use_msa_server=true). This avoids needing >1TB of local sequence "
    "databases but requires network access from the container. To disable, "
    "set 'use_msa_server': false in extra. Alternatively, provide pre-computed "
    "MSAs via extra['msa_paths'] (list of file paths, filenames should start "
    "with the chain ID, e.g., 'A.a3m').",
    # Key parameters
    "Key extra parameters: 'sampling_steps' (int, default 200 — diffusion "
    "iterations), 'recycling_steps' (int, default 3), 'step_scale' (float, "
    "1-2 recommended — higher = more confident but less diverse), "
    "'output_format' ('pdb' or 'mmcif', default 'mmcif'), 'seed' (int), "
    "'write_full_pae' (bool), 'write_full_pde' (bool), 'write_embeddings' "
    "(bool). num_models on the input maps to diffusion_samples.",
    # GPU memory
    "Boltz requires substantial GPU memory. For very large complexes, reduce "
    "'max_parallel_samples' (default 5) in extra to avoid OOM errors.",
)

_BOLTZ1_NOTES = _BOLTZ_SHARED_NOTES + (
    "Boltz-1 is the original model. It does NOT support affinity prediction. "
    "For affinity prediction, use boltz2 instead.",
)

_BOLTZ2_NOTES = _BOLTZ_SHARED_NOTES + (
    # Affinity prediction
    "Boltz-2 includes binding affinity prediction, enabled by default for "
    "protein-ligand complexes. Results appear as affinity_probability (0-1, "
    "probability of binding) and affinity_value (log10 IC50 in uM) on each "
    "PredictedStructure. Convert affinity_value to kcal/mol via "
    "(6 - value) * 1.364.",
    # Affinity parameters
    "Affinity-specific extra parameters: 'sampling_steps_affinity' (int, "
    "default 200), 'diffusion_samples_affinity' (int, default 5). Use "
    "'method' in extra to condition on experimental method (Boltz-2 only).",
)

TOOL_REGISTRY["boltz1"] = ToolEntry(
    image_tag="boltz:1.0.0",
    category=ToolCategory.STRUCTURE_PREDICTION,
    requires_gpu=True,
    gpu_count=1,
    input_schema=StructurePredictionInput,
    output_schema=StructurePredictionOutput,
    default_timeout=3600,
    supports_batch=False,
    description=(
        "Predict biomolecular structures using Boltz-1. Supports proteins, "
        "DNA, RNA, and ligand complexes with template-based and ab initio "
        "prediction."
    ),
    version="1.0.0",
    notes=_BOLTZ1_NOTES,
)

TOOL_REGISTRY["boltz2"] = ToolEntry(
    image_tag="boltz:1.0.0",
    category=ToolCategory.STRUCTURE_PREDICTION,
    requires_gpu=True,
    gpu_count=1,
    input_schema=StructurePredictionInput,
    output_schema=StructurePredictionOutput,
    default_timeout=7200,
    supports_batch=False,
    description=(
        "Predict biomolecular structures and binding affinity using Boltz-2. "
        "Supports proteins, DNA, RNA, and ligand complexes. Includes affinity "
        "prediction that approaches FEP accuracy at 1000x speed."
    ),
    version="1.0.0",
    notes=_BOLTZ2_NOTES,
)
