"""Chai-1 multi-modal structure prediction tool runner.

Chai-1 predicts biomolecular structures for proteins, DNA, RNA, ligands, and
glycans.  It supports restraints (contact, pocket) and covalent bonds via a
CSV constraint file, which is particularly useful for antibody-antigen
complexes with known epitopes and glycosylated proteins.

Simple protein predictions use the ``sequences`` dict on
``StructurePredictionInput``.  For multi-entity predictions (DNA, RNA, ligands)
agents specify ``extra["entity_types"]``.  For full control, provide a raw
FASTA via ``extra["chai_fasta"]``.

CLI-level args (``num_trunk_recycles``, ``num_diffn_timesteps``, etc.) are
passed through the ``extra`` dict on ``StructurePredictionInput``.
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

_CHAI_DOWNLOADS_DIR = "/app/chai/downloads"

# Valid entity type strings for Chai-1 FASTA headers.
_VALID_ENTITY_TYPES = frozenset({"protein", "dna", "rna", "ligand"})

# Keys in ``extra`` that are consumed by the runner and should NOT be
# flat-merged into config.json as CLI args.
_CONSUMED_EXTRA_KEYS = frozenset(
    {
        "entity_types",
        "constraints",
        "msa_directory",
        "chai_fasta",
    }
)

# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


class ChaiRunner(ToolRunner):
    """Runner for Chai-1 multi-modal structure prediction.

    ``prepare_workspace`` generates a FASTA input file from the standardised
    ``StructurePredictionInput`` fields and writes ``config.json``.
    ``parse_output`` reads the standardised ``result_data.json`` produced by
    the container's ``standardize.py``.
    """

    def prepare_workspace(self, input_data: BaseInput, workspace: Workspace) -> None:
        """Write config.json, FASTA input, and constraint files to the workspace."""
        assert isinstance(input_data, StructurePredictionInput)

        # -- Host-side validation (fail fast before container launch) --------
        self._validate_inputs(input_data)

        # -- Generate or pass through FASTA input ---------------------------
        if input_data.extra.get("chai_fasta"):
            fasta_content = input_data.extra["chai_fasta"]
        else:
            fasta_content = self._build_fasta(input_data)

        workspace.write_input_file("input.fasta", fasta_content.encode())

        # -- Copy constraint file into workspace ----------------------------
        constraint_path_config = None
        if "constraints" in input_data.extra:
            constraints = input_data.extra["constraints"]
            if "\n" in constraints or "," in constraints:
                # CSV content — write directly
                workspace.write_input_file("restraints.csv", constraints.encode())
            else:
                # File path — copy into workspace
                shutil.copy2(constraints, workspace.inputs_dir / "restraints.csv")
            constraint_path_config = "/workspace/inputs/restraints.csv"

        # -- Copy MSA directory into workspace ------------------------------
        msa_dir_config = None
        msa_directory = input_data.extra.get("msa_directory")
        if msa_directory:
            msa_dest = workspace.inputs_dir / "msa"
            shutil.copytree(msa_directory, msa_dest)
            msa_dir_config = "/workspace/inputs/msa"

        # -- Copy template files into workspace -----------------------------
        if input_data.templates:
            for tmpl_path in input_data.templates:
                shutil.copy2(tmpl_path, workspace.inputs_dir / tmpl_path.name)

        # -- Build config.json ----------------------------------------------
        config: dict[str, object] = {
            "fasta_path": "/workspace/inputs/input.fasta",
            "output_dir": "/workspace/outputs/raw",
            "downloads_dir": _CHAI_DOWNLOADS_DIR,
            "use_msa_server": True,
            "use_esm_embeddings": False,
        }

        # Map num_models → num_diffn_samples (always set — Chai-1 defaults to 5)
        config["num_diffn_samples"] = input_data.num_models

        # Optional paths (only set if provided)
        if constraint_path_config:
            config["constraint_path"] = constraint_path_config
        if msa_dir_config:
            config["msa_directory"] = msa_dir_config

        # Flat-merge extra dict for inference params (excluding consumed keys).
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
    def _build_fasta(input_data: StructurePredictionInput) -> str:
        """Generate a Chai-1 FASTA string from structured input fields.

        Each entry in ``sequences`` becomes a FASTA record with an entity-type
        header.  The default entity type is ``protein``; override per-chain via
        ``extra["entity_types"]``.

        Entity type values can be:
        - A string: ``"protein"``, ``"dna"``, ``"rna"``, ``"ligand"``
        - A dict for ligands: ``{"smiles": "CC(=O)..."}``

        For ligands specified as the string ``"ligand"``, the sequence value in
        the ``sequences`` dict is used as the SMILES string.
        """
        entity_types: dict = input_data.extra.get("entity_types", {})
        lines: list[str] = []

        # Sort by chain ID so Chai-1's alphabetical chain assignment aligns
        for chain_id in sorted(input_data.sequences):
            sequence = input_data.sequences[chain_id]
            etype = entity_types.get(chain_id, "protein")

            if isinstance(etype, dict):
                # Structured type: {"smiles": "CC..."}
                if "smiles" in etype:
                    lines.append(f">ligand|name={chain_id}")
                    lines.append(etype["smiles"])
                else:
                    raise AutobioError(
                        f"Unknown entity type dict for chain {chain_id!r}: {etype}. "
                        f"Expected {{'smiles': '...'}}."
                    )
            elif isinstance(etype, str) and etype == "ligand":
                # String "ligand" — use sequence value as SMILES
                lines.append(f">ligand|name={chain_id}")
                lines.append(sequence)
            elif isinstance(etype, str) and etype in _VALID_ENTITY_TYPES:
                lines.append(f">{etype}|name={chain_id}")
                lines.append(sequence)
            else:
                raise AutobioError(
                    f"Invalid entity_types value for chain {chain_id!r}: {etype!r}. "
                    f"Must be one of {sorted(_VALID_ENTITY_TYPES)} or a dict "
                    f"({{'smiles': '...'}})."
                )

        return "\n".join(lines) + "\n"

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
        has_chai_fasta = "chai_fasta" in input_data.extra

        if not has_chai_fasta and not input_data.sequences:
            raise AutobioError(
                "sequences must be non-empty, or provide a raw FASTA via extra['chai_fasta']."
            )

        # Validate template files exist
        if input_data.templates:
            for tmpl_path in input_data.templates:
                if not tmpl_path.exists():
                    raise AutobioError(f"Template file does not exist: {tmpl_path}")

        # Validate constraint file exists (if path, not CSV content)
        constraints = input_data.extra.get("constraints")
        if (
            constraints
            and "\n" not in constraints
            and "," not in constraints
            and not Path(constraints).exists()
        ):
            raise AutobioError(f"Constraint file does not exist: {constraints}")

        # Validate MSA directory exists
        msa_directory = input_data.extra.get("msa_directory")
        if msa_directory and not Path(msa_directory).is_dir():
            raise AutobioError(f"MSA directory does not exist: {msa_directory}")

        # Validate entity_types keys match sequences
        entity_types = input_data.extra.get("entity_types", {})
        if entity_types and not has_chai_fasta:
            unknown_chains = set(entity_types) - set(input_data.sequences)
            if unknown_chains:
                raise AutobioError(
                    f"entity_types references unknown chain IDs: {sorted(unknown_chains)}. "
                    f"Available chains: {sorted(input_data.sequences)}"
                )

            # Validate entity type values
            for chain_id, etype in entity_types.items():
                if isinstance(etype, dict):
                    if "smiles" not in etype:
                        raise AutobioError(
                            f"entity_types dict for chain {chain_id!r} must contain "
                            f"'smiles' key, got: {etype}"
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


# ---------------------------------------------------------------------------
# Registry entry — populated when this module is imported
# ---------------------------------------------------------------------------

_CHAI_INPUT_FORMAT = (
    # FASTA format overview
    "Chai-1 takes a multi-entity FASTA file as primary input. Each record has "
    "a header specifying entity type and chain name, followed by the sequence. "
    "Header format: >entity_type|name=chain_id where entity_type is one of: "
    "protein, dna, rna, ligand. Via the autobio API, each entry in the "
    "sequences dict becomes a FASTA entity (default type 'protein'). To "
    "specify other entity types, use extra['entity_types'] mapping chain IDs "
    "to types: {'B': 'dna', 'C': {'smiles': 'CC(=O)NC1=CC=C(O)C=C1'}}.",
    # Entity examples
    "FASTA entity examples — "
    "Protein: >protein|name=A\\nMKLLVVFLFL... "
    "DNA: >dna|name=B\\nATCGATCG... "
    "RNA: >rna|name=C\\nAUCGAUCG... "
    "Ligand (SMILES): >ligand|name=D\\nCC(=O)NC1=CC=C(O)C=C1. "
    "Via the API, ligands are specified in entity_types using either the "
    "string 'ligand' (sequence value used as SMILES) or a dict with a "
    "'smiles' key: {'smiles': 'CC(=O)NC1=CC=C(O)C=C1'}.",
    # Modified residues
    "Modified residues use parenthesised CCD codes inline in the protein "
    "sequence: e.g., 'AAA(SEP)AAA' for phosphoserine, 'MK(TPO)LL(SEP)VV' "
    "for multiple modifications. Chai-1 resolves these against the Chemical "
    "Component Dictionary.",
    # Glycans
    "Glycans are specified as ligand entities using CCD codes. Single sugar: "
    ">ligand|name=B\\nNAG. Multi-ring with bond notation: "
    ">ligand|name=B\\nNAG(4-1 NAG(4-1 BMA(3-1 MAN)(6-1 MAN))). "
    "Bond notation uses ATOM_NUMBER-ATOM_NUMBER between sugar units. Glycans "
    "must be connected to a protein via a covalent bond in the restraints CSV.",
    # Restraints CSV format
    "Restraints and covalent bonds use a CSV file (provide via "
    "extra['constraints'] as CSV content string or file path). Columns: "
    "chainA, res_idxA, chainB, res_idxB, connection_type, confidence, "
    "min_distance_angstrom, max_distance_angstrom, comment, restraint_id. "
    "Three connection types: "
    "'contact' — residue-to-residue distance restraint: "
    "A,C387,B,Y101,contact,1.0,0.0,5.5,interface contact,r1. "
    "'pocket' — any residue to a target residue (leave res_idxA empty): "
    "A,,B,Y101,pocket,1.0,0.0,8.0,binding pocket,r2. "
    "'covalent' — atom-level bond using RESIDUE@ATOM notation: "
    "A,N437@N,B,@C1,covalent,1.0,0.0,0.0,glycan bond,r3. "
    "For covalent bonds, protein residues use NUMBER@ATOM (e.g., N437@N), "
    "ligands use @ATOM (e.g., @C1). Chain IDs are assigned alphabetically "
    "by FASTA entity order.",
    # Complete example
    "Complete example — protein-ligand FASTA:\\n"
    ">protein|name=A\\n"
    "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQQIAATG\\n"
    ">ligand|name=B\\n"
    "CC(=O)NC1=CC=C(O)C=C1",
    # Raw override
    "For full control over the native FASTA format, provide the complete "
    "FASTA content via extra['chai_fasta'] (as a string). This bypasses "
    "automatic FASTA generation from the sequences dict.",
)

_CHAI_NOTES = (
    # MSA options
    "MSA generation via ColabFold's MMSeqs2 server is ENABLED BY DEFAULT "
    "(use_msa_server=true). This avoids needing >1TB of local sequence "
    "databases but requires network access from the container. To disable, "
    "set 'use_msa_server': false in extra. Alternatively, provide pre-computed "
    "MSAs via extra['msa_directory'] (path to directory with .aligned.pqt files).",
    # ESM embeddings
    "ESM protein language model embeddings are DISABLED BY DEFAULT "
    "(use_esm_embeddings=false). Enable via extra['use_esm_embeddings'] = True "
    "for potentially improved predictions at the cost of additional compute.",
    # Key parameters
    "Key extra parameters: 'num_trunk_recycles' (int, default 3), "
    "'num_diffn_timesteps' (int, default 200 — diffusion denoising steps), "
    "'seed' (int, default 42), 'low_memory' (bool, default false — moves "
    "components to GPU only during inference), 'use_templates_server' (bool, "
    "default false). num_models on the input maps to num_diffn_samples.",
    # GPU memory
    "Chai-1 requires substantial GPU memory. Recommended: A100 80GB or "
    "H100 80GB. Compatible: L40S 48GB, A10/A30 for smaller complexes. "
    "Set extra['low_memory'] = True to reduce peak VRAM usage.",
)

TOOL_REGISTRY["chai1"] = ToolEntry(
    image_tag="chai:1.0.0",
    category=ToolCategory.STRUCTURE_PREDICTION,
    requires_gpu=True,
    gpu_count=1,
    input_schema=StructurePredictionInput,
    output_schema=StructurePredictionOutput,
    default_timeout=3600,
    supports_batch=False,
    description=(
        "Predict biomolecular structures using Chai-1. Supports proteins, "
        "DNA, RNA, ligands, and glycans with restraints and covalent bonds."
    ),
    version="1.0.0",
    notes=_CHAI_NOTES,
    input_format=_CHAI_INPUT_FORMAT,
)
