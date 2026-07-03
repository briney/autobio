"""Chai-1 multi-modal structure prediction tool runner.

Chai-1 predicts biomolecular structures for proteins, DNA, RNA, ligands, and
glycans.  It supports restraints (contact, pocket) and covalent bonds via a
CSV constraint file, which is particularly useful for antibody-antigen
complexes with known epitopes and glycosylated proteins.

Simple protein predictions use the ``sequences`` dict on ``Chai1Input``. For
multi-entity predictions (DNA, RNA, ligands) agents specify ``entity_types``.
For full control, provide a raw FASTA via ``chai_fasta``.

CLI-level args (``num_trunk_recycles``, ``num_diffn_timesteps``, etc.) are
passed through the ``extra`` dict on ``Chai1Input``.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any

from autobio.core.catalog import Mode, Tool, register
from autobio.core.registry import ToolCategory
from autobio.core.result import AutobioError
from autobio.schemas.base import BaseInput  # noqa: TC001 - needed at runtime for isinstance
from autobio.schemas.structure_prediction import (
    Chai1Input,
    ConfidenceMetrics,
    PredictedStructure,
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

# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


class ChaiRunner(ToolRunner):
    """Runner for Chai-1 multi-modal structure prediction.

    ``prepare_workspace`` generates a FASTA input file from the standardised
    ``Chai1Input`` fields and writes ``config.json``. ``parse_output`` reads
    the standardised ``result_data.json`` produced by the container's
    ``standardize.py``.
    """

    def prepare_workspace(self, input_data: BaseInput, workspace: Workspace) -> None:
        """Write config.json, FASTA input, and constraint files to the workspace."""
        assert isinstance(input_data, Chai1Input)

        # -- Host-side validation (fail fast before container launch) --------
        self._validate_inputs(input_data)

        # -- Generate or pass through FASTA input ---------------------------
        fasta_content = input_data.chai_fasta or self._build_fasta(input_data)

        workspace.write_input_file("input.fasta", fasta_content.encode())

        # -- Copy constraint file into workspace ----------------------------
        constraint_path_config = None
        constraints = input_data.constraints
        if constraints:
            if "\n" in constraints or "," in constraints:
                # CSV content — write directly
                workspace.write_input_file("restraints.csv", constraints.encode())
            else:
                # File path — copy into workspace
                shutil.copy2(constraints, workspace.inputs_dir / "restraints.csv")
            constraint_path_config = "/workspace/inputs/restraints.csv"

        # -- Copy MSA directory into workspace ------------------------------
        msa_dir_config = None
        msa_directory = input_data.msa_directory
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
            "use_msa_server": input_data.use_msa_server,
            "use_esm_embeddings": input_data.use_esm_embeddings,
        }

        # Map num_models → num_diffn_samples (always set — Chai-1 defaults to 5)
        config["num_diffn_samples"] = input_data.num_models

        # Optional paths (only set if provided)
        if constraint_path_config:
            config["constraint_path"] = constraint_path_config
        if msa_dir_config:
            config["msa_directory"] = msa_dir_config

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
    def _build_fasta(input_data: Chai1Input) -> str:
        """Generate a Chai-1 FASTA string from structured input fields.

        Each entry in ``sequences`` becomes a FASTA record with an entity-type
        header.  The default entity type is ``protein``; override per-chain via
        ``entity_types``.

        Entity type values can be:
        - A string: ``"protein"``, ``"dna"``, ``"rna"``, ``"ligand"``
        - A dict for ligands: ``{"smiles": "CC(=O)..."}``

        For ligands specified as the string ``"ligand"``, the sequence value in
        the ``sequences`` dict is used as the SMILES string.
        """
        entity_types: dict[str, Any] = input_data.entity_types
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
    def _validate_inputs(input_data: Chai1Input) -> None:
        """Host-side validation — catch errors before container launch."""
        has_chai_fasta = bool(input_data.chai_fasta)

        if not has_chai_fasta and not input_data.sequences:
            raise AutobioError(
                "sequences must be non-empty, or provide a raw FASTA via the chai_fasta field."
            )

        # Validate template files exist
        if input_data.templates:
            for tmpl_path in input_data.templates:
                if not tmpl_path.exists():
                    raise AutobioError(f"Template file does not exist: {tmpl_path}")

        # Validate constraint file exists (if path, not CSV content)
        constraints = input_data.constraints
        if (
            constraints
            and "\n" not in constraints
            and "," not in constraints
            and not Path(constraints).exists()
        ):
            raise AutobioError(f"Constraint file does not exist: {constraints}")

        # Validate MSA directory exists
        msa_directory = input_data.msa_directory
        if msa_directory and not Path(msa_directory).is_dir():
            raise AutobioError(f"MSA directory does not exist: {msa_directory}")

        # Validate entity_types keys match sequences
        entity_types = input_data.entity_types
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
# Catalog registration — populated when this module is imported
# ---------------------------------------------------------------------------

_CHAI_NOTES = (
    # MSA options
    "MSA generation via ColabFold's MMSeqs2 server is ENABLED BY DEFAULT "
    "(use_msa_server=true). This avoids needing >1TB of local sequence "
    "databases but requires network access from the container. To disable, "
    "set the 'use_msa_server' field to false. Alternatively, provide pre-computed "
    "MSAs via the 'msa_directory' field (path to directory with .aligned.pqt files).",
    # ESM embeddings
    "ESM protein language model embeddings are DISABLED BY DEFAULT "
    "(use_esm_embeddings=false). Enable via the 'use_esm_embeddings' field "
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

CHAI1_TOOL = Tool(
    name="chai1",
    display_name="Chai-1",
    category=ToolCategory.STRUCTURE_PREDICTION,
    description=(
        "Predict biomolecular structures using Chai-1. Supports proteins, DNA, RNA, "
        "ligands, and glycans with restraints and covalent bonds."
    ),
    version="1.0.0",
    image_tag="chai:1.0.0",
    requires_gpu=True,
    gpu_count=1,
    default_mode="predict",
    modes={
        "predict": Mode(
            name="predict",
            display_name="Predict structure",
            description="Predict a biomolecular complex structure.",
            input_schema=Chai1Input,
            output_schema=StructurePredictionOutput,
            default_timeout=3600,
            notes=_CHAI_NOTES,
        )
    },
    keywords=("chai", "chai1", "structure prediction", "complex", "ligand", "glycan"),
)
"""Catalog Tool for Chai-1."""

register(CHAI1_TOOL)
