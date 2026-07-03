"""ProteinMPNN and LigandMPNN tool runners.

Both tools share a single Docker image (``autobio-mpnn``) and runner class.
The ``tool_name`` (``"proteinmpnn"`` or ``"ligandmpnn"``) determines which
model type and checkpoint are used.

LigandMPNN-specific parameters (``omit``, ``bias``, ``atomize_side_chains``,
etc.) are passed through the ``extra`` dict on ``MPNNInput``.
"""

from __future__ import annotations

import json
import shutil
from typing import TYPE_CHECKING

from autobio.core.catalog import Mode, Tool, register
from autobio.core.registry import ToolCategory
from autobio.schemas.base import BaseInput  # noqa: TC001 - needed at runtime for isinstance
from autobio.schemas.inverse_folding import (
    DesignedSequence,
    InverseFoldingOutput,
    MPNNInput,
)
from autobio.tools.base import ToolRunner

if TYPE_CHECKING:
    from autobio.core.workspace import Workspace

# ---------------------------------------------------------------------------
# Model configuration — maps tool name to model_type and default checkpoint
# ---------------------------------------------------------------------------

_CHECKPOINT_DIR = "/app/foundry/checkpoints"

_MODEL_CONFIG: dict[str, dict[str, str]] = {
    "proteinmpnn": {
        "model_type": "protein_mpnn",
        "checkpoint": "proteinmpnn_v_48_020.pt",
    },
    "ligandmpnn": {
        "model_type": "ligand_mpnn",
        "checkpoint": "ligandmpnn_v_32_010_25.pt",
    },
}

# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


class MPNNRunner(ToolRunner):
    """Shared runner for ProteinMPNN and LigandMPNN inverse folding tools.

    Both models use the same container image and three-phase protocol.
    ``prepare_workspace`` maps standardised ``MPNNInput`` fields
    to the foundry ``mpnn`` CLI configuration.  ``parse_output`` reads the
    standardised ``result_data.json`` produced by the container's
    ``standardize.py`` script.
    """

    def prepare_workspace(self, input_data: BaseInput, workspace: Workspace) -> None:
        """Write config.json and copy the input structure into the workspace."""
        assert isinstance(input_data, MPNNInput)
        model_cfg = _MODEL_CONFIG[self.tool_name]

        # Copy structure file into workspace inputs/
        src_path = input_data.structure_path
        dest_name = src_path.name
        shutil.copy2(src_path, workspace.inputs_dir / dest_name)

        # Build config.json for the container
        config: dict[str, object] = {
            "model_type": model_cfg["model_type"],
            "checkpoint_path": f"{_CHECKPOINT_DIR}/{model_cfg['checkpoint']}",
            "is_legacy_weights": True,
            "structure_path": f"/workspace/inputs/{dest_name}",
            "number_of_batches": input_data.num_sequences,
            "temperature": input_data.temperature,
        }

        # Map chains_to_design → --designed_chains (comma-separated)
        if input_data.chains_to_design is not None:
            config["designed_chains"] = ",".join(input_data.chains_to_design)

        # Map fixed_positions → --fixed_residues (e.g. "A35,A40,B10")
        # NOTE: designed_chains and fixed_residues are mutually exclusive
        # in the foundry CLI. fixed_positions takes precedence.
        if input_data.fixed_positions is not None:
            residue_ids = []
            for chain, positions in input_data.fixed_positions.items():
                for pos in positions:
                    residue_ids.append(f"{chain}{pos}")
            config["fixed_residues"] = ",".join(residue_ids)
            # Remove designed_chains if present — they're mutually exclusive
            config.pop("designed_chains", None)

        # Flat-merge extra dict for tool-specific params
        # (omit, seed, batch_size, bias, temperature_per_residue, etc.)
        self._apply_extra(config, input_data)

        workspace.write_config(config)

    def parse_output(self, workspace: Workspace) -> InverseFoldingOutput:
        """Read standardised outputs and return an ``InverseFoldingOutput``."""
        result_data_path = workspace.std_output_dir / "result_data.json"
        data = json.loads(result_data_path.read_text())

        designed_sequences = [
            DesignedSequence(
                rank=s["rank"],
                sequence=s["sequence"],
                score=s.get("score"),
                recovery=s.get("recovery"),
            )
            for s in data["designed_sequences"]
        ]

        # Placeholder metadata — overwritten by base class run()
        return InverseFoldingOutput(
            designed_sequences=designed_sequences,
            native_sequence=data.get("native_sequence"),
            metadata=self._build_metadata(workspace, 0.0, [], ""),
            raw_output_path=workspace.raw_output_dir,
        )


# ---------------------------------------------------------------------------
# Catalog registration — populated when this module is imported
# ---------------------------------------------------------------------------

_MPNN_NOTES = (
    "The foundry MPNN parser may resolve fewer residues than the PDB CA atom "
    "count. Expect ~3-5% fewer residues in designed sequences compared to the "
    "raw PDB SEQRES or ATOM record counts due to internal filtering of "
    "disordered or incomplete residues.",
    "PDB structures with multiple copies in the asymmetric unit (e.g., two Fab "
    "copies with chains H, L, M, P) can trigger an atomworks parser error: "
    "'Ambiguous residue annotations detected'. Use structures with unique chain "
    "IDs per sequence, or preprocess to extract a single copy.",
    "Output sequences are concatenated in alphabetical chain-ID order, not PDB "
    "encounter order. The standardize step handles this, but if you inspect raw "
    "FASTA output directly, be aware of this ordering.",
)

_LIGANDMPNN_LIGAND_NOTE = (
    "For protein-ligand complexes, the foundry parser separates non-polymer "
    "residues (ligands, ions) into synthetic chain IDs. Calcium ions (atom "
    "name 'CA') may be miscounted as protein residues by external PDB parsers, "
    "but the foundry parser handles them correctly."
)

PROTEINMPNN_TOOL = Tool(
    name="proteinmpnn",
    display_name="ProteinMPNN",
    category=ToolCategory.INVERSE_FOLDING,
    description="Design protein sequences for given backbone structures using ProteinMPNN.",
    version="1.0.0",
    image_tag="mpnn:1.0.0",
    requires_gpu=True,
    gpu_count=1,
    default_mode="design",
    modes={
        "design": Mode(
            name="design",
            display_name="Design sequences",
            description="Design protein sequences for a backbone structure.",
            input_schema=MPNNInput,
            output_schema=InverseFoldingOutput,
            default_timeout=600,
            notes=_MPNN_NOTES,
        )
    },
    keywords=("proteinmpnn", "inverse folding", "sequence design", "mpnn"),
)
"""Catalog Tool for ProteinMPNN."""

register(PROTEINMPNN_TOOL)

LIGANDMPNN_TOOL = Tool(
    name="ligandmpnn",
    display_name="LigandMPNN",
    category=ToolCategory.INVERSE_FOLDING,
    description="Design protein sequences with ligand awareness using LigandMPNN.",
    version="1.0.0",
    image_tag="mpnn:1.0.0",
    requires_gpu=True,
    gpu_count=1,
    default_mode="design",
    modes={
        "design": Mode(
            name="design",
            display_name="Design sequences",
            description="Design protein sequences (ligand-aware) for a backbone structure.",
            input_schema=MPNNInput,
            output_schema=InverseFoldingOutput,
            default_timeout=600,
            notes=_MPNN_NOTES + (_LIGANDMPNN_LIGAND_NOTE,),
        )
    },
    keywords=("ligandmpnn", "inverse folding", "sequence design", "ligand", "mpnn"),
)
"""Catalog Tool for LigandMPNN."""

register(LIGANDMPNN_TOOL)
