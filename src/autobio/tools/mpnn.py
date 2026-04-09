"""ProteinMPNN and LigandMPNN tool runners.

Both tools share a single Docker image (``autobio-mpnn``) and runner class.
The ``tool_name`` (``"proteinmpnn"`` or ``"ligandmpnn"``) determines which
model type and checkpoint are used.

LigandMPNN-specific parameters (``omit``, ``bias``, ``atomize_side_chains``,
etc.) are passed through the ``extra`` dict on ``InverseFoldingInput``.
"""

from __future__ import annotations

import json
import shutil
from typing import TYPE_CHECKING

from autobio.core.registry import TOOL_REGISTRY, ToolCategory, ToolEntry
from autobio.schemas.base import BaseInput  # noqa: TC001 - needed at runtime for isinstance
from autobio.schemas.inverse_folding import (
    DesignedSequence,
    InverseFoldingInput,
    InverseFoldingOutput,
)
from autobio.schemas.scoring import ScoredStructure, ScoringInput, ScoringOutput
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
    ``prepare_workspace`` maps standardised ``InverseFoldingInput`` fields
    to the foundry ``mpnn`` CLI configuration.  ``parse_output`` reads the
    standardised ``result_data.json`` produced by the container's
    ``standardize.py`` script.
    """

    def prepare_workspace(self, input_data: BaseInput, workspace: Workspace) -> None:
        """Write config.json and copy the input structure into the workspace."""
        assert isinstance(input_data, InverseFoldingInput)
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
        config.update(input_data.extra)

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
# Registry entries — populated when this module is imported
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

TOOL_REGISTRY["proteinmpnn"] = ToolEntry(
    image_tag="mpnn:1.0.0",
    category=ToolCategory.INVERSE_FOLDING,
    requires_gpu=True,
    gpu_count=1,
    input_schema=InverseFoldingInput,
    output_schema=InverseFoldingOutput,
    default_timeout=600,
    supports_batch=False,
    description="Design protein sequences for given backbone structures using ProteinMPNN.",
    version="1.0.0",
    notes=_MPNN_NOTES,
)

TOOL_REGISTRY["ligandmpnn"] = ToolEntry(
    image_tag="mpnn:1.0.0",
    category=ToolCategory.INVERSE_FOLDING,
    requires_gpu=True,
    gpu_count=1,
    input_schema=InverseFoldingInput,
    output_schema=InverseFoldingOutput,
    default_timeout=600,
    supports_batch=False,
    description="Design protein sequences with ligand awareness using LigandMPNN.",
    version="1.0.0",
    notes=_MPNN_NOTES
    + (
        "For protein-ligand complexes, the foundry parser separates non-polymer "
        "residues (ligands, ions) into synthetic chain IDs. Calcium ions (atom "
        "name 'CA') may be miscounted as protein residues by external PDB parsers, "
        "but the foundry parser handles them correctly.",
    ),
)

# ---------------------------------------------------------------------------
# Scoring runner — conditional log-likelihood scoring
# ---------------------------------------------------------------------------

_SCORE_CHECKPOINT_DIR = "/app/LigandMPNN/model_params"

_SCORE_MODEL_CONFIG: dict[str, dict[str, str]] = {
    "proteinmpnn_score": {
        "model_type": "protein_mpnn",
        "checkpoint": "proteinmpnn_v_48_020.pt",
    },
    "ligandmpnn_score": {
        "model_type": "ligand_mpnn",
        "checkpoint": "ligandmpnn_v_32_010_25.pt",
    },
}


class MPNNScoreRunner(ToolRunner):
    """Runner for ProteinMPNN/LigandMPNN sequence scoring (conditional log-likelihood).

    Maps ``ScoringInput`` fields to the container's config.json with
    ``mode="score"`` and parses the standardised ``result_data.json`` back
    into a ``ScoringOutput``.

    When ``sequences`` is ``None``, the container scores the native sequence
    extracted from the PDB.
    """

    def prepare_workspace(self, input_data: BaseInput, workspace: Workspace) -> None:
        """Write config.json and copy the input structure into the workspace."""
        assert isinstance(input_data, ScoringInput)
        model_cfg = _SCORE_MODEL_CONFIG[self.tool_name]

        # Copy structure file into workspace inputs/
        src_path = input_data.structure_path
        dest_name = src_path.name
        shutil.copy2(src_path, workspace.inputs_dir / dest_name)

        # Build config.json for the container
        config: dict[str, object] = {
            "mode": "score",
            "model_type": model_cfg["model_type"],
            "checkpoint_path": f"{_SCORE_CHECKPOINT_DIR}/{model_cfg['checkpoint']}",
            "structure_path": f"/workspace/inputs/{dest_name}",
            "sequences": input_data.sequences,
        }

        # Flat-merge extra dict for tool-specific params
        config.update(input_data.extra)

        workspace.write_config(config)

    def parse_output(self, workspace: Workspace) -> ScoringOutput:
        """Read standardised outputs and return a ``ScoringOutput``."""
        result_data_path = workspace.std_output_dir / "result_data.json"
        data = json.loads(result_data_path.read_text())

        scores = [
            ScoredStructure(
                total_score=s["total_score"],
                per_residue_scores=s.get("per_residue_scores"),
                score_breakdown=s.get("score_breakdown"),
                units=s.get("units"),
                structure_path=s.get("structure_path"),
                ddg=s.get("ddg"),
                mutations=s.get("mutations"),
            )
            for s in data["scores"]
        ]

        # Placeholder metadata — overwritten by base class run()
        return ScoringOutput(
            scores=scores,
            metadata=self._build_metadata(workspace, 0.0, [], ""),
            raw_output_path=workspace.raw_output_dir,
        )


# ---------------------------------------------------------------------------
# Scoring registry entries
# ---------------------------------------------------------------------------

_MPNN_SCORE_NOTES = (
    "Computes conditional log-likelihood of a sequence given a backbone structure "
    "using ProteinMPNN. Returns average negative log-likelihood (lower is better).",
    "When sequences is None, scores the native sequence from the PDB.",
    "Accepts PDB input. mmCIF support depends on the LigandMPNN parser.",
    "The score_breakdown includes per-chain mean NLL and perplexity.",
)

TOOL_REGISTRY["proteinmpnn_score"] = ToolEntry(
    image_tag="mpnn-score:1.0.0",
    category=ToolCategory.SCORING,
    requires_gpu=True,
    gpu_count=1,
    input_schema=ScoringInput,
    output_schema=ScoringOutput,
    default_timeout=300,
    supports_batch=False,
    description=(
        "Score protein sequences against backbone structures using ProteinMPNN "
        "conditional log-likelihood."
    ),
    version="1.0.0",
    notes=_MPNN_SCORE_NOTES,
)

_LIGANDMPNN_SCORE_NOTES = (
    "Computes conditional log-likelihood of a sequence given a backbone structure "
    "using LigandMPNN, with ligand-aware context.",
    "When sequences is None, scores the native sequence from the PDB.",
    "For protein-ligand complexes, ligand atoms are included as context for scoring.",
    "Accepts PDB input. mmCIF support depends on the LigandMPNN parser.",
    "The score_breakdown includes per-chain mean NLL and perplexity.",
)

TOOL_REGISTRY["ligandmpnn_score"] = ToolEntry(
    image_tag="mpnn-score:1.0.0",
    category=ToolCategory.SCORING,
    requires_gpu=True,
    gpu_count=1,
    input_schema=ScoringInput,
    output_schema=ScoringOutput,
    default_timeout=300,
    supports_batch=False,
    description=(
        "Score protein sequences against backbone structures using LigandMPNN "
        "conditional log-likelihood with ligand-aware context."
    ),
    version="1.0.0",
    notes=_LIGANDMPNN_SCORE_NOTES,
)
