"""ESMFold structure prediction tool runner.

ESMFold predicts protein structure from a single amino acid sequence using
the ESM-2 language model backbone with a folding trunk. No MSA or templates
are needed — it is a direct sequence-to-structure method.

Key limitations:
- Single-chain only (no multimer prediction).
- Deterministic — always produces exactly one structure.
- No template or MSA support.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from autobio.core.catalog import Mode, Tool, register
from autobio.core.registry import ToolCategory
from autobio.core.result import AutobioError
from autobio.schemas.base import BaseInput  # noqa: TC001 - needed at runtime for isinstance
from autobio.schemas.structure_prediction import (
    ConfidenceMetrics,
    ESMFoldInput,
    PredictedStructure,
    StructurePredictionOutput,
)
from autobio.tools.base import ToolRunner
from autobio.utils.sequences import validate_protein_sequence, write_fasta

if TYPE_CHECKING:
    from autobio.core.workspace import Workspace

# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------

_HF_CACHE = "/app/esmfold/hf_cache"
_MODEL_NAME = "facebook/esmfold_v1"

# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


class ESMFoldRunner(ToolRunner):
    """Runner for ESMFold single-sequence structure prediction.

    ESMFold uses the dedicated ``ESMFoldInput`` schema, which only supports
    a subset of the general structure-prediction feature set: single-chain
    sequences, no templates, and ``num_models=1`` (deterministic output).
    Unsupported inputs are rejected with clear error messages during
    host-side validation.
    """

    def prepare_workspace(self, input_data: BaseInput, workspace: Workspace) -> None:
        """Write config.json and input FASTA into the workspace."""
        assert isinstance(input_data, ESMFoldInput)

        # Host-side validation
        self._validate_inputs(input_data)

        # Write FASTA
        write_fasta(input_data.sequences, workspace.inputs_dir / "sequences.fasta")

        # Build config.json
        config: dict[str, object] = {
            "model_name": _MODEL_NAME,
            "input_fasta": "/workspace/inputs/sequences.fasta",
            "output_dir": "/workspace/outputs/raw",
            "hf_cache": _HF_CACHE,
        }

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

        confidence = ConfidenceMetrics(
            best_plddt_mean=data["confidence"].get("best_plddt_mean"),
            best_ptm=data["confidence"].get("best_ptm"),
            best_iptm=data["confidence"].get("best_iptm"),
        )

        # Placeholder metadata — overwritten by base class run()
        return StructurePredictionOutput(
            structures=structures,
            confidence=confidence,
            metadata=self._build_metadata(workspace, 0.0, [], ""),
            raw_output_path=workspace.raw_output_dir,
        )

    @staticmethod
    def _validate_inputs(input_data: ESMFoldInput) -> None:
        """Host-side validation — catch unsupported inputs before container launch."""
        if not input_data.sequences:
            raise AutobioError("sequences must be non-empty.")

        if len(input_data.sequences) > 1:
            raise AutobioError(
                "ESMFold is single-chain only. Received "
                f"{len(input_data.sequences)} chains "
                f"({sorted(input_data.sequences)}). For multimer structure "
                "prediction, use boltz2, chai1, or openfold3."
            )

        for seq_id, seq in input_data.sequences.items():
            if not validate_protein_sequence(seq):
                raise AutobioError(
                    f"Invalid protein sequence for {seq_id!r}: "
                    f"must contain only standard amino acid characters (ACDEFGHIKLMNPQRSTVWY)."
                )

        if input_data.templates:
            raise AutobioError(
                "ESMFold does not use templates. The templates field must be "
                "None or empty. For template-based prediction, use boltz2, "
                "chai1, or openfold3."
            )

        if input_data.num_models > 1:
            raise AutobioError(
                "ESMFold is deterministic and always produces exactly one "
                f"structure. num_models must be 1, got {input_data.num_models}."
            )

    @staticmethod
    def _resolve_container_path(container_path_str: str, workspace: Workspace) -> Path:
        """Map a container-internal ``/workspace/...`` path to the host workspace."""
        container_path = Path(container_path_str)
        try:
            relative = container_path.relative_to("/workspace")
        except ValueError:
            return container_path
        return workspace.root / relative


# ---------------------------------------------------------------------------
# Catalog registration — populated when this module is imported
# ---------------------------------------------------------------------------

_ESMFOLD_NOTES = (
    "ESMFold is single-chain only. It cannot predict multimer structures "
    "or protein-ligand complexes. For multimer prediction, use boltz2, "
    "chai1, or openfold3.",
    "ESMFold is deterministic — num_models must be 1. It always produces "
    "exactly one structure per input sequence.",
    "ESMFold does not use templates or MSAs. It is a direct sequence-to-"
    "structure method. The templates field must be None.",
    "ESMFold produces pLDDT and pTM confidence metrics but not ipTM "
    "(interface score requires multiple chains).",
    "Maximum sequence length is approximately 1200 residues due to GPU "
    "memory constraints. Longer sequences may cause OOM errors. The model "
    "requires 16-24GB GPU memory for medium-length sequences.",
)

ESMFOLD_TOOL = Tool(
    name="esmfold",
    display_name="ESMFold",
    category=ToolCategory.STRUCTURE_PREDICTION,
    description=(
        "Predict protein structure from a single sequence using ESMFold. "
        "No MSA or templates needed — direct sequence-to-structure prediction."
    ),
    version="1.0.0",
    image_tag="esmfold:1.0.0",
    requires_gpu=True,
    gpu_count=1,
    default_mode="predict",
    modes={
        "predict": Mode(
            name="predict",
            display_name="Predict structure",
            description="Predict a single-chain protein structure from sequence.",
            input_schema=ESMFoldInput,
            output_schema=StructurePredictionOutput,
            default_timeout=600,
            notes=_ESMFOLD_NOTES,
        )
    },
    keywords=("esmfold", "structure prediction", "protein folding", "single sequence"),
)
"""Catalog Tool for ESMFold — exposed for tests re-registering after CATALOG-clearing fixtures."""

register(ESMFOLD_TOOL)
