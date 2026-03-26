"""ESM-1b and ESM-2 protein language model embedding tool runners.

Both tools share a single Docker image (``autobio-esm``) and runner class.
The ``tool_name`` (``"esm1b"`` or ``"esm2"``) determines which model is used.

ESM-2 supports multiple checkpoint sizes (8M, 35M, 150M, 650M, 3B, 15B).
The default checkpoint is 650M. To select a different size, pass
``extra["checkpoint"]`` with a size code (e.g., ``"150M"``). The default
image includes 8M through 650M; 3B and 15B require dedicated builds.

Additional tool-specific parameters (``batch_size``, ``seed``, etc.) are
passed through the ``extra`` dict on ``EmbeddingInput``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from autobio.core.registry import TOOL_REGISTRY, ToolCategory, ToolEntry
from autobio.core.result import AutobioError
from autobio.schemas.base import BaseInput  # noqa: TC001 - needed at runtime for isinstance
from autobio.schemas.embedding import (
    EmbeddingInput,
    EmbeddingOutput,
    SequenceEmbedding,
)
from autobio.tools.base import ToolRunner
from autobio.utils.sequences import validate_protein_sequence, write_fasta

if TYPE_CHECKING:
    from autobio.core.workspace import Workspace

# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------

_HF_CACHE = "/app/esm/hf_cache"

_VALID_POOLING = frozenset({"mean", "cls", "per_residue"})

_ESM1B_CONFIG: dict[str, str | int] = {
    "model_name": "facebook/esm1b_t33_650M_UR50S",
    "num_layers": 33,
    "embedding_dim": 1280,
}

_ESM2_CHECKPOINTS: dict[str, dict[str, str | int]] = {
    "8M": {"model_name": "facebook/esm2_t6_8M_UR50D", "num_layers": 6, "embedding_dim": 320},
    "35M": {"model_name": "facebook/esm2_t12_35M_UR50D", "num_layers": 12, "embedding_dim": 480},
    "150M": {"model_name": "facebook/esm2_t30_150M_UR50D", "num_layers": 30, "embedding_dim": 640},
    "650M": {"model_name": "facebook/esm2_t33_650M_UR50D", "num_layers": 33, "embedding_dim": 1280},
    "3B": {"model_name": "facebook/esm2_t36_3B_UR50D", "num_layers": 36, "embedding_dim": 2560},
    "15B": {"model_name": "facebook/esm2_t48_15B_UR50D", "num_layers": 48, "embedding_dim": 5120},
}

_ESM2_DEFAULT_CHECKPOINT = "650M"

# Keys in ``extra`` consumed by the runner and NOT flat-merged into config.json.
_CONSUMED_EXTRA_KEYS = frozenset({"checkpoint"})

# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


class ESMRunner(ToolRunner):
    """Shared runner for ESM-1b and ESM-2 protein embedding tools.

    Both models use the same container image and three-phase protocol.
    ``prepare_workspace`` maps standardised ``EmbeddingInput`` fields to
    the container's ``config.json``.  ``parse_output`` reads the
    standardised ``result_data.json`` produced by the container's
    ``standardize.py`` script.
    """

    def prepare_workspace(self, input_data: BaseInput, workspace: Workspace) -> None:
        """Write config.json and input FASTA into the workspace."""
        assert isinstance(input_data, EmbeddingInput)

        # Resolve model config
        model_cfg = self._resolve_model_config(input_data)

        # Host-side validation
        self._validate_inputs(input_data, model_cfg)

        # Write FASTA
        write_fasta(input_data.sequences, workspace.inputs_dir / "sequences.fasta")

        # Build config.json
        config: dict[str, object] = {
            "model_name": model_cfg["model_name"],
            "input_fasta": "/workspace/inputs/sequences.fasta",
            "output_dir": "/workspace/outputs/raw",
            "layer": input_data.layer,
            "pooling": input_data.pooling or "mean",
            "hf_cache": _HF_CACHE,
        }

        # Flat-merge extra (excluding consumed keys)
        for key, value in input_data.extra.items():
            if key not in _CONSUMED_EXTRA_KEYS:
                config[key] = value

        workspace.write_config(config)

    def parse_output(self, workspace: Workspace) -> EmbeddingOutput:
        """Read standardised outputs and return an ``EmbeddingOutput``."""
        result_data_path = workspace.std_output_dir / "result_data.json"
        data = json.loads(result_data_path.read_text())

        embeddings = [
            SequenceEmbedding(
                sequence_id=e["sequence_id"],
                embedding_path=self._resolve_container_path(e["embedding_path"], workspace),
                dimension=e["dimension"],
                layer=e.get("layer"),
                pooling=e.get("pooling"),
            )
            for e in data["embeddings"]
        ]

        # Placeholder metadata — overwritten by base class run()
        return EmbeddingOutput(
            embeddings=embeddings,
            model_name=data["model_name"],
            embedding_dimension=data["embedding_dimension"],
            metadata=self._build_metadata(workspace, 0.0, [], ""),
            raw_output_path=workspace.raw_output_dir,
        )

    def _resolve_model_config(self, input_data: EmbeddingInput) -> dict[str, str | int]:
        """Return the model config dict for the current tool and checkpoint."""
        if self.tool_name == "esm1b":
            return _ESM1B_CONFIG

        # esm2 — resolve checkpoint
        checkpoint = input_data.extra.get("checkpoint", _ESM2_DEFAULT_CHECKPOINT)
        if checkpoint not in _ESM2_CHECKPOINTS:
            available = ", ".join(sorted(_ESM2_CHECKPOINTS))
            raise AutobioError(
                f"Unknown ESM-2 checkpoint {checkpoint!r}. Available checkpoints: {available}"
            )
        return _ESM2_CHECKPOINTS[checkpoint]

    @staticmethod
    def _validate_inputs(input_data: EmbeddingInput, model_cfg: dict[str, str | int]) -> None:
        """Host-side validation — catch errors before container launch."""
        if not input_data.sequences:
            raise AutobioError("sequences must be non-empty.")

        for seq_id, seq in input_data.sequences.items():
            if not validate_protein_sequence(seq):
                raise AutobioError(
                    f"Invalid protein sequence for {seq_id!r}: "
                    f"must contain only standard amino acid characters (ACDEFGHIKLMNPQRSTVWY)."
                )

        # Validate layer is in range for this model
        num_layers = int(model_cfg["num_layers"])
        if input_data.layer is not None and not (0 <= input_data.layer <= num_layers):
            raise AutobioError(
                f"layer must be between 0 and {num_layers} for "
                f"{model_cfg['model_name']}, got {input_data.layer}."
            )

        # Validate pooling strategy
        if input_data.pooling is not None and input_data.pooling not in _VALID_POOLING:
            raise AutobioError(
                f"pooling must be one of {sorted(_VALID_POOLING)}, got {input_data.pooling!r}."
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
# Registry entries — populated when this module is imported
# ---------------------------------------------------------------------------

_ESM_NOTES = (
    "Maximum sequence length is 1022 tokens (1024 minus BOS/EOS). Sequences "
    "longer than 1022 residues will be truncated by the tokenizer.",
    "Layer 0 is the input embedding layer. The final layer (default) is "
    "generally recommended for downstream tasks. Intermediate layers can "
    "capture different levels of structural and evolutionary information.",
    "Pooling strategies: 'mean' averages over non-padding token positions "
    "(shape (D,)); 'cls' uses the CLS token embedding (shape (D,)); "
    "'per_residue' returns the full per-position matrix (shape (L, D)). "
    "Embeddings are saved as NumPy .npy files.",
)

_ESM2_NOTES = _ESM_NOTES + (
    "ESM-2 checkpoint selection: pass extra['checkpoint'] with a size code — "
    "'8M', '35M', '150M', '650M' (default), '3B', or '15B'. The default "
    "image includes 8M through 650M. The 3B and 15B checkpoints require "
    "dedicated image builds and >24GB / >40GB GPU memory respectively.",
)

_ESM_INPUT_FORMAT = (
    "Input is a dict of sequence ID to amino acid sequence. All standard "
    "amino acid characters (ACDEFGHIKLMNPQRSTVWY) are accepted.",
    "Example: sequences={'heavy': 'EVQLVESGGGLVQPGG...', 'light': 'DIQMTQSPSSLSASVG...'}",
)

TOOL_REGISTRY["esm1b"] = ToolEntry(
    image_tag="esm:1.0.0",
    category=ToolCategory.EMBEDDING,
    requires_gpu=True,
    gpu_count=1,
    input_schema=EmbeddingInput,
    output_schema=EmbeddingOutput,
    default_timeout=600,
    supports_batch=True,
    description=(
        "Extract protein sequence embeddings using ESM-1b (650M parameters, 33 layers, 1280-dim)."
    ),
    version="1.0.0",
    notes=_ESM_NOTES,
    input_format=_ESM_INPUT_FORMAT,
)

TOOL_REGISTRY["esm2"] = ToolEntry(
    image_tag="esm:1.0.0",
    category=ToolCategory.EMBEDDING,
    requires_gpu=True,
    gpu_count=1,
    input_schema=EmbeddingInput,
    output_schema=EmbeddingOutput,
    default_timeout=600,
    supports_batch=True,
    description=(
        "Extract protein sequence embeddings using ESM-2. Default checkpoint: "
        "650M (33 layers, 1280-dim). Select other sizes via extra['checkpoint']: "
        "'8M', '35M', '150M', '3B', '15B'."
    ),
    version="1.0.0",
    notes=_ESM2_NOTES,
    input_format=_ESM_INPUT_FORMAT,
)
