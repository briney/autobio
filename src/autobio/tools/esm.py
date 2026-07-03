"""ESM-1b and ESM-2 protein language model embedding tool runners.

Both tools share a single Docker image (``autobio-esm``) and runner class.
``self.tool.name`` (``"esm1b"`` or ``"esm2"``) determines which model is used.

ESM-2 supports multiple checkpoint sizes (8M, 35M, 150M, 650M, 3B, 15B) via
the typed ``checkpoint`` field on ``ESM2Input``. The default checkpoint is
650M. The default image includes 8M through 650M; 3B and 15B require
dedicated builds.

Additional tool-specific parameters (``batch_size``, ``seed``, etc.) are
passed through the ``extra`` dict on ``ESMEmbedInput``.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from autobio.core.catalog import Mode, Tool, register
from autobio.core.registry import ToolCategory
from autobio.core.result import AutobioError
from autobio.schemas.base import BaseInput  # noqa: TC001 - needed at runtime for isinstance
from autobio.schemas.embedding import (
    EmbeddingOutput,
    ESM2Input,
    ESMEmbedInput,
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

# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


class ESMRunner(ToolRunner):
    """Shared runner for ESM-1b and ESM-2 protein embedding tools.

    Both models use the same container image and three-phase protocol.
    ``prepare_workspace`` maps standardised ``ESMEmbedInput`` fields to
    the container's ``config.json``.  ``parse_output`` reads the
    standardised ``result_data.json`` produced by the container's
    ``standardize.py`` script.
    """

    def prepare_workspace(self, input_data: BaseInput, workspace: Workspace) -> None:
        """Write config.json and input FASTA into the workspace."""
        assert isinstance(input_data, ESMEmbedInput)

        model_cfg = self._resolve_model_config(input_data)
        self._validate_inputs(input_data, model_cfg)

        write_fasta(input_data.sequences, workspace.inputs_dir / "sequences.fasta")

        config: dict[str, object] = {
            "model_name": model_cfg["model_name"],
            "input_fasta": "/workspace/inputs/sequences.fasta",
            "output_dir": "/workspace/outputs/raw",
            "layer": input_data.layer,
            "pooling": input_data.pooling,
            "hf_cache": _HF_CACHE,
        }
        self._apply_extra(config, input_data)

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

    def _resolve_model_config(self, input_data: ESMEmbedInput) -> dict[str, str | int]:
        """Return the model config for the current Tool and (esm2) checkpoint."""
        assert self.tool is not None
        if self.tool.name == "esm1b":
            return _ESM1B_CONFIG
        # esm2 — checkpoint is a validated Literal on ESM2Input
        assert isinstance(input_data, ESM2Input)
        checkpoint = input_data.checkpoint
        if checkpoint not in _ESM2_CHECKPOINTS:
            available = ", ".join(sorted(_ESM2_CHECKPOINTS))
            raise AutobioError(
                f"Unknown ESM-2 checkpoint {checkpoint!r}. Available checkpoints: {available}"
            )
        return _ESM2_CHECKPOINTS[checkpoint]

    @staticmethod
    def _validate_inputs(input_data: ESMEmbedInput, model_cfg: dict[str, str | int]) -> None:
        """Host-side validation — catch errors before container launch."""
        if not input_data.sequences:
            raise AutobioError("sequences must be non-empty.")
        for seq_id, seq in input_data.sequences.items():
            if not validate_protein_sequence(seq):
                raise AutobioError(
                    f"Invalid protein sequence for {seq_id!r}: "
                    f"must contain only standard amino acid characters (ACDEFGHIKLMNPQRSTVWY)."
                )
        num_layers = int(model_cfg["num_layers"])
        if input_data.layer is not None and not (0 <= input_data.layer <= num_layers):
            raise AutobioError(
                f"layer must be between 0 and {num_layers} for "
                f"{model_cfg['model_name']}, got {input_data.layer}."
            )


# ---------------------------------------------------------------------------
# Catalog registration — populated when this module is imported
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
    "ESM-2 checkpoint selection: set the 'checkpoint' field to a size code — "
    "'8M', '35M', '150M', '650M' (default), '3B', or '15B'. The default "
    "image includes 8M through 650M. The 3B and 15B checkpoints require "
    "dedicated image builds and >24GB / >40GB GPU memory respectively.",
)

ESM1B_TOOL = Tool(
    name="esm1b",
    display_name="ESM-1b",
    category=ToolCategory.EMBEDDING,
    description=(
        "Extract protein sequence embeddings using ESM-1b (650M params, 33 layers, 1280-dim)."
    ),
    version="1.0.0",
    image_tag="esm:1.0.0",
    requires_gpu=True,
    gpu_count=1,
    default_mode="embed",
    modes={
        "embed": Mode(
            name="embed",
            display_name="Embeddings",
            description="Extract per-sequence or per-residue embeddings.",
            input_schema=ESMEmbedInput,
            output_schema=EmbeddingOutput,
            default_timeout=600,
            supports_batch=True,
            notes=_ESM_NOTES,
        )
    },
    keywords=("esm", "embedding", "protein language model"),
)
"""The catalog Tool for ESM-1b — exposed for tests that re-register it after
CATALOG-clearing fixtures (e.g. CLI isolation tests)."""

register(ESM1B_TOOL)

ESM2_TOOL = Tool(
    name="esm2",
    display_name="ESM-2",
    category=ToolCategory.EMBEDDING,
    description=(
        "Extract protein sequence embeddings using ESM-2. Default checkpoint 650M "
        "(33 layers, 1280-dim); select 8M/35M/150M/3B/15B via the checkpoint field."
    ),
    version="1.0.0",
    image_tag="esm:1.0.0",
    requires_gpu=True,
    gpu_count=1,
    default_mode="embed",
    modes={
        "embed": Mode(
            name="embed",
            display_name="Embeddings",
            description="Extract per-sequence or per-residue embeddings.",
            input_schema=ESM2Input,
            output_schema=EmbeddingOutput,
            default_timeout=600,
            supports_batch=True,
            notes=_ESM2_NOTES,
        )
    },
    keywords=("esm", "esm2", "embedding", "protein language model"),
)
"""The catalog Tool for ESM-2 — exposed for tests that re-register it after
CATALOG-clearing fixtures (e.g. CLI isolation tests)."""

register(ESM2_TOOL)
