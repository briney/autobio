"""Antibody language model embedding and pseudo log-likelihood runners.

Supports CurrAb, BALM-paired, BALM-unpaired, ft-ESM, AbLang2, and
AntiBERTa2.  Each model has two tool variants: one for embedding
extraction and one for pseudo log-likelihood (PLL) scoring.  All twelve
tools share a single runner class (``AntibodyLMRunner``), dispatching on
``self.tool_name``.

Additional tool-specific parameters (``batch_size``, ``seed``, etc.) are
passed through the ``extra`` dict on ``AntibodyInput``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from autobio.core.registry import TOOL_REGISTRY, ToolCategory, ToolEntry
from autobio.core.result import AutobioError
from autobio.schemas.antibody import AntibodyInput, AntibodyPLLOutput, SequencePLL
from autobio.schemas.base import BaseInput  # noqa: TC001 - needed at runtime for isinstance
from autobio.schemas.embedding import EmbeddingOutput, SequenceEmbedding
from autobio.tools.base import ToolRunner
from autobio.utils.sequences import validate_antibody_sequence

if TYPE_CHECKING:
    from autobio.core.workspace import Workspace

# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------

_VALID_POOLING = frozenset({"mean", "cls", "per_residue"})

# Keys in ``extra`` consumed by the runner and NOT flat-merged into config.json.
_CONSUMED_EXTRA_KEYS = frozenset({"per_position"})


@dataclass(frozen=True)
class _ModelSpec:
    """Metadata for a single antibody language model."""

    model_name: str
    num_layers: int
    embedding_dim: int
    max_tokens: int
    model_family: str  # "esm", "roberta", "ablang2", "roformer"
    chain_separator: str  # "single_cls", "double_cls", "sep", "pipe", "sep_prefixed", "none"
    supports_paired: bool
    supports_unpaired: bool
    cache_path: str = "/app/antibody-lm/hf_cache"


_MODEL_CONFIG: dict[str, _ModelSpec] = {
    # CurrAb — ESM-2 architecture, single <cls> separator
    "currab": _ModelSpec(
        "brineylab/CurrAb",
        33,
        1280,
        320,
        "esm",
        "single_cls",
        True,
        True,
    ),
    "currab_pll": _ModelSpec(
        "brineylab/CurrAb",
        33,
        1280,
        320,
        "esm",
        "single_cls",
        True,
        True,
    ),
    # ft-ESM — ESM-2 fine-tuned, double <cls> separator
    "ft_esm": _ModelSpec(
        "brineylab/ft-ESM",
        33,
        1280,
        1024,
        "esm",
        "double_cls",
        True,
        True,
    ),
    "ft_esm_pll": _ModelSpec(
        "brineylab/ft-ESM",
        33,
        1280,
        1024,
        "esm",
        "double_cls",
        True,
        True,
    ),
    # BALM-paired — RoBERTa architecture, </s> separator
    "balm_paired": _ModelSpec(
        "brineylab/BALM-paired",
        24,
        1024,
        510,
        "roberta",
        "sep",
        True,
        False,
    ),
    "balm_paired_pll": _ModelSpec(
        "brineylab/BALM-paired",
        24,
        1024,
        510,
        "roberta",
        "sep",
        True,
        False,
    ),
    # BALM-unpaired — RoBERTa architecture, single chain only
    "balm_unpaired": _ModelSpec(
        "brineylab/BALM-unpaired",
        24,
        1024,
        254,
        "roberta",
        "none",
        False,
        True,
    ),
    "balm_unpaired_pll": _ModelSpec(
        "brineylab/BALM-unpaired",
        24,
        1024,
        254,
        "roberta",
        "none",
        False,
        True,
    ),
    # AbLang2 — custom ESM-2-derived architecture with RoPE + SwiGLU, pipe separator
    "ablang2": _ModelSpec(
        "ablang2-paired",
        12,
        480,
        512,
        "ablang2",
        "pipe",
        True,
        True,
        "/app/ablang2/weights",
    ),
    "ablang2_pll": _ModelSpec(
        "ablang2-paired",
        12,
        480,
        512,
        "ablang2",
        "pipe",
        True,
        True,
        "/app/ablang2/weights",
    ),
    # AntiBERTa2 — RoFormer with chain-prefix tokens and [SEP] separator
    "antiberta2": _ModelSpec(
        "alchemab/antiberta2",
        16,
        1024,
        250,
        "roformer",
        "sep_prefixed",
        True,
        True,
        "/app/antiberta2/hf_cache",
    ),
    "antiberta2_pll": _ModelSpec(
        "alchemab/antiberta2",
        16,
        1024,
        250,
        "roformer",
        "sep_prefixed",
        True,
        True,
        "/app/antiberta2/hf_cache",
    ),
}


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


class AntibodyLMRunner(ToolRunner):
    """Shared runner for all antibody language model tools.

    Supports both embedding extraction and pseudo log-likelihood scoring.
    The ``tool_name`` determines which model is loaded and whether
    embeddings or PLL scores are computed.
    """

    def prepare_workspace(self, input_data: BaseInput, workspace: Workspace) -> None:
        """Write config.json and input sequences into the workspace."""
        assert isinstance(input_data, AntibodyInput)

        spec = _MODEL_CONFIG[self.tool_name]

        # Host-side validation
        self._validate_inputs(input_data, spec)

        # Write sequences as JSON
        sequences_data = [
            {
                "id": seq.id,
                "heavy_chain": seq.heavy_chain,
                "light_chain": seq.light_chain,
            }
            for seq in input_data.sequences
        ]
        input_path = workspace.inputs_dir / "sequences.json"
        input_path.write_text(json.dumps(sequences_data, indent=2))

        # Build config.json
        mode = "pll" if self._is_pll_mode() else "embedding"
        config: dict[str, object] = {
            "model_name": spec.model_name,
            "model_family": spec.model_family,
            "chain_separator": spec.chain_separator,
            "input_file": "/workspace/inputs/sequences.json",
            "output_dir": "/workspace/outputs/raw",
            "mode": mode,
            "layer": input_data.layer,
            "pooling": input_data.pooling or "mean",
            "per_position": input_data.extra.get("per_position", False),
            "hf_cache": spec.cache_path,
        }

        # Flat-merge extra (excluding consumed keys)
        for key, value in input_data.extra.items():
            if key not in _CONSUMED_EXTRA_KEYS:
                config[key] = value

        workspace.write_config(config)

    def parse_output(self, workspace: Workspace) -> EmbeddingOutput | AntibodyPLLOutput:
        """Read standardised outputs and return the appropriate output model."""
        result_data_path = workspace.std_output_dir / "result_data.json"
        data = json.loads(result_data_path.read_text())

        if self._is_pll_mode():
            return self._parse_pll_output(data, workspace)
        return self._parse_embedding_output(data, workspace)

    def _is_pll_mode(self) -> bool:
        """Return True if this tool computes pseudo log-likelihood."""
        return self.tool_name.endswith("_pll")

    def _parse_embedding_output(
        self,
        data: dict[str, Any],
        workspace: Workspace,
    ) -> EmbeddingOutput:
        """Parse embedding results from result_data.json."""
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

        return EmbeddingOutput(
            embeddings=embeddings,
            model_name=data["model_name"],
            embedding_dimension=data["embedding_dimension"],
            metadata=self._build_metadata(workspace, 0.0, [], ""),
            raw_output_path=workspace.raw_output_dir,
        )

    def _parse_pll_output(
        self,
        data: dict[str, Any],
        workspace: Workspace,
    ) -> AntibodyPLLOutput:
        """Parse pseudo log-likelihood results from result_data.json."""
        scores = [
            SequencePLL(
                sequence_id=s["sequence_id"],
                pll=s["pll"],
                per_position_pll=s.get("per_position_pll"),
                sequence_length=s["sequence_length"],
            )
            for s in data["scores"]
        ]

        return AntibodyPLLOutput(
            scores=scores,
            model_name=data["model_name"],
            metadata=self._build_metadata(workspace, 0.0, [], ""),
            raw_output_path=workspace.raw_output_dir,
        )

    @staticmethod
    def _validate_inputs(input_data: AntibodyInput, spec: _ModelSpec) -> None:
        """Host-side validation — catch errors before container launch."""
        if not input_data.sequences:
            raise AutobioError("sequences must be non-empty.")

        for seq in input_data.sequences:
            # Validate chain combination against model capabilities
            is_paired = seq.heavy_chain is not None and seq.light_chain is not None
            is_heavy_only = seq.heavy_chain is not None and seq.light_chain is None
            is_light_only = seq.heavy_chain is None and seq.light_chain is not None

            if is_paired and not spec.supports_paired:
                raise AutobioError(
                    f"Sequence '{seq.id}': {spec.model_name} does not support paired "
                    f"sequences. Provide only one chain."
                )
            if (is_heavy_only or is_light_only) and not spec.supports_unpaired:
                raise AutobioError(
                    f"Sequence '{seq.id}': {spec.model_name} requires both heavy and light chains."
                )

            # Validate individual chain sequences
            if seq.heavy_chain is not None and not validate_antibody_sequence(seq.heavy_chain):
                raise AutobioError(
                    f"Sequence '{seq.id}' heavy_chain: contains invalid characters. "
                    f"Only amino acid characters are accepted."
                )
            if seq.light_chain is not None and not validate_antibody_sequence(seq.light_chain):
                raise AutobioError(
                    f"Sequence '{seq.id}' light_chain: contains invalid characters. "
                    f"Only amino acid characters are accepted."
                )

            # Validate combined token length
            total_len = len(seq.heavy_chain or "") + len(seq.light_chain or "")
            if total_len > spec.max_tokens:
                raise AutobioError(
                    f"Sequence '{seq.id}': combined chain length ({total_len}) exceeds "
                    f"maximum of {spec.max_tokens} tokens for {spec.model_name}."
                )

        # Validate layer
        if input_data.layer is not None and not (0 <= input_data.layer <= spec.num_layers):
            raise AutobioError(
                f"layer must be between 0 and {spec.num_layers} for "
                f"{spec.model_name}, got {input_data.layer}."
            )

        # Validate pooling
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

_ANTIBODY_NOTES = (
    "Input is a list of antibody sequences. Each sequence has an 'id' and "
    "optional 'heavy_chain' and 'light_chain' fields. At least one chain "
    "must be provided per sequence.",
    "Pooling strategies (embedding mode only): 'mean' averages over "
    "non-special token positions (shape (D,)); 'cls' uses the CLS token "
    "embedding (shape (D,)); 'per_residue' returns the full per-position "
    "matrix (shape (L, D)). Embeddings are saved as NumPy .npy files.",
)

_ANTIBODY_INPUT_FORMAT = (
    "Input sequences are provided as a list of AntibodySequence objects. "
    "Each has 'id', 'heavy_chain' (optional), and 'light_chain' (optional). "
    "At least one chain per sequence is required.",
    "Example: sequences=[AntibodySequence(id='ab1', heavy_chain='EVQLV...', "
    "light_chain='DIQMT...')]",
)

_PLL_NOTES = (
    "Pseudo log-likelihood (PLL) is computed by masking each non-special "
    "token position individually and summing the log-probabilities of the "
    "true tokens. Per-position scores are available via extra['per_position']=True.",
    "PLL computation requires N forward passes for an N-token sequence. "
    "This is significantly slower than embedding extraction.",
)

# -- CurrAb ----------------------------------------------------------------

TOOL_REGISTRY["currab"] = ToolEntry(
    image_tag="currab:1.0.0",
    category=ToolCategory.EMBEDDING,
    requires_gpu=True,
    gpu_count=1,
    input_schema=AntibodyInput,
    output_schema=EmbeddingOutput,
    default_timeout=600,
    supports_batch=True,
    description=(
        "Extract antibody sequence embeddings using CurrAb "
        "(650M parameters, 33 layers, 1280-dim). Supports paired and unpaired sequences."
    ),
    version="1.0.0",
    notes=_ANTIBODY_NOTES,
    input_format=_ANTIBODY_INPUT_FORMAT,
)

TOOL_REGISTRY["currab_pll"] = ToolEntry(
    image_tag="currab:1.0.0",
    category=ToolCategory.EMBEDDING,
    requires_gpu=True,
    gpu_count=1,
    input_schema=AntibodyInput,
    output_schema=AntibodyPLLOutput,
    default_timeout=1800,
    supports_batch=True,
    description=(
        "Compute pseudo log-likelihood for antibody sequences using CurrAb "
        "(650M parameters, 33 layers). Supports paired and unpaired sequences."
    ),
    version="1.0.0",
    notes=_ANTIBODY_NOTES + _PLL_NOTES,
    input_format=_ANTIBODY_INPUT_FORMAT,
)

# -- ft-ESM ----------------------------------------------------------------

TOOL_REGISTRY["ft_esm"] = ToolEntry(
    image_tag="ft-esm:1.0.0",
    category=ToolCategory.EMBEDDING,
    requires_gpu=True,
    gpu_count=1,
    input_schema=AntibodyInput,
    output_schema=EmbeddingOutput,
    default_timeout=600,
    supports_batch=True,
    description=(
        "Extract antibody sequence embeddings using ft-ESM "
        "(fine-tuned ESM-2, 650M parameters, 33 layers, 1280-dim). "
        "Supports paired and unpaired sequences."
    ),
    version="1.0.0",
    notes=_ANTIBODY_NOTES,
    input_format=_ANTIBODY_INPUT_FORMAT,
)

TOOL_REGISTRY["ft_esm_pll"] = ToolEntry(
    image_tag="ft-esm:1.0.0",
    category=ToolCategory.EMBEDDING,
    requires_gpu=True,
    gpu_count=1,
    input_schema=AntibodyInput,
    output_schema=AntibodyPLLOutput,
    default_timeout=1800,
    supports_batch=True,
    description=(
        "Compute pseudo log-likelihood for antibody sequences using ft-ESM "
        "(fine-tuned ESM-2, 650M parameters, 33 layers). "
        "Supports paired and unpaired sequences."
    ),
    version="1.0.0",
    notes=_ANTIBODY_NOTES + _PLL_NOTES,
    input_format=_ANTIBODY_INPUT_FORMAT,
)

# -- BALM-paired ------------------------------------------------------------

TOOL_REGISTRY["balm_paired"] = ToolEntry(
    image_tag="balm-paired:1.0.0",
    category=ToolCategory.EMBEDDING,
    requires_gpu=True,
    gpu_count=1,
    input_schema=AntibodyInput,
    output_schema=EmbeddingOutput,
    default_timeout=600,
    supports_batch=True,
    description=(
        "Extract paired antibody sequence embeddings using BALM-paired "
        "(304M parameters, 24 layers, 1024-dim). Requires both heavy and light chains."
    ),
    version="1.0.0",
    notes=_ANTIBODY_NOTES,
    input_format=_ANTIBODY_INPUT_FORMAT,
)

TOOL_REGISTRY["balm_paired_pll"] = ToolEntry(
    image_tag="balm-paired:1.0.0",
    category=ToolCategory.EMBEDDING,
    requires_gpu=True,
    gpu_count=1,
    input_schema=AntibodyInput,
    output_schema=AntibodyPLLOutput,
    default_timeout=1800,
    supports_batch=True,
    description=(
        "Compute pseudo log-likelihood for paired antibody sequences using BALM-paired "
        "(304M parameters, 24 layers). Requires both heavy and light chains."
    ),
    version="1.0.0",
    notes=_ANTIBODY_NOTES + _PLL_NOTES,
    input_format=_ANTIBODY_INPUT_FORMAT,
)

# -- BALM-unpaired ----------------------------------------------------------

TOOL_REGISTRY["balm_unpaired"] = ToolEntry(
    image_tag="balm-unpaired:1.0.0",
    category=ToolCategory.EMBEDDING,
    requires_gpu=True,
    gpu_count=1,
    input_schema=AntibodyInput,
    output_schema=EmbeddingOutput,
    default_timeout=600,
    supports_batch=True,
    description=(
        "Extract single-chain antibody sequence embeddings using BALM-unpaired "
        "(304M parameters, 24 layers, 1024-dim). Accepts one chain per sequence."
    ),
    version="1.0.0",
    notes=_ANTIBODY_NOTES,
    input_format=_ANTIBODY_INPUT_FORMAT,
)

TOOL_REGISTRY["balm_unpaired_pll"] = ToolEntry(
    image_tag="balm-unpaired:1.0.0",
    category=ToolCategory.EMBEDDING,
    requires_gpu=True,
    gpu_count=1,
    input_schema=AntibodyInput,
    output_schema=AntibodyPLLOutput,
    default_timeout=1800,
    supports_batch=True,
    description=(
        "Compute pseudo log-likelihood for single-chain antibody sequences using "
        "BALM-unpaired (304M parameters, 24 layers). Accepts one chain per sequence."
    ),
    version="1.0.0",
    notes=_ANTIBODY_NOTES + _PLL_NOTES,
    input_format=_ANTIBODY_INPUT_FORMAT,
)

# -- AbLang2 ---------------------------------------------------------------

TOOL_REGISTRY["ablang2"] = ToolEntry(
    image_tag="ablang2:1.0.0",
    category=ToolCategory.EMBEDDING,
    requires_gpu=True,
    gpu_count=1,
    input_schema=AntibodyInput,
    output_schema=EmbeddingOutput,
    default_timeout=600,
    supports_batch=True,
    description=(
        "Extract antibody sequence embeddings using AbLang2 "
        "(45M parameters, 12 layers, 480-dim). Supports paired and unpaired sequences."
    ),
    version="1.0.0",
    notes=_ANTIBODY_NOTES,
    input_format=_ANTIBODY_INPUT_FORMAT,
)

TOOL_REGISTRY["ablang2_pll"] = ToolEntry(
    image_tag="ablang2:1.0.0",
    category=ToolCategory.EMBEDDING,
    requires_gpu=True,
    gpu_count=1,
    input_schema=AntibodyInput,
    output_schema=AntibodyPLLOutput,
    default_timeout=1800,
    supports_batch=True,
    description=(
        "Compute pseudo log-likelihood for antibody sequences using AbLang2 "
        "(45M parameters, 12 layers). Supports paired and unpaired sequences."
    ),
    version="1.0.0",
    notes=_ANTIBODY_NOTES + _PLL_NOTES,
    input_format=_ANTIBODY_INPUT_FORMAT,
)

# -- AntiBERTa2 ------------------------------------------------------------

TOOL_REGISTRY["antiberta2"] = ToolEntry(
    image_tag="antiberta2:1.0.0",
    category=ToolCategory.EMBEDDING,
    requires_gpu=True,
    gpu_count=1,
    input_schema=AntibodyInput,
    output_schema=EmbeddingOutput,
    default_timeout=600,
    supports_batch=True,
    description=(
        "Extract antibody sequence embeddings using AntiBERTa2 "
        "(202M parameters, 16 layers, 1024-dim). Supports paired and unpaired sequences."
    ),
    version="1.0.0",
    notes=_ANTIBODY_NOTES,
    input_format=_ANTIBODY_INPUT_FORMAT,
)

TOOL_REGISTRY["antiberta2_pll"] = ToolEntry(
    image_tag="antiberta2:1.0.0",
    category=ToolCategory.EMBEDDING,
    requires_gpu=True,
    gpu_count=1,
    input_schema=AntibodyInput,
    output_schema=AntibodyPLLOutput,
    default_timeout=1800,
    supports_batch=True,
    description=(
        "Compute pseudo log-likelihood for antibody sequences using AntiBERTa2 "
        "(202M parameters, 16 layers). Supports paired and unpaired sequences."
    ),
    version="1.0.0",
    notes=_ANTIBODY_NOTES + _PLL_NOTES,
    input_format=_ANTIBODY_INPUT_FORMAT,
)
