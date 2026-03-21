# Tool Runner Specification — `TOOL_SPEC.md`

This document defines how to create a new host-side tool runner. A tool runner is a Python class that translates between autobio's standardized schemas and a specific tool's expected inputs and outputs. It is the bridge between the category-level interface that agents interact with and the tool-specific container.

---

## 1. Purpose

Each tool in autobio has:
- A **container** that runs the tool (see `containers/CONTAINER_SPEC.md`)
- A **runner** that prepares inputs for the container and parses its outputs

The runner is responsible for:
1. Translating a standardized schema input (e.g., `StructurePredictionInput`) into the tool's specific `config.json` and input file layout.
2. Deserializing the container's standardized output files into Pydantic model instances.
3. Registering the tool in the tool registry with its metadata.

The runner is NOT responsible for:
- Launching containers, managing GPUs, or handling timeouts (the base class does this).
- Understanding the tool's raw output format (the container's `standardize.sh` handles this).
- Validating the standardized output against the schema (Pydantic does this automatically during deserialization).

---

## 2. File Location and Naming

Tool runners live in `src/autobio/tools/`. Each tool gets its own module:

```
tools/
├── __init__.py
├── base.py              # ToolRunner ABC — do not modify per-tool
├── TOOL_SPEC.md         # this document
├── alphafold.py
├── boltz.py
├── esm2.py
├── ligandmpnn.py
└── <new_tool>.py
```

Module names use `snake_case` matching the tool's canonical name in the registry.

---

## 3. The ToolRunner Base Class

All runners inherit from `ToolRunner` defined in `base.py`. The base class provides the full execution lifecycle — runners only implement two abstract methods.

```python
from abc import ABC, abstractmethod
from pathlib import Path

from autobio.core.config import AutobioConfig
from autobio.core.container import ContainerManager
from autobio.core.gpu import GPUManager
from autobio.core.registry import TOOL_REGISTRY
from autobio.core.workspace import Workspace
from autobio.schemas.base import BaseInput, BaseOutput


class ToolRunner(ABC):
    """
    Abstract base for all tool runners. Subclasses implement
    prepare_workspace() and parse_output() only.
    """

    def __init__(self, tool_name: str, config: AutobioConfig):
        self.tool_name = tool_name
        self.entry = TOOL_REGISTRY[tool_name]
        self.config = config
        self._container = ContainerManager(config)
        self._gpu = GPUManager()

    @abstractmethod
    def prepare_workspace(
        self, input_data: BaseInput, workspace: Workspace
    ) -> None:
        """
        Write config.json and input files to the workspace.

        This method translates the standardized input schema into the
        tool-specific configuration and file layout expected by the
        container's run.sh.

        Args:
            input_data: Validated input conforming to the tool's category schema.
            workspace: Initialized workspace with directories created.
        """
        ...

    @abstractmethod
    def parse_output(self, workspace: Workspace) -> BaseOutput:
        """
        Read standardized outputs from the workspace into a Pydantic model.

        The container's standardize.sh has already coerced raw outputs into
        the schema format. This method reads outputs/standardized/result_data.json
        and any associated files, then returns a populated output model.

        Args:
            workspace: Workspace after successful container execution.

        Returns:
            Populated output model (e.g., StructurePredictionOutput).
        """
        ...

    def run(
        self,
        input_data: BaseInput,
        gpu: str | list[int] = "auto",
        timeout: int | None = None,
        output_dir: Path | None = None,
    ) -> BaseOutput:
        """
        Full execution lifecycle. Do NOT override this method.
        See DESIGN.md §2.3 for the detailed flow.
        """
        ...
```

---

## 4. Implementing a New Runner

### 4.1 Create the Module

Create `src/autobio/tools/<tool_name>.py`. Example for a hypothetical tool called "proteinx":

```python
import json
from pathlib import Path

from autobio.schemas.structure_prediction import (
    ConfidenceMetrics,
    PredictedStructure,
    StructurePredictionInput,
    StructurePredictionOutput,
)
from autobio.schemas.base import RunMetadata
from autobio.core.workspace import Workspace
from autobio.tools.base import ToolRunner


class ProteinXRunner(ToolRunner):
    """Runner for ProteinX structure prediction tool."""

    def prepare_workspace(
        self, input_data: StructurePredictionInput, workspace: Workspace
    ) -> None:
        # --- Translate standardized input to tool-specific config ---
        config = {
            # ProteinX uses "target_sequences" instead of "sequences"
            "target_sequences": [
                {"chain_id": chain, "sequence": seq}
                for chain, seq in input_data.sequences.items()
            ],
            # ProteinX calls this "n_predictions"
            "n_predictions": input_data.num_models,
            # Pass through any tool-specific params from extra
            **input_data.extra,
        }
        workspace.write_config(config)

        # --- Write input files if needed ---
        if input_data.templates:
            for template_path in input_data.templates:
                workspace.write_input_file(
                    template_path.name,
                    template_path.read_bytes(),
                )

    def parse_output(self, workspace: Workspace) -> StructurePredictionOutput:
        # --- Read the standardized output ---
        result_data_path = workspace.std_output_dir / "result_data.json"
        with open(result_data_path) as f:
            data = json.load(f)

        # --- Deserialize into schema objects ---
        structures = [
            PredictedStructure(
                model_rank=s["model_rank"],
                structure_path=workspace.root / s["structure_path"],
                plddt_per_residue=s.get("plddt_per_residue"),
                plddt_mean=s.get("plddt_mean"),
                ptm=s.get("ptm"),
                iptm=s.get("iptm"),
                chain_mapping=s.get("chain_mapping"),
            )
            for s in data["structures"]
        ]

        confidence = ConfidenceMetrics(**data["confidence"])

        # NOTE: metadata is populated by the base class run() method,
        # not by parse_output(). Use a placeholder here.
        return StructurePredictionOutput(
            structures=structures,
            confidence=confidence,
            metadata=self._build_metadata(workspace),
            raw_output_path=workspace.raw_output_dir,
        )
```

### 4.2 Key Implementation Details

**`prepare_workspace` must:**
- Write a `config.json` to `workspace.config_path` containing all parameters the container needs.
- Copy or write any input files (FASTA sequences, PDB structures, etc.) to `workspace.inputs_dir`.
- Merge `input_data.extra` into the config, typically via `**input_data.extra` or by inserting it under a dedicated key.
- NOT launch containers, manage GPUs, or handle errors — the base class does this.

**`parse_output` must:**
- Read `outputs/standardized/result_data.json` from the workspace.
- Resolve relative file paths in the JSON against the workspace root.
- Return a fully populated Pydantic output model.
- NOT read from `outputs/raw/` — the standardized output is the contract.
- NOT handle missing workspaces or container failures — the base class checks `result.json` before calling `parse_output`.

**Handling the `extra` dict:**

The `extra` dict provides a pass-through for tool-specific parameters. There are two strategies:

1. **Flat merge** — spread `extra` into the top level of `config.json`. Simple and works well when extra params don't conflict with standard ones:
   ```python
   config = {
       "sequences": ...,
       "num_models": ...,
       **input_data.extra,
   }
   ```

2. **Namespaced** — put extra under a dedicated key. Safer when the tool's config namespace is crowded:
   ```python
   config = {
       "sequences": ...,
       "num_models": ...,
       "advanced": input_data.extra,
   }
   ```

Choose whichever is more natural for the specific tool. Document the choice in the runner's docstring.

### 4.3 Batch-Aware Runners

For tools that support batch processing, the runner's `prepare_workspace` writes multiple inputs and configures batch mode:

```python
def prepare_workspace(
    self, input_data: EmbeddingInput, workspace: Workspace
) -> None:
    # Write each sequence as a separate input or as a combined FASTA
    fasta_content = "\n".join(
        f">{name}\n{seq}" for name, seq in input_data.sequences.items()
    )
    workspace.write_input_file("sequences.fasta", fasta_content)

    config = {
        "batch_mode": "batched",
        "batch_size": input_data.extra.get("batch_size", 32),
        "model_name": "esm2_t33_650M_UR50D",
        **input_data.extra,
    }
    workspace.write_config(config)
```

The `parse_output` method for batch tools reads a list of results:

```python
def parse_output(self, workspace: Workspace) -> EmbeddingOutput:
    result_data_path = workspace.std_output_dir / "result_data.json"
    with open(result_data_path) as f:
        data = json.load(f)

    embeddings = [
        SequenceEmbedding(
            sequence_id=e["sequence_id"],
            embedding_path=workspace.root / e["embedding_path"],
            dimension=e["dimension"],
        )
        for e in data["embeddings"]
    ]
    return EmbeddingOutput(
        embeddings=embeddings,
        metadata=self._build_metadata(workspace),
        raw_output_path=workspace.raw_output_dir,
    )
```

---

## 5. Registering the Tool

After implementing the runner, register the tool in `core/registry.py`:

```python
from autobio.schemas.structure_prediction import (
    StructurePredictionInput,
    StructurePredictionOutput,
)

TOOL_REGISTRY["proteinx"] = ToolEntry(
    image_tag="proteinx:1.0.0",
    category="structure-prediction",
    requires_gpu=True,
    gpu_count=1,
    input_schema=StructurePredictionInput,
    output_schema=StructurePredictionOutput,
    default_timeout=3600,
    supports_batch=False,
    description="Predict protein structures using ProteinX.",
    version="1.0.0",
    notes=(
        "Requires at least 16 GB GPU memory for sequences over 1000 residues.",
        "Template search is disabled by default; pass 'use_templates': true in extra.",
    ),
)
```

The `notes` field is a tuple of strings surfaced by `autobio info <tool>` (both table and JSON formats). Use it to record tool-specific quirks, parser limitations, known edge cases, or resource requirements — anything an agent or user would benefit from knowing before running the tool. Every `ToolEntry` should include `notes`; use an empty tuple `()` if there are no caveats to document. Good notes are:

- **Actionable**: tell the reader what to do or avoid, not just what the problem is.
- **Specific**: reference concrete error messages, thresholds, or input characteristics.
- **Discovered empirically**: capture things that aren't obvious from the tool's documentation — parser quirks, input format restrictions, performance cliffs, etc.

And register the runner class so the CLI can instantiate it. In `tools/__init__.py`:

```python
from autobio.tools.proteinx import ProteinXRunner

TOOL_RUNNERS: dict[str, type[ToolRunner]] = {
    "proteinx": ProteinXRunner,
    # ... other tools
}
```

---

## 6. Testing the Runner

Each runner should have tests at multiple tiers.

### Tier 1 — Unit Tests (`tests/unit/`)

Test `prepare_workspace` and `parse_output` in isolation, without Docker:

```python
def test_proteinx_prepare_workspace(tmp_path):
    """Verify config.json and input files are written correctly."""
    workspace = Workspace.create(tmp_path / "ws")
    runner = ProteinXRunner("proteinx", AutobioConfig.resolve())
    input_data = StructurePredictionInput(
        sequences={"A": "MKWVTFIS"},
        num_models=3,
    )
    runner.prepare_workspace(input_data, workspace)

    config = json.loads(workspace.config_path.read_text())
    assert config["n_predictions"] == 3
    assert len(config["target_sequences"]) == 1
    assert config["target_sequences"][0]["sequence"] == "MKWVTFIS"


def test_proteinx_parse_output(tmp_path):
    """Verify standardized JSON is correctly deserialized."""
    workspace = Workspace.create(tmp_path / "ws")
    # Write mock standardized output (what the container would produce)
    result_data = {
        "structures": [{
            "model_rank": 1,
            "structure_path": "outputs/standardized/model_1.pdb",
            "plddt_mean": 90.5,
            "ptm": 0.88,
        }],
        "confidence": {"best_plddt_mean": 90.5, "best_ptm": 0.88},
    }
    (workspace.std_output_dir / "result_data.json").write_text(
        json.dumps(result_data)
    )
    # Write the structure file so path validation passes
    (workspace.std_output_dir / "model_1.pdb").write_text("ATOM ...")

    runner = ProteinXRunner("proteinx", AutobioConfig.resolve())
    output = runner.parse_output(workspace)

    assert len(output.structures) == 1
    assert output.structures[0].plddt_mean == 90.5
    assert output.confidence.best_ptm == 0.88
```

### Tier 3 — Smoke Tests (`tests/smoke/`)

Run the full pipeline with minimal input (requires Docker, may require GPU):

```python
@pytest.mark.docker
@pytest.mark.gpu
def test_proteinx_smoke():
    """Full pipeline: workspace → container → parsed output."""
    input_data = StructurePredictionInput(
        sequences={"A": "GGGGGGGGG"},  # trivially small
        num_models=1,
    )
    runner = ProteinXRunner("proteinx", AutobioConfig.resolve())
    output = runner.run(input_data, gpu="auto", timeout=120)

    assert len(output.structures) == 1
    assert output.structures[0].structure_path.exists()
    assert output.metadata.tool_name == "proteinx"
```

---

## 7. Checklist for New Tool Runners

- [ ] Module created at `tools/<tool_name>.py`
- [ ] Runner class inherits from `ToolRunner`
- [ ] `prepare_workspace` writes valid `config.json` for the tool's container
- [ ] `prepare_workspace` handles `input_data.extra` (flat merge or namespaced)
- [ ] `prepare_workspace` copies required input files to `workspace.inputs_dir`
- [ ] `parse_output` reads from `outputs/standardized/result_data.json`
- [ ] `parse_output` resolves relative file paths against workspace root
- [ ] `parse_output` returns fully populated Pydantic output model
- [ ] Tool registered in `core/registry.py` with correct schemas and metadata
- [ ] `ToolEntry.notes` populated with known quirks, limitations, or usage caveats (empty tuple if none)
- [ ] Runner class registered in `tools/__init__.py`
- [ ] Unit tests cover `prepare_workspace` config generation
- [ ] Unit tests cover `parse_output` deserialization with mock standardized data
- [ ] Smoke test exercises full pipeline (Docker + optional GPU)
- [ ] Corresponding container exists (see `containers/CONTAINER_SPEC.md`)
- [ ] Runner docstring documents any `extra` dict conventions specific to this tool
