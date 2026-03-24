# Tool Runner Specification

This document defines how to create a new host-side tool runner. A tool runner is a Python class that translates between autobio's standardized schemas and a specific tool's expected inputs and outputs. It is the bridge between the category-level interface that agents interact with and the tool-specific container.

---

## 1. Purpose

Each tool in autobio has:
- A **container** that runs the tool (see `containers/CONTAINER_SPEC.md`)
- A **runner** that prepares inputs for the container and parses its outputs

The runner is responsible for:
1. Translating a standardized schema input (e.g., `InverseFoldingInput`) into the tool's specific `config.json` and input file layout.
2. Mapping schema field names to tool-specific config keys (e.g., `num_sequences` → `number_of_batches`).
3. Writing container-internal paths into `config.json` (e.g., `/workspace/inputs/structure.pdb`).
4. Deserializing the container's standardized output files into Pydantic model instances.
5. Registering the tool in the tool registry with its metadata and usage notes.

The runner is NOT responsible for:
- Launching containers, managing GPUs, or handling timeouts (the base class does this).
- Understanding the tool's raw output format (the container's `standardize.sh` handles this).
- Validating the standardized output against the schema (Pydantic does this automatically during deserialization).

---

## 2. File Location and Naming

Tool runners live in `src/autobio/tools/`. Each tool gets its own module. A single module can serve multiple tools if they share a runner class (e.g., `mpnn.py` serves both `proteinmpnn` and `ligandmpnn`):

```
src/autobio/tools/
├── __init__.py          # TOOL_RUNNERS dict + get_runner() factory
├── base.py              # ToolRunner ABC — do not modify per-tool
├── TOOL_SPEC.md         # this document
├── mpnn.py              # MPNNRunner — serves proteinmpnn + ligandmpnn
└── <new_tool>.py
```

Module names use `snake_case` matching the tool's canonical name in the registry.

---

## 3. The ToolRunner Base Class

All runners inherit from `ToolRunner` defined in `base.py`. The base class provides the full execution lifecycle — runners only implement two abstract methods.

```python
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
        """Write config.json and input files to the workspace."""
        ...

    @abstractmethod
    def parse_output(self, workspace: Workspace) -> BaseOutput:
        """Read standardized outputs into a Pydantic model."""
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

        Steps:
            1. Create workspace
            2. prepare_workspace (subclass hook)
            3. Resolve GPU allocation
            4. Ensure container image is available
            5. Run container
            6. Read result.json and check status
            7. parse_output (subclass hook)
            8. Attach metadata (overwrites placeholder from parse_output)
            9. Return output
        """
        ...
```

**Important:** The base class calls `parse_output` and then overwrites `output.metadata` with the real metadata (wall time, GPU IDs, image URI, timestamp). Your `parse_output` should return a placeholder metadata via `self._build_metadata(workspace, 0.0, [], "")` — it will be replaced.

---

## 4. Implementing a New Runner

### 4.1 Create the Module

Create `src/autobio/tools/<tool_name>.py`. Below is a simplified example based on the real MPNN runner:

```python
"""ProteinMPNN and LigandMPNN tool runners.

Both tools share a single Docker image (autobio-mpnn) and runner class.
The tool_name determines which model type and checkpoint are used.

Tool-specific params (omit, bias, atomize_side_chains, etc.) are passed
through the extra dict on InverseFoldingInput.
"""

import json
import shutil

from autobio.core.registry import TOOL_REGISTRY, ToolCategory, ToolEntry
from autobio.schemas.base import BaseInput
from autobio.schemas.inverse_folding import (
    DesignedSequence,
    InverseFoldingInput,
    InverseFoldingOutput,
)
from autobio.tools.base import ToolRunner

_CHECKPOINT_DIR = "/app/foundry/checkpoints"

_MODEL_CONFIG = {
    "proteinmpnn": {
        "model_type": "protein_mpnn",
        "checkpoint": "proteinmpnn_v_48_020.pt",
    },
    "ligandmpnn": {
        "model_type": "ligand_mpnn",
        "checkpoint": "ligandmpnn_v_32_010_25.pt",
    },
}


class MPNNRunner(ToolRunner):
    """Shared runner for ProteinMPNN and LigandMPNN."""

    def prepare_workspace(self, input_data: BaseInput, workspace) -> None:
        assert isinstance(input_data, InverseFoldingInput)
        model_cfg = _MODEL_CONFIG[self.tool_name]

        # Copy input files to workspace
        src_path = input_data.structure_path
        shutil.copy2(src_path, workspace.inputs_dir / src_path.name)

        # Build config.json with container-internal paths
        config = {
            "model_type": model_cfg["model_type"],
            "checkpoint_path": f"{_CHECKPOINT_DIR}/{model_cfg['checkpoint']}",
            "structure_path": f"/workspace/inputs/{src_path.name}",
            "number_of_batches": input_data.num_sequences,
            "temperature": input_data.temperature,
        }

        # Map schema fields to tool-specific config keys
        if input_data.chains_to_design is not None:
            config["designed_chains"] = ",".join(input_data.chains_to_design)

        # Handle mutually exclusive params
        if input_data.fixed_positions is not None:
            residue_ids = []
            for chain, positions in input_data.fixed_positions.items():
                for pos in positions:
                    residue_ids.append(f"{chain}{pos}")
            config["fixed_residues"] = ",".join(residue_ids)
            config.pop("designed_chains", None)  # mutually exclusive

        # Flat-merge extra dict
        config.update(input_data.extra)

        workspace.write_config(config)

    def parse_output(self, workspace) -> InverseFoldingOutput:
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
```

### 4.2 Key Implementation Details

**`prepare_workspace` must:**
- Write a `config.json` via `workspace.write_config(config)`.
- Copy input files to `workspace.inputs_dir` (use `shutil.copy2` for files, `workspace.write_input_file()` for generated content).
- Use **container-internal paths** in `config.json` (e.g., `/workspace/inputs/structure.pdb` not the host path). The runner knows these paths because the workspace is always mounted at `/workspace`.
- Reference **baked-in checkpoint paths** in `config.json` (e.g., `/app/checkpoints/model.pt`). The runner knows these because they are fixed at image build time (see `containers/CONTAINER_SPEC.md` §5.2).
- Map schema field names to tool-specific config keys. Document non-obvious mappings with comments.
- Handle mutually exclusive parameters (e.g., `designed_chains` vs `fixed_residues`).
- Merge `input_data.extra` into the config for tool-specific params.
- NOT launch containers, manage GPUs, or handle errors — the base class does this.

**`parse_output` must:**
- Read `outputs/standardized/result_data.json` from the workspace.
- Return a fully populated Pydantic output model with a placeholder metadata.
- NOT read from `outputs/raw/` — the standardized output is the contract.
- NOT handle missing workspaces or container failures — the base class checks `result.json` before calling `parse_output`.

**Handling the `extra` dict:**

The `extra` dict provides a pass-through for tool-specific parameters. There are two strategies:

1. **Flat merge** (preferred) — update the config dict with extra. Simple and works well when extra params don't conflict with standard ones:
   ```python
   config.update(input_data.extra)
   ```

2. **Namespaced** — put extra under a dedicated key. Safer when the tool's config namespace is crowded:
   ```python
   config["advanced"] = input_data.extra
   ```

Document the choice in the runner's module docstring.

### 4.3 Shared Runner Pattern

A single runner class can serve multiple tools when they share a container image and differ only in configuration. The `tool_name` attribute distinguishes them:

```python
_MODEL_CONFIG = {
    "proteinmpnn": {"model_type": "protein_mpnn", "checkpoint": "proteinmpnn_v_48_020.pt"},
    "ligandmpnn":  {"model_type": "ligand_mpnn",  "checkpoint": "ligandmpnn_v_32_010_25.pt"},
}

class MPNNRunner(ToolRunner):
    def prepare_workspace(self, input_data, workspace):
        model_cfg = _MODEL_CONFIG[self.tool_name]  # dispatches on tool_name
        ...
```

Both tools are registered separately in `TOOL_REGISTRY` (with their own descriptions and notes) and both map to the same runner class in `TOOL_RUNNERS`.

### 4.4 Batch-Aware Runners

For tools that support batch processing, the runner's `prepare_workspace` writes multiple inputs and configures batch mode:

```python
def prepare_workspace(self, input_data, workspace):
    fasta_content = "\n".join(
        f">{name}\n{seq}" for name, seq in input_data.sequences.items()
    )
    workspace.write_input_file("sequences.fasta", fasta_content)

    config = {
        "batch_mode": "batched",
        "batch_size": input_data.extra.get("batch_size", 32),
        **input_data.extra,
    }
    workspace.write_config(config)
```

---

## 5. Registering the Tool

Registration happens at the bottom of the tool module (not in `registry.py`). When the module is imported, the `ToolEntry` objects are added to `TOOL_REGISTRY`.

### 5.1 Registry Entry

```python
from autobio.core.registry import TOOL_REGISTRY, ToolCategory, ToolEntry

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
    notes=(
        "The foundry MPNN parser may resolve fewer residues than the PDB CA "
        "atom count due to internal filtering of disordered residues.",
        "PDB structures with multiple ASU copies can trigger an atomworks "
        "parser error. Use structures with unique chain IDs per sequence.",
    ),
)
```

**`image_tag`** is the tag portion only (e.g., `"mpnn:1.0.0"`). The host config's `image_prefix` is prepended at runtime to form the full URI (e.g., `ghcr.io/briney/autobio-mpnn:1.0.0`). Multiple tools can share the same `image_tag` when they use the same container.

**`notes`** is a tuple of strings surfaced by `autobio info <tool>` in both table and JSON formats. Use it to record operational guidance: tool-specific quirks, parser limitations, known edge cases, resource requirements, MSA/template options, key parameters — anything an agent or user would benefit from knowing about running and tuning the tool. Every `ToolEntry` must include `notes`; use an empty tuple `()` if there are no caveats. Good notes are:

- **Actionable**: tell the reader what to do or avoid, not just what the problem is.
- **Specific**: reference concrete error messages, thresholds, or input characteristics.
- **Discovered empirically**: capture things that aren't obvious from the tool's documentation — parser quirks, input format restrictions, performance cliffs, etc.

**`input_format`** is a tuple of strings documenting the tool's native input format. Use it to describe how to construct valid inputs: the file format (FASTA, YAML, JSON, etc.), entity specification syntax, special cases (ligands, modified residues, constraints), and concrete examples. This is surfaced by `autobio info <tool>` as a separate "Input Format" section (table mode) or `"input_format"` key (JSON mode), making it easy for agents to programmatically distinguish input construction guidance from operational notes. Use an empty tuple `()` for tools with trivial inputs (e.g., a single PDB file path). Good input_format entries:

- **Show the native syntax**: include actual format examples (FASTA headers, YAML structure, JSON hierarchy).
- **Cover all entity types**: proteins, DNA, RNA, ligands, and any special cases (glycans, non-canonical residues).
- **Include a complete example**: a full, valid input that an agent could adapt for a real prediction.
- **Document the raw override**: mention the `extra` key for bypassing auto-generation (e.g., `extra['chai_fasta']`).

### 5.2 Runner Registration

Register the runner class in `tools/__init__.py` so the `get_runner()` factory can instantiate it:

```python
from autobio.tools.mpnn import MPNNRunner

TOOL_RUNNERS: dict[str, type[ToolRunner]] = {
    "proteinmpnn": MPNNRunner,
    "ligandmpnn": MPNNRunner,  # same class, different tool_name
}
```

---

## 6. Testing the Runner

Each runner should have tests at two tiers.

### Tier 1 — Unit Tests (`tests/unit/`)

Test `prepare_workspace` and `parse_output` in isolation, without Docker. Mock the container manager and GPU manager.

```python
def test_mpnn_prepare_workspace(tmp_path):
    """Verify config.json and input files are written correctly."""
    workspace = Workspace.create(tmp_path / "ws")
    runner = MPNNRunner("proteinmpnn", AutobioConfig.resolve())
    input_data = InverseFoldingInput(
        structure_path=sample_pdb,
        num_sequences=3,
        temperature=0.2,
    )
    runner.prepare_workspace(input_data, workspace)

    config = json.loads(workspace.config_path.read_text())
    assert config["model_type"] == "protein_mpnn"
    assert config["number_of_batches"] == 3
    assert config["temperature"] == 0.2
    assert config["structure_path"].startswith("/workspace/inputs/")
    assert (workspace.inputs_dir / sample_pdb.name).exists()


def test_mpnn_parse_output(tmp_path):
    """Verify standardized JSON is correctly deserialized."""
    workspace = Workspace.create(tmp_path / "ws")
    result_data = {
        "designed_sequences": [{
            "rank": 1,
            "sequence": {"A": "MKWVTFIS"},
            "score": None,
            "recovery": 0.56,
        }],
        "native_sequence": {"A": "MKWVTFIS"},
    }
    (workspace.std_output_dir / "result_data.json").write_text(
        json.dumps(result_data)
    )
    runner = MPNNRunner("proteinmpnn", AutobioConfig.resolve())
    output = runner.parse_output(workspace)

    assert len(output.designed_sequences) == 1
    assert output.designed_sequences[0].recovery == 0.56
    assert output.native_sequence == {"A": "MKWVTFIS"}
```

### Tier 2 — Integration Tests (`tests/integration/`)

Run the full pipeline end-to-end with Docker and GPU. Download real structures from RCSB. These tests are marked with `@pytest.mark.docker` and `@pytest.mark.gpu` so they auto-skip unless explicitly selected with `-m docker`:

```python
pytestmark = [pytest.mark.docker, pytest.mark.gpu]

@pytest.fixture(scope="session")
def rcsb_pdb(tmp_path_factory):
    """Download a small PDB from RCSB."""
    cache = tmp_path_factory.mktemp("pdb_cache")
    pdb_path = cache / "1crn.pdb"
    urllib.request.urlretrieve(
        "https://files.rcsb.org/download/1CRN.pdb", pdb_path,
    )
    return pdb_path

def test_proteinmpnn_full_pipeline(rcsb_pdb, autobio_config, tmp_path):
    """Full pipeline: workspace → container → parsed output."""
    input_data = InverseFoldingInput(
        structure_path=rcsb_pdb,
        num_sequences=2,
        temperature=0.1,
    )
    runner = get_runner("proteinmpnn", autobio_config)
    output = runner.run(input_data, gpu="auto", output_dir=tmp_path / "ws")

    assert isinstance(output, InverseFoldingOutput)
    assert len(output.designed_sequences) == 2
    assert output.metadata.tool_name == "proteinmpnn"
```

Integration tests should exercise:
- Single-chain and multi-chain structures (different chain counts and IDs).
- Tool-specific features like `chains_to_design` or ligand-containing structures.
- Native sequence extraction.
- Use real PDB structures from RCSB, not synthetic data.

---

## 7. Checklist for New Tool Runners

- [ ] Module created at `src/autobio/tools/<tool_name>.py`
- [ ] Runner class inherits from `ToolRunner`
- [ ] `prepare_workspace` writes valid `config.json` with container-internal paths
- [ ] `prepare_workspace` references baked-in checkpoint paths (not host paths)
- [ ] `prepare_workspace` maps schema field names to tool-specific config keys
- [ ] `prepare_workspace` handles mutually exclusive parameters if applicable
- [ ] `prepare_workspace` merges `input_data.extra` (flat merge or namespaced, documented in docstring)
- [ ] `prepare_workspace` copies required input files to `workspace.inputs_dir`
- [ ] `parse_output` reads from `outputs/standardized/result_data.json`
- [ ] `parse_output` returns fully populated Pydantic output model with placeholder metadata
- [ ] `ToolEntry` registered in `TOOL_REGISTRY` at module level with correct schemas and metadata
- [ ] `ToolEntry.notes` populated with known quirks, limitations, or usage caveats (empty tuple if none)
- [ ] Runner class registered in `tools/__init__.py` `TOOL_RUNNERS` dict
- [ ] Unit tests cover `prepare_workspace` config generation and input file copying
- [ ] Unit tests cover `parse_output` deserialization with mock standardized data
- [ ] Integration tests exercise full pipeline with real PDB structures (Docker + GPU)
- [ ] Corresponding container exists (see `containers/CONTAINER_SPEC.md`)
- [ ] Module docstring documents `extra` dict conventions and any shared-runner patterns
