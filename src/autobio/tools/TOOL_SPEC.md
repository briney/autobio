# Tool Runner Specification

This document defines how to create a new host-side tool runner. A tool runner is a Python class that translates between autobio's standardized schemas and a specific tool's expected inputs and outputs. It is the bridge between the category-level interface that agents interact with and the tool-specific container.

Autobio organizes tools with a **catalog** of `Tool`/`Mode` objects (`src/autobio/core/catalog.py`), not a flat per-tool-name registry. A `Tool` is one coherent model or engine (e.g. `rosetta`, `esmfold`); a `Mode` is a named operation that Tool supports (e.g. `rosetta`'s `score`, `relax`, `minimize`, `flexddg`). Every Tool declares a `default_mode` so single-purpose tools (most of them) can be invoked without ever mentioning modes.

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
5. Dispatching on `self.current_mode.name` when the Tool supports more than one mode.
6. Registering the tool in the catalog with its `Mode`(s) and usage notes.

The runner is NOT responsible for:
- Launching containers, managing GPUs, or handling timeouts (the base class does this).
- Understanding the tool's raw output format (the container's `standardize.sh` handles this).
- Validating the standardized output against the schema (Pydantic does this automatically during deserialization).
- Resolving which mode is active (the base class does this in `run()`, before `prepare_workspace` is called).

---

## 2. File Location and Naming

Tool runners live in `src/autobio/tools/`. Each tool gets its own module. A single module can serve multiple tools if they share a runner class (e.g., `mpnn.py` serves both `proteinmpnn` and `ligandmpnn`):

```
src/autobio/tools/
├── __init__.py          # TOOL_RUNNERS dict + get_runner() factory
├── base.py              # ToolRunner ABC — do not modify per-tool
├── TOOL_SPEC.md         # this document
├── esmfold.py           # ESMFoldRunner — single-mode tool, registers ESMFOLD_TOOL
├── freesasa.py          # FreeSASARunner — multi-mode tool, registers FREESASA_TOOL
├── mpnn.py              # MPNNRunner — serves proteinmpnn + ligandmpnn
└── <new_tool>.py
```

Module names use `snake_case` matching the tool's canonical name in the catalog.

---

## 3. The ToolRunner Base Class

All runners inherit from `ToolRunner` defined in `base.py`. The base class provides the full execution lifecycle — runners only implement two abstract methods.

```python
class ToolRunner(ABC):
    """
    Abstract base for all tool runners. Subclasses implement
    prepare_workspace() and parse_output() only.
    """

    def __init__(self, tool_name: str, config: AutobioConfig) -> None:
        if tool_name not in CATALOG:
            available = ", ".join(sorted(CATALOG)) or "(none)"
            raise KeyError(f"Unknown tool {tool_name!r}. Available tools: {available}")
        self.tool: Tool = get_tool(tool_name)
        self.tool_name = tool_name
        self.config = config
        self.current_mode: Mode | None = None
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
        mode: str | None = None,
    ) -> BaseOutput:
        """
        Full execution lifecycle. Do NOT override this method.

        Steps:
            1. Resolve the Mode (mode name, else self.tool.default_mode)
            2. Create workspace
            3. prepare_workspace (subclass hook) — self.current_mode is set
            4. Resolve GPU allocation
            5. Ensure container image is available
            6. Run container
            7. Read result.json and check status
            8. parse_output (subclass hook)
            9. Attach metadata (overwrites placeholder from parse_output)
            10. Return output
        """
        ...
```

**`self.current_mode` is set by `run()` before `prepare_workspace`/`parse_output` are called** — it is resolved from the `mode` argument (or `self.tool.default_mode` when `mode` is `None`) and is guaranteed non-`None` for the rest of the call. Runners with more than one `Mode` dispatch on `self.current_mode.name`.

**Important:** The base class calls `parse_output` and then overwrites `output.metadata` with the real metadata (wall time, GPU IDs, image URI, timestamp, and the active mode name). Your `parse_output` should return a placeholder metadata via `self._build_metadata(workspace, 0.0, [], "")` — it will be replaced.

---

## 4. Implementing a New Runner

### 4.1 Single-Mode Tool — `esmfold.py`

Most tools have exactly one `Mode`. Below is the real ESMFold runner (`src/autobio/tools/esmfold.py`), trimmed of docstrings/validation detail, showing the shape every single-mode tool follows:

```python
"""ESMFold structure prediction tool runner."""

from __future__ import annotations

import json
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

_HF_CACHE = "/app/esmfold/hf_cache"
_MODEL_NAME = "facebook/esmfold_v1"


class ESMFoldRunner(ToolRunner):
    """Runner for ESMFold single-sequence structure prediction."""

    def prepare_workspace(self, input_data: BaseInput, workspace: Workspace) -> None:
        assert isinstance(input_data, ESMFoldInput)
        self._validate_inputs(input_data)

        write_fasta(input_data.sequences, workspace.inputs_dir / "sequences.fasta")

        config: dict[str, object] = {
            "model_name": _MODEL_NAME,
            "input_fasta": "/workspace/inputs/sequences.fasta",
            "output_dir": "/workspace/outputs/raw",
            "hf_cache": _HF_CACHE,
        }
        self._apply_extra(config, input_data)
        workspace.write_config(config)

    def parse_output(self, workspace: Workspace) -> StructurePredictionOutput:
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
        # ... additional checks (single-chain only, no templates, num_models == 1) ...


# ---------------------------------------------------------------------------
# Catalog registration — populated when this module is imported
# ---------------------------------------------------------------------------

_ESMFOLD_NOTES = (
    "ESMFold is single-chain only. It cannot predict multimer structures "
    "or protein-ligand complexes. For multimer prediction, use boltz2, "
    "chai1, or openfold3.",
    # ... additional notes ...
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

register(ESMFOLD_TOOL)
```

A single-mode `Tool` still declares `modes={...}` (one entry) and `default_mode` — there is no separate "no modes" code path. `autobio run esmfold --config ...` and `autobio run esmfold --config ... --mode predict` are equivalent.

### 4.2 Multi-Mode Tool — `freesasa.py`

When one engine exposes more than one operation, register a single `Tool` with multiple `Mode` entries, each with its own `input_schema`/`output_schema`/`default_timeout` (and optionally its own `image_tag`/`category` override). The runner dispatches on `self.current_mode.name`. This is the real FreeSASA runner (`src/autobio/tools/freesasa.py`), showing the `sasa` (default) / `bsa` split:

```python
"""FreeSASA tool — SASA and buried surface area calculation."""

from __future__ import annotations

import json
import shutil
from typing import TYPE_CHECKING, Any

from autobio.core.catalog import Mode, Tool, register
from autobio.core.registry import ToolCategory
from autobio.core.result import AutobioError
from autobio.schemas.base import BaseInput  # noqa: TC001 - needed at runtime for isinstance
from autobio.schemas.scoring import (
    FreeSASABaseInput,
    FreeSASABSAInput,
    FreeSASASASAInput,
    ScoredStructure,
    ScoringOutput,
)
from autobio.tools.base import ToolRunner

if TYPE_CHECKING:
    from autobio.core.workspace import Workspace


class FreeSASARunner(ToolRunner):
    """Runner for FreeSASA SASA and BSA calculations."""

    def prepare_workspace(self, input_data: BaseInput, workspace: Workspace) -> None:
        assert isinstance(input_data, FreeSASABaseInput)
        assert self.current_mode is not None
        is_bsa = self.current_mode.name == "bsa"  # dispatch on the active Mode

        self._validate_inputs(input_data, is_bsa=is_bsa)

        src_path = input_data.structure_path
        dest_name = src_path.name
        shutil.copy2(src_path, workspace.inputs_dir / dest_name)

        config: dict[str, Any] = {
            "mode": "bsa" if is_bsa else "sasa",
            "structure_path": f"/workspace/inputs/{dest_name}",
            "algorithm": input_data.algorithm,
            "probe_radius": input_data.probe_radius,
            "per_residue": input_data.per_residue,
            "output_dir": "/workspace/outputs/raw",
        }
        if is_bsa:
            assert isinstance(input_data, FreeSASABSAInput)
            config["partner1"] = input_data.partner1
            config["partner2"] = input_data.partner2

        self._apply_extra(config, input_data)
        workspace.write_config(config)

    def parse_output(self, workspace: Workspace) -> ScoringOutput:
        result_data_path = workspace.std_output_dir / "result_data.json"
        data = json.loads(result_data_path.read_text())

        scores = [
            ScoredStructure(
                total_score=s["total_score"],
                per_residue_scores=s.get("per_residue_scores"),
                score_breakdown=s.get("score_breakdown"),
                units=s.get("units"),
            )
            for s in data["scores"]
        ]

        # Placeholder metadata — overwritten by base class run()
        return ScoringOutput(
            scores=scores,
            metadata=self._build_metadata(workspace, 0.0, [], ""),
            raw_output_path=workspace.raw_output_dir,
        )

    @staticmethod
    def _validate_inputs(input_data: FreeSASABaseInput, *, is_bsa: bool) -> None:
        """Host-side validation — catch errors before container launch."""
        if not input_data.structure_path.exists():
            raise AutobioError(f"Input structure file does not exist: {input_data.structure_path}")
        # ... additional checks (BSA partner chain validation, etc.) ...


# ---------------------------------------------------------------------------
# Catalog registration — populated when this module is imported
# ---------------------------------------------------------------------------

FREESASA_TOOL = Tool(
    name="freesasa",
    display_name="FreeSASA",
    category=ToolCategory.SCORING,
    description=(
        "Solvent-accessible surface area (SASA) and buried surface area (BSA) via "
        "FreeSASA. CPU-only, no GPU required."
    ),
    version="2.2.1",
    image_tag="freesasa:2.2.1",
    requires_gpu=False,
    gpu_count=0,
    default_mode="sasa",
    modes={
        "sasa": Mode(
            name="sasa",
            display_name="SASA",
            description="Solvent-accessible surface area of a structure.",
            input_schema=FreeSASASASAInput,
            output_schema=ScoringOutput,
            default_timeout=300,
            notes=(...,),
        ),
        "bsa": Mode(
            name="bsa",
            display_name="BSA",
            description="Buried surface area at a protein-protein interface.",
            input_schema=FreeSASABSAInput,
            output_schema=ScoringOutput,
            default_timeout=300,
            notes=(...,),
        ),
    },
    keywords=("sasa", "bsa", "surface area", "interface", "freesasa"),
)

register(FREESASA_TOOL)
```

Each `Mode` on a multi-mode `Tool` can have a **distinct input schema** (`FreeSASASASAInput` vs. `FreeSASABSAInput` above) — the CLI (`autobio run <tool> --mode <mode>`) and `get_runner().run(..., mode=...)` both select the mode's `input_schema` to validate the config against before calling `prepare_workspace`. Modes commonly share an output schema (both above use `ScoringOutput`), but that isn't required.

Other real multi-mode tools use different dispatch styles worth knowing about:
- **Dict-driven dispatch** (`rosetta.py`, `complexa.py`, `openmm.py`, `evoef2.py`) — a module-level `dict` keyed by mode name holds per-mode config (binary/protocol paths, checkpoint names, variant flags, etc.), looked up once via `<CONFIG>[self.current_mode.name]` (`_MODE_CONFIG` in `rosetta.py`/`complexa.py`, `_VARIANT_CONFIG` in `openmm.py`, `_MODE_COMMAND` in `evoef2.py`), avoiding a long `if`/`elif` chain; a mode needing extra per-mode logic can still add a small `if self.current_mode.name == ...` block (as `evoef2.py` does for `binding`/`build_mutant`).
- **Binary `if`/`else` dispatch** (`esm_if1.py`, `antifold.py`) — for tools with exactly two modes (`design`/`score`) where the two code paths differ substantially.

Pick whichever shape best fits the number of modes and how much they diverge; `is_bsa`-style boolean flags (as above) work well for two modes with mostly-shared logic.

### 4.3 Key Implementation Details

**`prepare_workspace` must:**
- Write a `config.json` via `workspace.write_config(config)`.
- Copy input files to `workspace.inputs_dir` (use `shutil.copy2` for files, `workspace.write_input_file()` for generated content).
- Use **container-internal paths** in `config.json` (e.g., `/workspace/inputs/structure.pdb` not the host path). The runner knows these paths because the workspace is always mounted at `/workspace`.
- Reference **baked-in checkpoint paths** in `config.json` (e.g., `/app/checkpoints/model.pt`). The runner knows these because they are fixed at image build time (see `containers/CONTAINER_SPEC.md` §5.2).
- Map schema field names to tool-specific config keys. Document non-obvious mappings with comments.
- Handle mutually exclusive parameters (e.g., `designed_chains` vs `fixed_residues`).
- Dispatch on `self.current_mode.name` when the Tool has more than one `Mode` (see §4.2).
- Merge `input_data.extra` into the config via `self._apply_extra(config, input_data)` (see below).
- NOT launch containers, manage GPUs, or handle errors — the base class does this.

**`parse_output` must:**
- Read `outputs/standardized/result_data.json` from the workspace.
- Return a fully populated Pydantic output model with a placeholder metadata (`self._build_metadata(workspace, 0.0, [], "")`).
- Use `self._resolve_container_path(path_str, workspace)` (defined once on `ToolRunner`) to map any container-internal `/workspace/...` path in the standardized output back to a host path.
- NOT read from `outputs/raw/` — the standardized output is the contract.
- NOT handle missing workspaces or container failures — the base class checks `result.json` before calling `parse_output`.

**Handling the `extra` dict:**

The `extra` dict (inherited from `BaseInput`) is the escape hatch for tool-specific parameters not promoted to typed fields on the active mode's input schema. Runners must merge it via the base class helper:

```python
self._apply_extra(config, input_data)
```

`_apply_extra` merges `input_data.extra` into `config` in place, but **fails fast** (raising `AutobioError`) if any `extra` key collides with either a typed field on `self.current_mode.input_schema` or a key already written into `config` — this catches agents/users trying to set a typed parameter through `extra` instead of as a real field. Call it as the last step before `workspace.write_config(config)`, once all runner-derived keys are already in `config`.

### 4.4 Shared Runner Pattern

A single runner class can serve multiple **Tools** (not modes of the same Tool) when they share a container image and differ only in configuration. The `tool_name` attribute distinguishes them:

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

`proteinmpnn` and `ligandmpnn` are each registered as their own `Tool` (with their own `modes={"design": Mode(...)}`, descriptions, and notes) via two separate `register(...)` calls in `mpnn.py`, but both map to `MPNNRunner` in `TOOL_RUNNERS`. This is orthogonal to §4.2's multi-*mode* dispatch (`self.current_mode.name`) — a runner can dispatch on `self.tool_name` (which Tool), `self.current_mode.name` (which Mode of that Tool), or both.

### 4.5 Batch-Aware Runners

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
    }
    self._apply_extra(config, input_data)
    workspace.write_config(config)
```

---

## 5. Registering the Tool

Registration happens at the bottom of the tool module (not in `core/catalog.py` or `core/registry.py`). Build a `Tool` with one or more `Mode` entries and call `register(TOOL)` at module level — this runs once, when the module is imported, and adds the `Tool` to the global `CATALOG` dict.

### 5.1 Catalog Registration

```python
from autobio.core.catalog import Mode, Tool, register
from autobio.core.registry import ToolCategory

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
            description="Fixed-backbone sequence design.",
            input_schema=InverseFoldingInput,
            output_schema=InverseFoldingOutput,
            default_timeout=600,
            notes=(
                "The foundry MPNN parser may resolve fewer residues than the PDB CA "
                "atom count due to internal filtering of disordered residues.",
                "PDB structures with multiple ASU copies can trigger an atomworks "
                "parser error. Use structures with unique chain IDs per sequence.",
            ),
        ),
    },
    keywords=("proteinmpnn", "inverse folding", "sequence design"),
)

register(PROTEINMPNN_TOOL)
```

**`Tool.image_tag`** is the tag portion only (e.g., `"mpnn:1.0.0"`). The host config's `image_prefix` is prepended at runtime to form the full URI (e.g., `ghcr.io/briney/autobio-mpnn:1.0.0`). Multiple Tools can share the same `image_tag` when they use the same container. A `Mode` may set its own `image_tag`, which overrides the owning Tool's for that mode only — used by engines whose modes ship as separate container images.

**`Tool.category`** is the Tool's primary category, shown by `autobio list`. A `Mode` may set its own `category` override (e.g. `antifold`'s `score` mode is categorized under `scoring` even though the Tool's primary category is `inverse-folding`) — `autobio list --category <cat>` surfaces a Tool under every category any of its modes touches.

**`Mode.notes`** is a tuple of strings surfaced by `autobio info <tool>` in both table and JSON formats, scoped to that mode. Use it to record operational guidance: tool-specific quirks, parser limitations, known edge cases, resource requirements, MSA/template options, key parameters — anything an agent or user would benefit from knowing about running and tuning that mode. Every `Mode` must include `notes`; use an empty tuple `()` if there are no caveats. Good notes are:

- **Actionable**: tell the reader what to do or avoid, not just what the problem is.
- **Specific**: reference concrete error messages, thresholds, or input characteristics.
- **Discovered empirically**: capture things that aren't obvious from the tool's documentation — parser quirks, input format restrictions, performance cliffs, etc.

**Input-construction guidance goes in `notes`** — the `Mode` dataclass has no separate `input_format` field. When a mode has a non-trivial native input format, fold that guidance into its `notes` tuple alongside the operational caveats: show the native syntax (FASTA headers, YAML/JSON structure), cover all entity types (proteins, DNA, RNA, ligands, and special cases like glycans or non-canonical residues), include a complete adaptable example, and mention any `extra` raw-override key for bypassing auto-generation (e.g., `extra['chai_fasta']`). The richer typed input schema (with `x-autobio` field hints) is the primary machine-readable description of a mode's inputs; `notes` carries the prose that a schema can't.

### 5.2 Runner Registration

Register the runner class in `tools/__init__.py` so the `get_runner()` factory can instantiate it — one entry per **Tool name** (not per mode):

```python
from autobio.tools.mpnn import MPNNRunner

TOOL_RUNNERS: dict[str, type[ToolRunner]] = {
    "proteinmpnn": MPNNRunner,
    "ligandmpnn": MPNNRunner,  # same class, different tool_name
}
```

`get_runner("proteinmpnn", config)` instantiates `MPNNRunner("proteinmpnn", config)`; the runner's `__init__` (inherited from `ToolRunner`, defined in `base.py`) looks up `"proteinmpnn"` in `CATALOG` to set `self.tool`. Which `Mode` runs is decided later, when `.run(..., mode=...)` is called.

---

## 6. Testing the Runner

Each runner should have tests at two tiers.

### Tier 1 — Unit Tests (`tests/unit/`)

Test `prepare_workspace` and `parse_output` in isolation, without Docker. Mock the container manager and GPU manager. Since these methods are normally only reached via `run()` (which sets `self.current_mode` before calling them), unit tests that call them directly must set `runner.current_mode` themselves:

```python
from unittest.mock import patch

import pytest

from autobio.core.catalog import get_tool
from autobio.core.config import AutobioConfig
from autobio.core.workspace import Workspace
from autobio.schemas.inverse_folding import InverseFoldingInput
from autobio.tools.mpnn import MPNNRunner


@pytest.fixture
def runner() -> MPNNRunner:
    with (
        patch("autobio.tools.base.ContainerManager"),
        patch("autobio.tools.base.GPUManager"),
    ):
        runner = MPNNRunner("proteinmpnn", AutobioConfig.resolve())
    runner.current_mode = get_tool("proteinmpnn").modes["design"]
    return runner


def test_mpnn_prepare_workspace(runner, tmp_path, sample_pdb):
    """Verify config.json and input files are written correctly."""
    workspace = Workspace.create(tmp_path / "ws")
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


def test_mpnn_parse_output(runner, tmp_path):
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
    output = runner.parse_output(workspace)

    assert len(output.designed_sequences) == 1
    assert output.designed_sequences[0].recovery == 0.56
    assert output.native_sequence == {"A": "MKWVTFIS"}
```

For a multi-mode tool, parametrize the fixture (or write a small `_make_runner(mode_name)` helper — see `tests/unit/test_freesasa.py`) so each mode's `prepare_workspace`/`parse_output` behavior gets its own test coverage:

```python
def _make_runner(mode_name: str) -> FreeSASARunner:
    with patch("autobio.tools.base.ContainerManager"), patch("autobio.tools.base.GPUManager"):
        runner = FreeSASARunner("freesasa", AutobioConfig.resolve())
    runner.current_mode = get_tool("freesasa").modes[mode_name]
    return runner
```

Also add a registration test confirming the Tool is on `CATALOG` with the expected modes and default, and that the runner is wired into `TOOL_RUNNERS`:

```python
def test_freesasa_registered_with_expected_modes() -> None:
    assert "freesasa" in CATALOG
    assert set(get_tool("freesasa").modes) == {"sasa", "bsa"}
    assert get_tool("freesasa").default_mode == "sasa"
    assert "freesasa" in TOOL_RUNNERS
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

For a multi-mode tool, pass `mode=...` through to `run()` to exercise each mode's full pipeline (e.g. `runner.run(input_data, mode="bsa", ...)`), and assert `output.metadata.mode` matches.

Integration tests should exercise:
- Single-chain and multi-chain structures (different chain counts and IDs).
- Tool-specific features like `chains_to_design` or ligand-containing structures.
- Every `Mode` the Tool declares, not just the default.
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
- [ ] `prepare_workspace` dispatches on `self.current_mode.name` if the Tool has more than one Mode
- [ ] `prepare_workspace` merges `input_data.extra` via `self._apply_extra(config, input_data)`
- [ ] `prepare_workspace` copies required input files to `workspace.inputs_dir`
- [ ] `parse_output` reads from `outputs/standardized/result_data.json`
- [ ] `parse_output` returns fully populated Pydantic output model with placeholder metadata
- [ ] A `Tool` is built with one `Mode` per operation and passed to `register()` at module level
- [ ] Each `Mode.notes` populated with known quirks, limitations, or usage caveats (empty tuple if none)
- [ ] `Tool.default_mode` set and present among `Tool.modes`
- [ ] Runner class registered in `tools/__init__.py` `TOOL_RUNNERS` dict (one entry per Tool name)
- [ ] Unit tests cover `prepare_workspace` config generation and input file copying, for every Mode
- [ ] Unit tests cover `parse_output` deserialization with mock standardized data, for every Mode
- [ ] Unit tests confirm the Tool is registered in `CATALOG` with the expected modes/default and runner in `TOOL_RUNNERS`
- [ ] Integration tests exercise full pipeline with real PDB structures (Docker + GPU), for every Mode
- [ ] Corresponding container exists (see `containers/CONTAINER_SPEC.md`)
- [ ] Module docstring documents `extra` dict conventions and any shared-runner (§4.4) or multi-mode (§4.2) dispatch patterns
