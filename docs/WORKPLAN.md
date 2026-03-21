# Autobio Foundation & Harness — Implementation Plan

> **Status:** Approved
> **Scope:** Architectural foundation and harness only — no specific tool or container implementations

## Context

Autobio is a Python package providing a unified, agentic-friendly interface to computational biology tools (structure prediction, language models, inverse folding, scoring, docking). Each tool runs in its own Docker container, fully isolating dependencies. The host package has a minimal footprint — no PyTorch, no CUDA.

Four design documents (`docs/DESIGN.md`, `CONTAINER_SPEC.md`, `SCHEMA_SPEC.md`, `TOOL_SPEC.md`) define the architecture in detail. The current repo is a bare scaffold: empty `src/autobio/__init__.py`, a placeholder test, and no runtime dependencies.

**Goal of this phase:** Build the complete architectural foundation and harness — all core modules, schemas, CLI, tool runner ABC, utilities, container protocol infrastructure, and tests. No specific tool or container implementations yet.

**Key decisions:**
- Docker SDK: `python-on-whales`
- Include all category schemas (structure_prediction, embedding, inverse_folding, scoring)
- Include `containers/base-entrypoint.sh`
- Include all utility modules (logging, sequences, structures)

---

## Phase 0: Project Configuration

Update `pyproject.toml`:

- **Runtime dependencies:** `pydantic>=2.0,<3`, `typer>=0.9,<1`, `python-on-whales>=0.60,<1`, `pynvml>=11.5,<12`, `rich>=13.0,<14`
- **CLI entrypoint:** `[project.scripts] autobio = "autobio.cli.main:app"`
- **Test markers:** add `markers` to `[tool.pytest.ini_options]`: `slow`, `docker`, `gpu`
- **mypy overrides:** `[[tool.mypy.overrides]]` for `python_on_whales` and `pynvml` with `ignore_missing_imports = true`
- **Update description** to reflect the actual project purpose

Reinstall: `pip install -e ".[dev]"`

**Verify:** `python -c "import pydantic, typer, python_on_whales, rich; print('ok')"`

---

## Phase 1: Core Data Types & Utilities (No Internal Dependencies)

All modules here depend only on stdlib or third-party packages. No cross-dependencies within autobio.

### Files to create

| File | Contents |
|---|---|
| `src/autobio/core/__init__.py` | Package init, re-exports key classes |
| `src/autobio/core/config.py` | `AutobioConfig` dataclass with `resolve()` classmethod. 3 env vars: `AUTOBIO_DOCKER_HOST`, `AUTOBIO_IMAGE_PREFIX`, `AUTOBIO_LOG_LEVEL`. Precedence: runtime kwargs > env > defaults. Use `dataclasses.dataclass` (internal data, not serialization boundary). |
| `src/autobio/core/result.py` | Exception hierarchy: `AutobioError` base, `ContainerNotFoundError`, `GPUNotAvailableError`, `ToolExecutionError` (stores phase/exit_code/error_message/logs/wall_time), `ToolTimeoutError`. `RunResult` Pydantic model for deserializing `result.json` (status, exit_code, phase, error_type, error_message, wall_time_seconds, gpu_ids, completed, total, outputs). `ContainerResult` dataclass (exit_code, stdout_log path, stderr_log path). |
| `src/autobio/schemas/__init__.py` | Re-exports all public schema types with `__all__` |
| `src/autobio/schemas/base.py` | `RunMetadata(BaseModel)`, `BaseInput(BaseModel)` with `extra: dict[str, Any]`, `BaseOutput(BaseModel)` with `metadata` and `raw_output_path`. Exactly as specified in SCHEMA_SPEC.md §3. |
| `src/autobio/core/workspace.py` | `Workspace` class. Path properties: `config_path`, `inputs_dir`, `raw_output_dir`, `std_output_dir`, `logs_dir`, `result_path`. Methods: `create()` classmethod (temp or user dir), `write_config()`, `write_input_file()` (str or bytes), `read_result()` → `RunResult`, `cleanup()`. Tracks `_is_temp` for cleanup decisions. |
| `src/autobio/core/gpu.py` | `GPUManager`. `_discover_gpus()` via pynvml (graceful degradation if no driver/lib). `allocate(count, device_ids)` with `threading.Lock`. `release()`. Properties: `available_gpus`, `has_gpus`. Raises `GPUNotAvailableError` if requested GPUs unavailable. Non-blocking (no wait/retry). |
| `src/autobio/utils/__init__.py` | Package init |
| `src/autobio/utils/logging.py` | `setup_logging(level)`, `JsonLineFormatter`, `get_logger(name)` → namespaced logger. Stdlib `logging` only. |
| `src/autobio/utils/sequences.py` | `parse_fasta(path) → dict[str, str]`, `write_fasta(sequences, path)`, `validate_protein_sequence(seq) → bool`, `validate_nucleotide_sequence(seq, molecule)`. Alphabet constants: `AMINO_ACIDS`, `DNA_BASES`, `RNA_BASES`. Pure Python, no BioPython. |
| `src/autobio/utils/structures.py` | `read_pdb_sequences(path) → dict[str, str]`, `read_mmcif_sequences(path) → dict[str, str]`, `detect_structure_format(path) → str`, `count_residues(path) → int`. `THREE_TO_ONE` mapping. Pure Python parsing. |

### Tests

| File | Covers |
|---|---|
| `tests/conftest.py` | Shared fixtures: `tmp_workspace`, `sample_config`, `monkeypatch_no_gpu` |
| `tests/unit/__init__.py` | Package init |
| `tests/unit/test_config.py` | Defaults, env var overrides, runtime override precedence |
| `tests/unit/test_result.py` | Exception hierarchy, `RunResult` deserialization (success + failure), `ContainerResult` |
| `tests/unit/test_schemas_base.py` | `RunMetadata`/`BaseInput`/`BaseOutput` serialization round-trips, required field enforcement, `extra` dict |
| `tests/unit/test_workspace.py` | Directory creation, `write_config` round-trip, `write_input_file` str/bytes, `read_result`, missing result.json, `cleanup` |
| `tests/unit/test_gpu.py` | Mocked pynvml (simulate 2 GPUs), allocate/release, double-allocate error, graceful no-GPU degradation, properties |
| `tests/unit/test_logging.py` | Logger configuration, JSON formatter output validity, namespaced loggers |
| `tests/unit/test_sequences.py` | FASTA parse/write round-trip, multi-line sequences, validation (valid/invalid protein, DNA, RNA) |
| `tests/unit/test_structures.py` | PDB sequence extraction with small fragment, format detection, `THREE_TO_ONE` completeness |

**Verify:**
```bash
pytest tests/unit/ -v
ruff check src/autobio/core/ src/autobio/schemas/base.py src/autobio/utils/
mypy src/autobio/core/ src/autobio/schemas/base.py src/autobio/utils/
```

---

## Phase 2: Category Schemas (Depend on schemas/base)

All four modules depend only on `schemas/base.py`. Can be implemented in parallel.

### Files to create

| File | Key types |
|---|---|
| `src/autobio/schemas/structure_prediction.py` | `StructurePredictionInput(BaseInput)`: sequences dict, num_models, templates. `PredictedStructure`: rank, structure_path, plddt metrics, ptm, iptm, chain_mapping. `ConfidenceMetrics`: best scores. `StructurePredictionOutput(BaseOutput)`: structures list, confidence. |
| `src/autobio/schemas/embedding.py` | `EmbeddingInput(BaseInput)`: sequences dict, layer, pooling. `SequenceEmbedding`: sequence_id, embedding_path, dimension, layer, pooling. `EmbeddingOutput(BaseOutput)`: embeddings list, model_name, embedding_dimension. |
| `src/autobio/schemas/inverse_folding.py` | `InverseFoldingInput(BaseInput)`: structure_path, chains_to_design, num_sequences, temperature, fixed_positions. `DesignedSequence`: rank, sequence dict, score, recovery. `InverseFoldingOutput(BaseOutput)`: designed_sequences, native_sequence. |
| `src/autobio/schemas/scoring.py` | `ScoringInput(BaseInput)`: structure_path, sequences (optional). `ScoredStructure`: total_score, per_residue_scores, score_breakdown dict, units. `ScoringOutput(BaseOutput)`: scores. |

Update `src/autobio/schemas/__init__.py` to re-export all new types.

### Tests

| File | Covers |
|---|---|
| `tests/unit/test_schemas.py` | All four schemas: serialization round-trips, required field enforcement, optional defaults as None, extra dict passthrough, type validation errors, inheritance from BaseInput/BaseOutput. Use `@pytest.mark.parametrize` for shared patterns. |

**Verify:**
```bash
pytest tests/unit/test_schemas.py tests/unit/test_schemas_base.py -v
ruff check src/autobio/schemas/
mypy src/autobio/schemas/
```

---

## Phase 3: Registry & Container Manager (Depend on Phases 1-2)

### Files to create

| File | Contents |
|---|---|
| `src/autobio/core/registry.py` | `ToolCategory(StrEnum)`: `STRUCTURE_PREDICTION`, `EMBEDDING`, `INVERSE_FOLDING`, `SCORING`. `ToolEntry` dataclass: image_tag, category, requires_gpu, gpu_count, `input_schema: type[BaseInput]`, `output_schema: type[BaseOutput]`, default_timeout, supports_batch, description, version. `TOOL_REGISTRY: dict[str, ToolEntry] = {}` (empty). Helpers: `get_tool(name)` (raises KeyError with available tools), `list_tools(category=None)`. |
| `src/autobio/core/container.py` | `ContainerManager(config: AutobioConfig)`. Wraps `python_on_whales.DockerClient`. Methods: `ensure_image(uri)` (pull if missing, raise `ContainerNotFoundError` on failure), `run(image_uri, workspace, gpu_ids, timeout, memory_limit) → ContainerResult` (bind mount workspace at `/workspace`, GPU device spec, timeout handling via `ToolTimeoutError`, always remove container after), `list_images(prefix) → list[ImageInfo]`, `pull_image(uri)`. `ImageInfo` dataclass: uri, tag, size, created. |

### Tests

| File | Covers |
|---|---|
| `tests/unit/test_registry.py` | `ToolCategory` values, `ToolEntry` construction with mock schemas, `get_tool` success/KeyError, `list_tools` filtering, empty registry behavior |
| `tests/unit/test_container.py` | Mock `python_on_whales.DockerClient` throughout. `ensure_image` pull logic, `run` argument construction (bind mounts, GPU spec string), `ContainerNotFoundError` on pull failure, `ToolTimeoutError`, `list_images` parsing |

**Verify:**
```bash
pytest tests/unit/test_registry.py tests/unit/test_container.py -v
ruff check src/autobio/core/
mypy src/autobio/core/
```

---

## Phase 4: Tool Runner ABC (Depends on Phases 1-3)

### Files to create

| File | Contents |
|---|---|
| `src/autobio/tools/__init__.py` | `TOOL_RUNNERS: dict[str, type[ToolRunner]] = {}`. `get_runner(tool_name, config) → ToolRunner` (lookup + instantiate, KeyError if missing). |
| `src/autobio/tools/base.py` | `ToolRunner(ABC)`. Init: stores tool_name, looks up registry entry, creates ContainerManager and GPUManager. Abstract: `prepare_workspace(input_data, workspace)`, `parse_output(workspace) → BaseOutput`. Concrete `run()`: workspace creation → prepare → GPU resolve → ensure image → container run → read result → check status → parse output → return. Finally block: release GPUs, cleanup temp workspace. Helpers: `_resolve_gpu(gpu)` (handles "auto"/"none"/list/comma-string), `_build_metadata(workspace, wall_time, gpu_ids) → RunMetadata`, `_read_logs(workspace) → str`. |

### Tests

| File | Covers |
|---|---|
| `tests/unit/test_tool_runner.py` | Create `MockRunner(ToolRunner)` with trivial implementations. Fixture registers mock `ToolEntry`. Test init registry lookup + KeyError. Test `_resolve_gpu` modes (auto/none/list/string) with mocked GPUManager. Test `_build_metadata`. Test `run()` lifecycle with mocked ContainerManager + Workspace (verify call sequence). Test `ToolExecutionError` on failure result. Test GPU release in finally block. Test workspace cleanup vs. preservation. Test `get_runner()`. |

**Verify:**
```bash
pytest tests/unit/test_tool_runner.py -v
ruff check src/autobio/tools/
mypy src/autobio/tools/
```

---

## Phase 5: CLI (Depends on Everything)

### Files to create

| File | Contents |
|---|---|
| `src/autobio/cli/__init__.py` | Package docstring |
| `src/autobio/cli/formatters.py` | `OutputFormat(StrEnum)`: JSON, TABLE. Format functions: `format_tool_list`, `format_tool_info` (includes input schema JSON schema for agents), `format_run_result`, `format_workspace_result`, `format_image_list`. `print_error`. Use `rich.table.Table` for TABLE, `json.dumps` for JSON. |
| `src/autobio/cli/main.py` | `app = typer.Typer(...)`. Register all subcommands. Entry point for `[project.scripts]`. |
| `src/autobio/cli/list.py` | `autobio list [--category] [--format]`. Calls registry `list_tools()`, formats output. |
| `src/autobio/cli/info.py` | `autobio info <tool> [--format]`. Looks up tool, formats details. JSON includes input schema via `model_json_schema()`. Exits 1 on unknown tool. |
| `src/autobio/cli/run.py` | `autobio run <tool> --config --gpu --timeout --output-dir [--format]`. Loads config JSON, resolves runner, validates input against schema, calls `runner.run()`, formats output. Catches `AutobioError` subclasses. |
| `src/autobio/cli/result.py` | `autobio result <workspace-dir> [--format]`. Creates Workspace from path, reads result.json, formats. Handles missing result.json. |
| `src/autobio/cli/images.py` | `autobio pull <tool> [--all]` and `autobio images [--format]`. Uses ContainerManager for image operations. Progress via rich. |

### Tests

| File | Covers |
|---|---|
| `tests/unit/test_cli.py` | Use Typer's `CliRunner`. Test `list` empty + populated registry (JSON and table). Test `list --category` filter. Test `info` success + unknown tool exit 1. Test `result` with valid workspace + missing result.json. Test `run` arg parsing (mock execution). Test `pull`/`images` with mocked ContainerManager. Test `--format json` produces valid JSON. |
| `tests/unit/test_formatters.py` | Each format function with sample data. JSON validity. Table non-empty. `format_tool_info` JSON includes schema. |

**Verify:**
```bash
pytest tests/unit/test_cli.py tests/unit/test_formatters.py -v
ruff check src/autobio/cli/
mypy src/autobio/cli/
autobio --help
autobio list --format json
```

---

## Phase 6: Container Protocol Infrastructure

### Files to create

| File | Contents |
|---|---|
| `containers/base-entrypoint.sh` | Verbatim from CONTAINER_SPEC.md §3. Three-phase protocol: validate_config.sh → run.sh → standardize.sh. `write_result()` function producing result.json. Log capture via tee. Phase tracking. Requires jq + bc in container. |

### Tests

| File | Covers |
|---|---|
| `tests/unit/test_entrypoint.py` | Use `subprocess.run` with bash. Create temp mock hook scripts. Test successful 3-phase run. Test validate failure exits 1 with correct result.json. Test run.sh failure propagates exit code. Test result.json is valid JSON. Skip if jq/bc not available. |

**Verify:**
```bash
bash -n containers/base-entrypoint.sh
pytest tests/unit/test_entrypoint.py -v
```

---

## Phase 7: Wiring & Cleanup

- Update `src/autobio/__init__.py`: export public API (`AutobioConfig`, exceptions, base schemas, `__version__`)
- Update `src/autobio/core/__init__.py`: re-export core classes
- Remove `tests/test_placeholder.py`
- Update `CLAUDE.md` architecture section

**Verify (full suite):**
```bash
pip install -e ".[dev]"
pytest tests/ -v --tb=short
ruff check src/ tests/
ruff format --check src/ tests/
mypy src/
python -c "
import autobio
from autobio.core.config import AutobioConfig
from autobio.core.registry import TOOL_REGISTRY, ToolEntry, ToolCategory
from autobio.core.workspace import Workspace
from autobio.core.gpu import GPUManager
from autobio.core.container import ContainerManager
from autobio.core.result import AutobioError, RunResult
from autobio.schemas import (
    BaseInput, BaseOutput,
    StructurePredictionInput, StructurePredictionOutput,
    EmbeddingInput, EmbeddingOutput,
    InverseFoldingInput, InverseFoldingOutput,
    ScoringInput, ScoringOutput,
)
from autobio.tools.base import ToolRunner
from autobio.tools import TOOL_RUNNERS, get_runner
from autobio.utils.sequences import parse_fasta, validate_protein_sequence
from autobio.utils.structures import read_pdb_sequences
from autobio.utils.logging import setup_logging, get_logger
print('All imports successful')
"
autobio --help
autobio list --format json
```

---

## File Manifest

**Modified:** `pyproject.toml`, `src/autobio/__init__.py`, `CLAUDE.md`
**Deleted:** `tests/test_placeholder.py`

**New source (28 files):**
```
src/autobio/core/__init__.py
src/autobio/core/config.py
src/autobio/core/result.py
src/autobio/core/workspace.py
src/autobio/core/gpu.py
src/autobio/core/registry.py
src/autobio/core/container.py
src/autobio/schemas/__init__.py
src/autobio/schemas/base.py
src/autobio/schemas/structure_prediction.py
src/autobio/schemas/embedding.py
src/autobio/schemas/inverse_folding.py
src/autobio/schemas/scoring.py
src/autobio/tools/__init__.py
src/autobio/tools/base.py
src/autobio/cli/__init__.py
src/autobio/cli/main.py
src/autobio/cli/formatters.py
src/autobio/cli/list.py
src/autobio/cli/info.py
src/autobio/cli/run.py
src/autobio/cli/result.py
src/autobio/cli/images.py
src/autobio/utils/__init__.py
src/autobio/utils/logging.py
src/autobio/utils/sequences.py
src/autobio/utils/structures.py
containers/base-entrypoint.sh
```

**New tests (17 files):**
```
tests/conftest.py
tests/unit/__init__.py
tests/unit/test_config.py
tests/unit/test_result.py
tests/unit/test_schemas_base.py
tests/unit/test_schemas.py
tests/unit/test_workspace.py
tests/unit/test_gpu.py
tests/unit/test_registry.py
tests/unit/test_container.py
tests/unit/test_tool_runner.py
tests/unit/test_cli.py
tests/unit/test_formatters.py
tests/unit/test_logging.py
tests/unit/test_sequences.py
tests/unit/test_structures.py
tests/unit/test_entrypoint.py
```

---

## Potential Challenges

1. **mypy strict + python-on-whales:** May lack complete type stubs. Mitigation: `[[tool.mypy.overrides]]` for that package with `ignore_missing_imports = true`.
2. **pynvml on non-GPU machines:** `GPUManager._discover_gpus()` must catch both ImportError and pynvml init failures, returning `[]`. All GPU tests mock pynvml.
3. **Testing base-entrypoint.sh without Docker:** Requires `bash`, `jq`, `bc` on the test runner. Skip with `@pytest.mark.skipif` if unavailable.
4. **CLI `list` name shadowing:** Name the Typer command function `list_tools` to avoid shadowing the Python builtin within the module.
