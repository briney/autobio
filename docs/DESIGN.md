# Autobio — Technical Design Document

> **Status:** Draft v0.1
> **Working name:** `autobio` (placeholder — final package name TBD)
> **Python:** ≥ 3.10
> **CLI framework:** Typer

---

## 1. Overview

Autobio is a Python package that provides a unified, agentic-friendly interface to a broad range of computational biology tools — protein and antibody language models, structure prediction, inverse folding, energy scoring, docking, and more. Each tool runs inside its own Docker container, fully isolating tool-specific dependencies (CUDA versions, PyTorch builds, legacy Python runtimes, etc.) from the host package and from each other.

The package is designed primarily for autonomous agentic experimentation. An AI agent (or a human user) interacts with autobio through a CLI that accepts structured inputs, dispatches work to containers, and returns structured outputs. The host package itself has a minimal dependency footprint — no PyTorch, no CUDA, no tool-specific libraries.

### 1.1 Design Principles

1. **Zero-config setup.** Install the package, have Docker + NVIDIA Container Toolkit, run tools. No config files, no environment provisioning.
2. **Container isolation.** Every tool runs in its own Docker container. Tool dependencies never leak into the host or into other tools.
3. **Category-level interface standardization.** Tools of the same type (e.g., all structure predictors) share a common input/output schema, making them interchangeable from an agent's perspective.
4. **CLI-first for agents.** The primary interface is a CLI with structured JSON output. The Python API exists underneath and is available for custom orchestration.
5. **Robust error reporting.** Every run produces structured logs, timing data, and typed errors. Agents always have sufficient diagnostic context to understand failures.
6. **Real end-to-end testing.** Integration tests run real tools on real data — no mocking. A tiered test strategy accommodates both CI constraints and thoroughness requirements.

---

## 2. Architecture

### 2.1 Execution Model: Container as Function Call

Autobio uses a **volume-mount batch execution** model. For each tool invocation:

1. The host creates a workspace directory with a standardized layout.
2. The host writes input data and configuration to the workspace.
3. The host launches a Docker container with the workspace mounted as a volume.
4. The container reads inputs, executes the tool, standardizes outputs, and writes results.
5. The host reads the results from the workspace and returns structured output.

There is no persistent server, no HTTP/gRPC transport, no session state. Each invocation is a single `docker run` with a bind mount. This model is simple, idempotent, and trivially debuggable — after any run, the workspace directory contains the complete record of what happened.

For tools that benefit from amortizing model weight loading across multiple inputs (e.g., language models), the container accepts batch inputs within a single invocation. Weights are loaded once, and multiple inputs are processed sequentially or in GPU-batched groups. This provides the performance benefits of a persistent service without the operational complexity.

### 2.2 High-Level Data Flow

```
Agent/User
    │
    ▼
┌────────────────────────────────────────────────────────────────┐
│  CLI  (Typer)                                                  │
│  - parses args, resolves tool, formats output                  │
│  - `autobio run <tool> --config input.json --gpu auto`        │
│  - `autobio info <tool> --format json`                        │
│  - `autobio list --category structure-prediction`             │
└────────────┬───────────────────────────────────────────────────┘
             │
             ▼
┌────────────────────────────────────────────────────────────────┐
│  Core Library                                                  │
│                                                                │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────────────┐    │
│  │  ToolRunner   │  │  GPUManager  │  │  ContainerManager │    │
│  │  (per-tool)   │  │  (semaphore) │  │  (Docker SDK)     │    │
│  └──────┬───────┘  └──────┬───────┘  └────────┬──────────┘    │
│         │                 │                    │               │
│         ▼                 ▼                    ▼               │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Workspace Manager                                      │   │
│  │  - creates workspace dir                                │   │
│  │  - writes config.json + input files                     │   │
│  │  - reads result.json + standardized outputs             │   │
│  └─────────────────────────┬───────────────────────────────┘   │
│                            │                                   │
└────────────────────────────┼───────────────────────────────────┘
                             │
                    Docker volume mount
                             │
                             ▼
┌────────────────────────────────────────────────────────────────┐
│  Tool Container                                                │
│                                                                │
│  base-entrypoint.sh                                            │
│  ├─ Phase 1: validate_config.sh  (fail-fast)                  │
│  ├─ Phase 2: run.sh              (execute tool)               │
│  └─ Phase 3: standardize.sh      (coerce outputs)             │
│                                                                │
│  /workspace/                                                   │
│  ├── config.json                                               │
│  ├── inputs/                                                   │
│  ├── outputs/raw/                                              │
│  ├── outputs/standardized/                                     │
│  ├── logs/                                                     │
│  └── result.json                                               │
└────────────────────────────────────────────────────────────────┘
```

### 2.3 Lifecycle of a Single Tool Invocation

1. **CLI parsing.** Typer parses arguments. The tool name is resolved against the registry. Input data is loaded from a `--config` file or inline arguments (e.g., `--sequences`, `--input-fasta`).

2. **Workspace creation.** A temporary directory is created (or a user-specified `--output-dir` is used). Subdirectories (`inputs/`, `outputs/raw/`, `outputs/standardized/`, `logs/`) are initialized. `config.json` is written with tool parameters. Input files are copied into `inputs/`.

3. **GPU resolution.** If `--gpu auto`, the `GPUManager` queries available GPUs via `pynvml`, checks the semaphore for in-use devices, and allocates the first available. If `--gpu 0,1`, those specific devices are requested. If a GPU is required but unavailable, a clear error is raised before any container launch.

4. **Container launch.** `ContainerManager` pulls the image if not cached locally, creates the container with the workspace bind-mounted at `/workspace`, passes GPU device assignments, sets resource limits and timeouts, and starts the container.

5. **Container execution.** Inside the container, `base-entrypoint.sh` runs the three-phase protocol: config validation → tool execution → output standardization. Stdout/stderr are captured to log files throughout. `result.json` is written on completion or failure with status, timing, phase, and error details.

6. **Result collection.** The host reads `result.json`. On success, the tool runner's `parse_output()` method deserializes files from `outputs/standardized/` into Pydantic model instances. On failure, a typed exception is raised containing the structured error information from `result.json` plus captured logs.

7. **GPU release.** Allocated GPU IDs are returned to the semaphore.

8. **Output delivery.** The CLI serializes the Pydantic output to JSON and prints to stdout. If `--output-dir` was specified, the workspace persists on disk. Otherwise, the temp directory is cleaned up (configurable retention on failure for debugging).

---

## 3. Repository Structure

```
autobio/
│
├── DESIGN.md                              # this document
├── pyproject.toml
├── LICENSE
├── README.md
│
├── src/
│   └── autobio/
│       ├── __init__.py                    # version, top-level convenience imports
│       │
│       ├── cli/                           # --- CLI layer (Typer) ---
│       │   ├── __init__.py
│       │   ├── main.py                    # Typer app, top-level command group
│       │   ├── run.py                     # `autobio run <tool>` subcommand
│       │   ├── info.py                    # `autobio info <tool>` subcommand
│       │   ├── list.py                    # `autobio list` subcommand
│       │   ├── images.py                  # `autobio pull`, `autobio images`
│       │   ├── result.py                  # `autobio result <workspace>`
│       │   └── formatters.py             # JSON / human-readable output formatting
│       │
│       ├── core/                          # --- Core orchestration ---
│       │   ├── __init__.py
│       │   ├── container.py               # Docker SDK wrapper (sole Docker touchpoint)
│       │   ├── workspace.py               # workspace directory lifecycle
│       │   ├── gpu.py                     # GPU discovery + allocation semaphore
│       │   ├── config.py                  # AutobioConfig (env vars + defaults)
│       │   ├── registry.py                # tool name → image, category, schema, etc.
│       │   ├── result.py                  # result.json parsing, typed exceptions
│       │   └── cache.py                   # content-addressable output cache (future)
│       │
│       ├── schemas/                       # --- Standardized I/O schemas ---
│       │   ├── SCHEMA_SPEC.md             # schema authoring specification
│       │   ├── __init__.py
│       │   ├── base.py                    # RunMetadata, BaseInput, BaseOutput
│       │   ├── structure_prediction.py
│       │   ├── embedding.py
│       │   ├── inverse_folding.py
│       │   ├── scoring.py
│       │   └── ...                        # additional categories as needed
│       │
│       ├── tools/                         # --- Per-tool host-side runners ---
│       │   ├── TOOL_SPEC.md               # tool runner authoring specification
│       │   ├── __init__.py
│       │   ├── base.py                    # abstract ToolRunner base class
│       │   ├── alphafold.py
│       │   ├── boltz.py
│       │   ├── esm2.py
│       │   ├── ligandmpnn.py
│       │   └── ...
│       │
│       └── utils/
│           ├── __init__.py
│           ├── sequences.py               # FASTA parsing, sequence validation
│           ├── structures.py              # PDB/mmCIF read/write helpers
│           └── logging.py                 # structured logging (JSON lines)
│
├── containers/                            # --- Docker build contexts ---
│   ├── CONTAINER_SPEC.md                  # container authoring specification
│   ├── base-entrypoint.sh                 # shared entrypoint protocol script
│   ├── alphafold/
│   │   ├── Dockerfile
│   │   ├── validate_config.sh
│   │   ├── run.sh
│   │   ├── standardize.sh
│   │   └── test/
│   │       ├── inputs/                    # minimal test inputs
│   │       └── expected_outputs/          # golden standardized outputs
│   ├── esm2/
│   │   └── ...
│   ├── ligandmpnn/
│   │   └── ...
│   └── ...
│
└── tests/
    ├── conftest.py                        # shared fixtures
    ├── unit/                              # Tier 1
    │   ├── test_workspace.py
    │   ├── test_gpu.py
    │   ├── test_config.py
    │   ├── test_registry.py
    │   ├── test_schemas.py
    │   └── test_cli.py
    ├── container_build/                   # Tier 2
    │   └── test_builds.py
    ├── smoke/                             # Tier 3
    │   ├── test_esm2_smoke.py
    │   ├── test_alphafold_smoke.py
    │   └── ...
    └── integration/                       # Tier 4
        ├── test_esm2_integration.py
        ├── test_alphafold_integration.py
        └── ...
```

---

## 4. Host Package Components

### 4.1 Configuration (`core/config.py`)

Zero-config by default. Everything is auto-detected or has sensible defaults. Three optional environment variables provide escape hatches:

| Variable | Default | Purpose |
|---|---|---|
| `AUTOBIO_DOCKER_HOST` | `None` (system default) | Non-default Docker socket path |
| `AUTOBIO_IMAGE_PREFIX` | `ghcr.io/briney/autobio-` | Override image registry prefix |
| `AUTOBIO_LOG_LEVEL` | `INFO` | Logging verbosity |

```python
@dataclass
class AutobioConfig:
    docker_host: str | None = None
    image_prefix: str = "ghcr.io/briney/autobio-"
    log_level: str = "INFO"

    @classmethod
    def resolve(cls, **runtime_overrides) -> "AutobioConfig":
        """Precedence: runtime args > env vars > defaults."""
        return cls(
            docker_host=(
                runtime_overrides.get("docker_host")
                or os.environ.get("AUTOBIO_DOCKER_HOST")
            ),
            image_prefix=(
                runtime_overrides.get("image_prefix")
                or os.environ.get("AUTOBIO_IMAGE_PREFIX", cls.image_prefix)
            ),
            log_level=(
                runtime_overrides.get("log_level")
                or os.environ.get("AUTOBIO_LOG_LEVEL", cls.log_level)
            ),
        )
```

### 4.2 Tool Registry (`core/registry.py`)

A Python dictionary mapping tool names to their metadata. Ships with the package and is versioned alongside releases.

```python
@dataclass
class ToolEntry:
    image_tag: str              # e.g., "alphafold:2.3.2"
    category: str               # e.g., "structure-prediction"
    requires_gpu: bool
    gpu_count: int              # default number of GPUs
    input_schema: type          # Pydantic model class
    output_schema: type         # Pydantic model class
    default_timeout: int        # seconds
    supports_batch: bool
    description: str            # human/agent-readable summary
    version: str                # upstream tool version tracked by this image

TOOL_REGISTRY: dict[str, ToolEntry] = {
    "alphafold": ToolEntry(
        image_tag="alphafold:2.3.2",
        category="structure-prediction",
        requires_gpu=True,
        gpu_count=1,
        input_schema=StructurePredictionInput,
        output_schema=StructurePredictionOutput,
        default_timeout=3600,
        supports_batch=False,
        description="Predict protein structures from amino acid sequences.",
        version="2.3.2",
    ),
    # ... additional tools ...
}
```

Image tags are resolved to full URIs by prepending `AutobioConfig.image_prefix`. For example, `alphafold:2.3.2` becomes `ghcr.io/briney/autobio-alphafold:2.3.2`.

### 4.3 GPU Manager (`core/gpu.py`)

Lightweight semaphore-based GPU allocator.

```python
class GPUManager:
    def __init__(self):
        self._available: list[int] = self._discover_gpus()
        self._lock = threading.Lock()
        self._in_use: set[int] = set()

    def _discover_gpus(self) -> list[int]:
        """Query available NVIDIA GPUs via pynvml."""
        ...

    def allocate(
        self, count: int = 1, device_ids: list[int] | None = None
    ) -> list[int]:
        """
        Reserve GPUs. Blocks until requested devices are available.
        - count: number of GPUs to allocate (when device_ids is None)
        - device_ids: specific GPU IDs to request
        Returns list of allocated GPU IDs.
        """
        ...

    def release(self, device_ids: list[int]) -> None:
        """Return GPUs to the available pool."""
        ...

    @property
    def available_gpus(self) -> list[int]:
        """Currently unallocated GPU IDs."""
        ...
```

GPU resolution from CLI arguments:
- `--gpu auto` → `allocate(count=tool.gpu_count)`
- `--gpu 0,2` → `allocate(device_ids=[0, 2])`
- `--gpu none` → no GPU passed to container (CPU-only run)
- Omitted → `auto` if tool requires GPU, `none` otherwise

### 4.4 Container Manager (`core/container.py`)

The sole interface to the Docker SDK. No other module in the codebase imports Docker libraries directly.

Responsibilities:
- **Image management:** pull images, check local availability, list cached images.
- **Container lifecycle:** create, start, wait, remove. Configures bind mounts, GPU devices, resource limits (memory, CPU), and timeouts.
- **Log capture:** streams stdout/stderr and writes them to the workspace log directory.
- **Cleanup:** ensures containers are removed after completion (success or failure).

```python
class ContainerManager:
    def __init__(self, config: AutobioConfig):
        self._client = docker.DockerClient(base_url=config.docker_host)

    def ensure_image(self, image_uri: str) -> None:
        """Pull image if not present locally."""
        ...

    def run(
        self,
        image_uri: str,
        workspace: Path,
        gpu_ids: list[int] | None = None,
        timeout: int | None = None,
        memory_limit: str | None = None,
    ) -> ContainerResult:
        """
        Run container with workspace mounted at /workspace.
        Blocks until container exits or timeout is reached.
        Returns ContainerResult with exit_code and log paths.
        """
        ...

    def list_images(self, prefix: str) -> list[ImageInfo]:
        """List locally cached autobio images."""
        ...
```

### 4.5 Workspace Manager (`core/workspace.py`)

Handles creation and teardown of the standardized workspace directory.

```python
class Workspace:
    def __init__(self, root: Path):
        self.root = root
        self.config_path = root / "config.json"
        self.inputs_dir = root / "inputs"
        self.raw_output_dir = root / "outputs" / "raw"
        self.std_output_dir = root / "outputs" / "standardized"
        self.logs_dir = root / "logs"
        self.result_path = root / "result.json"

    @classmethod
    def create(cls, output_dir: Path | None = None) -> "Workspace":
        """
        Create workspace directory structure.
        If output_dir is None, creates a temp directory.
        """
        root = output_dir or Path(tempfile.mkdtemp(prefix="autobio-"))
        for subdir in [
            "inputs",
            "outputs/raw",
            "outputs/standardized",
            "logs",
        ]:
            (root / subdir).mkdir(parents=True, exist_ok=True)
        return cls(root)

    def write_config(self, config: dict) -> None:
        """Serialize tool configuration to config.json."""
        ...

    def write_input_file(self, filename: str, content: str | bytes) -> Path:
        """Write an input file to inputs/."""
        ...

    def read_result(self) -> RunResult:
        """Parse result.json into a RunResult object."""
        ...

    def cleanup(self) -> None:
        """Remove workspace if it was auto-created (temp dir)."""
        ...
```

### 4.6 Result Parsing and Exceptions (`core/result.py`)

Typed exception hierarchy:

```python
class AutobioError(Exception):
    """Base exception for all autobio errors."""
    pass

class ContainerNotFoundError(AutobioError):
    """Image not available and could not be pulled."""
    pass

class GPUNotAvailableError(AutobioError):
    """Requested GPU(s) not available."""
    pass

class ToolExecutionError(AutobioError):
    """Container exited with non-zero status."""
    def __init__(
        self,
        phase: str,
        exit_code: int,
        error_message: str,
        logs: str,
        wall_time: float,
    ):
        self.phase = phase          # "setup" | "execution" | "standardization"
        self.exit_code = exit_code
        self.error_message = error_message
        self.logs = logs
        self.wall_time = wall_time

class ToolTimeoutError(AutobioError):
    """Container exceeded timeout."""
    pass
```

The `result.json` contract (written by the container):

```json
{
    "status": "success | failed",
    "exit_code": 0,
    "phase": "complete | setup | execution | standardization",
    "error_type": null,
    "error_message": null,
    "wall_time_seconds": 142.7,
    "gpu_ids": [0],
    "completed": 1,
    "total": 1,
    "outputs": {
        "standardized_files": [
            "outputs/standardized/structures.json"
        ],
        "raw_files": [
            "outputs/raw/ranked_0.pdb",
            "outputs/raw/ranking_debug.json"
        ]
    }
}
```

### 4.7 Tool Runners (`tools/`)

Each tool has a runner class inheriting from `ToolRunner`. See `tools/TOOL_SPEC.md` for the full authoring specification.

```python
class ToolRunner(ABC):
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
        Translate standardized input into tool-specific config.json
        and input files within the workspace.
        """
        ...

    @abstractmethod
    def parse_output(self, workspace: Workspace) -> BaseOutput:
        """
        Deserialize standardized output files from the workspace
        into the appropriate Pydantic output model.
        """
        ...

    def run(
        self,
        input_data: BaseInput,
        gpu: str | list[int] = "auto",
        timeout: int | None = None,
        output_dir: Path | None = None,
    ) -> BaseOutput:
        """Full execution lifecycle (see §2.3)."""
        workspace = Workspace.create(output_dir)
        gpu_ids = []
        try:
            self.prepare_workspace(input_data, workspace)
            gpu_ids = self._resolve_gpu(gpu)
            image_uri = self.config.image_prefix + self.entry.image_tag
            self._container.ensure_image(image_uri)
            self._container.run(
                image_uri,
                workspace.root,
                gpu_ids,
                timeout or self.entry.default_timeout,
            )
            result = workspace.read_result()
            if result.status != "success":
                raise ToolExecutionError(
                    phase=result.phase,
                    exit_code=result.exit_code,
                    error_message=result.error_message,
                    logs=self._read_logs(workspace),
                    wall_time=result.wall_time_seconds,
                )
            return self.parse_output(workspace)
        finally:
            if gpu_ids:
                self._gpu.release(gpu_ids)
            if output_dir is None:
                workspace.cleanup()
```

### 4.8 CLI (`cli/`)

A thin Typer application that delegates all logic to the core library.

| Command | Description |
|---|---|
| `autobio list` | List available tools. Filterable by `--category`. |
| `autobio info <tool>` | Show tool details: params, GPU needs, schema. |
| `autobio run <tool>` | Execute a tool. Accepts `--config`, `--gpu`, `--timeout`, `--output-dir`. |
| `autobio result <dir>` | Inspect a previous run from its workspace directory. |
| `autobio pull <tool>` | Pull a tool's container image. `--all` for everything. |
| `autobio images` | List locally cached autobio container images. |

All commands support `--format json` for machine-readable output, which is the primary interface for agentic use.

The `autobio info --format json` output provides a complete input schema an agent can use to construct valid `autobio run` commands:

```json
{
    "name": "alphafold",
    "category": "structure-prediction",
    "description": "Predict protein structures from amino acid sequences.",
    "version": "2.3.2",
    "requires_gpu": true,
    "default_gpu_count": 1,
    "supports_batch": false,
    "default_timeout": 3600,
    "parameters": {
        "sequences": {
            "type": "dict[str, str]",
            "required": true,
            "description": "Mapping of chain ID to amino acid sequence."
        },
        "num_models": {
            "type": "int",
            "required": false,
            "default": 1,
            "range": [1, 5]
        },
        "extra": {
            "type": "dict",
            "required": false,
            "default": {},
            "description": "Tool-specific parameters passed through to the container."
        }
    },
    "output_schema": { "..." : "..." }
}
```

---

## 5. Container Architecture

### 5.1 Workspace Directory Contract

Every container receives a workspace mounted at `/workspace`:

```
/workspace/
├── config.json                # tool parameters (written by host)
├── inputs/                    # input files (written by host)
├── outputs/
│   ├── raw/                   # native tool outputs (written by container)
│   └── standardized/          # coerced to schema format (written by container)
├── logs/
│   ├── stdout.log             # captured stdout
│   ├── stderr.log             # captured stderr
│   └── tool.log               # optional tool-specific structured log
└── result.json                # run status and metadata (written by container)
```

### 5.2 Container Entrypoint Protocol

All containers use a shared `base-entrypoint.sh` that implements the three-phase execution protocol. Tool-specific logic is provided via hook scripts:

| Script | Phase | Purpose |
|---|---|---|
| `/opt/tool/validate_config.sh` | Setup | Validate `config.json`. Fail fast with clear errors. |
| `/opt/tool/run.sh` | Execution | Run the tool. Read from `config.json` and `inputs/`, write to `outputs/raw/`. |
| `/opt/tool/standardize.sh` | Standardization | Transform `outputs/raw/` into `outputs/standardized/` per the category schema. |

The base entrypoint handles log capture (tee stdout/stderr to files), phase tracking, timing, `result.json` generation, and graceful error handling. If `run.sh` fails, raw partial outputs are preserved. If `standardize.sh` fails, `outputs/raw/` is still intact.

See `containers/CONTAINER_SPEC.md` for the full container authoring specification.

### 5.3 Batch Execution

For tools that support batch processing, `config.json` includes:

```json
{
    "batch_mode": "sequential | batched",
    "batch_size": 32
}
```

The `inputs/` directory contains multiple input items. The container's `run.sh` iterates over them, writing outputs progressively rather than accumulating in memory. `result.json` is updated incrementally with a `completed` count. If the container crashes mid-batch, all completed outputs in `outputs/raw/` and `outputs/standardized/` are preserved.

### 5.4 Container Image Strategy

- Images are published to GHCR: `ghcr.io/briney/autobio-<tool>:<version>`
- Version tags track the upstream tool version
- For tools with existing official Docker images, the autobio image extends them with a thin wrapper layer
- `autobio pull <tool>` pulls the version pinned in the registry for the installed package version
- `--image <custom_uri>` overrides the registry at runtime

---

## 6. Schemas

### 6.1 Base Types (`schemas/base.py`)

```python
class RunMetadata(BaseModel):
    """Metadata attached to every tool output."""
    tool_name: str
    tool_version: str
    image_uri: str
    wall_time_seconds: float
    gpu_ids: list[int] | None = None
    workspace_path: Path
    timestamp: datetime

class BaseInput(BaseModel):
    """Base class for all tool inputs."""
    extra: dict[str, Any] = Field(
        default_factory=dict,
        description="Tool-specific parameters passed through to the container.",
    )

class BaseOutput(BaseModel):
    """Base class for all tool outputs."""
    metadata: RunMetadata
    raw_output_path: Path
```

### 6.2 Category Schemas (illustrative example)

Each tool category defines standardized input and output models. Fields common across most tools are explicit. Fields only some tools provide are `Optional`. Tool-specific parameters flow through `extra`.

```python
# -- structure_prediction.py --

class StructurePredictionInput(BaseInput):
    sequences: dict[str, str]              # chain_id → sequence
    num_models: int = 1
    templates: list[Path] | None = None

class PredictedStructure(BaseModel):
    model_rank: int
    structure_path: Path                   # in outputs/standardized/
    plddt_per_residue: list[float] | None = None
    plddt_mean: float | None = None
    ptm: float | None = None
    iptm: float | None = None
    chain_mapping: dict[str, str] | None = None

class ConfidenceMetrics(BaseModel):
    best_plddt_mean: float | None = None
    best_ptm: float | None = None
    best_iptm: float | None = None

class StructurePredictionOutput(BaseOutput):
    structures: list[PredictedStructure]
    confidence: ConfidenceMetrics
```

See `schemas/SCHEMA_SPEC.md` for the full specification on authoring new schemas.

---

## 7. Testing Strategy

### 7.1 Test Tiers

| Tier | Scope | Trigger | Runner | Docker | GPU |
|---|---|---|---|---|---|
| 1 — Unit | Schema validation, config, workspace, GPU logic, CLI parsing, registry | Every push / PR | GitHub Actions (CPU) | No | No |
| 2 — Container build | Each Dockerfile builds successfully | PR, nightly | GitHub Actions (CPU) | Build only | No |
| 3 — Smoke | Full pipeline with minimal inputs (10-residue peptide, single sequence) | `workflow_dispatch`, nightly | Self-hosted (GPU) / GHA (CPU tools) | Run | Per tool |
| 4 — Integration | Real-world inputs, realistic params, output correctness | `workflow_dispatch` (manual) | Self-hosted (GPU) | Run | Per tool |

### 7.2 CI Infrastructure

- Tiers 1–2 run on standard GitHub Actions runners.
- Tiers 3–4 requiring GPU use self-hosted runners registered on lab GPU machines, labeled `self-hosted, gpu`. Triggered via `workflow_dispatch` or configurable schedule.
- GPU tests are tagged with `@pytest.mark.gpu` and auto-skipped when no GPU is detected, so the full suite can run anywhere without failures.
- Each tool's `containers/<tool>/test/` directory contains minimal inputs and expected outputs for smoke tests.

### 7.3 Test Data

- **Smoke inputs:** committed to the repo. Tiny sequences and minimal structures designed to exercise the full pipeline in under 60 seconds per tool.
- **Integration data:** stored externally (shared storage or designated directory on self-hosted runner), referenced by tests, not committed to the repo.
- **Golden outputs:** committed for smoke tests. Assertions compare standardized output structure and values (with tolerance for floating-point results).

### 7.4 Test Logging

All tiers capture and report container stdout/stderr, `result.json` contents, and workspace directory listings on failure. Failed integration tests preserve the full workspace for post-mortem debugging.

---

## 8. Dependencies

### 8.1 Host Package

| Package | Purpose |
|---|---|
| `pydantic` | Schema validation and serialization |
| `typer` | CLI framework |
| `docker` or `python-on-whales` | Docker SDK |
| `pynvml` | GPU discovery |
| `rich` | Human-readable CLI formatting |

No PyTorch, no CUDA, no tool-specific dependencies.

### 8.2 System Requirements

- Python ≥ 3.10
- Docker Engine
- NVIDIA Container Toolkit (for GPU tools)
- NVIDIA drivers (for GPU tools)

---

## 9. Future Considerations

Deferred from initial implementation:

- **Content-addressable output cache** (`core/cache.py`): Hash inputs deterministically, skip runs when cached outputs exist. High value for iterative agentic workflows.
- **MCP adapter**: Expose autobio tools as MCP tool calls, built on top of the CLI.
- **Persistent service mode**: For tools where warm-up latency is a bottleneck beyond what batching solves.
- **Config file** (`.autobio.toml`): Introduced only if repeated override patterns emerge.
- **Custom tool registration**: Users point to a Docker image and provide a runner class for proprietary tools.
- **Async execution**: `autobio run --async` returns a job handle; `autobio status <job>` polls progress. Useful for long-running tools in agentic loops.

---

## Appendices

### A. Glossary

| Term | Definition |
|---|---|
| **Tool** | A computational biology model or program wrapped for use in autobio. |
| **Category** | A functional grouping of tools sharing an I/O schema (e.g., structure-prediction). |
| **Runner** | The host-side Python class for a specific tool. Translates between standardized schemas and tool-specific config. |
| **Container** | The Docker image and its internal scripts that execute a tool. |
| **Workspace** | The directory mounted into a container at `/workspace`. Contains all inputs, outputs, logs, and metadata for a single run. |
| **Registry** | The in-code mapping from tool names to their metadata. |
| **Standardized output** | Tool outputs coerced into the category's schema format by the container's `standardize.sh`. |
| **Raw output** | Native tool output files, preserved unmodified in `outputs/raw/`. |

### B. CLI Quick Reference

```bash
# List tools
autobio list
autobio list --category structure-prediction --format json

# Tool info
autobio info alphafold
autobio info esm2 --format json

# Run tools
autobio run alphafold --config input.json --gpu auto
autobio run esm2 --input-fasta seqs.fasta --gpu 0 --output-dir ./my_run
autobio run ligandmpnn --config design.json --gpu none

# Inspect results
autobio result ./my_run
autobio result ./my_run --format json

# Manage images
autobio pull alphafold
autobio pull --all
autobio images
```
