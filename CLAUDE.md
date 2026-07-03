# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Quick Reference

```bash
# Install (editable, with dev dependencies)
pip install -e ".[dev]"

# Run all tests
pytest

# Run a single test file or test
pytest tests/unit/test_mpnn.py
pytest tests/unit/test_mpnn.py::TestMPNNPrepareWorkspace::test_some_method -v

# Skip tests requiring Docker or GPU
pytest -m "not docker and not gpu"

# Lint and format
ruff check --fix src/ tests/
ruff format src/ tests/

# Type check
mypy src/

# CLI (after install)
autobio list
autobio info <tool>
autobio run <tool> --config config.json
```

## Code Conventions

- Python 3.11+ — use modern syntax (type unions with `|`, `match` statements, etc.)
- All public functions and classes need docstrings (Google style)
- Type hints on all function signatures
- Tests: `tests/unit/` and `tests/integration/`. Unit tests mock Docker/GPU; integration tests run containers
- Ruff handles formatting and linting — config is in `pyproject.toml`

## Before Committing

1. `ruff check --fix src/ tests/` — auto-fix lint issues
2. `ruff format src/ tests/` — format code
3. `pytest` — all tests pass
4. Commit message: `<component>: <what changed and why>`

## Architecture

Autobio provides a unified, agentic-friendly interface to computational biology tools. Each tool runs in its own Docker container, fully isolating heavyweight dependencies (PyTorch, CUDA, etc.) from the host. The host package has no ML dependencies.

**Key design principle:** Tool logic is split between **host-side preparation** (Python runner in `src/autobio/tools/`) and **container-side execution** (shell + Python scripts in `containers/`), connected by a standardized workspace directory and result protocol.

### Execution Flow

`ToolRunner.run()` orchestrates the full lifecycle:

1. Create `Workspace` (directory with `config.json`, `inputs/`, `outputs/`, `logs/`)
2. Call `prepare_workspace()` — subclass hook writes config and input files
3. Resolve GPUs via `GPUManager` (thread-safe allocation via pynvml)
4. Ensure container image via `ContainerManager` (pull if needed)
5. Run container with workspace bind-mounted at `/workspace`
6. Container executes three-phase protocol: `validate_config.sh` → `run.sh` → `standardize.sh`
7. Read `result.json`, call `parse_output()` — subclass hook returns typed output
8. Attach `RunMetadata` and return

### Catalog & Runners

- **`CATALOG`** (`core/catalog.py`) — maps tool names to `Tool` objects. Each `Tool` (one coherent model/engine) owns one or more named `Mode`s; per-mode metadata includes the input/output schemas, default timeout, and optional `image_tag`/`category` overrides, while image tag, GPU requirements, and primary category live on the `Tool`. `ToolCategory` (the category enum) also lives in `core/registry.py`.
- **`TOOL_RUNNERS`** (`tools/__init__.py`) — maps tool names to runner classes.

Multiple tool names can share a runner class (e.g., `proteinmpnn` and `ligandmpnn` both use `MPNNRunner`). A single Tool with multiple modes (e.g. `rosetta` with `score`/`relax`/`minimize`/`flexddg`) uses one runner that dispatches on `self.current_mode.name`.

### Schemas (`src/autobio/schemas/`)

One schema module per tool category: `structure_prediction`, `embedding`, `inverse_folding`, `scoring`, `simulation`, `structure_design`. Each defines typed Pydantic input/output models inheriting from `BaseInput`/`BaseOutput` in `base.py`. `BaseInput` includes an `extra` dict for tool-specific parameters not in the typed fields.

### Container Protocol (`containers/`)

Each tool has a directory under `containers/` containing: `Dockerfile`, `validate_config.sh`, `run.sh`, `standardize.sh`, `standardize.py`, and a `test/` directory. All containers use `containers/base-entrypoint.sh` as their shared entrypoint, which runs the three phases and produces `result.json` with status, timing, and error info.

### Adding a New Tool

1. **Schema** — use an existing category schema or create a new one in `src/autobio/schemas/`
2. **Runner** — create `src/autobio/tools/newtool.py` with a `ToolRunner` subclass implementing `prepare_workspace()` and `parse_output()`. Define a `Tool` (with its `Mode`s) and call `register(TOOL)` at module level (see `src/autobio/tools/TOOL_SPEC.md`)
3. **Wire up** — add the runner to `TOOL_RUNNERS` in `src/autobio/tools/__init__.py`
4. **Container** — create `containers/newtool/` with Dockerfile and the four protocol scripts
5. **Tests** — add `tests/unit/test_newtool.py` and `tests/integration/test_newtool_integration.py`
