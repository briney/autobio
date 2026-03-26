# autobio

A Python package.

## Quick Reference

```bash
# Install (editable, with dev dependencies)
pip install -e ".[dev]"

# Run tests
pytest

# Lint and format
ruff check src/ tests/
ruff format src/ tests/

# Type check
mypy src/
```

## Project Structure

```
src/autobio/     # Main package code
tests/           # Test suite (mirrors src structure)
```

## Code Conventions

- Python 3.11+ — use modern syntax (type unions with `|`, `match` statements, etc.)
- All public functions and classes need docstrings (Google style)
- Type hints on all function signatures
- Tests go in `tests/` mirroring the src structure: `src/autobio/bar.py` → `tests/test_bar.py`
- Ruff handles formatting and linting — don't override its defaults beyond pyproject.toml config

## Before Committing

1. `ruff check --fix src/ tests/` — auto-fix lint issues
2. `ruff format src/ tests/` — format code
3. `pytest` — all tests pass
4. Write a meaningful commit message: `<component>: <what changed and why>`

## Architecture

Autobio provides a unified, agentic-friendly interface to computational biology tools. Each tool runs in its own Docker container, fully isolating dependencies. The host package has a minimal footprint — no PyTorch, no CUDA.

### Core (`src/autobio/core/`)

- **`config.py`** — `AutobioConfig` dataclass with `resolve()` classmethod. Env vars: `AUTOBIO_DOCKER_HOST`, `AUTOBIO_IMAGE_PREFIX`, `AUTOBIO_LOG_LEVEL`. Precedence: runtime kwargs > env > defaults.
- **`result.py`** — Exception hierarchy (`AutobioError` → `ContainerNotFoundError`, `GPUNotAvailableError`, `ToolExecutionError`, `ToolTimeoutError`). `RunResult` Pydantic model for deserializing container `result.json`. `ContainerResult` dataclass.
- **`workspace.py`** — `Workspace` manages the run directory layout: `config.json`, `inputs/`, `raw_output/`, `std_output/`, `logs/`, `result.json`. Supports temp or user-specified directories.
- **`gpu.py`** — `GPUManager` discovers GPUs via pynvml (graceful degradation), allocates/releases with a thread lock.
- **`registry.py`** — `ToolCategory` StrEnum, `ToolEntry` dataclass, `TOOL_REGISTRY` dict, `get_tool()`/`list_tools()` helpers.
- **`container.py`** — `ContainerManager` wraps `python-on-whales` for image pull, container run (bind mounts, GPU device spec, timeout), and image listing. `ImageInfo` dataclass.

### Schemas (`src/autobio/schemas/`)

- **`base.py`** — `RunMetadata`, `BaseInput` (with `extra` dict), `BaseOutput` (with `metadata` and `raw_output_path`).
- **Category schemas** — `structure_prediction.py`, `embedding.py`, `inverse_folding.py`, `scoring.py`. Each defines typed input/output models inheriting from the base schemas.

### Tools (`src/autobio/tools/`)

- **`base.py`** — `ToolRunner` ABC. Concrete `run()` lifecycle: workspace → prepare → GPU resolve → ensure image → container run → result check → parse output → metadata. Subclasses implement `prepare_workspace()` and `parse_output()`.
- **`__init__.py`** — `TOOL_RUNNERS` dict and `get_runner()` factory.

### CLI (`src/autobio/cli/`)

Typer-based CLI with six subcommands: `list`, `info`, `run`, `result`, `pull`, `images`. Supports `--format json|table` output via Rich tables and `json.dumps`.

### Utilities (`src/autobio/utils/`)

- **`logging.py`** — `setup_logging()`, `JsonLineFormatter`, `get_logger()`.
- **`sequences.py`** — FASTA I/O, protein/nucleotide validation. Pure Python.
- **`structures.py`** — PDB/mmCIF sequence extraction, format detection. Pure Python.

### Container Protocol (`containers/`)

- **`base-entrypoint.sh`** — Shared entrypoint implementing a three-phase protocol: `validate_config.sh` → `run.sh` → `standardize.sh`. Produces `result.json` with status, timing, and error info.
