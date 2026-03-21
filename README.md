# autobio

Tools for autonomous biological research.

Autobio provides a unified interface to computational biology tools — structure prediction, inverse folding, protein embeddings, and more. Each tool runs in its own Docker container, fully isolating heavyweight dependencies (PyTorch, CUDA, tool-specific libraries) from your host environment. The host package is lightweight and has no ML dependencies.

Autobio is designed for both direct human use and as the computational backbone for AI agents performing autonomous biological research. All commands support `--format json` for structured, machine-readable output.

## Prerequisites

**Docker** is required. All tools run inside containers.

```bash
# Install Docker: https://docs.docker.com/engine/install/
docker --version
```

**GPU + NVIDIA drivers** are required for most tools. You'll need:
- An NVIDIA GPU with up-to-date drivers
- The [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html), which enables GPU access from inside Docker containers

```bash
# Verify GPU access from Docker
docker run --rm --gpus all nvidia/cuda:12.6.3-base-ubuntu24.04 nvidia-smi
```

## Installation

```bash
pip install autobio
```

### Development

```bash
git clone https://github.com/briney/autobio.git
cd autobio
pip install -e ".[dev]"
```

## Quick start

Pull a tool's container image, then run it:

```bash
# See what tools are available
autobio list

# Pull the ProteinMPNN container image
autobio pull proteinmpnn

# Run inverse folding on a PDB structure
autobio run proteinmpnn --config config.json
```

Where `config.json` contains the tool's input parameters:

```json
{
    "structure_path": "1abc.pdb",
    "num_sequences": 8,
    "temperature": 0.1,
    "chains_to_design": ["A"]
}
```

The `structure_path` can be an absolute path or relative to the config file's directory. Output is printed as a Rich table by default, or as JSON with `--format json`.

## CLI

The `autobio` command has six subcommands:

| Command | Description |
|---------|-------------|
| `autobio list` | List available tools, optionally filtered by `--category` |
| `autobio info <tool>` | Show tool details: description, schemas, GPU requirements |
| `autobio run <tool>` | Execute a tool with `--config`, `--gpu`, `--timeout`, `--output-dir` |
| `autobio result <dir>` | Inspect a previous run's result from its workspace directory |
| `autobio pull [tool]` | Pull a tool's container image (or `--all` for everything) |
| `autobio images` | List locally cached autobio container images |

All commands support `--format json` for machine-readable output.

For the full CLI reference, see [src/autobio/cli/README.md](src/autobio/cli/README.md).

## Python API

You can also use autobio as a library:

```python
from pathlib import Path

from autobio import AutobioConfig
from autobio.schemas.inverse_folding import InverseFoldingInput
from autobio.tools import get_runner

# Configure (picks up AUTOBIO_* env vars automatically)
config = AutobioConfig.resolve()

# Build a typed input
input_data = InverseFoldingInput(
    structure_path=Path("1abc.pdb"),
    num_sequences=8,
    temperature=0.1,
    chains_to_design=["A"],
)

# Run
runner = get_runner("proteinmpnn", config)
output = runner.run(input_data)

# Structured output with metadata
for seq in output.designed_sequences:
    print(f"Rank {seq.rank}: score={seq.score:.3f}, recovery={seq.recovery:.2%}")
    for chain_id, sequence in seq.sequence.items():
        print(f"  Chain {chain_id}: {sequence}")

# Run metadata is attached automatically
print(f"Wall time: {output.metadata.wall_time_seconds:.1f}s")
print(f"GPUs used: {output.metadata.gpu_ids}")
```

### Error handling

```python
from autobio import ToolExecutionError, ToolTimeoutError, GPUNotAvailableError

try:
    output = runner.run(input_data, gpu="auto", timeout=600)
except GPUNotAvailableError:
    # No GPUs available (or requested GPUs are busy)
    output = runner.run(input_data, gpu="none")
except ToolTimeoutError:
    # Container exceeded the timeout
    ...
except ToolExecutionError as exc:
    # Container exited with an error
    print(f"Failed in phase: {exc.phase}")
    print(f"Logs: {exc.logs}")
```

## Configuration

Autobio is configured through environment variables, with sensible defaults:

| Variable | Default | Description |
|----------|---------|-------------|
| `AUTOBIO_IMAGE_PREFIX` | `ghcr.io/briney/autobio-` | Container image registry prefix |
| `AUTOBIO_DOCKER_HOST` | *(system default)* | Custom Docker daemon socket |
| `AUTOBIO_LOG_LEVEL` | `INFO` | Logging verbosity |

Environment variables can be overridden at runtime:

```python
config = AutobioConfig.resolve(image_prefix="my-registry.io/autobio-")
```

## Available tools

| Tool | Category | GPU | Description |
|------|----------|-----|-------------|
| `proteinmpnn` | inverse-folding | Yes | Fixed-backbone sequence design with ProteinMPNN |
| `ligandmpnn` | inverse-folding | Yes | Ligand-aware sequence design with LigandMPNN |

More tools (structure prediction, embeddings, scoring) are in development.

## How it works

Each tool follows a three-phase container protocol:

1. **Validate** — check that the input configuration is well-formed
2. **Execute** — run the tool, writing raw outputs
3. **Standardize** — coerce raw outputs into a common schema

All phases log to a structured workspace directory. If any phase fails, partial outputs are preserved and errors are reported with the failing phase, exit code, and captured logs.

## License

MIT
