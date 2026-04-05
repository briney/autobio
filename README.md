# autobio

Tools for autonomous biological research.

Autobio provides a unified interface to computational biology tools — structure prediction, inverse folding, protein embeddings, structure design, mutant structure building, scoring, and molecular dynamics simulation. Each tool runs in its own Docker container, fully isolating heavyweight dependencies (PyTorch, CUDA, tool-specific libraries) from your host environment. The host package is lightweight and has no ML dependencies.

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

### Structure prediction

| Tool | GPU | Description |
|------|-----|-------------|
| `boltz1` | Yes | Predict biomolecular structures using Boltz-1 (proteins, DNA, RNA, ligand complexes) |
| `boltz2` | Yes | Predict biomolecular structures and binding affinity using Boltz-2 |
| `chai1` | Yes | Predict biomolecular structures using Chai-1 (proteins, DNA, RNA, ligands, glycans) |
| `esmfold` | Yes | Predict protein structure from a single sequence using ESMFold (no MSA needed) |
| `openfold3` | Yes | Predict biomolecular structures using OpenFold3 (open-source AlphaFold3) |

### Inverse folding

| Tool | GPU | Description |
|------|-----|-------------|
| `proteinmpnn` | Yes | Fixed-backbone sequence design with ProteinMPNN |
| `ligandmpnn` | Yes | Ligand-aware sequence design with LigandMPNN |

### Structure design

| Tool | GPU | Description |
|------|-----|-------------|
| `rfd3` | Yes | Generate novel protein backbone structures using RFDiffusion3 |
| `complexa` | Yes | Design novel protein binders for protein targets using Proteina-Complexa |
| `complexa_ligand` | Yes | Design protein binders for small-molecule ligand targets using Proteina-Complexa |
| `complexa_ame` | Yes | Scaffold functional motifs into complete proteins using Proteina-Complexa AME |

### Embedding

| Tool | GPU | Description |
|------|-----|-------------|
| `esm1b` | Yes | Extract protein sequence embeddings using ESM-1b (650M parameters) |
| `esm2` | Yes | Extract protein sequence embeddings using ESM-2 (selectable 8M–15B parameters) |
| `currab` | Yes | Extract antibody sequence embeddings using CurrAb (650M parameters). Paired and unpaired. |
| `currab_pll` | Yes | Compute pseudo log-likelihood for antibody sequences using CurrAb |
| `ft_esm` | Yes | Extract antibody sequence embeddings using ft-ESM (fine-tuned ESM-2, 650M parameters) |
| `ft_esm_pll` | Yes | Compute pseudo log-likelihood for antibody sequences using ft-ESM |
| `balm_paired` | Yes | Extract paired antibody sequence embeddings using BALM-paired (304M parameters) |
| `balm_paired_pll` | Yes | Compute pseudo log-likelihood for paired antibody sequences using BALM-paired |
| `balm_unpaired` | Yes | Extract single-chain antibody sequence embeddings using BALM-unpaired (304M parameters) |
| `balm_unpaired_pll` | Yes | Compute pseudo log-likelihood for single-chain antibody sequences using BALM-unpaired |
| `ablang2` | Yes | Extract antibody sequence embeddings using AbLang2 (45M parameters). Paired and unpaired. |
| `ablang2_pll` | Yes | Compute pseudo log-likelihood for antibody sequences using AbLang2 |
| `antiberta2` | Yes | Extract antibody sequence embeddings using AntiBERTa2 (202M parameters). Paired and unpaired. |
| `antiberta2_pll` | Yes | Compute pseudo log-likelihood for antibody sequences using AntiBERTa2 |

### Mutate structures

| Tool | GPU | Description |
|------|-----|-------------|
| `evoef2_build_mutant` | No | Build mutant protein structures and repack sidechains using EvoEF2's physics-based rotamer library |
| `ligandmpnn_build_mutant` | Yes | Build mutant protein structures and repack sidechains using LigandMPNN's neural network sidechain packing model |

### Scoring

| Tool | GPU | Description |
|------|-----|-------------|
| `rosetta_score` | No | Score a protein structure using Rosetta's energy function |
| `rosetta_relax` | No | Relax a protein structure using Rosetta's FastRelax protocol |
| `rosetta_minimize` | No | Minimize a protein structure using gradient-based energy minimization |
| `rosetta_flexddg` | No | Predict binding ΔΔG at protein-protein interfaces using flex-ddG |
| `stabddg` | Yes | Predict binding ΔΔG from mutations using StaB-ddG (ML-based, ProteinMPNN architecture) |
| `baddg` | Yes | Predict binding ΔΔG at protein-protein interfaces using BA-ddG (Boltzmann-aligned inverse folding) |
| `evoef2_repair` | No | Repair protein structures by rebuilding incomplete side chains using EvoEF2 |
| `evoef2_binding` | No | Compute protein-protein binding energy using EvoEF2's physics-based energy function |
| `openmm_amber_minimize` | No | Minimize a protein structure using OpenMM with the Amber force field (AlphaFold-style) |
| `openmm_amber_relax` | No | Relax a protein structure using OpenMM with Amber force field and explicit solvent |

### Simulation

| Tool | GPU | Description |
|------|-----|-------------|
| `openmm_md_simulate` | Yes | Run production molecular dynamics using OpenMM with Amber force field and explicit solvent |

## How it works

Each tool follows a three-phase container protocol:

1. **Validate** — check that the input configuration is well-formed
2. **Execute** — run the tool, writing raw outputs
3. **Standardize** — coerce raw outputs into a common schema

All phases log to a structured workspace directory. If any phase fails, partial outputs are preserved and errors are reported with the failing phase, exit code, and captured logs.

## License

MIT
