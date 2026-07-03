# autobio CLI Reference

All commands support `--format json` (or `-f json`) for machine-readable output. The default format is `table`, which uses Rich for human-readable tables.

## `autobio list`

List available tools.

```bash
# All tools
autobio list

# Filter by category
autobio list --category inverse-folding

# JSON output
autobio list -f json
```

**Options:**

| Flag | Description |
|------|-------------|
| `--category`, `-c` | Filter by tool category (`structure-prediction`, `embedding`, `inverse-folding`, `scoring`) |
| `--format`, `-f` | Output format: `table` (default) or `json` |

`autobio list` shows one row per tool, not per mode — a tool that supports multiple modes (e.g. `rosetta`) is listed once. Use `autobio info <tool>` to see its modes.

## `autobio info`

Show detailed information about a tool: its description, GPU requirements, and every mode it supports (each mode's own description, input/output schemas, default timeout, and notes). Tools with a single mode still list that one mode, which doubles as the default.

```bash
autobio info proteinmpnn
autobio info proteinmpnn -f json
```

**Arguments:**

| Argument | Description |
|----------|-------------|
| `TOOL` | Tool name (required) |

**Options:**

| Flag | Description |
|------|-------------|
| `--format`, `-f` | Output format: `table` (default) or `json` |

## `autobio run`

Execute a tool. Reads input parameters from a JSON config file, runs the tool inside its container, and prints the structured output.

```bash
# Basic usage
autobio run proteinmpnn --config design.json

# Select a non-default mode (rosetta defaults to "score"; here we run "relax" instead)
autobio run rosetta --config relax.json --mode relax

# Specify GPU and timeout
autobio run proteinmpnn --config design.json --gpu 0,1 --timeout 3600

# Save workspace for later inspection
autobio run proteinmpnn --config design.json --output-dir ./my_run

# No GPU (for tools that support CPU fallback)
autobio run proteinmpnn --config design.json --gpu none

# JSON output (useful for piping to other tools or agents)
autobio run proteinmpnn --config design.json -f json
```

**Arguments:**

| Argument | Description |
|----------|-------------|
| `TOOL` | Tool name (required) |

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--config` | *(required)* | Path to input config JSON file |
| `--mode` | Tool's default mode | Mode to run, for tools with more than one (e.g. `relax` for `rosetta`). See `autobio info <tool>` for available modes |
| `--gpu` | `auto` | GPU spec: `auto` (use tool's default), `none`, or comma-separated device IDs (e.g., `0,1`) |
| `--timeout` | Tool default | Maximum wall-clock seconds before the container is killed |
| `--output-dir` | *(temp dir)* | Persist the workspace to this directory. Without this flag, the workspace is cleaned up after the run |
| `--format`, `-f` | `table` | Output format: `table` or `json` |

Each mode may declare its own input schema, so the required `config.json` fields can differ between modes of the same tool (e.g. `freesasa`'s `bsa` mode requires `partner1`/`partner2` chain groups that its default `sasa` mode does not). Use `autobio info <tool>` to see the schema for each mode.

### Config file format

The config file contains the tool's input parameters as JSON. The exact fields depend on the tool. Use `autobio info <tool>` to see the input schema.

Example for `proteinmpnn`:

```json
{
    "structure_path": "structures/1abc.pdb",
    "num_sequences": 8,
    "temperature": 0.1,
    "chains_to_design": ["A"],
    "fixed_positions": {"A": [1, 2, 3, 45, 46]}
}
```

## `autobio result`

Inspect a previous run from its workspace directory. Reads and displays the `result.json` file, which contains status, timing, error info, and output file paths.

```bash
autobio result ./my_run
autobio result ./my_run -f json
```

This is useful for inspecting runs that were saved with `--output-dir`, or for debugging failed runs.

**Arguments:**

| Argument | Description |
|----------|-------------|
| `WORKSPACE_DIR` | Path to workspace directory (required) |

**Options:**

| Flag | Description |
|------|-------------|
| `--format`, `-f` | Output format: `table` (default) or `json` |

## `autobio pull`

Pull container images from the registry.

```bash
# Pull a single tool's image
autobio pull proteinmpnn

# Pull all registered tool images
autobio pull --all
```

**Arguments:**

| Argument | Description |
|----------|-------------|
| `TOOL` | Tool name (optional if `--all` is used) |

**Options:**

| Flag | Description |
|------|-------------|
| `--all` | Pull images for all registered tools |

## `autobio images`

List locally cached autobio container images, showing image URI, tag, size, and creation date.

```bash
autobio images
autobio images -f json
```

**Options:**

| Flag | Description |
|------|-------------|
| `--format`, `-f` | Output format: `table` (default) or `json` |
