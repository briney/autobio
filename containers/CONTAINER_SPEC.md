# Container Specification

This document defines how to create a new tool container for autobio. A container encapsulates a computational biology tool and its dependencies, exposing it through a standardized three-phase execution protocol.

Containers are **fully self-contained**: they include the tool, all runtime dependencies, model weights/checkpoints, and the logic to standardize the tool's native outputs into autobio's schema format. The host system provides only Docker, a GPU, and the workspace directory — nothing else.

---

## 1. Purpose

Each tool container is responsible for:
1. **Validating** the configuration provided by the host.
2. **Executing** the tool on the provided inputs.
3. **Standardizing** the tool's native outputs into the category schema format.
4. **Reporting** the result status, timing, and any errors.

Containers are language-agnostic. The entrypoint protocol is shell-based. The tool itself, the validation logic, and the standardization logic can be written in any language the tool requires (Python 2, Python 3, R, C++, etc.).

---

## 2. Directory Layout

Each tool has a build context directory under `containers/`:

```
containers/
├── CONTAINER_SPEC.md               # this document
├── base-entrypoint.sh              # shared — do not modify per-tool
├── <tool_name>/
│   ├── Dockerfile
│   ├── validate_config.sh          # Phase 1: config validation
│   ├── run.sh                      # Phase 2: tool execution
│   ├── standardize.sh              # Phase 3: output standardization
│   ├── standardize.py              # (optional) Python standardization script
│   └── test/
│       ├── inputs/                 # minimal test inputs for smoke tests
│       │   ├── config.json
│       │   └── structure.pdb       # (example — varies by tool)
│       └── expected_outputs/       # golden standardized outputs
│           └── result_data.json
```

---

## 3. The Shared Entrypoint

`base-entrypoint.sh` is copied into every container and serves as the `ENTRYPOINT`. It implements the execution protocol and calls three tool-specific hook scripts. **Do not modify this file per-tool.**

```bash
#!/usr/bin/env bash
#
# base-entrypoint.sh — Autobio container entrypoint protocol
#
# This script is shared across all tool containers. Do not modify.
# Tool-specific logic goes in validate_config.sh, run.sh, and standardize.sh.
#
set -Euo pipefail

WORKSPACE="/workspace"
LOGS_DIR="$WORKSPACE/logs"
RESULT_FILE="$WORKSPACE/result.json"
PHASE_FILE="$LOGS_DIR/phase.json"

START_EPOCH=$(date +%s%N)

# ─── Helper: write result.json ──────────────────────────────────────────
write_result() {
    local status="$1"
    local phase="$2"
    local error_message="${3:-}"
    local exit_code="${4:-0}"
    local completed="${5:-0}"
    local total="${6:-1}"

    END_EPOCH=$(date +%s%N)
    WALL_TIME=$(echo "scale=3; ($END_EPOCH - $START_EPOCH) / 1000000000" | bc)

    # Collect output file lists
    local raw_files="[]"
    local std_files="[]"
    if [ -d "$WORKSPACE/outputs/raw" ]; then
        raw_files=$(find "$WORKSPACE/outputs/raw" -type f \
            | sort \
            | jq -R . \
            | jq -s .)
    fi
    if [ -d "$WORKSPACE/outputs/standardized" ]; then
        std_files=$(find "$WORKSPACE/outputs/standardized" -type f \
            | sort \
            | jq -R . \
            | jq -s .)
    fi

    cat > "$RESULT_FILE" <<EOF
{
    "status": "$status",
    "exit_code": $exit_code,
    "phase": "$phase",
    "error_type": $([ -n "$error_message" ] && echo '"runtime"' || echo 'null'),
    "error_message": $([ -n "$error_message" ] && echo "\"$error_message\"" || echo 'null'),
    "wall_time_seconds": $WALL_TIME,
    "completed": $completed,
    "total": $total,
    "outputs": {
        "raw_files": $raw_files,
        "standardized_files": $std_files
    }
}
EOF
}

# ─── Setup ───────────────────────────────────────────────────────────────
mkdir -p "$WORKSPACE"/{outputs/raw,outputs/standardized,logs}

# Redirect stdout/stderr to log files while preserving terminal output
exec > >(tee "$LOGS_DIR/stdout.log") 2> >(tee "$LOGS_DIR/stderr.log" >&2)

# ─── Phase 1: Config validation ─────────────────────────────────────────
echo '{"phase":"setup"}' > "$PHASE_FILE"
echo "[autobio] Phase 1: Validating configuration..."
if ! /opt/tool/validate_config.sh "$WORKSPACE/config.json"; then
    echo "[autobio] ERROR: Config validation failed." >&2
    write_result "failed" "setup" "Config validation failed" "1"
    exit 1
fi

# ─── Phase 2: Tool execution ────────────────────────────────────────────
echo '{"phase":"execution"}' > "$PHASE_FILE"
echo "[autobio] Phase 2: Executing tool..."
if ! /opt/tool/run.sh "$WORKSPACE"; then
    RUN_EXIT=$?
    echo "[autobio] ERROR: Tool execution failed with exit code $RUN_EXIT." >&2
    write_result "failed" "execution" "Tool exited with code $RUN_EXIT" "$RUN_EXIT"
    exit $RUN_EXIT
fi

# ─── Phase 3: Output standardization ────────────────────────────────────
echo '{"phase":"standardization"}' > "$PHASE_FILE"
echo "[autobio] Phase 3: Standardizing outputs..."
if ! /opt/tool/standardize.sh "$WORKSPACE"; then
    echo "[autobio] ERROR: Output standardization failed." >&2
    echo "[autobio] Raw outputs are preserved in outputs/raw/." >&2
    write_result "failed" "standardization" "Output standardization failed" "1"
    exit 1
fi

# ─── Success ─────────────────────────────────────────────────────────────
echo "[autobio] Complete."
write_result "success" "complete" "" "0" "1" "1"
exit 0
```

### 3.1 Dependencies of the Base Entrypoint

The base entrypoint requires `jq` and `bc`. Many base images lack these, so the Dockerfile must install them explicitly. Also install `wget` if the Dockerfile downloads model weights at build time (see §5.2):

```dockerfile
RUN apt-get update && apt-get install -y --no-install-recommends jq bc wget \
    && rm -rf /var/lib/apt/lists/*
```

---

## 4. Tool-Specific Hook Scripts

### 4.1 `validate_config.sh`

**Purpose:** Validate that `config.json` contains all required fields and that values are within acceptable ranges. Fail fast with a clear error message rather than letting the tool crash mid-execution.

**Contract:**
- Receives one argument: the path to `config.json`.
- Exit code 0 = valid. Non-zero = invalid.
- Print clear error messages to stderr on validation failure.
- Validate that referenced files (structures, checkpoints) exist at the paths given in the config.

**Example** (from `containers/mpnn/validate_config.sh`):

```bash
#!/usr/bin/env bash
set -euo pipefail

CONFIG_FILE="$1"

# Check required fields exist
for field in structure_path model_type checkpoint_path; do
    if ! jq -e ".$field" "$CONFIG_FILE" > /dev/null 2>&1; then
        echo "ERROR: config.json missing required field '$field'" >&2
        exit 1
    fi
done

# Validate enum field
MODEL_TYPE=$(jq -r '.model_type' "$CONFIG_FILE")
if [[ "$MODEL_TYPE" != "protein_mpnn" && "$MODEL_TYPE" != "ligand_mpnn" ]]; then
    echo "ERROR: model_type must be 'protein_mpnn' or 'ligand_mpnn', got '$MODEL_TYPE'" >&2
    exit 1
fi

# Validate referenced files exist
STRUCTURE_PATH=$(jq -r '.structure_path' "$CONFIG_FILE")
if [ ! -f "$STRUCTURE_PATH" ]; then
    echo "ERROR: structure file not found: $STRUCTURE_PATH" >&2
    exit 1
fi

echo "Config validation passed."
exit 0
```

### 4.2 `run.sh`

**Purpose:** Execute the tool. Read configuration from `config.json` and input files from `inputs/`. Write native outputs to `outputs/raw/`.

**Contract:**
- Receives one argument: the workspace path (`/workspace`).
- Must read parameters from `$WORKSPACE/config.json`.
- Must read input files from `$WORKSPACE/inputs/`.
- Must write all outputs to `$WORKSPACE/outputs/raw/`.
- Exit code 0 = success. Non-zero = failure.
- Should capture tool output to `$WORKSPACE/logs/tool.log` via `tee`.

**Example:**

```bash
#!/usr/bin/env bash
set -euo pipefail

WORKSPACE="$1"
CONFIG="$WORKSPACE/config.json"
OUTPUT_DIR="$WORKSPACE/outputs/raw"

# Read parameters from config.json
STRUCTURE_PATH=$(jq -r '.structure_path' "$CONFIG")
CHECKPOINT_PATH=$(jq -r '.checkpoint_path' "$CONFIG")
TEMPERATURE=$(jq -r '.temperature // 0.1' "$CONFIG")

# Build and run the tool command
CMD=(
    my_tool predict
    --structure "$STRUCTURE_PATH"
    --checkpoint "$CHECKPOINT_PATH"
    --output-dir "$OUTPUT_DIR"
    --temperature "$TEMPERATURE"
)

echo "[tool] Running: ${CMD[*]}"
"${CMD[@]}" 2>&1 | tee "$WORKSPACE/logs/tool.log"
```

**Pattern: optional config fields.** Use `jq -r '.field // empty'` with a guard to conditionally add CLI flags:

```bash
SEED=$(jq -r '.seed // empty' "$CONFIG" 2>/dev/null || true)
if [ -n "$SEED" ] && [ "$SEED" != "null" ]; then
    CMD+=(--seed "$SEED")
fi
```

### 4.3 `standardize.sh`

**Purpose:** Transform native outputs in `outputs/raw/` into the category schema format in `outputs/standardized/`. This is the key script that decouples the tool's native output format from autobio's standardized interface.

**Contract:**
- Receives one argument: the workspace path.
- Reads from `$WORKSPACE/outputs/raw/` and `$WORKSPACE/config.json` (for input structure metadata).
- Writes to `$WORKSPACE/outputs/standardized/`.
- MUST produce a `result_data.json` file in `outputs/standardized/` conforming to the category's output schema.
- MUST copy or convert any referenced files (PDB structures, embeddings, etc.) into `outputs/standardized/`.
- Exit code 0 = success. Non-zero = failure.

**Preferred pattern:** delegate to a Python script that receives the full workspace path. This gives the standardization script access to both the raw outputs and the input config (which often contains metadata needed for correct parsing, such as chain IDs or sequence lengths).

```bash
#!/usr/bin/env bash
set -euo pipefail
python3 /opt/tool/standardize.py --workspace "$1"
```

```python
#!/usr/bin/env python3
"""Standardize tool outputs into autobio schema format."""

import argparse
import json
from pathlib import Path


def standardize(workspace: Path) -> None:
    raw_dir = workspace / "outputs" / "raw"
    std_dir = workspace / "outputs" / "standardized"
    config = json.loads((workspace / "config.json").read_text())

    # Read tool-native outputs from raw_dir...
    # Use config for context (input structure path, chain info, etc.)...
    # Write result_data.json to std_dir...

    result_data = { ... }
    (std_dir / "result_data.json").write_text(json.dumps(result_data, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    args = parser.parse_args()
    standardize(args.workspace)
```

---

## 5. Dockerfile

### 5.1 Build Context

The build context is `containers/` (the parent directory, not the tool subdirectory). This is required because `base-entrypoint.sh` lives at `containers/base-entrypoint.sh` and must be accessible during the build. All Dockerfiles use `-f` to specify the path:

```bash
docker build -f containers/<tool_name>/Dockerfile containers/ -t autobio-<tool_name>:<version>
```

COPY paths in the Dockerfile are relative to the build context (`containers/`):

```dockerfile
# Shared entrypoint — from containers/base-entrypoint.sh
COPY base-entrypoint.sh /opt/autobio/base-entrypoint.sh

# Tool-specific scripts — from containers/<tool_name>/
COPY <tool_name>/validate_config.sh /opt/tool/validate_config.sh
COPY <tool_name>/run.sh             /opt/tool/run.sh
COPY <tool_name>/standardize.sh     /opt/tool/standardize.sh
COPY <tool_name>/standardize.py     /opt/tool/standardize.py
```

Every Dockerfile should include the build command in a header comment for easy reference.

### 5.2 Self-Contained Images: Baking in Model Weights

Containers must be fully self-contained. Model weights and checkpoints are downloaded at image build time, not mounted from the host at runtime. This ensures:

- **No host state dependency** — the container works identically on any machine with Docker and a GPU.
- **Reproducibility** — the exact weights used are pinned in the image layer.
- **Simpler orchestration** — no need to manage weight caches, download scripts, or volume mounts for models.

Download weights in a dedicated `RUN` layer so they are cached across rebuilds:

```dockerfile
# Download model checkpoints at build time
RUN mkdir -p /app/checkpoints \
    && wget -q -O /app/checkpoints/model_v1.pt \
       https://example.com/weights/model_v1.pt \
    && wget -q -O /app/checkpoints/model_v2.pt \
       https://example.com/weights/model_v2.pt
```

The `config.json` written by the host-side runner should reference these baked-in paths. The runner knows the container-internal checkpoint path (e.g., `/app/checkpoints/model_v1.pt`) and writes it into the config. See `TOOL_SPEC.md` for the runner side of this contract.

If the upstream project provides an official Docker image with weights already included, prefer extending that image over downloading separately.

### 5.3 Full Dockerfile Structure

```dockerfile
# autobio-<tool_name> — <brief description>
#
# Build context: containers/
#   docker build -f containers/<tool_name>/Dockerfile containers/ -t autobio-<tool_name>:<version>

# ─── Base image ──────────────────────────────────────────────────────────
# Option A: extend an existing official image (preferred when available)
FROM ghcr.io/official-org/tool-image:1.0.0

# Option B: build from scratch
# FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04
# RUN ... install tool and dependencies ...

# ─── Autobio protocol dependencies ──────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends jq bc wget \
    && rm -rf /var/lib/apt/lists/*

# ─── Model weights (baked into image) ───────────────────────────────────
RUN mkdir -p /app/checkpoints \
    && wget -q -O /app/checkpoints/model_v1.pt \
       https://example.com/weights/model_v1.pt

# ─── Shared entrypoint ──────────────────────────────────────────────────
COPY base-entrypoint.sh /opt/autobio/base-entrypoint.sh
RUN chmod +x /opt/autobio/base-entrypoint.sh

# ─── Tool-specific hook scripts ─────────────────────────────────────────
COPY <tool_name>/validate_config.sh /opt/tool/validate_config.sh
COPY <tool_name>/run.sh             /opt/tool/run.sh
COPY <tool_name>/standardize.sh     /opt/tool/standardize.sh
COPY <tool_name>/standardize.py     /opt/tool/standardize.py
RUN chmod +x /opt/tool/*.sh

ENTRYPOINT ["/opt/autobio/base-entrypoint.sh"]
```

### 5.4 Image Tagging

Images are tagged as `autobio-<tool_name>:<version>`:
- `autobio-mpnn:1.0.0`
- `autobio-alphafold:2.3.2`
- `autobio-esm2:2.0.0`

The full registry URI is: `ghcr.io/briney/autobio-<tool_name>:<version>`

When the autobio wrapper changes but the upstream tool version doesn't, append a build number: `autobio-alphafold:2.3.2-build3`.

Note: a single image can serve multiple autobio tools. For example, `autobio-mpnn:1.0.0` serves both `proteinmpnn` and `ligandmpnn` — the `config.json` written by the runner determines which model type and checkpoint are used.

---

## 6. Batch Execution in Containers

For tools that support batch processing, `run.sh` should:

1. Detect the number of inputs in `inputs/` or the batch parameters in `config.json`.
2. Process inputs iteratively or in batches.
3. Write outputs progressively — individual files or JSONL append, not all-at-once.
4. Update `result.json` incrementally with `completed` count if practical.

**Progressive output pattern:**

```bash
#!/usr/bin/env bash
# run.sh for a batch-capable tool
set -euo pipefail

WORKSPACE="$1"
INPUT_FASTA="$WORKSPACE/inputs/sequences.fasta"
OUTPUT_DIR="$WORKSPACE/outputs/raw"
BATCH_SIZE=$(jq -r '.batch_size // 32' "$WORKSPACE/config.json")

# Tool writes one output file per input sequence
my_tool embed \
    --input "$INPUT_FASTA" \
    --output-dir "$OUTPUT_DIR" \
    --batch-size "$BATCH_SIZE" \
    --save-per-sequence
```

The standardization script then collects all per-sequence outputs and assembles `result_data.json`.

---

## 7. Testing Containers

### 7.1 Test Artifacts

Each container's `test/` directory contains:

- **`test/inputs/`**: A minimal, valid workspace input set. Config with minimal parameters, smallest possible input data (short sequences, small structures). These should exercise the full pipeline in under 60 seconds.

- **`test/expected_outputs/`**: Golden standardized outputs. `result_data.json` with expected structure and approximate values. Used for both smoke tests and standalone container testing.

### 7.2 Standalone Container Testing

A container can be tested independently of the host package:

```bash
# Create a test workspace
mkdir -p /tmp/test_ws/{inputs,outputs/raw,outputs/standardized,logs}
cp containers/<tool_name>/test/inputs/* /tmp/test_ws/inputs/
cp containers/<tool_name>/test/inputs/config.json /tmp/test_ws/

# Run the container
docker run --rm \
    -v /tmp/test_ws:/workspace \
    --gpus '"device=0"' \
    autobio-<tool_name>:<version>

# Check results
cat /tmp/test_ws/result.json
diff <(jq -S . /tmp/test_ws/outputs/standardized/result_data.json) \
     <(jq -S . containers/<tool_name>/test/expected_outputs/result_data.json)
```

### 7.3 Standardization-Only Testing

The standardization phase can be tested without running the tool by providing pre-computed raw outputs:

```bash
# Populate outputs/raw/ with known tool outputs (from a previous run)
mkdir -p /tmp/std_test/{outputs/raw,outputs/standardized,logs}
cp /path/to/saved/raw_outputs/* /tmp/std_test/outputs/raw/
cp /path/to/config.json /tmp/std_test/config.json

# Run only standardize.sh inside the container
docker run --rm \
    -v /tmp/std_test:/workspace \
    --entrypoint /opt/tool/standardize.sh \
    autobio-<tool_name>:<version> \
    /workspace

# Verify standardized output
cat /tmp/std_test/outputs/standardized/result_data.json
```

This allows fast iteration on the standardization logic without re-running expensive tool computations.

---

## 8. Error Handling Guidelines

### 8.1 In `validate_config.sh`

- Check for all required config fields.
- Validate types and ranges where practical.
- Verify that referenced files (input structures, checkpoints) exist.
- Print specific, actionable error messages to stderr:
  ```
  ERROR: config.json missing required field 'structure_path'
  ERROR: model_type must be 'protein_mpnn' or 'ligand_mpnn', got 'invalid'
  ERROR: structure file not found: /workspace/inputs/missing.pdb
  ERROR: checkpoint file not found: /app/checkpoints/missing.pt
  ```

### 8.2 In `run.sh`

- Let the tool's native error messages propagate (they go to stderr -> `stderr.log`).
- For known failure modes, add contextual messages:
  ```bash
  if ! command -v my_tool &>/dev/null; then
      echo "ERROR: my_tool binary not found in PATH" >&2
      exit 127
  fi
  ```
- Use `tee` to capture tool output to `logs/tool.log` while still showing it in stdout/stderr.

### 8.3 In `standardize.sh`

- Verify that expected raw output files exist before attempting to parse them:
  ```python
  fasta_files = sorted(raw_dir.glob("*.fa")) + sorted(raw_dir.glob("*.fasta"))
  if not fasta_files:
      raise RuntimeError(
          f"No FASTA files found in {raw_dir}. "
          f"Files present: {[f.name for f in raw_dir.iterdir()]}"
      )
  ```
- If the tool produced partial outputs, standardize what's available and note the incompleteness in `result_data.json`.

---

## 9. Checklist for New Containers

- [ ] Directory created at `containers/<tool_name>/`
- [ ] `Dockerfile` builds successfully
- [ ] Build command documented in Dockerfile header comment
- [ ] `base-entrypoint.sh` is copied and set as `ENTRYPOINT`
- [ ] `jq`, `bc`, and `wget` are installed in the image
- [ ] Model weights/checkpoints are baked into the image (not mounted from host)
- [ ] `validate_config.sh` checks all required config fields and verifies referenced files exist
- [ ] `run.sh` reads from `config.json` and `inputs/`, writes to `outputs/raw/`
- [ ] `standardize.sh` reads from `outputs/raw/` (and `config.json` for context), writes to `outputs/standardized/`
- [ ] `standardize.sh` produces `outputs/standardized/result_data.json` conforming to the category schema
- [ ] All hook scripts are executable (`chmod +x`)
- [ ] `test/inputs/` contains minimal valid inputs
- [ ] `test/expected_outputs/` contains golden `result_data.json`
- [ ] Standalone container test passes (§7.2)
- [ ] Standardization-only test passes (§7.3)
- [ ] Smoke test passes with Docker + GPU
- [ ] Image tagged following convention: `autobio-<tool>:<version>`
- [ ] Corresponding host-side runner exists (see `docs/TOOL_SPEC.md`)
