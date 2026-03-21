# Container Specification — `CONTAINER_SPEC.md`

This document defines how to create a new tool container for autobio. A container encapsulates a computational biology tool and its dependencies, exposing it through a standardized three-phase execution protocol. Containers are fully self-contained — they include the tool itself, all of its dependencies, and the logic to standardize the tool's native outputs into autobio's schema format.

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
├── base-entrypoint.sh              # shared — do not modify per-tool
├── <tool_name>/
│   ├── Dockerfile
│   ├── validate_config.sh          # Phase 1: config validation
│   ├── run.sh                      # Phase 2: tool execution
│   ├── standardize.sh              # Phase 3: output standardization
│   ├── requirements.txt            # (optional) Python deps for standardization
│   ├── standardize.py              # (optional) Python standardization script
│   └── test/
│       ├── inputs/                 # minimal test inputs for smoke tests
│       │   ├── config.json
│       │   └── sequences.fasta     # (example — varies by tool)
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

The base entrypoint requires `jq` and `bc` to be available in the container. The Dockerfile must install these (they are often already present in base images, but should be explicitly ensured):

```dockerfile
RUN apt-get update && apt-get install -y --no-install-recommends jq bc \
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

**Example:**

```bash
#!/usr/bin/env bash
set -euo pipefail

CONFIG_FILE="$1"

# Check required fields exist
if ! jq -e '.sequences' "$CONFIG_FILE" > /dev/null 2>&1; then
    echo "ERROR: config.json missing required field 'sequences'" >&2
    exit 1
fi

# Validate field values
NUM_MODELS=$(jq -r '.num_models // 1' "$CONFIG_FILE")
if [ "$NUM_MODELS" -lt 1 ] || [ "$NUM_MODELS" -gt 5 ]; then
    echo "ERROR: num_models must be between 1 and 5, got $NUM_MODELS" >&2
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
- May write additional tool-specific logs to `$WORKSPACE/logs/tool.log`.

**Example:**

```bash
#!/usr/bin/env bash
set -euo pipefail

WORKSPACE="$1"
CONFIG="$WORKSPACE/config.json"
INPUT_DIR="$WORKSPACE/inputs"
OUTPUT_DIR="$WORKSPACE/outputs/raw"

# Parse config
SEQUENCES_FILE="$INPUT_DIR/sequences.fasta"
NUM_MODELS=$(jq -r '.num_models // 1' "$CONFIG")

# Run the tool (example: hypothetical tool CLI)
proteinx predict \
    --input "$SEQUENCES_FILE" \
    --output-dir "$OUTPUT_DIR" \
    --num-models "$NUM_MODELS" \
    2>&1 | tee "$WORKSPACE/logs/tool.log"
```

### 4.3 `standardize.sh`

**Purpose:** Transform native outputs in `outputs/raw/` into the category schema format in `outputs/standardized/`. This is the key script that decouples the tool's native output format from autobio's standardized interface.

**Contract:**
- Receives one argument: the workspace path.
- Reads from `$WORKSPACE/outputs/raw/`.
- Writes to `$WORKSPACE/outputs/standardized/`.
- MUST produce a `result_data.json` file in `outputs/standardized/` conforming to the category's output schema (see `SCHEMA_SPEC.md`).
- MUST copy or convert any referenced files (PDB structures, embeddings, etc.) into `outputs/standardized/`.
- Exit code 0 = success. Non-zero = failure.

**Example:**

```bash
#!/usr/bin/env bash
set -euo pipefail

WORKSPACE="$1"
RAW_DIR="$WORKSPACE/outputs/raw"
STD_DIR="$WORKSPACE/outputs/standardized"

# For complex standardization, delegate to a Python script
python3 /opt/tool/standardize.py \
    --raw-dir "$RAW_DIR" \
    --std-dir "$STD_DIR"
```

Where `standardize.py` reads tool-native outputs and produces `result_data.json`:

```python
#!/usr/bin/env python3
"""
Standardize ProteinX outputs into autobio schema format.
Reads from outputs/raw/, writes to outputs/standardized/.
"""
import argparse
import json
import shutil
from pathlib import Path


def standardize(raw_dir: Path, std_dir: Path) -> None:
    # Read tool-native ranking file
    ranking = json.loads((raw_dir / "ranking.json").read_text())

    structures = []
    best_plddt = 0.0
    best_ptm = 0.0

    for i, entry in enumerate(ranking["models"]):
        # Copy PDB to standardized dir with consistent naming
        src_pdb = raw_dir / entry["pdb_filename"]
        dst_pdb = std_dir / f"model_{i + 1}.pdb"
        shutil.copy2(src_pdb, dst_pdb)

        plddt_mean = entry["metrics"]["avg_plddt"]
        ptm = entry["metrics"].get("ptm")

        structures.append({
            "model_rank": i + 1,
            "structure_path": f"outputs/standardized/model_{i + 1}.pdb",
            "plddt_mean": plddt_mean,
            "plddt_per_residue": entry["metrics"].get("per_residue_plddt"),
            "ptm": ptm,
            "iptm": entry["metrics"].get("iptm"),
            "chain_mapping": entry.get("chain_mapping"),
        })

        if plddt_mean > best_plddt:
            best_plddt = plddt_mean
        if ptm and ptm > best_ptm:
            best_ptm = ptm

    result_data = {
        "structures": structures,
        "confidence": {
            "best_plddt_mean": best_plddt,
            "best_ptm": best_ptm if best_ptm > 0 else None,
            "best_iptm": None,  # ProteinX doesn't provide this
        },
    }

    (std_dir / "result_data.json").write_text(
        json.dumps(result_data, indent=2)
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--std-dir", type=Path, required=True)
    args = parser.parse_args()
    standardize(args.raw_dir, args.std_dir)
```

---

## 5. Dockerfile

### 5.1 Structure

```dockerfile
# ─── Option A: Extend an existing official image ────────────────────────
FROM ghcr.io/official-org/tool-image:1.0.0 AS base

# ─── Option B: Build from scratch ───────────────────────────────────────
# FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04 AS base
# RUN ... install tool dependencies ...

# ─── Autobio protocol layer (same for all containers) ──────────────────
RUN apt-get update && apt-get install -y --no-install-recommends jq bc \
    && rm -rf /var/lib/apt/lists/*

# Copy shared entrypoint
COPY ../base-entrypoint.sh /opt/autobio/base-entrypoint.sh
RUN chmod +x /opt/autobio/base-entrypoint.sh

# Copy tool-specific hook scripts
COPY validate_config.sh /opt/tool/validate_config.sh
COPY run.sh             /opt/tool/run.sh
COPY standardize.sh     /opt/tool/standardize.sh
COPY standardize.py     /opt/tool/standardize.py
RUN chmod +x /opt/tool/*.sh

ENTRYPOINT ["/opt/autobio/base-entrypoint.sh"]
```

### 5.2 Build Context Considerations

The Dockerfile's build context is `containers/<tool_name>/`. Since `base-entrypoint.sh` lives one level up in `containers/`, there are two options:

1. **Build from the `containers/` root** with `-f`:
   ```bash
   docker build -f containers/proteinx/Dockerfile containers/
   ```

2. **Copy `base-entrypoint.sh` into each tool directory during CI** (simpler Dockerfiles, no context tricks).

The CI pipeline should use whichever approach is cleaner. Document the chosen convention in the repo's Makefile or CI config.

### 5.3 Image Tagging

Images are tagged as `<tool_name>:<upstream_version>`:
- `autobio-alphafold:2.3.2`
- `autobio-esm2:2.0.0`
- `autobio-ligandmpnn:1.0.0`

The full URI is: `ghcr.io/briney/autobio-<tool_name>:<version>`

When the autobio wrapper changes but the upstream tool version doesn't, append a build number: `autobio-alphafold:2.3.2-build3`.

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
cp containers/proteinx/test/inputs/* /tmp/test_ws/inputs/
cp containers/proteinx/test/inputs/config.json /tmp/test_ws/

# Run the container
docker run --rm \
    -v /tmp/test_ws:/workspace \
    --gpus '"device=0"' \
    ghcr.io/briney/autobio-proteinx:1.0.0

# Check results
cat /tmp/test_ws/result.json
diff <(jq -S . /tmp/test_ws/outputs/standardized/result_data.json) \
     <(jq -S . containers/proteinx/test/expected_outputs/result_data.json)
```

### 7.3 Standardization-Only Testing

The standardization phase can be tested without running the tool by providing pre-computed raw outputs:

```bash
# Populate outputs/raw/ with known tool outputs (from a previous run)
mkdir -p /tmp/std_test/{outputs/raw,outputs/standardized,logs}
cp /path/to/saved/raw_outputs/* /tmp/std_test/outputs/raw/

# Run only standardize.sh inside the container
docker run --rm \
    -v /tmp/std_test:/workspace \
    --entrypoint /opt/tool/standardize.sh \
    ghcr.io/briney/autobio-proteinx:1.0.0 \
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
- Print specific, actionable error messages to stderr:
  ```
  ERROR: config.json missing required field 'sequences'
  ERROR: num_models must be between 1 and 5, got 12
  ERROR: input file 'inputs/template.pdb' not found
  ```

### 8.2 In `run.sh`

- Let the tool's native error messages propagate (they go to stderr → `stderr.log`).
- For known failure modes, add contextual messages:
  ```bash
  if ! command -v proteinx &>/dev/null; then
      echo "ERROR: proteinx binary not found in PATH" >&2
      exit 127
  fi
  ```
- Use `tee` to capture tool output to `logs/tool.log` while still showing it in stdout/stderr.

### 8.3 In `standardize.sh`

- Verify that expected raw output files exist before attempting to parse them:
  ```python
  ranking_file = raw_dir / "ranking.json"
  if not ranking_file.exists():
      print(f"ERROR: Expected {ranking_file} not found. "
            f"Raw output files: {list(raw_dir.iterdir())}", file=sys.stderr)
      sys.exit(1)
  ```
- If the tool produced partial outputs, standardize what's available and note the incompleteness in `result_data.json`.

---

## 9. Checklist for New Containers

- [ ] Directory created at `containers/<tool_name>/`
- [ ] `Dockerfile` builds successfully
- [ ] `base-entrypoint.sh` is copied and set as `ENTRYPOINT`
- [ ] `jq` and `bc` are installed in the image
- [ ] `validate_config.sh` checks all required config fields
- [ ] `run.sh` reads from `config.json` and `inputs/`, writes to `outputs/raw/`
- [ ] `standardize.sh` reads from `outputs/raw/`, writes to `outputs/standardized/`
- [ ] `standardize.sh` produces `outputs/standardized/result_data.json` conforming to the category schema
- [ ] All hook scripts are executable (`chmod +x`)
- [ ] `test/inputs/` contains minimal valid inputs
- [ ] `test/expected_outputs/` contains golden `result_data.json`
- [ ] Standalone container test passes (§7.2)
- [ ] Standardization-only test passes (§7.3)
- [ ] Container builds in CI (Tier 2 test)
- [ ] Smoke test passes (Tier 3)
- [ ] Image tagged following convention: `autobio-<tool>:<upstream_version>`
- [ ] Corresponding host-side runner exists (see `TOOL_SPEC.md`)
