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
    # bc omits leading zero for values < 1 (.003 not 0.003) — fix for JSON
    [[ "$WALL_TIME" == .* ]] && WALL_TIME="0$WALL_TIME"

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
RUN_EXIT=0
/opt/tool/run.sh "$WORKSPACE" || RUN_EXIT=$?
if [ $RUN_EXIT -ne 0 ]; then
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
