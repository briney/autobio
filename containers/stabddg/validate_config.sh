#!/usr/bin/env bash
# validate_config.sh — Phase 1: verify config.json for StaB-ddG
set -euo pipefail

CONFIG_FILE="$1"
ERRORS=0

error() {
    echo "ERROR: $1" >&2
    ERRORS=$((ERRORS + 1))
}

# Required fields
for field in pdb_path mutations chains checkpoint_path; do
    if ! jq -e ".$field" "$CONFIG_FILE" > /dev/null 2>&1; then
        error "config.json missing required field '$field'"
    fi
done

# Validate PDB file exists
PDB_PATH=$(jq -r '.pdb_path // empty' "$CONFIG_FILE" 2>/dev/null || true)
if [ -n "$PDB_PATH" ] && [ "$PDB_PATH" != "null" ]; then
    if [ ! -f "$PDB_PATH" ]; then
        error "PDB file not found: $PDB_PATH"
    fi
fi

# Validate checkpoint file exists
CKPT_PATH=$(jq -r '.checkpoint_path // empty' "$CONFIG_FILE" 2>/dev/null || true)
if [ -n "$CKPT_PATH" ] && [ "$CKPT_PATH" != "null" ]; then
    if [ ! -f "$CKPT_PATH" ]; then
        error "Checkpoint file not found: $CKPT_PATH"
    fi
fi

# Validate chains format (must contain exactly one underscore)
CHAINS=$(jq -r '.chains // empty' "$CONFIG_FILE" 2>/dev/null || true)
if [ -n "$CHAINS" ] && [ "$CHAINS" != "null" ]; then
    UNDERSCORE_COUNT=$(echo "$CHAINS" | tr -cd '_' | wc -c)
    if [ "$UNDERSCORE_COUNT" -ne 1 ]; then
        error "chains must be in 'binder1_binder2' format (exactly one underscore), got '$CHAINS'"
    fi
fi

# Validate mutations is non-empty
MUTATIONS=$(jq -r '.mutations // empty' "$CONFIG_FILE" 2>/dev/null || true)
if [ -z "$MUTATIONS" ] || [ "$MUTATIONS" = "null" ]; then
    error "mutations must be a non-empty comma-separated string"
fi

# Validate numeric parameters if present
MC_SAMPLES=$(jq -r '.mc_samples // empty' "$CONFIG_FILE" 2>/dev/null || true)
if [ -n "$MC_SAMPLES" ] && [ "$MC_SAMPLES" != "null" ]; then
    if ! echo "$MC_SAMPLES" | grep -qE '^[0-9]+$'; then
        error "mc_samples must be a positive integer, got '$MC_SAMPLES'"
    fi
fi

if [ "$ERRORS" -gt 0 ]; then
    echo "Config validation failed with $ERRORS error(s)." >&2
    exit 1
fi

echo "Config validation passed."
exit 0
