#!/usr/bin/env bash
# validate_config.sh — Phase 1: verify config.json for ANTIPASTI
set -euo pipefail

CONFIG_FILE="$1"
ERRORS=0

error() {
    echo "ERROR: $1" >&2
    ERRORS=$((ERRORS + 1))
}

# Required fields
for field in pdb_path heavy_chain light_chain antigen_chains checkpoint_path antipasti_dir; do
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

# Validate chain IDs are non-empty strings
HEAVY=$(jq -r '.heavy_chain // empty' "$CONFIG_FILE" 2>/dev/null || true)
if [ -z "$HEAVY" ] || [ "$HEAVY" = "null" ]; then
    error "heavy_chain must be a non-empty string"
fi

LIGHT=$(jq -r '.light_chain // empty' "$CONFIG_FILE" 2>/dev/null || true)
if [ -z "$LIGHT" ] || [ "$LIGHT" = "null" ]; then
    error "light_chain must be a non-empty string"
fi

# Validate antigen_chains is a non-empty JSON array
AG_COUNT=$(jq -r '.antigen_chains | length' "$CONFIG_FILE" 2>/dev/null || echo 0)
if [ "$AG_COUNT" -eq 0 ]; then
    error "antigen_chains must be a non-empty JSON array of chain IDs"
fi

# Validate ANTIPASTI directory exists
ANTIPASTI_DIR=$(jq -r '.antipasti_dir // empty' "$CONFIG_FILE" 2>/dev/null || true)
if [ -n "$ANTIPASTI_DIR" ] && [ "$ANTIPASTI_DIR" != "null" ]; then
    if [ ! -d "$ANTIPASTI_DIR" ]; then
        error "ANTIPASTI directory not found: $ANTIPASTI_DIR"
    fi
fi

if [ "$ERRORS" -gt 0 ]; then
    echo "Config validation failed with $ERRORS error(s)." >&2
    exit 1
fi

echo "Config validation passed."
exit 0
