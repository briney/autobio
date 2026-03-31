#!/usr/bin/env bash
# validate_config.sh — Phase 1: verify config.json for BA-ddG
set -euo pipefail

CONFIG_FILE="$1"
ERRORS=0

error() {
    echo "ERROR: $1" >&2
    ERRORS=$((ERRORS + 1))
}

# Required fields
for field in pdb_path mutations chains mpnn_checkpoint_path ddg_checkpoint_path; do
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

# Validate MPNN checkpoint file exists
MPNN_CKPT=$(jq -r '.mpnn_checkpoint_path // empty' "$CONFIG_FILE" 2>/dev/null || true)
if [ -n "$MPNN_CKPT" ] && [ "$MPNN_CKPT" != "null" ]; then
    if [ ! -f "$MPNN_CKPT" ]; then
        error "MPNN checkpoint file not found: $MPNN_CKPT"
    fi
fi

# Validate DDG checkpoint file exists
DDG_CKPT=$(jq -r '.ddg_checkpoint_path // empty' "$CONFIG_FILE" 2>/dev/null || true)
if [ -n "$DDG_CKPT" ] && [ "$DDG_CKPT" != "null" ]; then
    if [ ! -f "$DDG_CKPT" ]; then
        error "DDG checkpoint file not found: $DDG_CKPT"
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

# Validate n_folds if present (must be 1-3)
N_FOLDS=$(jq -r '.n_folds // empty' "$CONFIG_FILE" 2>/dev/null || true)
if [ -n "$N_FOLDS" ] && [ "$N_FOLDS" != "null" ]; then
    if ! echo "$N_FOLDS" | grep -qE '^[1-3]$'; then
        error "n_folds must be 1, 2, or 3, got '$N_FOLDS'"
    fi
fi

if [ "$ERRORS" -gt 0 ]; then
    echo "Config validation failed with $ERRORS error(s)." >&2
    exit 1
fi

echo "Config validation passed."
exit 0
