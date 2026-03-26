#!/usr/bin/env bash
# validate_config.sh — Phase 1: verify config.json for OpenFold3
#
# Catches config errors before the slow model loads.
set -euo pipefail

CONFIG_FILE="$1"
ERRORS=0

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

error() {
    echo "ERROR: $1" >&2
    ERRORS=$((ERRORS + 1))
}

# ---------------------------------------------------------------------------
# Required fields
# ---------------------------------------------------------------------------

for field in query_json_path output_dir; do
    if ! jq -e ".$field" "$CONFIG_FILE" > /dev/null 2>&1; then
        error "config.json missing required field '$field'"
    fi
done

# ---------------------------------------------------------------------------
# File existence: query_json_path
# ---------------------------------------------------------------------------

QUERY_JSON_PATH=$(jq -r '.query_json_path // empty' "$CONFIG_FILE" 2>/dev/null || true)
if [ -n "$QUERY_JSON_PATH" ] && [ "$QUERY_JSON_PATH" != "null" ]; then
    if [ ! -f "$QUERY_JSON_PATH" ]; then
        error "query JSON file not found: $QUERY_JSON_PATH"
    fi
fi

# ---------------------------------------------------------------------------
# Optional file existence: checkpoint_path
# ---------------------------------------------------------------------------

CKPT_PATH=$(jq -r '.checkpoint_path // empty' "$CONFIG_FILE" 2>/dev/null || true)
if [ -n "$CKPT_PATH" ] && [ "$CKPT_PATH" != "null" ]; then
    if [ ! -f "$CKPT_PATH" ]; then
        error "checkpoint file not found: $CKPT_PATH"
    fi
fi

# ---------------------------------------------------------------------------
# Positive integers
# ---------------------------------------------------------------------------

for numfield in num_diffusion_samples num_model_seeds seed; do
    VAL=$(jq -r ".$numfield // empty" "$CONFIG_FILE" 2>/dev/null || true)
    if [ -n "$VAL" ] && [ "$VAL" != "null" ]; then
        if ! [[ "$VAL" =~ ^[0-9]+$ ]] || [ "$VAL" -lt 1 ]; then
            error "$numfield must be a positive integer, got '$VAL'"
        fi
    fi
done

# ---------------------------------------------------------------------------
# Booleans
# ---------------------------------------------------------------------------

for bool_field in use_msa_server use_templates pae_enabled low_memory; do
    VAL=$(jq -r ".$bool_field // empty" "$CONFIG_FILE" 2>/dev/null || true)
    if [ -n "$VAL" ] && [ "$VAL" != "null" ] && [ "$VAL" != "true" ] && [ "$VAL" != "false" ]; then
        error "$bool_field must be a boolean, got '$VAL'"
    fi
done

# ---------------------------------------------------------------------------
# Enum: output_format
# ---------------------------------------------------------------------------

OUTPUT_FMT=$(jq -r '.output_format // empty' "$CONFIG_FILE" 2>/dev/null || true)
if [ -n "$OUTPUT_FMT" ] && [ "$OUTPUT_FMT" != "null" ]; then
    if [ "$OUTPUT_FMT" != "cif" ] && [ "$OUTPUT_FMT" != "pdb" ]; then
        error "output_format must be 'cif' or 'pdb', got '$OUTPUT_FMT'"
    fi
fi

# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

if [ "$ERRORS" -gt 0 ]; then
    echo "Config validation failed with $ERRORS error(s)." >&2
    exit 1
fi

echo "Config validation passed."
exit 0
