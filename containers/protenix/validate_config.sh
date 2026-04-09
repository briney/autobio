#!/usr/bin/env bash
# validate_config.sh — Phase 1: verify config.json for Protenix
set -euo pipefail

CONFIG_FILE="$1"
ERRORS=0

error() {
    echo "ERROR: $1" >&2
    ERRORS=$((ERRORS + 1))
}

# --- Required fields ---
for field in input_json_path output_dir; do
    if ! jq -e ".$field" "$CONFIG_FILE" > /dev/null 2>&1; then
        error "config.json missing required field '$field'"
    fi
done

# --- File existence ---
INPUT_JSON=$(jq -r '.input_json_path // empty' "$CONFIG_FILE" 2>/dev/null || true)
if [ -n "$INPUT_JSON" ] && [ "$INPUT_JSON" != "null" ]; then
    if [ ! -f "$INPUT_JSON" ]; then
        error "input JSON file not found: $INPUT_JSON"
    fi
fi

# --- Enum: dtype ---
DTYPE=$(jq -r '.dtype // empty' "$CONFIG_FILE" 2>/dev/null || true)
if [ -n "$DTYPE" ] && [ "$DTYPE" != "null" ]; then
    if [ "$DTYPE" != "bf16" ] && [ "$DTYPE" != "fp32" ] && [ "$DTYPE" != "fp16" ]; then
        error "dtype must be 'bf16', 'fp16', or 'fp32', got '$DTYPE'"
    fi
fi

# --- Positive integers ---
for numfield in diffusion_samples diffusion_steps pairformer_cycles; do
    VAL=$(jq -r ".$numfield // empty" "$CONFIG_FILE" 2>/dev/null || true)
    if [ -n "$VAL" ] && [ "$VAL" != "null" ]; then
        if ! [[ "$VAL" =~ ^[0-9]+$ ]] || [ "$VAL" -lt 1 ]; then
            error "$numfield must be a positive integer, got '$VAL'"
        fi
    fi
done

# --- Booleans ---
for bool_field in use_msa use_template use_tfg_guidance; do
    VAL=$(jq -r ".$bool_field // empty" "$CONFIG_FILE" 2>/dev/null || true)
    if [ -n "$VAL" ] && [ "$VAL" != "null" ] && \
       [ "$VAL" != "true" ] && [ "$VAL" != "false" ]; then
        error "$bool_field must be a boolean, got '$VAL'"
    fi
done

# --- Seeds validation (comma-separated string) ---
SEEDS=$(jq -r '.seeds // empty' "$CONFIG_FILE" 2>/dev/null || true)
if [ -n "$SEEDS" ] && [ "$SEEDS" != "null" ]; then
    if ! [[ "$SEEDS" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
        error "seeds must be a comma-separated list of integers, got '$SEEDS'"
    fi
fi

if [ "$ERRORS" -gt 0 ]; then
    echo "Config validation failed with $ERRORS error(s)." >&2
    exit 1
fi

echo "Config validation passed."
exit 0
