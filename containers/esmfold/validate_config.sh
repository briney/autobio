#!/usr/bin/env bash
# validate_config.sh — Phase 1: verify config.json for ESMFold
set -euo pipefail

CONFIG_FILE="$1"
ERRORS=0

error() {
    echo "ERROR: $1" >&2
    ERRORS=$((ERRORS + 1))
}

# ---------------------------------------------------------------------------
# Required fields
# ---------------------------------------------------------------------------

for field in model_name input_fasta output_dir hf_cache; do
    if ! jq -e ".$field" "$CONFIG_FILE" > /dev/null 2>&1; then
        error "config.json missing required field '$field'"
    fi
done

# ---------------------------------------------------------------------------
# Validate model_name
# ---------------------------------------------------------------------------

MODEL_NAME=$(jq -r '.model_name // empty' "$CONFIG_FILE" 2>/dev/null || true)
if [ -n "$MODEL_NAME" ] && [ "$MODEL_NAME" != "null" ]; then
    if [ "$MODEL_NAME" != "facebook/esmfold_v1" ]; then
        error "Unknown model_name: $MODEL_NAME (expected 'facebook/esmfold_v1')"
    fi

    # Verify model exists in HF cache
    HF_CACHE=$(jq -r '.hf_cache // "/app/esmfold/hf_cache"' "$CONFIG_FILE")
    CACHE_DIR_NAME=$(echo "$MODEL_NAME" | sed 's|/|--|g')
    EXPECTED_DIR="$HF_CACHE/hub/models--$CACHE_DIR_NAME"
    if [ ! -d "$EXPECTED_DIR" ]; then
        error "Model not found in HF cache: $MODEL_NAME (expected $EXPECTED_DIR)"
    fi
fi

# ---------------------------------------------------------------------------
# File existence: input_fasta
# ---------------------------------------------------------------------------

INPUT_FASTA=$(jq -r '.input_fasta // empty' "$CONFIG_FILE" 2>/dev/null || true)
if [ -n "$INPUT_FASTA" ] && [ "$INPUT_FASTA" != "null" ]; then
    if [ ! -f "$INPUT_FASTA" ]; then
        error "FASTA file not found: $INPUT_FASTA"
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
