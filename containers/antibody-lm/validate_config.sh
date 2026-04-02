#!/usr/bin/env bash
# validate_config.sh — Phase 1: verify config.json for antibody LM tools
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

for field in model_name model_family chain_separator input_file output_dir mode hf_cache; do
    if ! jq -e ".$field" "$CONFIG_FILE" > /dev/null 2>&1; then
        error "config.json missing required field '$field'"
    fi
done

# ---------------------------------------------------------------------------
# Validate model_name is a known antibody LM
# ---------------------------------------------------------------------------

MODEL_NAME=$(jq -r '.model_name // empty' "$CONFIG_FILE" 2>/dev/null || true)
if [ -n "$MODEL_NAME" ] && [ "$MODEL_NAME" != "null" ]; then
    case "$MODEL_NAME" in
        brineylab/CurrAb|\
        brineylab/ft-ESM|\
        brineylab/BALM-paired|\
        brineylab/BALM-unpaired)
            ;;
        *)
            error "Unknown model_name: $MODEL_NAME"
            ;;
    esac

    # Verify the model exists in the HuggingFace cache
    HF_CACHE=$(jq -r '.hf_cache // "/app/antibody-lm/hf_cache"' "$CONFIG_FILE")
    CACHE_DIR_NAME=$(echo "$MODEL_NAME" | sed 's|/|--|g')
    EXPECTED_DIR="$HF_CACHE/hub/models--$CACHE_DIR_NAME"
    if [ ! -d "$EXPECTED_DIR" ]; then
        error "Model not found in HF cache: $MODEL_NAME (expected $EXPECTED_DIR)"
    fi
fi

# ---------------------------------------------------------------------------
# Validate mode
# ---------------------------------------------------------------------------

MODE=$(jq -r '.mode // empty' "$CONFIG_FILE" 2>/dev/null || true)
if [ -n "$MODE" ] && [ "$MODE" != "null" ]; then
    case "$MODE" in
        embedding|pll) ;;
        *) error "mode must be 'embedding' or 'pll', got '$MODE'" ;;
    esac
fi

# ---------------------------------------------------------------------------
# Validate chain_separator
# ---------------------------------------------------------------------------

CHAIN_SEP=$(jq -r '.chain_separator // empty' "$CONFIG_FILE" 2>/dev/null || true)
if [ -n "$CHAIN_SEP" ] && [ "$CHAIN_SEP" != "null" ]; then
    case "$CHAIN_SEP" in
        single_cls|double_cls|sep|none) ;;
        *) error "chain_separator must be 'single_cls', 'double_cls', 'sep', or 'none', got '$CHAIN_SEP'" ;;
    esac
fi

# ---------------------------------------------------------------------------
# File existence: input_file
# ---------------------------------------------------------------------------

INPUT_FILE=$(jq -r '.input_file // empty' "$CONFIG_FILE" 2>/dev/null || true)
if [ -n "$INPUT_FILE" ] && [ "$INPUT_FILE" != "null" ]; then
    if [ ! -f "$INPUT_FILE" ]; then
        error "Input file not found: $INPUT_FILE"
    fi
fi

# ---------------------------------------------------------------------------
# Validate pooling (embedding mode only)
# ---------------------------------------------------------------------------

POOLING=$(jq -r '.pooling // empty' "$CONFIG_FILE" 2>/dev/null || true)
if [ -n "$POOLING" ] && [ "$POOLING" != "null" ]; then
    case "$POOLING" in
        mean|cls|per_residue) ;;
        *) error "pooling must be 'mean', 'cls', or 'per_residue', got '$POOLING'" ;;
    esac
fi

# ---------------------------------------------------------------------------
# Validate layer (non-negative integer or null)
# ---------------------------------------------------------------------------

LAYER=$(jq -r '.layer // empty' "$CONFIG_FILE" 2>/dev/null || true)
if [ -n "$LAYER" ] && [ "$LAYER" != "null" ]; then
    if ! [[ "$LAYER" =~ ^[0-9]+$ ]]; then
        error "layer must be a non-negative integer, got '$LAYER'"
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
