#!/usr/bin/env bash
# run.sh — Phase 2: execute Protenix structure prediction
#
# Two sub-phases:
#   1. MSA fetch (if use_msa=true and no pre-computed MSAs in input JSON)
#   2. protenix pred
set -euo pipefail

WORKSPACE="$1"
CONFIG="$WORKSPACE/config.json"

# --- Phase 2a: MSA fetch (conditional) ---
USE_MSA=$(jq -r '.use_msa // "true"' "$CONFIG")

if [ "$USE_MSA" = "true" ]; then
    INPUT_JSON_PATH=$(jq -r '.input_json_path' "$CONFIG")

    # Check if any proteinChain entries already have MSA paths
    HAS_MSA=$(python3 -c "
import json, sys
data = json.loads(open('$INPUT_JSON_PATH').read())
has_msa = any(
    'pairedMsaPath' in entity.get('proteinChain', {})
    or 'unpairedMsaPath' in entity.get('proteinChain', {})
    for job in data
    for entity in job.get('sequences', [])
    if 'proteinChain' in entity
)
print('true' if has_msa else 'false')
" 2>/dev/null || echo "false")

    if [ "$HAS_MSA" = "false" ]; then
        MSA_SERVER_URL=$(jq -r '.msa_server_url // "https://api.colabfold.com"' "$CONFIG")
        echo "[protenix] Fetching MSAs from $MSA_SERVER_URL..."
        python3 /opt/tool/fetch_msa.py \
            --workspace "$WORKSPACE" \
            --server-url "$MSA_SERVER_URL" \
            2>&1 | tee -a "$WORKSPACE/logs/tool.log"
        echo "[protenix] MSA fetch complete."
    else
        echo "[protenix] Pre-computed MSAs found, skipping fetch."
    fi
else
    echo "[protenix] MSA disabled (use_msa=false)."
fi

# --- Phase 2b: Protenix inference ---
INPUT_JSON_PATH=$(jq -r '.input_json_path' "$CONFIG")
OUTPUT_DIR=$(jq -r '.output_dir' "$CONFIG")

CMD=(
    protenix pred
    -i "$INPUT_JSON_PATH"
    -o "$OUTPUT_DIR"
)

# Model name
MODEL_NAME=$(jq -r '.model_name // "protenix_base_default_v1.0.0"' "$CONFIG")
CMD+=(-n "$MODEL_NAME")

# Seeds (comma-separated string)
SEEDS=$(jq -r '.seeds // empty' "$CONFIG" 2>/dev/null || true)
if [ -n "$SEEDS" ] && [ "$SEEDS" != "null" ]; then
    CMD+=(-s "$SEEDS")
fi

# Helper: append optional arg if set in config
append_if_set() {
    local config_key="$1"
    local cli_flag="$2"
    local val
    val=$(jq -r ".$config_key // empty" "$CONFIG" 2>/dev/null || true)
    if [ -n "$val" ] && [ "$val" != "null" ]; then
        CMD+=("$cli_flag" "$val")
    fi
}

# Protenix CLI flag mapping:
#   -c = pairformer cycles (not diffusion samples!)
#   -p = diffusion steps
#   -e = diffusion samples (number of samples per seed)
#   -d = dtype
append_if_set pairformer_cycles  "-c"
append_if_set diffusion_steps    "-p"
append_if_set diffusion_samples  "-e"
append_if_set dtype              "-d"

# Boolean flags — Protenix CLI uses Click BOOLEAN type which takes explicit
# True/False values (not --flag/--no-flag style)
append_bool() {
    local config_key="$1"
    local cli_flag="$2"
    local val
    val=$(jq -r ".$config_key // empty" "$CONFIG" 2>/dev/null || true)
    if [ -n "$val" ] && [ "$val" != "null" ]; then
        CMD+=("$cli_flag" "$val")
    fi
}

append_bool use_template    "--use_template"
append_bool use_tfg_guidance "--use_tfg_guidance"

# MSA flag — tell Protenix whether to use the MSAs we've prepared
if [ "$USE_MSA" = "true" ]; then
    CMD+=(--use_msa True)
else
    CMD+=(--use_msa False)
fi

echo "[protenix] Running: ${CMD[*]}"
"${CMD[@]}" 2>&1 | tee "$WORKSPACE/logs/tool.log"
