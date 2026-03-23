#!/usr/bin/env bash
# run.sh — Phase 2: execute Boltz structure prediction
set -euo pipefail

WORKSPACE="$1"
CONFIG="$WORKSPACE/config.json"

# --- Required fields ----------------------------------------------------------
INPUT_PATH=$(jq -r '.input_path' "$CONFIG")
MODEL=$(jq -r '.model' "$CONFIG")
OUTPUT_DIR=$(jq -r '.output_dir' "$CONFIG")
CACHE_DIR=$(jq -r '.cache_dir // "/app/boltz/cache"' "$CONFIG")

# --- Build base command -------------------------------------------------------
CMD=(
    boltz predict "$INPUT_PATH"
    --model "$MODEL"
    --out_dir "$OUTPUT_DIR"
    --cache "$CACHE_DIR"
    --devices 1
)

# --- Helper: append CLI arg if present in config ------------------------------
append_if_set() {
    local config_key="$1"
    local cli_flag="$2"
    local val
    val=$(jq -r ".$config_key // empty" "$CONFIG" 2>/dev/null || true)
    if [ -n "$val" ] && [ "$val" != "null" ]; then
        CMD+=("$cli_flag" "$val")
    fi
}

# --- Helper: append boolean flag if true in config ----------------------------
append_if_true() {
    local config_key="$1"
    local cli_flag="$2"
    local val
    val=$(jq -r ".$config_key // empty" "$CONFIG" 2>/dev/null || true)
    if [ "$val" = "true" ]; then
        CMD+=("$cli_flag")
    fi
}

# --- Structure prediction options ---------------------------------------------
append_if_set diffusion_samples          --diffusion_samples
append_if_set sampling_steps             --sampling_steps
append_if_set recycling_steps            --recycling_steps
append_if_set output_format              --output_format
append_if_set seed                       --seed
append_if_set step_scale                 --step_scale
append_if_set max_parallel_samples       --max_parallel_samples
append_if_set max_msa_seqs               --max_msa_seqs
append_if_set method                     --method

# --- Affinity options ---------------------------------------------------------
append_if_set sampling_steps_affinity    --sampling_steps_affinity
append_if_set diffusion_samples_affinity --diffusion_samples_affinity

# --- Custom checkpoint paths --------------------------------------------------
append_if_set checkpoint                 --checkpoint
append_if_set affinity_checkpoint        --affinity_checkpoint

# --- MSA options --------------------------------------------------------------
append_if_true use_msa_server            --use_msa_server
append_if_set  msa_server_url            --msa_server_url
append_if_set  msa_pairing_strategy      --msa_pairing_strategy
append_if_set  api_key_header            --api_key_header
append_if_set  api_key_value             --api_key_value
append_if_true subsample_msa             --subsample_msa
append_if_set  num_subsampled_msa        --num_subsampled_msa

# --- Boolean flags ------------------------------------------------------------
append_if_true use_potentials            --use_potentials
append_if_true override                  --override
append_if_true write_full_pae            --write_full_pae
append_if_true write_full_pde            --write_full_pde
append_if_true write_embeddings          --write_embeddings

echo "[boltz] Running: ${CMD[*]}"
"${CMD[@]}" 2>&1 | tee "$WORKSPACE/logs/tool.log"
