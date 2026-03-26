#!/usr/bin/env bash
# run.sh — Phase 2: execute ProteinMPNN or LigandMPNN
set -euo pipefail

WORKSPACE="$1"
CONFIG="$WORKSPACE/config.json"
OUTPUT_DIR="$WORKSPACE/outputs/raw"

# --- Required fields ----------------------------------------------------------
STRUCTURE_PATH=$(jq -r '.structure_path' "$CONFIG")
CHECKPOINT_PATH=$(jq -r '.checkpoint_path' "$CONFIG")
MODEL_TYPE=$(jq -r '.model_type' "$CONFIG")

# --- Optional fields with defaults --------------------------------------------
TEMPERATURE=$(jq -r '.temperature // 0.1' "$CONFIG")
NUMBER_OF_BATCHES=$(jq -r '.number_of_batches // 1' "$CONFIG")
BATCH_SIZE=$(jq -r '.batch_size // 1' "$CONFIG")

# Build base command
CMD=(
    mpnn
    --structure_path "$STRUCTURE_PATH"
    --checkpoint_path "$CHECKPOINT_PATH"
    --model_type "$MODEL_TYPE"
    --is_legacy_weights True
    --out_directory "$OUTPUT_DIR"
    --write_fasta True
    --write_structures False
    --temperature "$TEMPERATURE"
    --number_of_batches "$NUMBER_OF_BATCHES"
    --batch_size "$BATCH_SIZE"
)

# --- Optional: seed -----------------------------------------------------------
SEED=$(jq -r '.seed // empty' "$CONFIG" 2>/dev/null || true)
if [ -n "$SEED" ] && [ "$SEED" != "null" ]; then
    CMD+=(--seed "$SEED")
fi

# --- Optional: designed_chains ------------------------------------------------
# NOTE: designed_chains, fixed_chains, fixed_residues, and designed_residues
# are mutually exclusive in the foundry CLI. The host runner ensures only one
# of these is present in config.json.
DESIGNED_CHAINS=$(jq -r '.designed_chains // empty' "$CONFIG" 2>/dev/null || true)
if [ -n "$DESIGNED_CHAINS" ] && [ "$DESIGNED_CHAINS" != "null" ]; then
    CMD+=(--designed_chains "$DESIGNED_CHAINS")
fi

# --- Optional: fixed_residues -------------------------------------------------
FIXED_RESIDUES=$(jq -r '.fixed_residues // empty' "$CONFIG" 2>/dev/null || true)
if [ -n "$FIXED_RESIDUES" ] && [ "$FIXED_RESIDUES" != "null" ]; then
    CMD+=(--fixed_residues "$FIXED_RESIDUES")
fi

# --- Optional: omit -----------------------------------------------------------
OMIT=$(jq -r '.omit // empty' "$CONFIG" 2>/dev/null || true)
if [ -n "$OMIT" ] && [ "$OMIT" != "null" ]; then
    CMD+=(--omit "$OMIT")
fi

# --- Optional: structure_noise ------------------------------------------------
STRUCTURE_NOISE=$(jq -r '.structure_noise // empty' "$CONFIG" 2>/dev/null || true)
if [ -n "$STRUCTURE_NOISE" ] && [ "$STRUCTURE_NOISE" != "null" ]; then
    CMD+=(--structure_noise "$STRUCTURE_NOISE")
fi

# --- Optional: homo_oligomer_chains -------------------------------------------
HOMO_OLIGOMER=$(jq -r '.homo_oligomer_chains // empty' "$CONFIG" 2>/dev/null || true)
if [ -n "$HOMO_OLIGOMER" ] && [ "$HOMO_OLIGOMER" != "null" ]; then
    CMD+=(--homo_oligomer_chains "$HOMO_OLIGOMER")
fi

# --- Optional: atomize_side_chains (LigandMPNN) ------------------------------
ATOMIZE=$(jq -r '.atomize_side_chains // empty' "$CONFIG" 2>/dev/null || true)
if [ -n "$ATOMIZE" ] && [ "$ATOMIZE" != "null" ]; then
    CMD+=(--atomize_side_chains "$ATOMIZE")
fi

echo "[mpnn] Running: ${CMD[*]}"
"${CMD[@]}" 2>&1 | tee "$WORKSPACE/logs/tool.log"
