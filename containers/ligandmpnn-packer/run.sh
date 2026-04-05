#!/usr/bin/env bash
# run.sh — Phase 2: apply mutations and pack sidechains with LigandMPNN
set -euo pipefail

WORKSPACE="$1"
CONFIG="$WORKSPACE/config.json"

# --- Step 1: Apply mutations to PDB -----------------------------------------
echo "[ligandmpnn-packer] Applying mutations..."
python3 /opt/tool/apply_mutations.py "$CONFIG" 2>&1 | tee -a "$WORKSPACE/logs/tool.log"

# --- Step 2: Run sidechain packing ------------------------------------------
echo "[ligandmpnn-packer] Running sidechain packing..."
python3 /opt/tool/pack_sidechains.py "$CONFIG" 2>&1 | tee -a "$WORKSPACE/logs/tool.log"
