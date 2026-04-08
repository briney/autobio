#!/usr/bin/env bash
set -euo pipefail

WORKSPACE="$1"

echo "[antifold] Running via Python API..."
python3 /opt/tool/run_antifold.py --workspace "$WORKSPACE"
