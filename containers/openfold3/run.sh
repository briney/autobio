#!/usr/bin/env bash
# run.sh — Phase 2: execute OpenFold3 structure prediction
#
# Delegates to a Python wrapper because the OpenFold3 CLI does not expose
# all inference parameters (MSA server URL, PAE settings, etc.).  The Python
# API (InferenceExperimentRunner) provides full control.
set -euo pipefail

WORKSPACE="$1"

echo "[openfold3] Running inference via Python API..."
python3 /opt/tool/run_openfold3.py --workspace "$WORKSPACE" 2>&1 | tee "$WORKSPACE/logs/tool.log"
