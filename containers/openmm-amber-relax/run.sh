#!/usr/bin/env bash
# run.sh — Phase 2: relax a structure with OpenMM Amber force field
set -euo pipefail
python3 /opt/tool/run.py "$1" 2>&1 | tee "$1/logs/tool.log"
