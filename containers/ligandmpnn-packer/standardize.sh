#!/usr/bin/env bash
# standardize.sh — Phase 3: convert packing outputs to standardized format
set -euo pipefail
python3 /opt/tool/standardize.py --workspace "$1"
