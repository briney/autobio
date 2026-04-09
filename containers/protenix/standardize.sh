#!/usr/bin/env bash
# standardize.sh — Phase 3: convert Protenix outputs to autobio schema
set -euo pipefail
python3 /opt/tool/standardize.py --workspace "$1"
