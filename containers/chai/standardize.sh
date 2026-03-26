#!/usr/bin/env bash
# standardize.sh — Phase 3: transform Chai-1 outputs into autobio schema format
set -euo pipefail

python3 /opt/tool/standardize.py --workspace "$1"
