#!/usr/bin/env bash
set -euo pipefail

WORKSPACE="$1"

python3 /opt/tool/standardize.py --workspace "$WORKSPACE"
