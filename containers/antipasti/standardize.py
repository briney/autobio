#!/usr/bin/env python3
"""Standardize ANTIPASTI outputs into autobio BindingAffinityOutput format.

ANTIPASTI inference.py writes output.json with log10_kd and metadata. This
script reads that JSON and writes result_data.json in the standard binding
affinity schema format.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def standardize(workspace: Path) -> None:
    """Parse ANTIPASTI output JSON and write result_data.json."""
    raw_dir = workspace / "outputs" / "raw"
    std_dir = workspace / "outputs" / "standardized"
    config = json.loads((workspace / "config.json").read_text())

    output_file = raw_dir / "output.json"
    if not output_file.exists():
        raise RuntimeError(
            f"output.json not found in {raw_dir}. "
            f"Files present: {[f.name for f in raw_dir.iterdir() if f.is_file()]}"
        )

    data = json.loads(output_file.read_text())

    prediction = {
        "log10_kd": data["log10_kd"],
        "kd_molar": data.get("kd_molar"),
        "units": "log10(Kd) [M]",
        "score_breakdown": {
            "heavy_chain": config.get("heavy_chain"),
            "light_chain": config.get("light_chain"),
            "antigen_chains": config.get("antigen_chains"),
            "modes": data.get("modes", "all"),
            "checkpoint": data.get("checkpoint"),
            "pdb_id": data.get("pdb_id"),
        },
    }

    result_data = {"predictions": [prediction]}
    (std_dir / "result_data.json").write_text(json.dumps(result_data, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    args = parser.parse_args()
    standardize(args.workspace)
