#!/usr/bin/env python3
"""Standardize ESMFold outputs into autobio schema format.

Reads the PDB and metadata from outputs/raw/ and writes result_data.json
plus the structure file to outputs/standardized/.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def standardize(workspace: Path) -> None:
    raw_dir = workspace / "outputs" / "raw"
    std_dir = workspace / "outputs" / "standardized"

    # Read metadata written by run_esmfold.py
    metadata_path = raw_dir / "metadata.json"
    if not metadata_path.exists():
        raise RuntimeError(
            f"metadata.json not found in {raw_dir}. "
            f"Files present: {[f.name for f in raw_dir.iterdir()]}"
        )
    metadata = json.loads(metadata_path.read_text())

    # Copy PDB to standardized/
    raw_pdb = Path(metadata["pdb_path"])
    std_pdb = std_dir / "prediction.pdb"
    shutil.copy2(raw_pdb, std_pdb)

    # Build result_data.json conforming to StructurePredictionOutput schema
    result_data = {
        "structures": [
            {
                "model_rank": 1,
                "structure_path": "/workspace/outputs/standardized/prediction.pdb",
                "plddt_per_residue": metadata["plddt_per_residue"],
                "plddt_mean": metadata["plddt_mean"],
                "ptm": metadata["ptm"],
                "iptm": None,
                "chain_mapping": None,
            }
        ],
        "confidence": {
            "best_plddt_mean": metadata["plddt_mean"],
            "best_ptm": metadata["ptm"],
            "best_iptm": None,
        },
    }

    (std_dir / "result_data.json").write_text(json.dumps(result_data, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    args = parser.parse_args()
    standardize(args.workspace)
